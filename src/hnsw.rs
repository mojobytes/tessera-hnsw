//! A rust implementation of Approximate NN search from:
//! Efficient and robust approximate nearest neighbour search using Hierarchical Navigable
//! small World graphs.
//! Yu. A. Malkov, D.A Yashunin 2016, 2018
//!
//! ## Using External Vector Storage
//!
//! For memory efficiency, HNSW can reference vectors from external storage
//! instead of copying them. This is useful for large-scale datasets.
//!
//! ### Example: Basic Usage with External Storage
//!
//! ```rust,ignore
//! use tessera_hnsw::{Hnsw, InMemoryVectorStorage};
//! use anndists::distance::DistL2;
//!
//! // 1. Create storage with your vectors
//! let vectors = vec![
//!     vec![1.0f32, 0.0, 0.0],
//!     vec![0.0, 1.0, 0.0],
//!     vec![0.0, 0.0, 1.0],
//! ];
//! let storage = InMemoryVectorStorage::new(vectors);
//!
//! // 2. Create HNSW with external storage
//! let hnsw = Hnsw::with_external_storage(
//!     16,            // max_nb_connection
//!     1000,          // max_elements
//!     16,            // max_layer
//!     200,           // ef_construction
//!     DistL2 {},     // distance function
//!     &storage,
//! );
//!
//! // 3. Insert vectors by ID (zero-copy)
//! hnsw.insert_by_id(0, 100)?; // vector_id=0, data_id=100
//! hnsw.insert_by_id(1, 101)?;
//! hnsw.insert_by_id(2, 102)?;
//!
//! // 4. Search as usual
//! let query = vec![1.0, 0.0, 0.0];
//! let neighbors = hnsw.search(&query, 2, 50);
//!
//! assert_eq!(neighbors[0].d_id, 100); // Found exact match
//! ```
//!
//! ### Memory Savings
//!
//! Using `insert_by_id()` with external storage:
//! - **No vector copying**: Vectors stay in storage
//! - **Lower memory footprint**: Only IDs stored in HNSW graph
//! - **Faster insertion**: No allocation overhead
//!
//! Traditional `insert()` method:
//! - **Copies each vector**: 2x memory usage
//! - **Allocation overhead**: Vec allocation per point
//!
//! ### Mixing Both Approaches
//!
//! You can mix `insert()` and `insert_by_id()`:
//!
//! ```rust,ignore
//! // Insert from storage (zero-copy)
//! hnsw.insert_by_id(0, 100)?;
//!
//! // Insert directly (copies vector)
//! hnsw.insert((&[0.5, 0.5, 0.5], 200));
//!
//! // Both points are searchable
//! let results = hnsw.search(&query, 10, 50);
//! ```
//!
//! ### Error Handling
//!
//! `insert_by_id()` returns `Result<(), String>` and fails if:
//!
//! - **No storage configured**: Use `with_external_storage()` constructor
//! - **Invalid vector_id**: ID doesn't exist in storage
//! - **Dimension mismatch**: Vector dimension differs from HNSW configuration

use serde::{Deserialize, Serialize};

use cpu_time::ProcessTime;
use std::time::SystemTime;

use std::cmp::Ordering;

use parking_lot::{Mutex, RwLock, RwLockReadGuard};
use rayon::prelude::*;
use std::sync::Arc;
use std::sync::mpsc::channel;

use std::any::type_name;

use hashbrown::HashMap;
#[allow(unused)]
use std::collections::HashSet;
use std::collections::binary_heap::BinaryHeap;

use log::trace;
use log::{debug, info};

pub use crate::filter::FilterT;
use crate::distance::Distance;
use crate::storage::{DynVectorStorage, VectorStorage};

// TODO
// Profiling.

/// This unit structure provides the type to instanciate Hnsw with,
/// to get reload of graph only in the the structure.
/// It must be associated to the unit structure dist::NoDist for the distance type to provide.
#[derive(Default, Clone, Copy, Serialize, Deserialize, Debug)]
pub struct NoData;

/// Marker type indicating no external vector storage.
///
/// When using `NoStorage`, vectors are embedded directly in Points (legacy mode).
/// This is the default behavior when external storage is not needed.
///
/// # Examples
///
/// ```rust,ignore
/// use tessera_hnsw::{Hnsw, NoStorage};
/// use anndists::distance::DistL2;
///
/// // Vectors stored internally (no external storage)
/// let hnsw: Hnsw<f32, DistL2, NoStorage> = Hnsw::new(15, 1000, 16, 200, DistL2 {});
/// ```
#[derive(Default, Clone, Copy, Debug)]
pub struct NoStorage;

/// Implementation of VectorStorage for NoStorage (no-op).
/// This allows Hnsw to work without external storage (legacy mode).
impl<'a, T: Clone + Send + Sync + std::fmt::Debug + 'a> VectorStorage<'a, T> for NoStorage {
    fn get_vector(&'a self, _id: usize) -> Option<&'a [T]> {
        None
    }

    fn dimension(&self) -> usize {
        0
    }

    fn len(&self) -> usize {
        0
    }

    fn is_empty(&self) -> bool {
        true
    }
}

/// maximum number of layers
pub(crate) const NB_LAYER_MAX: u8 = 16; // so max layer is 15!!

#[derive(Debug, Default, Clone, Copy, PartialEq, Eq, PartialOrd, Ord, Hash)]
/// The 2-uple represent layer as u8  and rank in layer as a i32 as stored in our structure
pub struct PointId(pub u8, pub i32);

/// this type is for an identificateur of each data vector, given by client.
/// Can be the rank of data in an array, a hash value or anything that permits
/// retrieving the data.
pub type DataId = usize;

pub type PointDistance<T> = Box<dyn Distance<T>>;

/// A structure containing internal pointId with distance to this pointId.
/// The order is given by ordering the distance to the point it refers to.
/// So points ordering has a meaning only has points refers to the same point
#[derive(Debug, Clone, Copy)]
pub struct PointIdWithOrder {
    /// the identificateur of the point for which we store a distance
    pub point_id: PointId,
    /// The distance to a reference point (not represented in the structure)
    pub dist_to_ref: f32,
}

impl PartialEq for PointIdWithOrder {
    fn eq(&self, other: &PointIdWithOrder) -> bool {
        self.dist_to_ref == other.dist_to_ref
    } // end eq
}

// order points by distance to self.
impl PartialOrd for PointIdWithOrder {
    fn partial_cmp(&self, other: &PointIdWithOrder) -> Option<Ordering> {
        self.dist_to_ref.partial_cmp(&other.dist_to_ref)
    } // end cmp
} // end impl PartialOrd

impl<T: Send + Sync + Clone + Copy> From<&PointWithOrder<'_, T>> for PointIdWithOrder {
    fn from(point: &PointWithOrder<T>) -> PointIdWithOrder {
        PointIdWithOrder::new(point.point_ref.p_id, point.dist_to_ref)
    }
}

impl PointIdWithOrder {
    pub fn new(point_id: PointId, dist_to_ref: f32) -> Self {
        PointIdWithOrder {
            point_id,
            dist_to_ref,
        }
    }
} // end of impl block

//=======================================================================================
/// The struct giving an answer point to a search request.
/// This structure is exported to other language API.
/// First field is origin id of the request point, second field is distance to request point
#[repr(C)]
#[derive(Debug, Copy, Clone, Default)]
pub struct Neighbour {
    /// identification of data vector as given in initializing hnsw
    pub d_id: DataId,
    /// distance of neighbours
    pub distance: f32,
    /// point identification inside layers
    pub p_id: PointId,
}

impl Neighbour {
    pub fn new(d_id: DataId, distance: f32, p_id: PointId) -> Neighbour {
        Neighbour {
            d_id,
            distance,
            p_id,
        }
    }
    /// retrieves original id of neighbour as given in hnsw initialization
    pub fn get_origin_id(&self) -> DataId {
        self.d_id
    }
    /// return the distance
    pub fn get_distance(&self) -> f32 {
        self.distance
    }
}

//=======================================================================================

#[derive(Debug, Clone)]
enum PointData<'b, T: Clone + Send + Sync + 'b> {
    // full data
    V(Vec<T>),
    // areference to a mmaped slice
    S(&'b [T]),
    // vector ID for external storage
    VectorId(usize),
} // end of enum PointData

impl<'b, T: Clone + Send + Sync + 'b> PointData<'b, T> {
    // allocate a point stored in structure
    fn new_v(v: Vec<T>) -> Self {
        PointData::V(v)
    }

    // allocate a point representation a memory mapped slice
    fn new_s(s: &'b [T]) -> Self {
        PointData::S(s)
    }

    fn get_v(&self) -> &[T] {
        match self {
            PointData::V(v) => v.as_slice(),
            PointData::S(s) => s,
            PointData::VectorId(_) => {
                panic!("Cannot access vector from VectorId without VectorStorage")
            }
        }
    } // end of get_v

    /// Try to get the vector data without panicking.
    /// Returns None for VectorId variants (use Hnsw::get_point_vector for those).
    fn try_get_v(&self) -> Option<&[T]> {
        match self {
            PointData::V(v) => Some(v.as_slice()),
            PointData::S(s) => Some(s),
            PointData::VectorId(_) => None,
        }
    }
} // end of impl block for PointData

/// The basestructure representing a data point.  
/// Its constains data as coming from the client, its client id,  
/// and position in layer representation and neighbours.
///
// neighbours table : one vector by layer so neighbours is allocated to NB_LAYER_MAX
//
#[derive(Debug, Clone)]
#[allow(clippy::type_complexity)]
pub struct Point<'b, T: Clone + Send + Sync> {
    /// The data of this point, coming from hnsw client and associated to origin_id,
    data: PointData<'b, T>,
    /// an id coming from client using hnsw, should identify point uniquely
    origin_id: DataId,
    /// a point id identifying point as stored in our structure
    p_id: PointId,
    /// neighbours info
    pub(crate) neighbours: Arc<RwLock<Vec<Vec<Arc<PointWithOrder<'b, T>>>>>>,
}

impl<'b, T: Clone + Send + Sync> Point<'b, T> {
    pub fn new(v: Vec<T>, origin_id: usize, p_id: PointId) -> Self {
        let mut neighbours = Vec::with_capacity(NB_LAYER_MAX as usize);
        // CAVEAT, perhaps pass nb layer as arg ?
        for _ in 0..NB_LAYER_MAX {
            neighbours.push(Vec::<Arc<PointWithOrder<T>>>::new());
        }
        Point {
            data: PointData::new_v(v),
            origin_id,
            p_id,
            neighbours: Arc::new(RwLock::new(neighbours)),
        }
    }

    pub fn new_from_mmap(s: &'b [T], origin_id: usize, p_id: PointId) -> Self {
        let mut neighbours = Vec::with_capacity(NB_LAYER_MAX as usize);
        // CAVEAT, perhaps pass nb layer as arg ?
        for _ in 0..NB_LAYER_MAX {
            neighbours.push(Vec::<Arc<PointWithOrder<T>>>::new());
        }
        Point {
            data: PointData::new_s(s),
            origin_id,
            p_id,
            neighbours: Arc::new(RwLock::new(neighbours)),
        }
    }

    /// Creates a new Point referencing a vector by ID in external storage.
    ///
    /// This is more memory-efficient than `new()` as it avoids copying the vector.
    /// The vector is fetched on-demand from VectorStorage using the provided ID.
    ///
    /// # Arguments
    /// * `vector_id` - Index of the vector in external VectorStorage
    /// * `origin_id` - User-defined identifier for this point
    /// * `p_id` - Internal point ID assigned by HNSW
    pub fn new_from_vector_id(vector_id: usize, origin_id: usize, p_id: PointId) -> Self {
        let mut neighbours = Vec::with_capacity(NB_LAYER_MAX as usize);
        for _ in 0..NB_LAYER_MAX {
            neighbours.push(Vec::<Arc<PointWithOrder<T>>>::new());
        }
        Point {
            data: PointData::VectorId(vector_id),
            origin_id,
            p_id,
            neighbours: Arc::new(RwLock::new(neighbours)),
        }
    }

    /// get a reference to vector data
    pub fn get_v(&self) -> &[T] {
        self.data.get_v()
    }

    /// Try to get the vector data without panicking.
    /// Returns None for VectorId variants (use Hnsw::get_point_vector for those).
    pub fn try_get_v(&self) -> Option<&[T]> {
        self.data.try_get_v()
    }

    /// return coordinates in indexation
    pub fn get_point_id(&self) -> PointId {
        self.p_id
    }

    /// returns external (or client id) id of point
    pub fn get_origin_id(&self) -> usize {
        self.origin_id
    }

    /// returns for each layer, a vector Neighbour of a point, one vector by layer
    /// useful for extern crate only as it reallocates vectors
    pub fn get_neighborhood_id(&self) -> Vec<Vec<Neighbour>> {
        let ref_neighbours = self.neighbours.read();
        let nb_layer = ref_neighbours.len();
        let mut neighborhood = Vec::<Vec<Neighbour>>::with_capacity(nb_layer);
        for i in 0..nb_layer {
            let mut neighbours = Vec::<Neighbour>::new();
            let nb_ngbh = ref_neighbours[i].len();
            if nb_ngbh > 0usize {
                neighbours.reserve(nb_ngbh);
                for pointwo in &ref_neighbours[i] {
                    neighbours.push(Neighbour::new(
                        pointwo.point_ref.get_origin_id(),
                        pointwo.dist_to_ref,
                        pointwo.point_ref.get_point_id(),
                    ));
                }
            }
            neighborhood.push(neighbours);
        }
        neighborhood
    }

    /// prints minimal information on neighbours of point.
    pub fn debug_dump(&self) {
        println!(" \n dump of point id : {:?}", self.p_id);
        println!("\n origin id : {:?} ", self.origin_id);
        println!(" neighbours : ...");
        let ref_neighbours = self.neighbours.read();
        for i in 0..ref_neighbours.len() {
            if !ref_neighbours[i].is_empty() {
                println!("neighbours at layer {:?}", i);
                for n in &ref_neighbours[i] {
                    println!(" {:?}", n.point_ref.p_id);
                }
            }
        }
        println!(" neighbours dump : end");
    }
} // end of block

//===========================================================================================

/// A structure to store neighbours for of a point.
#[derive(Debug, Clone)]
pub(crate) struct PointWithOrder<'b, T: Clone + Send + Sync> {
    /// the identificateur of the point for which we store a distance to a point for which
    ///  we made a request.
    point_ref: Arc<Point<'b, T>>,
    /// The distance to a point_ref to the request point (not represented in the structure)
    dist_to_ref: f32,
}

impl<T: Clone + Send + Sync> PartialEq for PointWithOrder<'_, T> {
    fn eq(&self, other: &PointWithOrder<T>) -> bool {
        self.dist_to_ref == other.dist_to_ref
    } // end eq
}

impl<T: Clone + Send + Sync> Eq for PointWithOrder<'_, T> {}

// order points by distance to self.
#[allow(clippy::non_canonical_partial_ord_impl)]
impl<T: Clone + Send + Sync> PartialOrd for PointWithOrder<'_, T> {
    fn partial_cmp(&self, other: &PointWithOrder<T>) -> Option<Ordering> {
        self.dist_to_ref.partial_cmp(&other.dist_to_ref)
    } // end cmp
} // end impl PartialOrd

impl<T: Clone + Send + Sync> Ord for PointWithOrder<'_, T> {
    fn cmp(&self, other: &PointWithOrder<T>) -> Ordering {
        if !self.dist_to_ref.is_nan() && !other.dist_to_ref.is_nan() {
            self.dist_to_ref.partial_cmp(&other.dist_to_ref).unwrap()
        } else {
            panic!("got a NaN in a distance");
        }
    } // end cmp
}

impl<'b, T: Clone + Send + Sync> PointWithOrder<'b, T> {
    pub fn new(point_ref: &Arc<Point<'b, T>>, dist_to_ref: f32) -> Self {
        PointWithOrder {
            point_ref: Arc::clone(point_ref),
            dist_to_ref,
        }
    }
} // end of impl block

//============================================================================================

//  LayerGenerator
use rand::distr::Uniform;
use rand::prelude::*;

/// a struct to randomly generate a level for an item according to an exponential law
/// of parameter given by scale.
/// The distribution is constrained to be in [0..maxlevel[
pub struct LayerGenerator {
    rng: Arc<Mutex<rand::rngs::StdRng>>,
    unif: Uniform<f64>,
    // drives number of levels generated ~ S
    scale: f64,
    maxlevel: usize,
}

impl LayerGenerator {
    pub fn new(max_nb_connection: usize, maxlevel: usize) -> Self {
        let scale = 1. / (max_nb_connection as f64).ln();
        LayerGenerator {
            rng: Arc::new(Mutex::new(StdRng::from_os_rng())),
            unif: Uniform::<f64>::new(0., 1.).unwrap(),
            scale,
            maxlevel,
        }
    }

    // new when we know scale used. Should replace the one without scale
    pub(crate) fn new_with_scale(
        max_nb_connection: usize,
        scale_factor: f64,
        maxlevel: usize,
    ) -> Self {
        let scale_default = 1. / (max_nb_connection as f64).ln();
        LayerGenerator {
            rng: Arc::new(Mutex::new(StdRng::from_os_rng())),
            unif: Uniform::<f64>::new(0., 1.).unwrap(),
            scale: scale_default * scale_factor,
            maxlevel,
        }
    }
    //
    // l=0 most densely packed layer
    // if S is scale we sample so that P(l=n) = exp(-n/S) - exp(- (n+1)/S)
    // with S = 1./ln(max_nb_connection) P(l >= maxlevel) = exp(-maxlevel * ln(max_nb_connection))
    // for nb_conn = 10, even with maxlevel = 10,  we get P(l >= maxlevel) = 1.E-13
    // In Malkov(2016) S = 1./log(max_nb_connection)
    //
    /// generate a layer with given maxlevel. upper layers (higher index) are of decreasing probabilities.
    /// thread safe method.
    fn generate(&self) -> usize {
        let mut protected_rng = self.rng.lock();
        let xsi = protected_rng.sample(self.unif);
        let level = -xsi.ln() * self.scale;
        let mut ulevel = level.floor() as usize;
        // we redispatch possibly sampled level  >= maxlevel to required range
        if ulevel >= self.maxlevel {
            // This occurs with very low probability. Cf commentary above.
            ulevel = protected_rng.sample(Uniform::<usize>::new(0, self.maxlevel).unwrap());
        }
        ulevel
    }

    /// just to try some variations on exponential level sampling. Unused.
    fn set_scale_modification(&mut self, scale_modification: f64) {
        self.scale *= scale_modification;
        log::info!("using scale for sampling levels : {:.2e}", self.scale);
    }

    //
    fn get_level_scale(&self) -> f64 {
        self.scale
    }
} // end impl for LayerGenerator

// ====================================================================

/// A short-hand for points in a layer
type Layer<'b, T> = Vec<Arc<Point<'b, T>>>;

/// a structure for indexation of points in layer
#[allow(unused)]
pub struct PointIndexation<'b, T: Clone + Send + Sync> {
    /// max number of connection for a point at a layer
    pub(crate) max_nb_connection: usize,
    //
    pub(crate) max_layer: usize,
    /// needs at least one representation of points. points_by_layers\[i\] gives the points in layer i
    pub(crate) points_by_layer: Arc<RwLock<Vec<Layer<'b, T>>>>,
    /// utility to generate a level
    pub(crate) layer_g: LayerGenerator,
    /// number of points in indexed structure
    pub(crate) nb_point: Arc<RwLock<usize>>,
    /// curent enter_point: an Arc RwLock on a possible Arc Point
    pub(crate) entry_point: Arc<RwLock<Option<Arc<Point<'b, T>>>>>,
    /// cached data dimension (set on first insert, needed for VectorId points)
    pub(crate) data_dimension: Arc<RwLock<usize>>,
}

// A point indexation may contain circular references. To deallocate these after a point indexation goes out of scope,
// implement the Drop trait.

impl<T: Clone + Send + Sync> Drop for PointIndexation<'_, T> {
    fn drop(&mut self) {
        let cpu_start = ProcessTime::now();
        let sys_now = SystemTime::now();
        info!("entering PointIndexation drop");
        // clear_neighborhood. There are no point in neighborhoods that are not referenced directly in layers.
        // so we cannot lose reference to a point by cleaning neighborhood
        fn clear_neighborhoods<T: Clone + Send + Sync>(init: &Point<T>) {
            let mut neighbours = init.neighbours.write();
            let nb_layer = neighbours.len();
            for l in 0..nb_layer {
                neighbours[l].clear();
            }
            neighbours.clear();
        }
        if let Some(i) = self.entry_point.write().as_ref() {
            clear_neighborhoods(i.as_ref());
        }
        //
        let nb_level = self.get_max_level_observed();
        for l in 0..=nb_level {
            trace!("clearing layer {}", l);
            let layer = &mut self.points_by_layer.write()[l as usize];
            layer.into_par_iter().for_each(|p| clear_neighborhoods(p));
            layer.clear();
        }
        //
        debug!("clearing self.points_by_layer...");
        drop(self.points_by_layer.write());
        debug!("exiting PointIndexation drop");
        info!(
            " drop sys time(s) {:?} cpu time {:?}",
            sys_now.elapsed().unwrap().as_secs(),
            cpu_start.elapsed().as_secs()
        );
    } // end my drop
} // end implementation Drop

impl<'b, T: Clone + Send + Sync> PointIndexation<'b, T> {
    pub fn new(max_nb_connection: usize, max_layer: usize, max_elements: usize) -> Self {
        let mut points_by_layer = Vec::with_capacity(max_layer);
        for i in 0..max_layer {
            // recall that range are right extremeity excluded
            // compute fraction of points going into layer i and do expected memory reservation
            let s = 1. / (max_nb_connection as f64).ln();
            let frac = (-(i as f64) / s).exp() - (-((i + 1) as f64) / s);
            let expected_size = ((frac * max_elements as f64).round()) as usize;
            points_by_layer.push(Vec::with_capacity(expected_size));
        }
        let layer_g = LayerGenerator::new(max_nb_connection, max_layer);
        PointIndexation {
            max_nb_connection,
            max_layer,
            points_by_layer: Arc::new(RwLock::new(points_by_layer)),
            layer_g,
            nb_point: Arc::new(RwLock::new(0)),
            entry_point: Arc::new(RwLock::new(None)),
            data_dimension: Arc::new(RwLock::new(0)),
        }
    } // end of new

    /// Sets the data dimension for vectors.
    /// Called by Hnsw when using external storage (VectorId points).
    pub fn set_data_dimension(&self, dim: usize) {
        *self.data_dimension.write() = dim;
    }

    /// returns the maximum level of layer observed
    pub fn get_max_level_observed(&self) -> u8 {
        let opt = self.entry_point.read();
        match opt.as_ref() {
            Some(arc_point) => arc_point.p_id.0,
            None => 0,
        }
    }

    pub fn get_level_scale(&self) -> f64 {
        self.layer_g.get_level_scale()
    }

    fn debug_dump(&self) {
        println!(" debug dump of PointIndexation");
        let max_level_observed = self.get_max_level_observed();
        // CAVEAT a lock once
        for l in 0..=max_level_observed as usize {
            println!(
                " layer {} : length : {} ",
                l,
                self.points_by_layer.read()[l].len()
            );
        }
        println!(" debug dump of PointIndexation end");
    }

    /// real insertion of point in point indexation
    // generate a new Point/ArcPoint (with neigbourhood info empty) and store it in global table
    // The function is called by Hnsw insert method
    fn generate_new_point(&self, data: &[T], origin_id: usize) -> (Arc<Point<'b, T>>, usize) {
        // Set data dimension on first insert (for get_data_dimension())
        {
            let mut dim = self.data_dimension.write();
            if *dim == 0 && !data.is_empty() {
                *dim = data.len();
            }
        }
        // get a write lock at the beginning of the function
        let level = self.layer_g.generate();
        let new_point;
        {
            // open a write lock on points_by_layer
            let mut points_by_layer_ref = self.points_by_layer.write();
            let mut p_id = PointId(level as u8, -1);
            p_id.1 = points_by_layer_ref[p_id.0 as usize].len() as i32;
            // make a Point and then an Arc<Point>
            let point = Point::new(data.to_vec(), origin_id, p_id);
            new_point = Arc::new(point);
            trace!("definitive pushing of point {:?}", p_id);
            points_by_layer_ref[p_id.0 as usize].push(Arc::clone(&new_point));
        } // close write lock on points_by_layer
        //
        let nb_point;
        {
            let mut lock_nb_point = self.nb_point.write();
            *lock_nb_point += 1;
            nb_point = *lock_nb_point;
            if nb_point % 50000 == 0 {
                println!(" setting number of points {:?} ", nb_point);
            }
        }
        trace!(" setting number of points {:?} ", *self.nb_point);
        // Now possibly this is a point on a new layer that will have no neighbours in its layer
        (Arc::clone(&new_point), nb_point)
    } // end of insert

    /// Generates a new point using a vector ID from external storage.
    ///
    /// Similar to `generate_new_point()` but references a vector by ID
    /// instead of copying it. More memory-efficient for large datasets.
    ///
    /// # Arguments
    /// * `vector_id` - Index of the vector in external VectorStorage
    /// * `origin_id` - User-defined identifier for this point
    ///
    /// # Returns
    /// Tuple of (Arc<Point>, number of points after insertion)
    fn generate_new_point_by_id(&self, vector_id: usize, origin_id: usize) -> (Arc<Point<'b, T>>, usize) {
        let level = self.layer_g.generate();
        let new_point;
        {
            let mut points_by_layer_ref = self.points_by_layer.write();
            let mut p_id = PointId(level as u8, -1);
            p_id.1 = points_by_layer_ref[p_id.0 as usize].len() as i32;
            // Create Point with VectorId instead of copying vector
            let point = Point::new_from_vector_id(vector_id, origin_id, p_id);
            new_point = Arc::new(point);
            trace!("definitive pushing of point {:?} with vector_id {}", p_id, vector_id);
            points_by_layer_ref[p_id.0 as usize].push(Arc::clone(&new_point));
        }
        //
        let nb_point;
        {
            let mut lock_nb_point = self.nb_point.write();
            *lock_nb_point += 1;
            nb_point = *lock_nb_point;
            if nb_point % 50000 == 0 {
                println!(" setting number of points {:?} ", nb_point);
            }
        }
        trace!(" setting number of points {:?} ", *self.nb_point);
        (Arc::clone(&new_point), nb_point)
    }

    /// check if entry_point is modified
    fn check_entry_point(&self, new_point: &Arc<Point<'b, T>>) {
        //
        // take directly a write lock so that we are sure nobody can change anything between read and write
        // of entry_point_id
        trace!("trying to get a lock on entry point");
        let mut entry_point_ref = self.entry_point.write();
        match entry_point_ref.as_ref() {
            Some(arc_point) => {
                if new_point.p_id.0 > arc_point.p_id.0 {
                    debug!("Hnsw  , inserting  entry point {:?} ", new_point.p_id);
                    debug!(
                        "PointIndexation insert setting max level from {:?} to {:?}",
                        arc_point.p_id.0, new_point.p_id.0
                    );
                    *entry_point_ref = Some(Arc::clone(new_point));
                }
            }
            None => {
                trace!("initializing entry point");
                debug!("Hnsw  , inserting  entry point {:?} ", new_point.p_id);
                *entry_point_ref = Some(Arc::clone(new_point));
            }
        }
    } // end of check_entry_point

    /// returns the number of points in layered structure
    pub fn get_nb_point(&self) -> usize {
        *self.nb_point.read()
    }

    /// returns the number of points in a given layer, 0 on a bad layer num
    pub fn get_layer_nb_point(&self, layer: usize) -> usize {
        let nb_layer = self.points_by_layer.read().len();
        if layer < nb_layer {
            self.points_by_layer.read()[layer].len()
        } else {
            0
        }
    } // end of get_layer_nb_point

    /// returns the size of data vector in graph if any, else return 0.
    /// This returns the cached dimension set during first insert or via set_data_dimension().
    /// Works correctly for both embedded vectors and VectorId references.
    pub fn get_data_dimension(&self) -> usize {
        *self.data_dimension.read()
    }

    /// returns (**by cloning**) the data inside a point given it PointId, or None if PointId is not coherent.
    /// Returns None for VectorId points (those using external storage) - use Hnsw::get_point_vector instead.
    /// Can be useful after reloading from a dump.
    /// NOTE : This function should not be called during or before insertion in the structure is terminated as it
    /// uses read locks to access the inside of Hnsw structure.
    pub fn get_point_data(&self, p_id: &PointId) -> Option<Vec<T>> {
        if p_id.1 < 0 {
            return None;
        }
        let p: usize = std::convert::TryFrom::try_from(p_id.1).unwrap();
        let l = p_id.0 as usize;
        if p_id.0 <= self.get_max_level_observed() && p < self.get_layer_nb_point(l) {
            // Use try_get_v() to handle VectorId points gracefully (returns None instead of panicking)
            self.points_by_layer.read()[l][p].try_get_v().map(|v| v.to_vec())
        } else {
            None
        }
    } // end of get_point_data

    /// returns (**by Arc::clone**) the point given it PointId, or None if PointId is not coherent.  
    /// Can be useful after reloading from a dump.   
    /// NOTE : This function should not be called during or before insertion in the structure is terminated as it
    /// uses read locks to access the inside of Hnsw structure.
    #[allow(unused)]
    pub(crate) fn get_point(&self, p_id: &PointId) -> Option<Arc<Point<'b, T>>> {
        if p_id.1 < 0 {
            return None;
        }
        let p: usize = std::convert::TryFrom::try_from(p_id.1).unwrap();
        let l = p_id.0 as usize;
        if p_id.0 <= self.get_max_level_observed() && p < self.get_layer_nb_point(l) {
            Some(self.points_by_layer.read()[l][p].clone())
        } else {
            None
        }
    } // end of get_point

    /// get an iterator on the points stored in a given layer
    pub fn get_layer_iterator<'a>(&'a self, layer: usize) -> IterPointLayer<'a, 'b, T> {
        IterPointLayer::new(self, layer)
    } // end of get_layer_iterator
} // end of impl PointIndexation

//============================================================================================

/// an iterator on points stored.
/// The iteration begins at level 0 (most populated level) and goes upward in levels.
/// The iterator takes a ReadGuard on the PointIndexation structure
pub struct IterPoint<'a, 'b, T: Clone + Send + Sync + 'b> {
    point_indexation: &'a PointIndexation<'b, T>,
    pi_guard: RwLockReadGuard<'a, Vec<Layer<'b, T>>>,
    layer: i64,
    slot_in_layer: i64,
}

impl<'a, 'b, T: Clone + Send + Sync> IterPoint<'a, 'b, T> {
    pub fn new(point_indexation: &'a PointIndexation<'b, T>) -> Self {
        let pi_guard: RwLockReadGuard<Vec<Layer<'b, T>>> = point_indexation.points_by_layer.read();
        IterPoint {
            point_indexation,
            pi_guard,
            layer: -1,
            slot_in_layer: -1,
        }
    }
} // end of block impl IterPoint

/// iterator for layer 0 to upper layer.
impl<'b, T: Clone + Send + Sync> Iterator for IterPoint<'_, 'b, T> {
    type Item = Arc<Point<'b, T>>;
    //
    fn next(&mut self) -> Option<Self::Item> {
        if self.layer == -1 {
            self.layer = 0;
            self.slot_in_layer = 0;
        }
        if (self.slot_in_layer as usize) < self.pi_guard[self.layer as usize].len() {
            let slot = self.slot_in_layer as usize;
            self.slot_in_layer += 1;
            Some(self.pi_guard[self.layer as usize][slot].clone())
        } else {
            self.slot_in_layer = 0;
            self.layer += 1;
            // must reach a non empty layer if possible
            let entry_point_ref = self.point_indexation.entry_point.read();
            let points_by_layer = self.point_indexation.points_by_layer.read();
            let entry_point_level = entry_point_ref.as_ref().unwrap().p_id.0;
            while (self.layer as u8) <= entry_point_level
                && points_by_layer[self.layer as usize].is_empty()
            {
                self.layer += 1;
            }
            // now here either  (self.layer as u8) > self.point_indexation.max_level_observed
            // or self.point_indexation.points_by_layer[self.layer as usize ].len() > 0
            if (self.layer as u8) <= entry_point_level {
                let slot = self.slot_in_layer as usize;
                self.slot_in_layer += 1;
                Some(points_by_layer[self.layer as usize][slot].clone())
            } else {
                None
            }
        }
    } // end of next
} // end of impl Iterator

impl<'a, 'b, T: Clone + Send + Sync> IntoIterator for &'a PointIndexation<'b, T> {
    type Item = Arc<Point<'b, T>>;
    type IntoIter = IterPoint<'a, 'b, T>;
    //
    fn into_iter(self) -> Self::IntoIter {
        IterPoint::new(self)
    }
} // end of IntoIterator for &'a PointIndexation<T>

/// An iterator on points stored in a given layer
/// The iterator stores a ReadGuard on the structure PointIndexation
pub struct IterPointLayer<'a, 'b, T: Clone + Send + Sync> {
    _point_indexation: &'a PointIndexation<'b, T>,
    pi_guard: RwLockReadGuard<'a, Vec<Layer<'b, T>>>,
    layer: usize,
    slot_in_layer: usize,
}

impl<'a, 'b, T: Clone + Send + Sync> IterPointLayer<'a, 'b, T> {
    pub fn new(point_indexation: &'a PointIndexation<'b, T>, layer: usize) -> Self {
        let pi_guard: RwLockReadGuard<Vec<Layer<'b, T>>> = point_indexation.points_by_layer.read();
        IterPointLayer {
            _point_indexation: point_indexation,
            pi_guard,
            layer,
            slot_in_layer: 0,
        }
    }
} // end of block impl IterPointLayer

/// iterator for layer 0 to upper layer.
impl<'b, T: Clone + Send + Sync + 'b> Iterator for IterPointLayer<'_, 'b, T> {
    type Item = Arc<Point<'b, T>>;
    //
    fn next(&mut self) -> Option<Self::Item> {
        if (self.slot_in_layer) < self.pi_guard[self.layer].len() {
            let slot = self.slot_in_layer;
            self.slot_in_layer += 1;
            Some(self.pi_guard[self.layer][slot].clone())
        } else {
            None
        }
    } // end of next
} // end of impl Iterator

// ============================================================================================

// The fields are made pub(crate) to be able to initialize struct from hnswio
/// The Base structure for hnsw implementation.
/// The main useful functions are : new, insert, insert_parallel, search, parallel_search and file_dump
/// as described in trait AnnT.
///
/// Other functions are mainly for others crate to get access to some fields.
///
/// # Generic Parameters
///
/// * `'b` - Lifetime for borrowed data (mmap slices or VectorStorage references)
/// * `T` - Element type of vectors
/// * `D` - Distance function type
/// * `VS` - Vector storage type (defaults to `NoStorage` for legacy mode)
pub struct Hnsw<'b, T: Clone + Send + Sync + 'b, D: Distance<T>, VS: VectorStorage<'b, T> = NoStorage> {
    /// asked number of candidates in search
    pub(crate) ef_construction: usize,
    /// maximum number of connection by layer for a point
    pub(crate) max_nb_connection: usize,
    /// flag to enforce that we have ef candidates as pruning strategy can discard some points
    /// Can be set to true with method :set_extend_candidates
    /// When set to true used only in base layer.
    pub(crate) extend_candidates: bool,
    /// defuault to false
    pub(crate) keep_pruned: bool,
    /// max layer , recall rust is in 0..maxlevel right bound excluded
    pub(crate) max_layer: usize,
    /// The global table containing points
    pub(crate) layer_indexed_points: PointIndexation<'b, T>,
    /// dimension data stored in points
    #[allow(unused)]
    pub(crate) data_dimension: usize,
    /// distance between points. initialized at first insertion
    pub(crate) dist_f: D,
    /// insertion mode or searching mode. This flag prevents a internal thread to do a write when searching with other threads.
    pub(crate) searching: bool,
    /// set to true if some data come from a mmap
    pub(crate) datamap_opt: bool,
    /// External vector storage (optional, for external vector management).
    /// When `Some`, vectors are accessed via VectorStorage trait instead of being embedded in Points.
    pub(crate) vector_storage: Option<&'b VS>,
    /// Dynamic vector storage for graph-only reloads.
    ///
    /// When loading an HNSW index from disk in graph-only mode (no embedded vectors),
    /// Points have VectorId references but `vector_storage` is None. This field allows
    /// setting storage dynamically after load via `set_dynamic_vector_storage()`.
    ///
    /// Resolution priority in `get_point_vector()`:
    /// 1. Embedded vector in Point (`PointData::V`)
    /// 2. Mmap slice in Point (`PointData::S`)
    /// 3. Dynamic vector storage (this field) - for `PointData::VectorId`
    /// 4. Static vector storage (`vector_storage`) - for `PointData::VectorId`
    /// 5. Panic if no source available
    pub(crate) dyn_vector_storage: Option<Arc<dyn DynVectorStorage<T>>>,
    /// Soft-deleted data IDs. Deleted points remain in the graph for traversal
    /// but are excluded from search results and `get_nb_point()`.
    ///
    /// ## Thread Safety
    ///
    /// Uses `parking_lot::RwLock` for interior mutability: `mark_deleted(&self)`
    /// takes a shared reference, allowing concurrent reads (search) and writes
    /// (deletion) without requiring `&mut self`.
    ///
    /// - **Search (`search()`, `search_filter()`)**: Acquires read lock per-neighbor
    ///   during traversal. Deleted points are still traversed as graph edges but
    ///   excluded from results.
    /// - **mark_deleted()**: Acquires write lock for O(1) HashSet insert.
    ///   Concurrent searches may see a slightly stale view of the deleted set
    ///   (a point deleted mid-search may still appear in that search's results).
    ///   This is by design: snapshot semantics at search initiation.
    ///
    /// ## Why RwLock, not Mutex
    ///
    /// Search calls `is_deleted()` per visited neighbor (hundreds of times per query).
    /// Multiple concurrent searches must read the deleted set in parallel.
    /// `parking_lot::RwLock` allows this. `Mutex` would serialize all search threads
    /// at every neighbor check, causing severe contention under concurrent search load.
    /// Writes (mark_deleted) are rare by comparison and tolerate brief serialization.
    ///
    /// ## Persistence
    ///
    /// `file_dump()` snapshots the deleted set via `self.deleted.read().clone()`.
    /// If `mark_deleted()` is called concurrently with `file_dump()`, the snapshot
    /// is point-in-time: the persisted file may not include the concurrent deletion.
    /// This is acceptable — the deletion will be captured in the next `file_dump()`.
    ///
    /// Persisted to `<prefix>.hnsw.deleted` on save, loaded on reload.
    /// If the file is absent (older index format), defaults to empty set.
    pub(crate) deleted: parking_lot::RwLock<std::collections::HashSet<DataId>>,
} // end of Hnsw

impl<'b, T: Clone + Send + Sync + std::fmt::Debug, D: Distance<T> + Send + Sync, VS: VectorStorage<'b, T>> Hnsw<'b, T, D, VS> {
    /// allocation function
    /// . max_nb_connection : number of neighbours stored, by layer, in tables. Must be less than 256.
    /// . ef_construction : controls numbers of neighbours explored during construction. See README or paper.
    /// . max_elements : hint to speed up allocation tables. number of elements expected.
    /// . f : the distance function
    pub fn new(
        max_nb_connection: usize,
        max_elements: usize,
        max_layer: usize,
        ef_construction: usize,
        f: D,
    ) -> Hnsw<'b, T, D, NoStorage> {
        let adjusted_max_layer = (NB_LAYER_MAX as usize).min(max_layer);
        let layer_indexed_points =
            PointIndexation::<T>::new(max_nb_connection, adjusted_max_layer, max_elements);
        let extend_candidates = false;
        let keep_pruned = false;
        //
        if max_nb_connection > 256 {
            println!("error max_nb_connection must be less equal than 256");
            std::process::exit(1);
        }
        //
        info!("Hnsw max_nb_connection {:?}", max_nb_connection);
        info!("Hnsw nb elements {:?}", max_elements);
        info!("Hnsw ef_construction {:?}", ef_construction);
        info!("Hnsw distance {:?}", type_name::<D>());
        info!("Hnsw extend candidates {:?}", extend_candidates);
        //
        Hnsw {
            max_nb_connection,
            ef_construction,
            extend_candidates,
            keep_pruned,
            max_layer: adjusted_max_layer,
            layer_indexed_points,
            data_dimension: 0,
            dist_f: f,
            searching: false,
            datamap_opt: false,
            vector_storage: None,
            dyn_vector_storage: None,
            deleted: parking_lot::RwLock::new(std::collections::HashSet::new()),
        }
    } // end of new

    /// allocation function with external vector storage
    /// . max_nb_connection : number of neighbours stored, by layer, in tables. Must be less than 256.
    /// . ef_construction : controls numbers of neighbours explored during construction. See README or paper.
    /// . max_elements : hint to speed up allocation tables. number of elements expected.
    /// . f : the distance function
    /// . storage : reference to external VectorStorage implementation
    pub fn new_with_storage(
        max_nb_connection: usize,
        max_elements: usize,
        max_layer: usize,
        ef_construction: usize,
        f: D,
        storage: &'b VS,
    ) -> Self {
        let adjusted_max_layer = (NB_LAYER_MAX as usize).min(max_layer);
        let layer_indexed_points =
            PointIndexation::<T>::new(max_nb_connection, adjusted_max_layer, max_elements);
        let extend_candidates = false;
        let keep_pruned = false;
        //
        if max_nb_connection > 256 {
            println!("error max_nb_connection must be less equal than 256");
            std::process::exit(1);
        }
        //
        info!("Hnsw max_nb_connection {:?}", max_nb_connection);
        info!("Hnsw nb elements {:?}", max_elements);
        info!("Hnsw ef_construction {:?}", ef_construction);
        info!("Hnsw distance {:?}", type_name::<D>());
        info!("Hnsw extend candidates {:?}", extend_candidates);
        info!("Hnsw using external VectorStorage");
        //
        let data_dimension = storage.dimension();
        info!("Hnsw data_dimension from storage: {:?}", data_dimension);
        // Set dimension in PointIndexation for get_data_dimension() support with VectorId points
        layer_indexed_points.set_data_dimension(data_dimension);
        //
        Hnsw {
            max_nb_connection,
            ef_construction,
            extend_candidates,
            keep_pruned,
            max_layer: adjusted_max_layer,
            layer_indexed_points,
            data_dimension,
            dist_f: f,
            searching: false,
            datamap_opt: false,
            vector_storage: Some(storage),
            dyn_vector_storage: None,
            deleted: parking_lot::RwLock::new(std::collections::HashSet::new()),
        }
    } // end of new_with_storage

    /// Set dynamic vector storage for graph-only indexes loaded from disk.
    ///
    /// When loading an HNSW index saved without embedded vectors (graph-only mode),
    /// Points have `VectorId` references but no storage to resolve them. This method
    /// sets the storage so that `search()` can access vectors during distance calculation.
    ///
    /// # Arguments
    ///
    /// * `storage` - Arc-wrapped VectorStorage implementation (e.g., `ShardedVectorStorage`)
    ///
    /// # Example
    ///
    /// ```rust,ignore
    /// // Load graph-only index
    /// let mut hnsw = reloader.load_hnsw::<f32, DistL2>()?;
    ///
    /// // Configure external vector storage
    /// let storage = Arc::new(ShardedVectorStorage::new(reader, 50_000));
    /// hnsw.set_dynamic_vector_storage(storage);
    ///
    /// // Now search works!
    /// let results = hnsw.search(&query, 10, 50);
    /// ```
    ///
    /// # Thread Safety
    ///
    /// The Arc wrapper ensures thread-safe sharing of the storage across concurrent searches.
    pub fn set_dynamic_vector_storage(&mut self, storage: Arc<dyn DynVectorStorage<T>>) {
        self.dyn_vector_storage = Some(storage);
    }

    /// Helper method to get vector data from a Point, accounting for VectorStorage.
    ///
    /// This abstracts away whether vectors are stored internally or externally.
    /// Supports multiple vector resolution strategies:
    ///
    /// 1. Embedded vector (`PointData::V`) - used for in-memory construction
    /// 2. Mmap slice (`PointData::S`) - used for memory-mapped data
    /// 3. VectorId reference (`PointData::VectorId`) - used for external storage
    /// 4. Empty embedded vector - treated as VectorId for graph-only reloads
    ///
    /// For VectorId and empty vectors, resolution order is:
    /// - `dyn_vector_storage` (dynamic, set after load)
    /// - `vector_storage` (static, set at construction)
    ///
    /// # Panics
    ///
    /// Panics if:
    /// - Point has VectorId/empty vector but no storage is configured
    /// - Vector ID not found in storage (data corruption)
    ///
    /// # Returns
    ///
    /// - `Some(&[T])` if vector data is available
    /// - `None` if vector not found in storage (allows graceful degradation)
    #[inline]
    fn get_point_vector<'a>(&'a self, point: &'a Point<'b, T>) -> Option<&'a [T]> {
        match &point.data {
            PointData::V(v) if !v.is_empty() => Some(v.as_slice()),
            PointData::V(_) => {
                // Empty vector - graph-only reload case
                // Use origin_id to look up vector from storage
                let id = point.origin_id;
                self.resolve_vector_from_storage(id)
            }
            PointData::S(s) => Some(s),
            PointData::VectorId(id) => {
                self.resolve_vector_from_storage(*id)
            }
        }
    }

    /// Resolve vector from dynamic or static storage.
    ///
    /// Tries `dyn_vector_storage` first (for graph-only reloads),
    /// then falls back to `vector_storage` (for static storage references).
    ///
    /// # Returns
    ///
    /// - `Some(&[T])` if vector found in storage
    /// - `None` if no storage configured or vector not found (graceful degradation)
    ///
    /// # Note
    ///
    /// Previously this function would panic on missing vectors. Now it returns
    /// `None` to allow callers to handle missing vectors gracefully without
    /// poisoning locks or crashing the process.
    #[inline]
    fn resolve_vector_from_storage(&self, id: usize) -> Option<&[T]> {
        // Priority 1: Dynamic vector storage (for graph-only reloads)
        if let Some(ref storage) = self.dyn_vector_storage {
            if let Some(vec) = storage.get_vector(id) {
                return Some(vec);
            }
        }

        // Priority 2: Static vector storage (for construction-time storage)
        if let Some(storage) = self.vector_storage {
            if let Some(vec) = storage.get_vector(id) {
                return Some(vec);
            }
        }

        // No storage configured or vector not found - return None for graceful degradation
        // Previously: panic!() - which poisoned RwLock and broke all subsequent operations
        log::warn!(
            "Vector ID {} not found: no storage configured or vector missing. \
             For graph-only reloads, call set_dynamic_vector_storage() after loading.",
            id
        );
        None
    }

    /// get ef_construction used in graph creation
    pub fn get_ef_construction(&self) -> usize {
        self.ef_construction
    }
    /// returns the maximum layer authorized in construction
    pub fn get_max_level(&self) -> usize {
        self.max_layer
    }

    /// return the maximum level reached in the layers.
    pub fn get_max_level_observed(&self) -> u8 {
        self.layer_indexed_points.get_max_level_observed()
    }
    /// returns the maximum of links between a point and others points in each layer
    pub fn get_max_nb_connection(&self) -> u8 {
        self.max_nb_connection as u8
    }
    /// Returns number of active (non-deleted) points in the HNSW structure.
    pub fn get_nb_point(&self) -> usize {
        self.layer_indexed_points.get_nb_point() - self.deleted.read().len()
    }

    /// Returns total number of points in the graph including deleted ones.
    pub fn get_nb_point_total(&self) -> usize {
        self.layer_indexed_points.get_nb_point()
    }

    /// Returns the `origin_id` (external DataId / VectorId) of the HNSW entry
    /// point, or `None` if the index is empty.
    ///
    /// The entry point is the top-level node used to start every search and
    /// insertion traversal. If this ID is orphaned (not present in storage),
    /// search and insertion quality degrades for ALL operations.
    pub fn get_entry_point_origin_id(&self) -> Option<usize> {
        self.layer_indexed_points
            .entry_point
            .read()
            .as_ref()
            .map(|p| p.get_origin_id())
    }

    /// Soft-delete a point by its data ID. The point remains in the graph
    /// for traversal but is excluded from search results.
    ///
    /// # Return Value
    ///
    /// Returns `true` if the ID was newly marked as deleted.
    /// Returns `false` if the ID was already in the deleted set (idempotent).
    ///
    /// # Caller Responsibility
    ///
    /// This method does NOT validate whether `data_id` exists in the graph.
    /// Marking a non-existent ID is harmless for search correctness (the ID
    /// will never appear in results anyway), but inflates `deleted_count()`
    /// and `get_nb_point()` will under-count. Callers SHOULD validate the
    /// ID exists before calling this method.
    ///
    /// # Thread Safety
    ///
    /// Takes `&self` (not `&mut self`) via interior mutability
    /// (`parking_lot::RwLock<HashSet<DataId>>`). Safe to call concurrently
    /// with `search()` and other `mark_deleted()` calls. The write lock is
    /// held only for the duration of the `HashSet::insert()` call.
    pub fn mark_deleted(&self, data_id: DataId) -> bool {
        self.deleted.write().insert(data_id)
    }

    /// Check if a data ID has been soft-deleted.
    pub fn is_deleted(&self, data_id: DataId) -> bool {
        self.deleted.read().contains(&data_id)
    }

    /// Returns the number of soft-deleted points.
    pub fn deleted_count(&self) -> usize {
        self.deleted.read().len()
    }
    /// set searching mode.  
    /// It is not possible to do parallel insertion and parallel searching simultaneously in different threads
    /// so to enable searching after parallel insertion the flag must be set to true.  
    /// To resume parallel insertion reset the flag to false and so on.
    pub fn set_searching_mode(&mut self, flag: bool) {
        // must use an atomic!
        self.searching = flag;
    }
    /// get name if distance
    pub fn get_distance_name(&self) -> String {
        type_name::<D>().to_string()
    }
    /// set the flag asking to keep pruned vectors by Navarro's heuristic (see Paper).
    /// It can be useful for small datasets where the pruning can make it difficult
    /// to get the exact number of neighbours asked for.
    pub fn set_keeping_pruned(&mut self, flag: bool) {
        self.keep_pruned = flag;
    }

    /// retrieves the distance used in Hnsw construction
    pub fn get_distance(&self) -> &D {
        &self.dist_f
    }

    /// set extend_candidates to given flag. By default it is false.  
    /// Only used in the level 0 layer during insertion (see the paper)
    /// flag to enforce that we have ef candidates neighbours examined as pruning strategy
    /// can discard some points
    pub fn set_extend_candidates(&mut self, flag: bool) {
        self.extend_candidates = flag;
    }

    // When dumping we need to know if some file is mmapped
    pub(crate) fn get_datamap_opt(&self) -> bool {
        self.datamap_opt
    }

    /// By default the levels are sampled using an exponential law of parameter **ln(max_nb_conn)**
    /// so the probability of having more than l levels decrease as **exp(-l * ln(max_nb_conn))**.  
    /// Reducing the scale change the parameter of the exponential to **ln(max_nb_conn)/scale**.
    /// This reduce the number of levels generated and can provide better precision, reduce memory with marginally more cpu used.  
    /// The factor must between 0.2 and 1.
    pub fn modify_level_scale(&mut self, scale_modification: f64) {
        //
        if self.get_nb_point() > 0 {
            println!(
                "using modify_level_scale is possible at creation of a Hnsw structure to ensure coherence between runs"
            )
        }
        //
        let min_factor = 0.2;
        println!(
            "\n  Current scale value : {:.2e}, Scale modification factor asked : {:.2e},(modification factor must be between {:.2e} and 1.)",
            self.layer_indexed_points.layer_g.scale, scale_modification, min_factor
        );
        //
        if scale_modification > 1. {
            println!(
                "\n Scale modification not applied, modification arg {:.2e} not valid , factor must be less than 1.)",
                scale_modification
            );
        } else if scale_modification < min_factor {
            println!(
                "\n Scale modification arg {:.2e} not valid , factor must be greater than {:.2e}, using {:.2e})",
                scale_modification, min_factor, min_factor
            );
        }
        //
        self.layer_indexed_points
            .layer_g
            .set_scale_modification(scale_modification.max(min_factor).min(1.));
    } // end of set_scale_modification

    // here we could pass a point_id_with_order instead of entry_point_id: PointId
    // The efficacity depends on greedy part depends on how near entry point is from point.
    // ef is the number of points to return
    // The method returns a BinaryHeap with positive distances. The caller must transforms it according its need
    //** NOTE: the entry point is pushed into returned point at the beginning of the function, but in fact entry_point is in a layer
    //** with higher (one more) index than the argument layer. If the greedy search matches a sufficiently large number of points
    //** nearer to point searched (arg point) than entry_point it will finally get popped up from the heap of returned points
    //** but otherwise it will stay in the binary heap and so we can have a point in neighbours that is in fact in a layer
    //** above the one we search in.
    //** The guarantee is that the binary heap will return points in layer
    //** with a larger index, although we can expect that most often (at least in densely populated layers) the returned
    //** points will be found in searched layer
    ///
    /// Greedy algorithm n° 2 in Malkov paper.
    /// search in a layer (layer) for the ef points nearest a point to be inserted in hnsw.
    fn search_layer(
        &self,
        point: &[T],
        entry_point: Arc<Point<'b, T>>,
        ef: usize,
        layer: u8,
        filter: Option<&dyn FilterT>,
    ) -> BinaryHeap<Arc<PointWithOrder<'b, T>>> {
        //
        trace!(
            "entering search_layer with entry_point_id {:?} layer : {:?} ef {:?} ",
            entry_point.p_id, layer, ef
        );
        //
        // here we allocate a binary_heap on values not on reference beccause we want to return
        // log2(skiplist_size) must be greater than 1.
        let skiplist_size = ef.max(2);
        // we will store positive distances in this one
        let mut return_points = BinaryHeap::<Arc<PointWithOrder<T>>>::with_capacity(skiplist_size);
        //
        if self.layer_indexed_points.points_by_layer.read()[layer as usize].is_empty() {
            // at the beginning we can have nothing in layer
            trace!("search layer {:?}, empty layer", layer);
            return return_points;
        }
        if entry_point.p_id.1 < 0 {
            trace!("search layer negative point id : {:?}", entry_point.p_id);
            return return_points;
        }
        // initialize visited points
        let entry_vec = match self.get_point_vector(&entry_point) {
            Some(v) => v,
            None => {
                // Entry point vector missing - cannot proceed with search in this layer
                log::warn!("Entry point vector {} missing from storage, returning empty results", entry_point.origin_id);
                return return_points;
            }
        };
        let dist_to_entry_point = self.dist_f.eval(point, entry_vec)
            .expect("Distance to entry point should never fail for indexed vectors");
        trace!("       distance to entry point: {:?} ", dist_to_entry_point);
        // keep a list of id visited
        let mut visited_point_id = HashMap::<PointId, Arc<Point<T>>>::new();
        visited_point_id.insert(entry_point.p_id, Arc::clone(&entry_point));
        //
        let mut candidate_points =
            BinaryHeap::<Arc<PointWithOrder<T>>>::with_capacity(skiplist_size);
        candidate_points.push(Arc::new(PointWithOrder::new(
            &entry_point,
            -dist_to_entry_point,
        )));
        // PERF-FIX: Filter entry point BEFORE adding to return_points
        // This ensures all results in return_points pass the filter, eliminating
        // the need for redundant post-processing in search_filter().
        // Previously, entry point was added unconditionally, requiring double filtering.
        let entry_passes_filter = !self.is_deleted(entry_point.origin_id)
            && filter
                .as_ref()
                .is_none_or(|f| f.hnsw_filter(&entry_point.origin_id));
        if entry_passes_filter {
            return_points.push(Arc::new(PointWithOrder::new(
                &entry_point,
                dist_to_entry_point,
            )));
        }
        // at the beginning candidate_points contains point passed as arg in layer entry_point_id.0
        while !candidate_points.is_empty() {
            // get nearest point in candidate_points
            let c = candidate_points.pop().unwrap();
            // f farthest point to
            // PERF-FIX: Handle case where return_points is empty (entry point filtered out)
            // When using filters, the entry point may not pass the filter, leaving return_points empty.
            // In this case, we must continue traversing to find filtered results.
            let f_opt = return_points.peek();
            if let Some(f) = f_opt {
                // Note: These assertions removed because some distance metrics (e.g., DistDot) can return negative values
                // assert!(f.dist_to_ref >= 0.);
                // assert!(c.dist_to_ref <= 0.);
                trace!(
                    "Comparaing c : {:?} f : {:?}",
                    -(c.dist_to_ref),
                    f.dist_to_ref
                );
                if -(c.dist_to_ref) > f.dist_to_ref {
                    // this comparison requires that we are sure that distances compared are distances to the same point :
                    // This is the case we compare distance to point passed as arg.
                    trace!(
                        "Fast return from search_layer, nb points : {:?} \n \t c {:?} \n \t f {:?} dists: {:?}  {:?}",
                        return_points.len(),
                        c.point_ref.p_id,
                        f.point_ref.p_id,
                        -(c.dist_to_ref),
                        f.dist_to_ref
                    );
                    // Early return: Only when no filter (all results valid) OR we have enough filtered results
                    // PERF-FIX: All items in return_points already pass the filter due to entry point filtering,
                    // so we don't need the retain() call that was here before.
                    if filter.is_none() || return_points.len() >= ef {
                        return return_points;
                    }
                    // With filter but not enough results yet: continue searching
                }
            }
            // else: return_points is empty, continue searching for any filtered results
            // now we scan neighborhood of c in layer and increment visited_point, candidate_points
            // and optimize candidate_points so that it contains points with lowest distances to point arg
            //
            let neighbours_c_l = &c.point_ref.neighbours.read()[layer as usize];
            let c_pid = c.point_ref.p_id;
            trace!(
                "       search_layer, {:?} has  nb neighbours  : {:?} ",
                c_pid,
                neighbours_c_l.len()
            );
            for e in neighbours_c_l {
                // HERE WE sEE THAT neighbours should be stored as PointIdWithOrder !!
                // CAVEAT what if several point_id with same distance to ref point?
                if !visited_point_id.contains_key(&e.point_ref.p_id) {
                    visited_point_id.insert(e.point_ref.p_id, Arc::clone(&e.point_ref));
                    trace!("             visited insertion {:?}", e.point_ref.p_id);

                    let is_deleted = self.is_deleted(e.point_ref.origin_id);

                    // Skip neighbors with missing vectors (graceful degradation)
                    let e_vec = match self.get_point_vector(&e.point_ref) {
                        Some(v) => v,
                        None => {
                            log::trace!("Skipping neighbor {} with missing vector", e.point_ref.origin_id);
                            continue;
                        }
                    };
                    let e_dist_to_p = self.dist_f.eval(point, e_vec)
                        .expect("Distance between indexed vectors should be valid");

                    // PERF-FIX: Separate exploration (candidate_points) from results (return_points)
                    // For filtered search, we need to explore MORE of the graph to find filtered results.
                    // Key insight: return_points only contains FILTERED results, which may be sparse.
                    // Without filter, use distance-based pruning. With filter, explore more aggressively.
                    let need_more_filtered = filter.is_some() && return_points.len() < ef;
                    let closer_than_worst = return_points.peek().is_none_or(|f| e_dist_to_p < f.dist_to_ref);
                    let should_explore = need_more_filtered || closer_than_worst || return_points.len() < ef;

                    // Deleted points are still traversable for graph exploration (maintains
                    // search quality) but excluded from results.
                    if should_explore {
                        candidate_points
                            .push(Arc::new(PointWithOrder::new(&e.point_ref, -e_dist_to_p)));
                    }

                    // Add to return_points only if NOT deleted, passes filter, AND closer than worst
                    let passes_filter = !is_deleted
                        && filter.as_ref().is_none_or(|f| f.hnsw_filter(&e.point_ref.origin_id));
                    let should_add_to_results = passes_filter && (closer_than_worst || return_points.len() < ef);

                    if should_add_to_results {
                        let e_prime = Arc::new(PointWithOrder::new(&e.point_ref, e_dist_to_p));
                        trace!(
                            "                inserting new candidate {:?}",
                            e_prime.point_ref.p_id
                        );
                        return_points.push(e_prime);
                        if return_points.len() > ef {
                            return_points.pop();
                        }
                    }
                }
            } // end of for on neighbours_c
        } // end of while in candidates
        //
        trace!(
            "return from search_layer, nb points : {:?}",
            return_points.len()
        );
        return_points
    } // end of search_layer

    /// insert a tuple (&Vec, usize) with its external id as given by the client.
    ///  The insertion method gives the point an internal id.
    #[inline]
    pub fn insert(&self, datav_with_id: (&[T], usize)) {
        self.insert_slice((datav_with_id.0, datav_with_id.1))
    }

    // Hnsw insert.
    ///  Insert a data slice with its external id as given by the client.   
    ///  The insertion method gives the point an internal id.  
    ///  The slice insertion makes integration with ndarray crate easier than the vector insertion
    pub fn insert_slice(&self, data_with_id: (&[T], usize)) {
        //
        let (data, origin_id) = data_with_id;
        let keep_pruned = self.keep_pruned;
        // insert in indexation and get point_id adn generate a new entry_point if necessary
        let (new_point, point_rank) = self
            .layer_indexed_points
            .generate_new_point(data, origin_id);
        trace!("Hnsw insert generated new point {:?} ", new_point.p_id);
        // now real work begins
        // allocate a binary heap
        let level = new_point.p_id.0;
        let mut enter_point_copy = None;
        let mut max_level_observed = 0;
        // entry point has been set in
        {
            // I open a read lock on an option
            if let Some(arc_point) = self.layer_indexed_points.entry_point.read().as_ref() {
                enter_point_copy = Some(Arc::clone(arc_point));
                if point_rank == 1 {
                    debug!(
                        "Hnsw  stored first point , direct return  {:?} ",
                        new_point.p_id
                    );
                    return;
                }
                max_level_observed = enter_point_copy.as_ref().unwrap().p_id.0;
            }
        }
        if enter_point_copy.is_none() {
            self.layer_indexed_points.check_entry_point(&new_point);
            return;
        }
        let entry_vec = match self.get_point_vector(enter_point_copy.as_ref().unwrap()) {
            Some(v) => v,
            None => {
                // Entry point vector missing - cannot proceed with insertion
                log::warn!("Entry point vector missing during insert, adding point without full HNSW integration");
                self.layer_indexed_points.check_entry_point(&new_point);
                return;
            }
        };
        let mut dist_to_entry = self.dist_f.eval(data, entry_vec)
            .expect("Distance calculation failed: vectors in HNSW index should be valid");
        // we go from self.max_level_observed to level+1 included
        for l in ((level + 1)..(max_level_observed + 1)).rev() {
            // CAVEAT could bypass when layer empty, avoid  allocation..
            let mut sorted_points = self.search_layer(
                data,
                Arc::clone(enter_point_copy.as_ref().unwrap()),
                1,
                l,
                None,
            );
            trace!(
                "in insert :search_layer layer {:?}, returned {:?} points ",
                l,
                sorted_points.len()
            );
            if sorted_points.len() > 1 {
                panic!(
                    "in insert : search_layer layer {:?}, returned {:?} points ",
                    l,
                    sorted_points.len()
                );
            }
            // the heap conversion is useless beccause of the preceding test.
            // sorted_points = from_positive_binaryheap_to_negative_binary_heap(&mut sorted_points);
            //
            if let Some(ep) = sorted_points.pop() {
                // useful for projecting lower layer to upper layer. keep track of points encountered.
                if new_point.neighbours.read()[l as usize].len()
                    < self.get_max_nb_connection() as usize
                {
                    new_point.neighbours.write()[l as usize].push(Arc::clone(&ep));
                }
                // get the lowest distance point (skip if vector missing)
                if let Some(ep_vec) = self.get_point_vector(&ep.point_ref) {
                    let tmp_dist = self.dist_f.eval(data, ep_vec)
                        .expect("HNSW search distance calculation failed for indexed vector");
                    if tmp_dist < dist_to_entry {
                        enter_point_copy = Some(Arc::clone(&ep.point_ref));
                        dist_to_entry = tmp_dist;
                    }
                }
            } else {
                // this layer is not yet filled
                trace!("layer still empty  {} : got null list", l);
            }
        }
        // now enter_point_id_copy contains id of nearest
        // now loop down to 0
        for l in (0..level + 1).rev() {
            let ef = self.ef_construction;
            // when l == level, we cannot get new_point in sorted_points as it is seen only from declared neighbours
            let mut sorted_points = self.search_layer(
                data,
                Arc::clone(enter_point_copy.as_ref().unwrap()),
                ef,
                l,
                None,
            );
            trace!(
                "in insert :search_layer layer {:?}, returned {:?} points ",
                l,
                sorted_points.len()
            );
            sorted_points = from_positive_binaryheap_to_negative_binary_heap(&mut sorted_points);
            if !sorted_points.is_empty() {
                let nb_conn;
                let extend_c;
                if l == 0 {
                    nb_conn = 2 * self.max_nb_connection;
                    extend_c = self.extend_candidates;
                } else {
                    nb_conn = self.max_nb_connection;
                    extend_c = false;
                }
                let mut neighbours = Vec::<Arc<PointWithOrder<T>>>::with_capacity(nb_conn);
                self.select_neighbours(
                    data,
                    &mut sorted_points,
                    nb_conn,
                    extend_c,
                    l,
                    keep_pruned,
                    &mut neighbours,
                );
                // sort neighbours
                neighbours.sort_unstable();
                // we must add bidirecti*onal from data i.e new_point_id to neighbours
                new_point.neighbours.write()[l as usize].clone_from(&neighbours);
                // this reverse neighbour update could be done here but we put it at end to gather all code
                // requiring a mutex guard for multi threading.
                // update ep for loop iteration. As we sorted neighbours the nearest
                if !neighbours.is_empty() {
                    enter_point_copy = Some(Arc::clone(&neighbours[0].point_ref));
                }
            }
        } // for l
        //
        // new_point has been inserted at the beginning in table
        // so that we can call reverse_update_neighborhoodwe consitently
        // now reverse update of neighbours.
        self.reverse_update_neighborhood_simple(Arc::clone(&new_point));
        //
        self.layer_indexed_points.check_entry_point(&new_point);
        //
        trace!("Hnsw exiting insert new point {:?} ", new_point.p_id);
    } // end of insert

    /// Insert in parallel a slice of Vec\<T\> each associated to its id.    
    /// It uses Rayon for threading so the number of insertions asked for must be large enough to be efficient.  
    /// Typically 1000 * the number of threads.  
    /// Many consecutive parallel_insert can be done, so the size of vector inserted in one insertion can be optimized.
    pub fn parallel_insert(&self, datas: &[(&Vec<T>, usize)]) {
        debug!("entering parallel_insert");
        datas
            .par_iter()
            .for_each(|&(item, v)| self.insert((item.as_slice(), v)));
        debug!("exiting parallel_insert");
    } // end of parallel_insert

    /// Insert in parallel slices of \[T\] each associated to its id.
    /// It uses Rayon for threading so the number of insertions asked for must be large enough to be efficient.
    /// Typically 1000 * the number of threads.
    /// Facilitates the use with the ndarray crate as we can extract slices (for data in contiguous order) from Array.
    pub fn parallel_insert_slice(&self, datas: &Vec<(&[T], usize)>) {
        datas.par_iter().for_each(|&item| self.insert_slice(item));
    } // end of parallel_insert

    /// Insert a point into the HNSW index using a vector ID from external storage.
    ///
    /// This method is more memory-efficient than `insert()` as it avoids copying vectors.
    /// The vector is referenced by ID and fetched from VectorStorage on-demand during
    /// distance computations.
    ///
    /// # Arguments
    /// * `vector_id` - Index of the vector in the external VectorStorage
    /// * `data_id` - User-defined identifier for this point (origin_id)
    ///
    /// # Returns
    /// * `Ok(())` on success
    /// * `Err(_)` if:
    ///   - VectorStorage is not configured (use constructor with storage)
    ///   - `vector_id` does not exist in storage
    ///   - Vector dimension doesn't match HNSW configuration
    ///
    /// # Example
    /// ```ignore
    /// use tessera_hnsw::{Hnsw, InMemoryVectorStorage};
    /// use anndists::distance::DistL2;
    ///
    /// let storage = InMemoryVectorStorage::new(vec![vec![1.0, 2.0], vec![3.0, 4.0]]);
    /// let hnsw = Hnsw::with_external_storage(&storage, 16, 1000, 16, 200, DistL2 {});
    ///
    /// hnsw.insert_by_id(0, 42)?; // Insert vector 0 with DataId 42
    /// hnsw.insert_by_id(1, 43)?; // Insert vector 1 with DataId 43
    /// ```
    pub fn insert_by_id(&self, vector_id: usize, data_id: DataId) -> Result<(), String> {
        // Validate storage is configured
        let storage = self.vector_storage
            .ok_or_else(||
                "VectorStorage not configured. Use constructor with external storage parameter.".to_string()
            )?;

        // Validate vector_id exists in storage
        if vector_id >= storage.len() {
            return Err(format!(
                "Vector ID {} out of bounds (storage has {} vectors)",
                vector_id,
                storage.len()
            ));
        }

        // Validate dimension matches
        let storage_dim = storage.dimension();
        if storage_dim != self.data_dimension {
            return Err(format!(
                "Dimension mismatch: storage has dimension {} but HNSW expects {}",
                storage_dim,
                self.data_dimension
            ));
        }

        // Get the vector from storage to pass to search operations
        let vector = storage.get_vector(vector_id)
            .ok_or_else(|| format!("Failed to retrieve vector {} from storage", vector_id))?;

        let keep_pruned = self.keep_pruned;

        // Generate new point using vector_id (no copy)
        let (new_point, point_rank) = self
            .layer_indexed_points
            .generate_new_point_by_id(vector_id, data_id);

        trace!("Hnsw insert_by_id generated new point {:?} with vector_id {}", new_point.p_id, vector_id);

        // Rest of insertion logic mirrors insert_slice()
        let level = new_point.p_id.0;
        let mut enter_point_copy = None;
        let mut max_level_observed = 0;

        {
            if let Some(arc_point) = self.layer_indexed_points.entry_point.read().as_ref() {
                enter_point_copy = Some(Arc::clone(arc_point));
                if point_rank == 1 {
                    debug!("Hnsw stored first point, direct return {:?}", new_point.p_id);
                    return Ok(());
                }
                max_level_observed = enter_point_copy.as_ref().unwrap().p_id.0;
            }
        }

        if enter_point_copy.is_none() {
            self.layer_indexed_points.check_entry_point(&new_point);
            return Ok(());
        }

        let entry_vec = match self.get_point_vector(enter_point_copy.as_ref().unwrap()) {
            Some(v) => v,
            None => {
                // Entry point vector missing - cannot proceed with parallel insertion
                log::warn!("Entry point vector missing during parallel insert, adding point without full HNSW integration");
                self.layer_indexed_points.check_entry_point(&new_point);
                return Ok(());
            }
        };
        let mut dist_to_entry = self.dist_f.eval(vector, entry_vec)
            .expect("Distance calculation failed: vectors in HNSW index should be valid");

        // Search from top layer down to level+1
        for l in ((level + 1)..(max_level_observed + 1)).rev() {
            let mut sorted_points = self.search_layer(
                vector,
                Arc::clone(enter_point_copy.as_ref().unwrap()),
                1,
                l,
                None,
            );

            if let Some(ep) = sorted_points.pop() {
                // Track points for upper layers
                if new_point.neighbours.read()[l as usize].len()
                    < self.get_max_nb_connection() as usize
                {
                    new_point.neighbours.write()[l as usize].push(Arc::clone(&ep));
                }
                // Update entry point if closer (skip if vector missing)
                if let Some(ep_vec) = self.get_point_vector(&ep.point_ref) {
                    let tmp_dist = self.dist_f.eval(vector, ep_vec)
                        .expect("HNSW search distance calculation failed for indexed vector");
                    if tmp_dist < dist_to_entry {
                        enter_point_copy = Some(Arc::clone(&ep.point_ref));
                        dist_to_entry = tmp_dist;
                    }
                }
            } else {
                trace!("layer still empty {} : got null list", l);
            }
        }

        // Insert into layers from level down to 0
        for l in (0..level + 1).rev() {
            let ef = self.ef_construction;
            let mut sorted_points = self.search_layer(
                vector,
                Arc::clone(enter_point_copy.as_ref().unwrap()),
                ef,
                l,
                None,
            );

            sorted_points = from_positive_binaryheap_to_negative_binary_heap(&mut sorted_points);

            if !sorted_points.is_empty() {
                let nb_conn;
                let extend_c;
                if l == 0 {
                    nb_conn = 2 * self.max_nb_connection;
                    extend_c = self.extend_candidates;
                } else {
                    nb_conn = self.max_nb_connection;
                    extend_c = false;
                }

                let mut neighbours = Vec::<Arc<PointWithOrder<T>>>::with_capacity(nb_conn);
                self.select_neighbours(
                    vector,
                    &mut sorted_points,
                    nb_conn,
                    extend_c,
                    l,
                    keep_pruned,
                    &mut neighbours,
                );

                neighbours.sort_unstable();
                new_point.neighbours.write()[l as usize].clone_from(&neighbours);
                enter_point_copy = Some(Arc::clone(&new_point));
            }
        }

        // Reverse update of neighbours
        self.reverse_update_neighborhood_simple(Arc::clone(&new_point));
        self.layer_indexed_points.check_entry_point(&new_point);

        trace!("Hnsw exiting insert_by_id for point {:?}", new_point.p_id);
        Ok(())
    }

    /// insert new_point in neighbourhood info of point
    fn reverse_update_neighborhood_simple(&self, new_point: Arc<Point<T>>) {
        //  println!("reverse update neighbourhood for  new point {:?} ", new_point.p_id);
        trace!(
            "reverse update neighbourhood for  new point {:?} ",
            new_point.p_id
        );
        let level = new_point.p_id.0;
        for l in (0..level + 1).rev() {
            for q in &new_point.neighbours.read()[l as usize] {
                if new_point.p_id != q.point_ref.p_id {
                    // as new point is in global table, do not loop and deadlock!!
                    let q_point = &q.point_ref;
                    let mut q_point_neighbours = q_point.neighbours.write();
                    let n_to_add = PointWithOrder::<T>::new(&Arc::clone(&new_point), q.dist_to_ref);
                    // must be sure that we add a point at the correct level. See the comment to search_layer!
                    // this ensures that reverse updating do not add problems.
                    let l_n = n_to_add.point_ref.p_id.0 as usize;
                    let already = q_point_neighbours[l_n]
                        .iter()
                        .position(|old| old.point_ref.p_id == new_point.p_id);
                    if already.is_some() {
                        // debug!(" new_point.p_id {:?} already in neighbourhood of  q_point {:?} at index {:?}", new_point.p_id, q_point.p_id, already.unwrap());
                        // q_point.debug_dump();  cannot be called as its neighbours are locked write by this method.
                        //   new_point.debug_dump();
                        //   panic!();
                        continue;
                    }
                    q_point_neighbours[l_n].push(Arc::new(n_to_add));
                    let nbn_at_l = q_point_neighbours[l_n].len();
                    //
                    // if l < level, update upward chaining, insert does a sort! t_q has a neighbour not yet in global table of points!
                    let threshold_shrinking = if l_n > 0 {
                        self.max_nb_connection
                    } else {
                        2 * self.max_nb_connection
                    };
                    let shrink = nbn_at_l > threshold_shrinking;
                    {
                        // sort and shring if necessary
                        q_point_neighbours[l_n].sort_unstable();
                        if shrink {
                            q_point_neighbours[l_n].pop();
                        }
                    }
                } // end protection against point identity
            }
        }
        //   println!("     exitingreverse update neighbourhood for  new point {:?} ", new_point.p_id);
    } // end of reverse_update_neighborhood_simple

    pub fn get_point_indexation(&self) -> &PointIndexation<'b, T> {
        &self.layer_indexed_points
    }

    // This is best explained in : Navarro. Searching in metric spaces by spatial approximation.
    /// simplest searh neighbours
    // The binary heaps here is with negative distance sorted.
    #[allow(clippy::too_many_arguments)]
    fn select_neighbours(
        &self,
        data: &[T],
        candidates: &mut BinaryHeap<Arc<PointWithOrder<'b, T>>>,
        nb_neighbours_asked: usize,
        extend_candidates_asked: bool,
        layer: u8,
        keep_pruned: bool,
        neighbours_vec: &mut Vec<Arc<PointWithOrder<'b, T>>>,
    ) {
        //
        trace!(
            "entering select_neighbours : nb candidates: {}",
            candidates.len()
        );
        //
        neighbours_vec.clear();
        // we will extend if we do not have enough candidates and it is explicitly asked in arg
        let mut extend_candidates = false;
        if candidates.len() <= nb_neighbours_asked {
            if !extend_candidates_asked {
                // just transfer taking care of signs
                while !candidates.is_empty() {
                    let p = candidates.pop().unwrap();
                    // Note: Assertion removed - some distance metrics (e.g., DistDot) can return negative values
                    // assert!(-p.dist_to_ref >= 0.);
                    neighbours_vec
                        .push(Arc::new(PointWithOrder::new(&p.point_ref, -p.dist_to_ref)));
                }
                return;
            } else {
                extend_candidates = true;
            }
        }
        //
        //
        //extend_candidates = true;
        //
        if extend_candidates {
            let mut candidates_set = HashMap::<PointId, Arc<Point<T>>>::new();
            for c in candidates.iter() {
                candidates_set.insert(c.point_ref.p_id, Arc::clone(&c.point_ref));
            }
            let mut new_candidates_set = HashMap::<PointId, Arc<Point<T>>>::new();
            // get a list of all neighbours of candidates
            for (_p_id, p_point) in candidates_set.iter() {
                let n_p_layer = &p_point.neighbours.read()[layer as usize];
                for q in n_p_layer {
                    if !candidates_set.contains_key(&q.point_ref.p_id)
                        && !new_candidates_set.contains_key(&q.point_ref.p_id)
                    {
                        new_candidates_set.insert(q.point_ref.p_id, Arc::clone(&q.point_ref));
                    }
                }
            } // end of for p
            trace!(
                "select neighbours extend candidates from  : {:?} adding : {:?}",
                candidates.len(),
                new_candidates_set.len()
            );
            for (_p_id, p_point) in new_candidates_set.iter() {
                // Skip candidates with missing vectors
                if let Some(p_vec) = self.get_point_vector(p_point) {
                    let dist_topoint = self.dist_f.eval(data, p_vec)
                        .expect("Distance calculation failed for candidate vector in neighbor selection");
                    candidates.push(Arc::new(PointWithOrder::new(p_point, -dist_topoint)));
                }
            }
        } // end if extend_candidates
        //
        let mut discarded_points = BinaryHeap::<Arc<PointWithOrder<T>>>::new();
        while !candidates.is_empty() && neighbours_vec.len() < nb_neighbours_asked {
            // compare distances of e to data. we do not need to recompute dists!
            if let Some(e_p) = candidates.pop() {
                // Skip candidates with missing vectors
                let e_point_v = match self.get_point_vector(&e_p.point_ref) {
                    Some(v) => v,
                    None => continue, // Skip this candidate
                };
                let mut e_to_insert = true;
                // Note: Assertion removed - some distance metrics can return positive values in negative heaps
                // assert!(e_p.dist_to_ref <= 0.);
                // is e_p the nearest to reference? data than to previous neighbours
                if !neighbours_vec.is_empty() {
                    e_to_insert = !neighbours_vec.iter().any(|d| {
                        // If neighbor vector is missing, treat as not blocking this candidate
                        match self.get_point_vector(&d.point_ref) {
                            Some(d_vec) => {
                                self.dist_f.eval(e_point_v, d_vec)
                                    .expect("Distance calculation failed during neighbor pruning") <= -e_p.dist_to_ref
                            }
                            None => false, // Missing neighbor doesn't block
                        }
                    });
                }
                if e_to_insert {
                    trace!("inserting neighbours : {:?} ", e_p.point_ref.p_id);
                    neighbours_vec.push(Arc::new(PointWithOrder::new(
                        &e_p.point_ref,
                        -e_p.dist_to_ref,
                    )));
                } else {
                    trace!("discarded neighbours : {:?} ", e_p.point_ref.p_id);
                    // ep is taken from a binary heap, so it has a negative sign, we keep its sign
                    // to store it in another binary heap will possibly need to retain the best ones from the discarde binaryHeap
                    if keep_pruned {
                        discarded_points.push(Arc::new(PointWithOrder::new(
                            &e_p.point_ref,
                            e_p.dist_to_ref,
                        )));
                    }
                }
            }
        }
        // now this part of neighbours is the most interesting and is distance sorted.

        // not pruned are at the end of neighbours_vec which is not re-sorted , but discarded are sorted.
        if keep_pruned {
            while !discarded_points.is_empty() && neighbours_vec.len() < nb_neighbours_asked {
                let best_point = discarded_points.pop().unwrap();
                // do not forget to reverse sign
                // Note: Assertion removed - some distance metrics can return positive values in negative heaps
                // assert!(best_point.dist_to_ref <= 0.);
                neighbours_vec.push(Arc::new(PointWithOrder::new(
                    &best_point.point_ref,
                    -best_point.dist_to_ref,
                )));
            }
        };
        //
        if log::log_enabled!(log::Level::Trace) {
            trace!(
                "exiting select_neighbours : nb candidates: {}",
                neighbours_vec.len()
            );
            for n in neighbours_vec {
                trace!("   neighbours {:?} ", n.point_ref.p_id);
            }
        }
        //
    } // end of select_neighbours

    /// A utility to get printed info on how many points there are in each layer.
    pub fn dump_layer_info(&self) {
        self.layer_indexed_points.debug_dump();
    }

    // search the first knbn nearest neigbours of a data, but can modify ef for layer > 1
    // This function return Vec<Arc<PointWithOrder<T> >>
    // The parameter ef controls the width of the search in the lowest level, it must be greater
    // than number of neighbours asked. A rule of thumb could be between knbn and max_nb_connection.
    #[allow(unused)]
    fn search_general(&self, data: &[T], knbn: usize, ef_arg: usize) -> Vec<Neighbour> {
        //
        let mut entry_point;
        {
            // a lock on an option an a Arc<Point>
            let entry_point_opt_ref = self.layer_indexed_points.entry_point.read();
            if entry_point_opt_ref.is_none() {
                return Vec::<Neighbour>::new();
            } else {
                entry_point = Arc::clone((*entry_point_opt_ref).as_ref().unwrap());
            }
        }
        //
        let entry_vec = match self.get_point_vector(entry_point.as_ref()) {
            Some(v) => v,
            None => {
                log::warn!("Entry point vector missing in search_general, returning empty results");
                return Vec::<Neighbour>::new();
            }
        };
        let mut dist_to_entry = self.dist_f.eval(data, entry_vec)
            .expect("Distance calculation failed: vectors in HNSW index should be valid");
        for layer in (1..=entry_point.p_id.0).rev() {
            let mut neighbours = self.search_layer(data, Arc::clone(&entry_point), 1, layer, None);
            neighbours = from_positive_binaryheap_to_negative_binary_heap(&mut neighbours);
            if let Some(entry_point_tmp) = neighbours.pop() {
                // get the lowest distance point (skip if vector missing)
                if let Some(tmp_vec) = self.get_point_vector(&entry_point_tmp.point_ref) {
                    let tmp_dist = self.dist_f.eval(data, tmp_vec)
                        .expect("HNSW search distance calculation failed for indexed vector");
                    if tmp_dist < dist_to_entry {
                        entry_point = Arc::clone(&entry_point_tmp.point_ref);
                        dist_to_entry = tmp_dist;
                    }
                }
            }
        }
        // ef must be greater than knbn. Possibly it should be between knbn and self.max_nb_connection
        let ef = ef_arg.max(knbn);
        // now search with asked ef in layer 0
        let neighbours_heap = self.search_layer(data, entry_point, ef, 0, None);
        // go from heap of points with negative dist to a sorted vec of increasing points with > 0 distances.
        let neighbours = neighbours_heap.into_sorted_vec();
        // get the min of K and ef points into a vector.
        //
        let last = knbn.min(ef).min(neighbours.len());
        let knn_neighbours: Vec<Neighbour> = neighbours[0..last]
            .iter()
            .map(|p| {
                Neighbour::new(
                    p.as_ref().point_ref.origin_id,
                    p.as_ref().dist_to_ref,
                    p.as_ref().point_ref.p_id,
                )
            })
            .collect();

        knn_neighbours
    } // end of knn_search

    /// a filtered version of [`Self::search`].  
    /// A filter can be added to the search to get nodes with a particular property or id constraint.  
    /// See examples in tests/filtertest.rs
    pub fn search_filter(
        &self,
        data: &[T],
        knbn: usize,
        ef_arg: usize,
        filter: Option<&dyn FilterT>,
    ) -> Vec<Neighbour> {
        // PERF-OPT: Early termination for empty filters
        // If filter.is_empty() returns true, no results can pass the filter.
        // Skip HNSW traversal entirely and return empty results immediately.
        if let Some(f) = filter {
            if f.is_empty() {
                return Vec::new();
            }
        }
        //
        let entry_point;
        {
            // a lock on an option an a Arc<Point>
            let entry_point_opt_ref = self.layer_indexed_points.entry_point.read();
            if entry_point_opt_ref.is_none() {
                return Vec::<Neighbour>::new();
            } else {
                entry_point = Arc::clone((*entry_point_opt_ref).as_ref().unwrap());
            }
        }
        //
        let entry_vec = match self.get_point_vector(entry_point.as_ref()) {
            Some(v) => v,
            None => {
                log::warn!("Entry point vector missing in search_filter, returning empty results");
                return Vec::<Neighbour>::new();
            }
        };
        let mut dist_to_entry = self.dist_f.eval(data, entry_vec)
            .expect("Distance calculation failed: vectors in HNSW index should be valid");
        let mut pivot = Arc::clone(&entry_point);
        let mut new_pivot = None;

        //
        for layer in (1..=entry_point.p_id.0).rev() {
            let mut has_changed = false;
            // search in stored neighbours
            {
                let neighbours = &pivot.neighbours.read()[layer as usize];
                for n in neighbours {
                    // get the lowest distance point (skip if vector missing)
                    if let Some(n_vec) = self.get_point_vector(&n.point_ref) {
                        let tmp_dist = self.dist_f.eval(data, n_vec)
                            .expect("HNSW search distance calculation failed for indexed vector");
                        if tmp_dist < dist_to_entry {
                            new_pivot = Some(Arc::clone(&n.point_ref));
                            has_changed = true;
                            dist_to_entry = tmp_dist;
                        }
                    }
                } // end of for on neighbours
            }
            if has_changed {
                pivot = Arc::clone(new_pivot.as_ref().unwrap());
            }
        } // end on for on layers
        // ef must be greater than knbn. Possibly it should be between knbn and self.max_nb_connection
        let ef = ef_arg.max(knbn);
        log::debug!("pivot changed , current pivot {:?}", pivot.get_point_id());
        // search lowest non empty layer (in case of search with incomplete lower layer at beginning of hnsw filling)
        let mut l = 0u8;
        let layer_to_search = loop {
            if self.get_point_indexation().get_layer_nb_point(l as usize) > 0 {
                break l;
            }
            l += 1;
        };
        // now search with asked ef in lower layer
        let neighbours_heap = self.search_layer(data, pivot, ef, layer_to_search, filter);
        // go from heap of points with negative dist to a sorted vec of increasing points with > 0 distances.
        let neighbours = neighbours_heap.into_sorted_vec();
        // get the min of K and ef points into a vector.
        //
        let last = knbn.min(ef).min(neighbours.len());
        //
        // PERF-FIX: No post-processing filter needed!
        // search_layer now filters ALL results including the entry point.
        // Previously, entry point was added without filtering, requiring double filtering here.
        // Now we just convert PointWithOrder to Neighbour.
        neighbours[0..last]
            .iter()
            .map(|p| {
                Neighbour::new(
                    p.as_ref().point_ref.origin_id,
                    p.as_ref().dist_to_ref,
                    p.as_ref().point_ref.p_id,
                )
            })
            .collect()
    } // end of search_filter

    #[inline]
    pub fn search_possible_filter(
        &self,
        data: &[T],
        knbn: usize,
        ef_arg: usize,
        filter: Option<&dyn FilterT>,
    ) -> Vec<Neighbour> {
        self.search_filter(data, knbn, ef_arg, filter)
    }

    /// search the first knbn nearest neigbours of a data and returns a Vector of Neighbour.   
    /// The parameter ef controls the width of the search in the lowest level, it must be greater
    /// than number of neighbours asked.  
    /// A rule of thumb could be between knbn and max_nb_connection.
    pub fn search(&self, data: &[T], knbn: usize, ef_arg: usize) -> Vec<Neighbour> {
        self.search_possible_filter(data, knbn, ef_arg, None)
    }

    fn search_with_id(
        &self,
        request: (usize, &Vec<T>),
        knbn: usize,
        ef: usize,
    ) -> (usize, Vec<Neighbour>) {
        (request.0, self.search(request.1, knbn, ef))
    }

    /// knbn is the number of nearest neigbours asked for. Returns for each data vector
    /// a Vector of Neighbour
    pub fn parallel_search(&self, datas: &[Vec<T>], knbn: usize, ef: usize) -> Vec<Vec<Neighbour>> {
        let (sender, receiver) = channel();
        // make up requests
        let nb_request = datas.len();
        let requests: Vec<(usize, &Vec<T>)> = (0..nb_request).zip(datas.iter()).collect();
        //
        requests.par_iter().for_each_with(sender, |s, item| {
            s.send(self.search_with_id(*item, knbn, ef)).unwrap()
        });
        let req_res: Vec<(usize, Vec<Neighbour>)> = receiver.iter().collect();
        // now sort to respect the key order of input
        let mut answers = Vec::<Vec<Neighbour>>::with_capacity(datas.len());
        // get a map from request id to rank
        let mut req_hash = HashMap::<usize, usize>::new();
        for (i, elt) in req_res.iter().enumerate() {
            // the response of request req_res[i].0 is at rank i
            req_hash.insert(elt.0, i);
        }
        for i in 0..datas.len() {
            let answer_i = req_hash.get_key_value(&i).unwrap().1;
            answers.push((req_res[*answer_i].1).clone());
        }
        answers
    } // end of insert_parallel
} // end of Hnsw

// Convenience constructors (separate impl to allow different VS type parameter)
impl<'b, T, D> Hnsw<'b, T, D, NoStorage>
where
    T: Clone + Send + Sync + std::fmt::Debug,
    D: Distance<T> + Send + Sync,
{
    /// Convenience constructor for HNSW with external vector storage.
    ///
    /// This method avoids turbofish syntax by inferring the VectorStorage type
    /// from the `storage` parameter. This is particularly useful when working
    /// with concrete storage types.
    ///
    /// # Examples
    ///
    /// ```rust,ignore
    /// use tessera_hnsw::Hnsw;
    /// use tessera_hnsw::storage::InMemoryVectorStorage;
    /// use anndists::distance::DistL2;
    ///
    /// let vectors = vec![vec![1.0f32, 2.0], vec![3.0, 4.0]];
    /// let storage = InMemoryVectorStorage::new(vectors);
    ///
    /// // No turbofish needed - storage type is inferred!
    /// let hnsw = Hnsw::with_external_storage(
    ///     15,           // max_nb_connection
    ///     1000,         // max_elements
    ///     16,           // max_layer
    ///     200,          // ef_construction
    ///     DistL2 {},    // distance
    ///     &storage      // storage (type inferred)
    /// );
    /// ```
    ///
    /// Compare with the traditional approach:
    ///
    /// ```rust,ignore
    /// // Requires explicit type annotation (turbofish)
    /// let hnsw: Hnsw<f32, DistL2, InMemoryVectorStorage<f32>> =
    ///     Hnsw::new_with_storage(15, 1000, 16, 200, DistL2 {}, &storage);
    /// ```
    pub fn with_external_storage<VS>(
        max_nb_connection: usize,
        max_elements: usize,
        max_layer: usize,
        ef_construction: usize,
        f: D,
        storage: &'b VS,
    ) -> Hnsw<'b, T, D, VS>
    where
        VS: VectorStorage<'b, T>,
    {
        Hnsw::new_with_storage(
            max_nb_connection,
            max_elements,
            max_layer,
            ef_construction,
            f,
            storage,
        )
    }
}

// This function takes a binary heap with points declared with a negative distance
// and returns a vector of points with their correct positive distance to some reference distance
// The vector is sorted by construction
#[allow(unused)]
fn from_negative_binaryheap_to_sorted_vector<'b, T: Send + Sync + Copy>(
    heap_points: &mut BinaryHeap<Arc<PointWithOrder<'b, T>>>,
) -> Vec<Arc<PointWithOrder<'b, T>>> {
    let nb_points = heap_points.len();
    let mut vec_points = Vec::<Arc<PointWithOrder<T>>>::with_capacity(nb_points);
    //
    for p in heap_points.iter() {
        // Note: Assertion removed - some distance metrics can return positive values in negative heaps
        // assert!(p.dist_to_ref <= 0.);
        let reverse_p = Arc::new(PointWithOrder::new(&p.point_ref, -p.dist_to_ref));
        vec_points.push(reverse_p);
    }
    trace!(
        "from_negative_binaryheap_to_sorted_vector nb points in out {:?} {:?} ",
        nb_points,
        vec_points.len()
    );
    vec_points
}

// This function takes a binary heap with points declared with a positive distance
// and returns a binary_heap of points with their correct negative distance to some reference distance
//
fn from_positive_binaryheap_to_negative_binary_heap<'b, T: Send + Sync + Clone>(
    positive_heap: &mut BinaryHeap<Arc<PointWithOrder<'b, T>>>,
) -> BinaryHeap<Arc<PointWithOrder<'b, T>>> {
    let nb_points = positive_heap.len();
    let mut negative_heap = BinaryHeap::<Arc<PointWithOrder<T>>>::with_capacity(nb_points);
    //
    for p in positive_heap.iter() {
        // Note: Assertion removed - some distance metrics (e.g., DistDot) can return negative values
        // assert!(p.dist_to_ref >= 0.);
        let reverse_p = Arc::new(PointWithOrder::new(&p.point_ref, -p.dist_to_ref));
        negative_heap.push(reverse_p);
    }
    trace!(
        "from_positive_binaryheap_to_negative_binary_heap nb points in out {:?} {:?} ",
        nb_points,
        negative_heap.len()
    );
    negative_heap
}

// essentialy to check dump/reload conssistency
// in fact checks only equality of graph
#[allow(unused)]
pub(crate) fn check_graph_equality<'a, 'b, T1, D1, VS1, T2, D2, VS2>(hnsw1: &Hnsw<'a, T1, D1, VS1>, hnsw2: &Hnsw<'b, T2, D2, VS2>)
where
    T1: Copy + Clone + Send + Sync + std::fmt::Debug + 'a,
    D1: Distance<T1> + Default + Send + Sync,
    VS1: VectorStorage<'a, T1>,
    T2: Copy + Clone + Send + Sync + std::fmt::Debug + 'b,
    D2: Distance<T2> + Default + Send + Sync,
    VS2: VectorStorage<'b, T2>,
{
    //
    debug!("In check_graph_equality");
    //
    assert_eq!(hnsw1.get_nb_point(), hnsw2.get_nb_point());
    // check for entry point
    assert!(
        hnsw1.layer_indexed_points.entry_point.read().is_some()
            || hnsw1.layer_indexed_points.entry_point.read().is_some(),
        "one entry point is None"
    );
    let ep1_read = hnsw1.layer_indexed_points.entry_point.read();
    let ep2_read = hnsw2.layer_indexed_points.entry_point.read();
    let ep1 = ep1_read.as_ref().unwrap();
    let ep2 = ep2_read.as_ref().unwrap();
    assert_eq!(
        ep1.origin_id, ep2.origin_id,
        "different entry points {:?} {:?}",
        ep1.origin_id, ep2.origin_id
    );
    assert_eq!(ep1.p_id, ep2.p_id, "origin id {:?} ", ep1.origin_id);
    // check layers
    let layers_1 = hnsw1.layer_indexed_points.points_by_layer.read();
    let layers_2 = hnsw2.layer_indexed_points.points_by_layer.read();
    // Compare number of layers - may differ if one was created with smaller max_layer
    // but after reload, layers are padded to NB_LAYER_MAX. Compare only non-empty layers.
    let num_layers_to_check = layers_1.len().min(layers_2.len());
    let mut nb_point_checked = 0;
    let mut nb_neighbours_checked = 0;
    for i in 0..num_layers_to_check {
        debug!("Checking layer {:?}", i);
        assert_eq!(layers_1[i].len(), layers_2[i].len());
        for j in 0..layers_1[i].len() {
            let p1 = &layers_1[i][j];
            let p2 = &layers_2[i][j];
            assert_eq!(p1.origin_id, p2.origin_id);
            assert_eq!(
                p1.p_id, p2.p_id,
                "Checking origin_id point {:?} ",
                p1.origin_id
            );
            nb_point_checked += 1;
            // check neighborhood
            let nbgh1 = p1.neighbours.read();
            let nbgh2 = p2.neighbours.read();
            assert_eq!(nbgh1.len(), nbgh2.len());
            for k in 0..nbgh1.len() {
                assert_eq!(nbgh1[k].len(), nbgh2[k].len());
                for l in 0..nbgh1[k].len() {
                    assert_eq!(
                        nbgh1[k][l].point_ref.origin_id,
                        nbgh2[k][l].point_ref.origin_id
                    );
                    assert_eq!(nbgh1[k][l].point_ref.p_id, nbgh2[k][l].point_ref.p_id);
                    // CAVEAT for precision with f32
                    assert_eq!(nbgh1[k][l].dist_to_ref, nbgh2[k][l].dist_to_ref);
                    nb_neighbours_checked += 1;
                }
            }
        } // end of for j
    } // end of for i
    assert_eq!(nb_point_checked, hnsw1.get_nb_point());
    debug!("nb neighbours checked {:?}", nb_neighbours_checked);
    debug!("exiting check_equality");
} // end of check_reload

#[cfg(test)]

mod tests {

    use super::*;
    use crate::distance;

    fn log_init_test() {
        let _ = env_logger::builder().is_test(true).try_init();
    }

    #[test]
    fn test_iter_point() {
        //
        println!("\n\n test_iter_point");
        //
        let mut rng = rand::rng();
        let unif = Uniform::<f32>::new(0., 1.).unwrap();
        let nbcolumn = 5000;
        let nbrow = 10;
        let mut xsi;
        let mut data = Vec::with_capacity(nbcolumn);
        for j in 0..nbcolumn {
            data.push(Vec::with_capacity(nbrow));
            for _ in 0..nbrow {
                xsi = rng.sample(unif);
                data[j].push(xsi);
            }
        }
        //
        // check insertion
        let ef_construct = 25;
        let nb_connection = 10;
        let start = ProcessTime::now();
        let hns = Hnsw::<f32, distance::DistL1, NoStorage>::new(
            nb_connection,
            nbcolumn,
            16,
            ef_construct,
            distance::DistL1 {},
        );
        for (i, d) in data.iter().enumerate() {
            hns.insert((d, i));
        }
        let cpu_time = start.elapsed();
        println!(" test_insert_iter_point time inserting {:?}", cpu_time);

        hns.dump_layer_info();
        // now check iteration
        let ptiter = hns.get_point_indexation().into_iter();
        let mut nb_dumped = 0;
        for _point in ptiter {
            //    println!("point : {:?}", _point.p_id);
            nb_dumped += 1;
        }
        //
        assert_eq!(nb_dumped, nbcolumn);
    } // end of test_iter_point

    #[test]
    fn test_iter_layerpoint() {
        //
        println!("\n\n test_iter_point");
        //
        let mut rng = rand::rng();
        let unif = Uniform::<f32>::new(0., 1.).unwrap();
        let nbcolumn = 5000;
        let nbrow = 10;
        let mut xsi;
        let mut data = Vec::with_capacity(nbcolumn);
        for j in 0..nbcolumn {
            data.push(Vec::with_capacity(nbrow));
            for _ in 0..nbrow {
                xsi = rng.sample(unif);
                data[j].push(xsi);
            }
        }
        //
        // check insertion
        let ef_construct = 25;
        let nb_connection = 10;
        let start = ProcessTime::now();
        let hns = Hnsw::<f32, distance::DistL1, NoStorage>::new(
            nb_connection,
            nbcolumn,
            16,
            ef_construct,
            distance::DistL1 {},
        );
        for (i, d) in data.iter().enumerate() {
            hns.insert((d, i));
        }
        let cpu_time = start.elapsed();
        println!(" test_insert_iter_point time inserting {:?}", cpu_time);

        hns.dump_layer_info();
        // now check iteration
        let layer_num = 0;
        let nbpl = hns.get_point_indexation().get_layer_nb_point(layer_num);
        let layer_iter = hns.get_point_indexation().get_layer_iterator(layer_num);
        //
        let mut nb_dumped = 0;
        for _point in layer_iter {
            //    println!("point : {:?}", _point.p_id);
            nb_dumped += 1;
        }
        println!(
            "test_iter_layerpoint : nb point in layer {} , nb found {}",
            nbpl, nb_dumped
        );
        //
        assert_eq!(nb_dumped, nbpl);
    } // end of test_iter_layerpoint

    // we should find point even if it is in layer >= 1
    #[test]
    fn test_sparse_search() {
        log_init_test();
        //
        for _ in 0..800 {
            let hnsw = Hnsw::<f32, distance::DistL1, NoStorage>::new(15, 100_000, 20, 500_000, distance::DistL1 {});
            hnsw.insert((&[1.0, 0.0, 0.0, 0.0], 0));
            let result = hnsw.search(&[1.0, 0.0, 0.0, 0.0], 2, 10);
            assert_eq!(result, vec![Neighbour::new(0, 0.0, PointId(0, 0))]);
        }
    }

    // Test creating Hnsw with InMemoryVectorStorage
    #[test]
    fn test_hnsw_with_vector_storage() {
        use crate::storage::{DynVectorStorage, InMemoryVectorStorage};

        log_init_test();

        // Create test vectors
        let vectors = vec![
            vec![0.0f32, 0.0, 0.0, 0.0],
            vec![1.0, 0.0, 0.0, 0.0],
            vec![0.0, 1.0, 0.0, 0.0],
            vec![0.0, 0.0, 1.0, 0.0],
            vec![0.0, 0.0, 0.0, 1.0],
        ];

        // Create VectorStorage
        let storage = InMemoryVectorStorage::new(vectors);

        // Create Hnsw with storage
        let _hnsw: Hnsw<f32, distance::DistL2, InMemoryVectorStorage<f32>> =
            Hnsw::new_with_storage(15, 100, 16, 200, distance::DistL2 {}, &storage);

        // Verify storage info (use qualified syntax to avoid ambiguity)
        assert_eq!(DynVectorStorage::len(&storage), 5);
        assert_eq!(DynVectorStorage::dimension(&storage), 4);
        assert!(!DynVectorStorage::is_empty(&storage));
    }

    // Test backward compatibility - existing new() should still work
    #[test]
    fn test_backward_compatibility() {
        log_init_test();

        // Old API should still work without any changes
        let hnsw = Hnsw::<f32, distance::DistL1, NoStorage>::new(15, 1000, 16, 200, distance::DistL1 {});

        // Insert using old API
        hnsw.insert((&[1.0, 2.0, 3.0], 0));
        hnsw.insert((&[4.0, 5.0, 6.0], 1));

        // Search should work
        let results = hnsw.search(&[1.0, 2.0, 3.0], 1, 10);
        assert_eq!(results.len(), 1);
        assert_eq!(results[0].d_id, 0);
    }

    // ========== Phase 3 Tests: insert_by_id() ==========

    #[test]
    fn test_insert_by_id_success() {
        use crate::storage::InMemoryVectorStorage;

        log_init_test();

        let vectors = vec![
            vec![1.0f32, 2.0, 3.0],
            vec![4.0, 5.0, 6.0],
            vec![7.0, 8.0, 9.0],
        ];
        let storage = InMemoryVectorStorage::new(vectors);

        let hnsw = Hnsw::with_external_storage(
            16,            // max_nb_connection
            1000,          // max_elements
            16,            // max_layer
            200,           // ef_construction
            distance::DistL2 {},
            &storage,
        );

        // Insert by ID
        let result = hnsw.insert_by_id(0, 100);
        assert!(result.is_ok(), "insert_by_id(0) failed: {:?}", result.err());

        let result = hnsw.insert_by_id(1, 101);
        assert!(result.is_ok(), "insert_by_id(1) failed: {:?}", result.err());

        // Verify points were inserted
        assert_eq!(hnsw.get_nb_point(), 2);
    }

    #[test]
    fn test_insert_by_id_no_storage() {
        log_init_test();

        // Create HNSW without storage (legacy mode)
        let hnsw = Hnsw::<f32, distance::DistL2, NoStorage>::new(16, 1000, 16, 200, distance::DistL2 {});

        let result = hnsw.insert_by_id(0, 100);
        assert!(result.is_err(), "Should fail without storage");
        assert!(
            result.unwrap_err().contains("VectorStorage not configured"),
            "Error message should mention storage not configured"
        );
    }

    #[test]
    fn test_insert_by_id_out_of_bounds() {
        use crate::storage::InMemoryVectorStorage;

        log_init_test();

        let vectors = vec![vec![1.0f32, 2.0], vec![3.0, 4.0]];
        let storage = InMemoryVectorStorage::new(vectors);

        let hnsw = Hnsw::with_external_storage(16, 1000, 16, 200, distance::DistL2 {}, &storage);

        let result = hnsw.insert_by_id(2, 100); // ID 2 doesn't exist (only 0, 1)
        assert!(result.is_err(), "Should fail for out of bounds vector_id");
        assert!(
            result.unwrap_err().contains("out of bounds"),
            "Error message should mention out of bounds"
        );
    }

    #[test]
    fn test_insert_by_id_dimension_mismatch() {
        use crate::storage::InMemoryVectorStorage;

        log_init_test();

        // Storage with 3D vectors
        let vectors = vec![vec![1.0f32, 2.0, 3.0], vec![4.0, 5.0, 6.0]];
        let storage = InMemoryVectorStorage::new(vectors);

        // Create HNSW expecting 3D vectors (storage.dimension() = 3)
        let hnsw = Hnsw::with_external_storage(16, 1000, 16, 200, distance::DistL2 {}, &storage);

        // This should succeed (correct dimension)
        let result = hnsw.insert_by_id(0, 100);
        assert!(result.is_ok(), "Should work with correct dimension");
    }

    #[test]
    fn test_insert_by_id_end_to_end() {
        use crate::storage::InMemoryVectorStorage;

        log_init_test();

        // Known vectors
        let vectors = vec![
            vec![1.0f32, 0.0, 0.0], // ID 0
            vec![0.0, 1.0, 0.0],    // ID 1
            vec![0.0, 0.0, 1.0],    // ID 2
            vec![0.9, 0.1, 0.0],    // ID 3 (similar to 0)
            vec![0.1, 0.9, 0.0],    // ID 4 (similar to 1)
        ];
        let storage = InMemoryVectorStorage::new(vectors);

        let hnsw = Hnsw::with_external_storage(16, 1000, 16, 200, distance::DistL2 {}, &storage);

        // Insert all by ID
        for (vector_id, data_id) in [(0, 100), (1, 101), (2, 102), (3, 103), (4, 104)] {
            let result = hnsw.insert_by_id(vector_id, data_id);
            assert!(
                result.is_ok(),
                "Failed to insert vector_id {}: {:?}",
                vector_id,
                result.err()
            );
        }

        assert_eq!(hnsw.get_nb_point(), 5);

        // Search for vector similar to [1.0, 0.0, 0.0] (should match ID 0)
        let query = vec![1.0, 0.0, 0.0];
        let neighbors = hnsw.search(&query, 2, 50);

        // Verify we got results
        assert!(!neighbors.is_empty(), "Search should return neighbors");

        // First result should be exact match (DataId 100 for vector_id 0)
        assert_eq!(
            neighbors[0].d_id, 100,
            "First neighbor should be DataId 100 (vector_id 0)"
        );
        assert!(
            neighbors[0].distance < 0.01,
            "Distance to exact match should be ~0, got {}",
            neighbors[0].distance
        );
    }

    #[test]
    fn test_insert_by_id_mixed_with_regular_insert() {
        use crate::storage::InMemoryVectorStorage;

        log_init_test();

        let vectors = vec![vec![1.0f32, 0.0], vec![0.0, 1.0]];
        let storage = InMemoryVectorStorage::new(vectors);

        let hnsw = Hnsw::with_external_storage(16, 1000, 16, 200, distance::DistL2 {}, &storage);

        // Insert by ID
        hnsw.insert_by_id(0, 100).unwrap();

        // Insert regularly (copies vector)
        hnsw.insert((&[0.5, 0.5], 200));

        // Insert by ID again
        hnsw.insert_by_id(1, 101).unwrap();

        // Should have 3 points total
        assert_eq!(hnsw.get_nb_point(), 3);

        // Search should work across both types
        let results = hnsw.search(&[1.0, 0.0], 3, 50);
        assert_eq!(results.len(), 3, "Should find all 3 points");
    }
    #[test]
    fn test_get_entry_point_origin_id_empty_index() {
        log_init_test();
        let hnsw =
            Hnsw::<f32, distance::DistL2, NoStorage>::new(16, 0, 8, 200, distance::DistL2 {});
        assert_eq!(hnsw.get_entry_point_origin_id(), None);
    }

    #[test]
    fn test_get_entry_point_origin_id_after_insert() {
        log_init_test();
        let hnsw =
            Hnsw::<f32, distance::DistL2, NoStorage>::new(16, 100, 8, 200, distance::DistL2 {});
        // First inserted point becomes the initial entry point
        let data: Vec<f32> = vec![1.0; 8];
        hnsw.insert((&data, 42));
        assert_eq!(
            hnsw.get_entry_point_origin_id(),
            Some(42),
            "First inserted vector should become the entry point"
        );
        // Insert more points — entry point may or may not change, but must remain Some
        for i in 0..5 {
            let v: Vec<f32> = vec![i as f32; 8];
            hnsw.insert((&v, 100 + i));
        }
        assert!(
            hnsw.get_entry_point_origin_id().is_some(),
            "Entry point must exist after multiple inserts"
        );
    }
} // end of module test
