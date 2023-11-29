use crate::model::ModelTrait;
use anyhow::Result;
use candle_core::Tensor;
use fxhash::FxHashMap;

pub type StateIdType = u32;

pub trait StateTrait<T> {
    fn get_id(&self) -> StateIdType;
    fn get_tensor(&self) -> Result<&Tensor>;
    fn get_forward_flow(
        &self,
        next_state: &impl StateTrait<T>,
        model: &impl ModelTrait<T>,
    ) -> Result<Tensor>;
    fn get_previous_flow(
        &self,
        previous_state: &impl StateTrait<T>,
        model: &impl ModelTrait<T>,
    ) -> Result<Tensor>;
    fn get_data(&self) -> T;
}
#[derive(Debug)]
pub struct State<T> {
    pub id: StateIdType,
    pub data: T,
    pub tensor: Tensor,
    pub is_terminal: bool,
    pub reward: f32,
}



pub struct StateCollection<I, S> {
    pub map: FxHashMap<I, Box<S>>,
}
impl<I, S> StateCollection<I, S> {
    pub fn new() -> Self {
        let map = FxHashMap::<I, Box<S>>::default();
        Self { map }
    }
}

impl<I, S> Default for StateCollection<I, S> {
    fn default() -> Self {
        Self::new()
    }
}
