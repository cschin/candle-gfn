use crate::state::StateTrait;
use anyhow::Result;
use candle_core::Tensor;

pub trait ModelTrait<T>: Sized {
    fn forward_ssp(&self, source: &impl StateTrait<T>, sink: &impl StateTrait<T>) -> Result<Tensor>;
    fn reverse_ssp(&self, source: &impl StateTrait<T>, sink: &impl StateTrait<T>) -> Result<Tensor>;
}
