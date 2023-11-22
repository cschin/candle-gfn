use crate::state::StateTrait;
use anyhow::Result;
use candle_core::Tensor;

pub trait ModelTrait: Sized {
    fn forward_ssp(&self, source: &impl StateTrait, sink: &impl StateTrait) -> Result<Tensor>;
    fn reverse_ssp(&self, source: &impl StateTrait, sink: &impl StateTrait) -> Result<Tensor>;
}
