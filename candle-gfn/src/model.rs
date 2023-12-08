use crate::state::{StateTrait, StateIdType};
use anyhow::Result;
use candle_core::Tensor;
pub trait ModelTrait<T>: Sized {
    fn forward_ss_flow(
        &self,
        source: &impl StateTrait<T>,
        sink: &impl StateTrait<T>,
    ) -> Result<Tensor>;
    fn reverse_ss_flow(
        &self,
        source: &impl StateTrait<T>,
        sink: &impl StateTrait<T>,
    ) -> Result<Tensor>;

    fn forward_ss_flow_batch(
        &self,
        ss_pairs: &[(StateIdType, StateIdType)],
        batch_size: usize,
    ) -> Result<Tensor>;
    fn reverse_ss_flow_batch(
        &self,
        ss_pairs: &[(StateIdType, StateIdType)],
        batch_size: usize,
    ) -> Result<Tensor>;

    fn get_f0(&self) -> Result<&Tensor>;
}
