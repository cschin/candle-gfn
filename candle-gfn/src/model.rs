
use crate::state::{StateTrait, StateIdType};
use anyhow::Result;
use candle_core::Tensor;
pub trait ModelTrait<T>: Sized {
    fn forward_ss_flow(
        &self,
        _source: &impl StateTrait<T>,
        _sink: &impl StateTrait<T>,
    ) -> Result<Tensor> { unimplemented!() }

    fn reverse_ss_flow(
        &self,
        _source: &impl StateTrait<T>,
        _sink: &impl StateTrait<T>,
    ) -> Result<Tensor> {  unimplemented!() }

    fn forward_ss_flow_batch(
        &self,
        _ss_pairs: &[(StateIdType, StateIdType)],
        _batch_size: usize,
    ) -> Result<Tensor> {
        unimplemented!()
    }

    fn forward_ss_flow_batch_with_states(
        &self,
        _ss_pairs: &[(&impl StateTrait<T>, &impl StateTrait<T>)],
        _batch_size: usize,
    ) -> Result<Tensor> {
        unimplemented!()
    }

    fn reverse_ss_flow_batch(
        &self,
        _ss_pairs: &[(StateIdType, StateIdType)],
        _batch_size: usize,
    ) -> Result<Tensor> {
        unimplemented!()
    }

    fn get_f0(&self) -> Result<&Tensor>;
}
