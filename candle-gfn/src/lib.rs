pub mod model;
pub mod sampler;
pub mod simple_grid_gfn;
pub mod state;
pub mod trajectory;

#[cfg(test)]
mod tests {

    use candle_core::{Tensor, DType};
    use fxhash::FxHashMap;

    use crate::sampler::{MDPTrait, Sampling};

    use super::*;
    #[test]
    fn test_simple_grid_mdp() {
        use simple_grid_gfn::*;
        let c = (0u32, 0u32);
        let mut state_id = 0_u32;
        use candle_core::Device;
        let device = Device::new_cuda(0).expect("no cuda device available");
        let parameters = &SimpleGridParameters {
            max_x: 64,
            max_y: 64,
            number_trajectories: 1000,
            terminate_states: vec![],
            rewards: FxHashMap::default()
        };
        let state: SimpleGridState = SimpleGridState::new(0, c, false, 0.0, &device, parameters);
        let collection = &mut SimpleGridStateCollection::default();
        collection.map.insert(state_id, Box::new(state));
        let mdp = SimpleGridMDP::new();

        while let Some(s) = mdp.mdp_next_one_uniform(state_id, collection, &device, parameters) {
            println!("state_id: {:?}", s);
            state_id = s;
        }
    }

    #[test]
    fn generate_trajectory() {
        use simple_grid_gfn::*;
        let c = (0u32, 0u32);
        let mut state_id = 0_u32;
        use candle_core::Device;
        let device = Device::new_cuda(0).expect("no cuda device available");
        let parameters = &SimpleGridParameters {
            max_x: 64,
            max_y: 64,
            number_trajectories: 1000,
            terminate_states: vec![],
            rewards: FxHashMap::default()

        };
        let state: SimpleGridState = SimpleGridState::new(0, c, false, 0.0, &device, parameters);
        let collection = &mut SimpleGridStateCollection::default();
        let mut traj = trajectory::Trajectory::new();
        traj.push(state_id);
        collection.map.insert(state_id, Box::new(state));
        let mdp = SimpleGridMDP::new();

        while let Some(s) = mdp.mdp_next_one_uniform(state_id, collection, &device, parameters) {
            println!("state_id: {:?}", s);
            state_id = s;
            traj.push(state_id);
        }
        println!("{:?}", traj.get_parent_offspring_pairs());
    }

    #[test]
    fn test_simple_grid_model() {
        use candle_core::Device;
        use model::*;
        use simple_grid_gfn::*;
        let device = Device::new_cuda(0).expect("no cuda device available");
        let parameters = &SimpleGridParameters {
            max_x: 64,
            max_y: 64,
            number_trajectories: 1000,
            terminate_states: vec![],
            rewards: FxHashMap::default()

        };
        let s0 = SimpleGridState::new(0, (0, 0), false, 0.0, &device, parameters);
        let s1 = SimpleGridState::new(1, (1, 1), false, 0.0, &device, parameters);
        let model = SimpleGridModel::new(&device, parameters).unwrap();
        let out = model.forward_ssp(&s0, &s1).unwrap();
        println!("{:?} {}", out, out);
    }

    #[test]
    fn test_simple_grid_model_with_a_trajectory() {
        use crate::model::ModelTrait;
        use candle_core::Device;
        use simple_grid_gfn::*;
        let c = (0u32, 0u32);

        let state_id = 0_u32;
        let device = &Device::new_cuda(0).expect("no cuda device available");
        let parameters = &SimpleGridParameters {
            max_x: 64,
            max_y: 64,
            number_trajectories: 1000,
            terminate_states: vec![],
            rewards: FxHashMap::default()

        };
        let state: SimpleGridState = SimpleGridState::new(0, c, false, 0.0, device, parameters);
        let collection = &mut SimpleGridStateCollection::default();
        collection.map.insert(state_id, Box::new(state));

        let model = SimpleGridModel::new(device, parameters).unwrap();

        let mut sampler = SimpleGridSampler::new();
        let mdp = &mut SimpleGridMDP::new();

        let mut config = SimpleGridSamplingConfiguration {
            begin: 0,
            collection,
            mdp,
            model: Some(&model),
            device,
            parameters,
        };

        let traj = sampler.sample_a_new_trajectory(&mut config);
        //let logp_sum = &mut Tensor::zeros((1, 1), DType::F32, device).unwrap();
        let mut logp_sum = Vec::<_>::new();
        traj.get_parent_offspring_pairs()
            .iter_mut()
            .for_each(|(s0id, s1id)| {
                let s0 = config.collection.map.get(s0id).unwrap().as_ref();
                let s1 = config.collection.map.get(s1id).unwrap().as_ref();
                let out = model.forward_ssp(s0, s1).unwrap();
                println!("out: {}", out);
                logp_sum.push(out);
                //println!("XX: {} {} {:?}", s0id, s1id, out);
            });
            let log_sum = Tensor::stack(&logp_sum[..], 0).unwrap();
            let log_sum = log_sum.sum_all().unwrap();
            let gs = log_sum.backward().unwrap();
            println!("dbg: {:?}",gs);
        sampler.sample_trajectories(&mut config);
    }
}
