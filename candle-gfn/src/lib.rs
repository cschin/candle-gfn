pub mod model;
pub mod sampler;
pub mod simple_grid_gfn;
pub mod state;
pub mod trajectory;

#[cfg(test)]
mod tests {

    use candle_core::{shape, DType, Tensor};
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
            rewards: FxHashMap::default(),
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
            number_trajectories: 100,
            terminate_states: vec![],
            rewards: FxHashMap::default(),
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
            rewards: FxHashMap::default(),
        };
        let s0 = SimpleGridState::new(0, (0, 0), false, 0.0, &device, parameters);
        let s1 = SimpleGridState::new(1, (1, 1), false, 0.0, &device, parameters);
        let model = SimpleGridModel::new(&device, parameters).unwrap();
        let out = model.forward_ss_flow(&s0, &s1).unwrap();
        println!("{:?} {}", out, out);
    }

    #[test]
    fn test_simple_grid_model_with_a_trajectory() {
        use crate::model::ModelTrait;
        use candle_core::Device;
        use candle_nn::*;
        use simple_grid_gfn::*;
        use std::sync::Mutex;
        let c = (0u32, 0u32);

        let state_id = 0_u32;
        let device = &Device::new_cuda(0).expect("no cuda device available");
        let parameters = &SimpleGridParameters {
            max_x: 64,
            max_y: 64,
            number_trajectories: 50,
            terminate_states: vec![],
            rewards: FxHashMap::default(),
        };
        let state: SimpleGridState = SimpleGridState::new(0, c, false, 0.0, device, parameters);
        let collection = &mut SimpleGridStateCollection::default();
        collection.map.insert(state_id, Box::new(state));

        let state_id = 32 * parameters.max_x + 32;
        let state: SimpleGridState =
            SimpleGridState::new(state_id, (32, 32), true, 10.0, device, parameters);
        collection.map.insert(state_id, Box::new(state));

        let state_id = 12 * parameters.max_x + 12;
        let state: SimpleGridState =
            SimpleGridState::new(state_id, (12, 12), true, 10.0, device, parameters);
        collection.map.insert(state_id, Box::new(state));

        let state_id = 63 * parameters.max_x + 63;
        let state: SimpleGridState =
            SimpleGridState::new(state_id, (63, 63), true, 0.5, device, parameters);
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

        let mut opt = candle_nn::SGD::new(model.varmap.all_vars(), 0.00005).unwrap();

        (0..100).for_each(|i| {
            sampler.trajectories.clear();
            sampler.sample_trajectories(&mut config);

            let mut losses = Vec::<_>::new();
            // let mut losses_sum = Vec::<_>::new();
            sampler.trajectories.iter().for_each(|traj| {
                // println!("traj len: {}", traj.trajectory.len());

                let trajectory = &traj.trajectory;

                // println!("trajectory len: {}", trajectory.len());

                trajectory[1..trajectory.len() - 1]
                    .iter()
                    .for_each(|&state_id| {
                        let mut out_flow = Vec::<_>::new();

                        if let Some(next_state_ids) = config.mdp.mdp_next_possible_states(
                            state_id,
                            config.collection,
                            device,
                            parameters,
                        ) {
                            next_state_ids.into_iter().for_each(|next_state_id| {
                                // println!("DBG: {} {}", state_id, next_state_id);
                                let s0 = config.collection.map.get(&state_id).unwrap().as_ref();
                                let s1 =
                                    config.collection.map.get(&next_state_id).unwrap().as_ref();
                                out_flow.push(model.forward_ss_flow(s0, s1).unwrap());
                            });
                        }

                        let out_flow = Tensor::stack(&out_flow[..], 0).unwrap();
                        let out_flow = out_flow
                            .sum_all()
                            .unwrap()
                            .log()
                            .unwrap();

                        let mut in_flow = Vec::<_>::new();

                        if let Some(previous_state_ids) = config.mdp.mdp_previous_possible_states(
                            state_id,
                            config.collection,
                            device,
                            parameters,
                        ) {
                            previous_state_ids
                                .into_iter()
                                .for_each(|previous_state_id| {
                                    let s0 = config
                                        .collection
                                        .map
                                        .get(&previous_state_id)
                                        .unwrap()
                                        .as_ref();
                                    let s1 = config.collection.map.get(&state_id).unwrap().as_ref();
                                    in_flow.push(model.forward_ss_flow(s0, s1).unwrap());
                                });
                        }

                        let in_flow = Tensor::stack(&in_flow[..], 0).unwrap();
                        let in_flow = in_flow
                            .sum_all()
                            .unwrap()
                            .log()
                            .unwrap();

                        let loss_at_s = out_flow.sub(&in_flow).unwrap().sqr().unwrap();
                        // println!(
                        //     "state_id: {} in: {} out: {} loss: {}",
                        //     state_id, in_flow, out_flow, loss_at_s
                        // );

                        losses.push(loss_at_s);
                    });

                let terminal_state_id = trajectory[trajectory.len() - 1];
                let reward = config
                    .collection
                    .map
                    .get(&terminal_state_id)
                    .unwrap()
                    .reward;

                let mut in_flow = Vec::<_>::new();
                if let Some(previous_state_ids) = config.mdp.mdp_previous_possible_states(
                    terminal_state_id,
                    config.collection,
                    device,
                    parameters,
                ) {
                    previous_state_ids
                        .into_iter()
                        .for_each(|previous_state_id| {
                            let s0 = config
                                .collection
                                .map
                                .get(&previous_state_id)
                                .unwrap()
                                .as_ref();
                            let s1 = config.collection.map.get(&state_id).unwrap().as_ref();
                            in_flow.push(model.forward_ss_flow(s0, s1).unwrap());
                        });
                }

                let in_flow = Tensor::stack(&in_flow[..], 0).unwrap();
                let in_flow = in_flow
                    .sum_all()
                    .unwrap()
                    .log()
                    .unwrap();

                let r = Tensor::from_slice(&[1.0f32; 1], shape::SCALAR, device);

                let reward = Tensor::from_slice(&[reward; 1], shape::SCALAR, device)
                    .unwrap()
                    .log()
                    .unwrap();

                let loss_at_s = reward.sub(&in_flow).unwrap().sqr().unwrap();

                losses.push(loss_at_s);
            });

            let losses = Tensor::stack(&losses[..], 0)
                .unwrap()
                .sum_all()
                .unwrap();
            opt.backward_step(&losses);
            println!("batch {}, total loss: {}", i, losses);

            println!("# of trajectories: {}", sampler.trajectories.len());
        });
        (0..63).for_each(|x| {
            (0..63).for_each(|y| {
                if x < 63 {
                    let s0id = x * parameters.max_x + y;
                    let s1id = (x + 1) * parameters.max_x + y;

                    if !config.collection.map.contains_key(&s0id)
                        || !config.collection.map.contains_key(&s1id)
                    {
                        return;
                    }

                    let s0 = config.collection.map.get(&s0id).unwrap().as_ref();
                    let s1 = config.collection.map.get(&s1id).unwrap().as_ref();
                    let out0 = model.forward_ss_flow(s0, s1).unwrap();
                    // println!("out: {}", out0);
                    let out0: f32 = out0
                        .squeeze(1)
                        .unwrap()
                        .squeeze(0)
                        .unwrap()
                        .to_scalar()
                        .unwrap();
                    println!("{} {} -> {} {} : {}", x, y, x + 1, y, out0);
                }
                if y < 63 {
                    let s0id = x * parameters.max_x + y;
                    let s1id = x * parameters.max_x + y + 1;

                    if !config.collection.map.contains_key(&s0id)
                        || !config.collection.map.contains_key(&s1id)
                    {
                        return;
                    }

                    let s0 = config.collection.map.get(&s0id).unwrap().as_ref();
                    let s1 = config.collection.map.get(&s1id).unwrap().as_ref();
                    let out0: f32 = model
                        .forward_ss_flow(s0, s1)
                        .unwrap()
                        .squeeze(1)
                        .unwrap()
                        .squeeze(0) 
                        .unwrap()
                        .to_scalar()
                        .unwrap();
                    println!("{} {} -> {} {} : {}", x, y, x, y + 1, out0);
                }
            });
        });
    }
}
