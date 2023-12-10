
pub struct Trajectory<I: Copy> {
    pub trajectory: Vec<I>,
}

impl<I: Copy> Default for Trajectory<I> {
    fn default() -> Self {
        Self::new()
    }
}

impl<I: Copy> Trajectory<I> {
    pub fn new() -> Self {
        let trajectory = Vec::<I>::new();
        Self { trajectory }
    }

    pub fn clear(&mut self) {
        self.trajectory.clear();
    }

    pub fn get_parent_offspring_pairs(&self) -> Vec<(I, I)> {
        assert!(self.trajectory.len() > 1);
        (0..self.trajectory.len() - 1)
            .map(|idx| (self.trajectory[idx], self.trajectory[idx + 1]))
            .collect::<Vec<_>>()
    }

    pub fn push(&mut self, id: I) {
        self.trajectory.push(id);
    }
}
