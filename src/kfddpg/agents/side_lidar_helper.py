    def _side_min_lidar_norm(self, state: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor] | None:
        """
        Extract min distance for Left and Right sectors.
        Assumption: 90 pts, 0=Front, increases CCW.
        Front Sector w=23 => [0:23] U [67:90].
        Left Sector: [23:45] (approx 90 deg ~ 180 deg)
        Right Sector: [45:67] (approx 180 deg ~ 270 deg / -90 deg)
        Normalized inputs.
        """
        if state.shape[-1] < 90:
            return None, None
        lidar = state[..., 0:90]
        w = int(self.cfg_cls.FRONT_SECTOR_HALF_WIDTH)
        # Left: [w, 45]
        left_sector = lidar[..., w:45]
        # Right: [45, 90-w]
        right_sector = lidar[..., 45:90-w]
        
        min_left = left_sector.min(dim=-1).values
        min_right = right_sector.min(dim=-1).values
        return min_left, min_right
