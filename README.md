# Dynamic 3D Gaussians Datasets
This is a collection of kubric scripts used to generate dynamic multi-view scene datasets.

## MOVi-F Multicam
Modified script from Kubric's MOVi-F dataset used to generate the same from multiple angles around a dome.

### Usage
Assuming you have docker installed:
```
git clone 
docker pull kubricdockerhub/kubruntu
docker run --rm --interactive \
  --user $(id -u):$(id -g)    \
  --volume "$(pwd):/kubric"   \
  kubricdockerhub/kubruntu    \
  /usr/bin/python3 path/to/dynamic3dgaussiansdatasets/movi-f-mult-cam/worker.py \
  --camera=fixed_camera
  --max_motion_blur=2.0
  --save_state
```