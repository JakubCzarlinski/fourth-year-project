# Fourth Year Project

- `./implementations/`: Contains paper implementations with modifications and
  improvements for portability. Each implementation is a separate directory, and
  contains a `README.md` file with instructions on how to run the code. It
  should contain a `pyproject.toml` instead of a `requirements.txt` file. This
  makes it easier to install dependencies with `poetry`. `poetry` is a
  dependency manager that is similar to `go mod` and `npm` - it is very exact
  about the dependencies it installs, and it creates a `poetry.lock` file that
  specifies the exact versions of the dependencies that were installed. This
  makes it easier to reproduce the environment in which the code was run.

- `./lint.sh`: A script that runs `yapf` and `isort` on the files in the
  project. To install `yapf` and `isort`, run `pip install yapf isort`. To run
  the script, run `./lint.sh`.

Current optimal version of the code uses +1 quality for each iteration and grad rep mod 80 +20 to reach a range of 20-100. Additionally, in each grad rep, the quality for that rep is adjusted by a normally distributed ~[-5,+5] using `np.random.randint` seeded to 1003. This is done to introduce some extra noise over the repititions in training. Additionally, this version used the `COS_NORMED` loss criterion; this sets the same base criteria parameter as the `COS` loss criterion defined in regular DDD with several additions as follows:
- In the setting of `self.criteria` a new parameter `self.temp` is used to scale up the loss function (currently set to 0.1). This makes the loss function more sensitive to changes in the cosine similarity.
- The `COS_NORMED` loss criterion also normalises the a and b inputs before calculating with the criterion as we care mostly about the direction of how the model might view the adversarial images - moving to a more different representation is more beneficial than increasing magnitude in a more similar direction??? (Not quite sure if this is how it actually works)
- The criterion also uses an L2 loss term weighted at 0.1 to keep track of the pixelwise differences and not just the direction of the image. This was done as MSE performed better than vanilla COS in some cases and this allows us to incorporate a similar idea to MSE into COS to allow it to perform better in these cases.
- The layers are also weighted at `{256: 1, 64: 0.5}`. This is meant to prioritise the deeper layers that are more likely to contain image semantics within the unet but this is a GPT suggestion and I don't fully understand how it works - It does seem to have made it quite a lot better though.

Possible loss depths are:
- 64
- 256
- 1024
- 4096
- 1007
Theoretically the higher headed layers should capture more of the semantic information so disrupting those will push the scene composition further away but will focus less on things within the scene being distorted looking. From testing this does appear to be true - only using the 4096 headed layers gave perturbations that tended to effect the scene as a whole but were less impactful if you look at a specific subsection. E.g. in 004 adding what looks like a mesh or chain link fence across half the image or replacing the houses with roads in 003.

### Tested versions
- Using vanilla COS with only temperature added and 20-100: improved over the MSE versions
- Using COS_NORMED as described above: current optimal version
- COS_NORMED without layer weighting: worse than with layer weighting
- Reducing the weighting of the 64 layers to 0.3: this was slightly worse than at 0.5
- Changing the count only per iteration: noticably worse than changing per iteration and grad rep
- Expanding the distribution to ~[-10,+10]: slightly worse than ~[-5,+5]
- Using a 1-100 quality range: noticably worse than 20-100
- Using a 10-90 quality range: slightly worse than 20-100; potentially the really low quality factors cause the model to not learn as well since they are so different from the rest of the data
- Increasing eps to 25 from 13: significant improvement in all apart from 011 but the perturbations are much more noticeable
- Using 0.2 l2 weight: better on some results but worse on others, overall less consistent
- Additionally using the 1007 headed layers weighted at 2 and 268 total iterations: slight improvement (maybe beating optimal)
- Adding in the other layers too {4096:0.5, 1024:0.75, 1007:2, 256: 1, 64: 0.5}: better in some cases but worse in others probably due to weighting
- All layers with l2 only on layers with fewer heads: marginally better than overall l2 but still not as good as optimal
- {4096:0.5, 1024:0.75, 1007:2, 256: 1, 64: 0.5} with l2 only on smaller layers: best results yet but smaller test set
- Increasing iterations to 750: not much impact
- self.layer_weights = {4096:0.2, 1024:1, 1007:3, 256: 2, 64: 0.5} and 268 iterations with fixed randomness: NEW BEST, primarily better on 003 with some improvement to 004 and 011