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