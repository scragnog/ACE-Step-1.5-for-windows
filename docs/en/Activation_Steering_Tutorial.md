# Activation Steering Tutorial

> ⚠️ **Experimental Feature:** This feature is currently in-progress, experimental, and may not work as intended. Latent space drift (especially when using adapters) and precision limits can cause unexpected audio artifacts.

Activation Steering (also known as TADA - Task Adaptive Directional Activation) is a powerful feature in ACE-Step that allows you to guide the music generation process *without* extending the text prompt. This is especially useful for maintaining a consistent style, genre, or mood without overloading the main `genre` prompt with too many descriptive adjectives, which can often confuse the model.

Instead of retraining the model (like a LoRA), Activation Steering works by finding the "direction" of a concept in the model's internal activations and pushing the generation towards or away from that concept during inference.

---

## What is Activation Steering?

Imagine you want to generate a rock song, but you want it to have a specific "skate punk" energy. You could add "skate punk" to the prompt, but sometimes the model ignores it or changes other acoustic properties in an undesirable way.

With Activation Steering, you first **compute a steering vector** for the concept of "skate punk". The system does this by running the model on a set of neutral base prompts (e.g., "a rock song", "a pop song"), and then again on those same prompts with your concept added (e.g., "a rock song with skate punk energy"). It subtracts the internal brain states (activations) of the two runs to isolate the pure mathematical essence of "skate punk energy".

During generation, you can then inject this isolated "skate punk energy" vector directly into the model's brain (the transformer layers) while it generates *any* song.

---

## How to use Activation Steering in the UI

The Activation Steering controls are located in the **Create Panel** under the **Expert Controls** accordion.

### 1. Computing a New Concept

If the concept you want isn't already available, you need to compute it first. This process takes a few minutes (depending on your GPU), but you only ever have to do it once per concept. 

1. Expand the **Activation Steering** section.
2. Click **Compute New**.
3. In the text area, enter one concept per line. For best results, phrase them as additions to a base prompt, for example:
   * `with heavy metalcore breakdowns`
   * `with a driving disco beat`
   * `with ethereal synthwave pads`
4. *(Optional)* **Custom Base Prompts**: By default, the system uses 50 built-in generic genres to compute the contrast. If your concept is highly specific to a certain genre, you can override these. Check "Custom base prompts" and provide your own (e.g., `a rock song`, `a metal song`).
5. Click **Compute Queue**.
6. Wait for the computation to finish. The new vectors will be saved to your `steering_vectors` folder as `.pkl` files and will become available to load.

### 2. Loading and Applying Concepts

Once you have concepts available (either built-in or custom), you can apply them to your generation.

1. Under **Available Concepts**, find the concept you want to use.
2. Click the concept button to **Load** it (it will turn purple and show a checkmark).
3. Under **Active Concept Strengths**, fine-tune how the concept is applied:
   * **Strength (Alpha):** This determines how hard to push the model towards the concept. 
      * Positive values (e.g., `0.5`, `1.5`) amplify the concept.
      * Negative values (e.g., `-0.5`, `-1.5`) suppress the concept.
      * `0.0` disables the effect.
   * **Target Layers:** Determines which parts of the model's brain are affected. `tf7` generally works best for high-level musical concepts, but you can experiment with `tf6`, `tf6 + tf7`, or `all`.

You can load and apply multiple concepts simultaneously!

### 3. Deleting Concepts

If you are unhappy with a computed concept or no longer need it, you can delete it from your disk to save space.

1. Locate the custom concept under **Available Concepts**.
2. If it is currently loaded (purple), click it to unload it first.
3. Click the red **X** icon next to the concept name. 

---

## Tips for Best Results

* **Concept Phrasing:** Phrasing your concept starting with "with" (e.g., `with heavy chugging guitars`) often yields cleaner vectors than just typing `heavy chugging guitars`.
* **Strength Calibration:** Start with a low Strength (Alpha) around `0.3` to `0.8`. Pushing the alpha too high (above `2.0`) can cause the audio to degrade into noise or extreme distortion.
* **Negative Steering:** Try computing a concept for something you *don't* want (like `with lo-fi muffled vocals`), load it, and set the Strength to a negative number (e.g., `-1.0`). This pushes the model away from that sound.
* **Base Prompts Matter:** If you are trying to compute a vector for "black metal blast beats" but the default base prompts include "a classical symphony", the model might struggle to find the isolated contrast. Using custom base prompts like "a heavy metal song" and "a rock song" will help isolate the "blast beat" aspect much cleaner.
