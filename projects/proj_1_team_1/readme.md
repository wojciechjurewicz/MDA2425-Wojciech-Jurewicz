# Data Analysis and Modeling Project

This project focuses on exploring and applying various machine learning techniques on the Mushroom Dataset.

## Mushroom Dataset

We will use the [Mushroom Classification Dataset](https://archive.ics.uci.edu/dataset/73/mushroom). Commonly used for classification tasks and benchmarking machine learning algorithms due to its balanced complexity and clear target structure.

This dataset, originally from the UCI Machine Learning Repository, provides detailed characteristics of various mushroom species, with the primary goal of classifying them as edible or poisonous based on their physical features.

The dataset contains 8,124 instances and 22 categorical attributes describing various physical characteristics of mushrooms. The target variable is class, which indicates whether a mushroom is edible (e) or poisonous (p).

All features in the dataset are categorical, encoded as single characters. There are no true missing values, although some attributes (such as stalk-root) use the character ? to indicate unknown values. Additionally, the dataset contains no numerical or continuous features.

### Column Descriptions

Below are the 22 features included in the dataset, along with their possible values (encoded using single letters):

- **cap-shape**: Shape of the mushroom cap.  
  Values: b = bell, c = conical, x = convex, f = flat, k = knobbed, s = sunken.

- **cap-surface**: Surface texture of the cap.  
  Values: f = fibrous, g = grooves, y = scaly, s = smooth.

- **cap-color**: Color of the cap.  
  Values: n = brown, b = buff, c = cinnamon, g = gray, r = green, p = pink, u = purple, e = red, w = white, y = yellow.

- **bruises**: Whether the mushroom has bruises or not.  
  Values: t = bruises, f = no bruises.

- **odor**: Smell of the mushroom.  
  Values: a = almond, l = anise, c = creosote, y = fishy, f = foul, m = musty, n = none, p = pungent, s = spicy.

- **gill-attachment**: How the gills attach to the stalk.  
  Values: a = attached, d = descending, f = free, n = notched.

- **gill-spacing**: Spacing between the gills.  
  Values: c = close, w = crowded, d = distant.

- **gill-size**: Size of the gills.  
  Values: b = broad, n = narrow.

- **gill-color**: Color of the gills.  
  Values: k = black, n = brown, b = buff, h = chocolate, g = gray, r = green, o = orange, p = pink, u = purple, e = red, w = white, y = yellow.

- **stalk-shape**: Shape of the stalk.  
  Values: e = enlarging, t = tapering.

- **stalk-root**: Type of root (some values are missing and marked as ?).  
  Values: b = bulbous, c = club, u = cup, e = equal, z = rhizomorphs, r = rooted, ? = missing.

- **stalk-surface-above-ring**: Surface texture above the ring.  
  Values: f = fibrous, y = scaly, k = silky, s = smooth.

- **stalk-surface-below-ring**: Surface texture below the ring.  
  Same values as above.

- **stalk-color-above-ring**: Color of the stalk above the ring.  
  Values: n = brown, b = buff, c = cinnamon, g = gray, o = orange, p = pink, e = red, w = white, y = yellow.

- **stalk-color-below-ring**: Color of the stalk below the ring.  
  Same values as above.

- **veil-type**: Type of veil (only one value: p = partial).  
  This column has no variation and can be dropped.

- **veil-color**: Color of the veil.  
  Values: n = brown, o = orange, w = white, y = yellow.

- **ring-number**: Number of rings on the stalk.  
  Values: n = none, o = one, t = two.

- **ring-type**: Type of ring.  
  Values: c = cobwebby, e = evanescent, f = flaring, l = large, n = none, p = pendant, s = sheathing, z = zone.

- **spore-print-color**: Color of the spore print.  
  Values: k = black, n = brown, b = buff, h = chocolate, r = green, o = orange, u = purple, w = white, y = yellow.

- **population**: Estimated population size where the mushroom is found.  
  Values: a = abundant, c = clustered, n = numerous, s = scattered, v = several, y = solitary.

- **habitat**: Natural environment where the mushroom grows.  
  Values: g = grasses, l = leaves, m = meadows, p = paths, u = urban, w = waste, d = woods.