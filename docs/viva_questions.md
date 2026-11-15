# RIntelliCart — Viva Questions and Answers (44)

Answers are short on purpose: say them in your own words.

## A. Project and Python
**1. What is your project about?**
RIntelliCart analyses product ratings, prices, brands and features and recommends suitable products to a customer, with an explanation for each recommendation.

**2. Why did you choose these libraries?**
They are the ones we studied and they fit together: Pandas for data, NumPy for calculations, Matplotlib for charts, scikit-learn for machine learning.

**3. Why Python?**
It is simple to read, has strong data-science libraries, and Streamlit lets us build a web app in pure Python.

**4. What is Streamlit?**
A Python library that turns a script into an interactive web app. Every click re-runs the script from top to bottom.

**5. What is session state and why do you use it?**
`st.session_state` is a dictionary that survives re-runs. We store the active dataset, the trained model and the last recommendations in it.

**6. Is your data real?**
No. `products.csv` is demonstration data written for this project. Prices, ratings and review counts are illustrative, and there are no real customer reviews.

## B. NumPy
**7. What is NumPy?**
A library for fast calculations on arrays of numbers.

**8. Where do you use NumPy?**
Rating statistics (`mean`, `median`, `std`, `max`, `min`), weighted average, histogram bins, percentile for the weighted rating, min-max scaling, and the weighted hybrid score.

**9. Difference between `np.mean` and `np.average`?**
`mean` treats every value equally; `average` can take weights, so a product with more reviews counts more.

**10. What is standard deviation?**
It shows how spread out the ratings are around the average. A small value means most ratings are close to the mean (ours is about 0.22).

**11. Why is NumPy faster than a Python loop?**
It runs the operation on the whole array in optimised compiled code instead of one element at a time.

## C. Pandas and data cleaning
**12. What is a DataFrame?**
A table with rows and named columns, like a spreadsheet in Python.

**13. How do you filter rows in Pandas?**
With boolean masks, for example `df[(df["price"] <= 30000) & (df["rating"] >= 4)]`.

**14. What does `groupby` do?**
It splits the data into groups (for example by category) and lets us calculate something for each group, like the average rating.

**15. What data cleaning do you do?**
Check required columns, strip spaces, convert numbers, drop rows missing key values, fill missing review counts with 0, remove duplicates, clip ratings to 0–5, and produce a report of what changed.

**16. Why remove duplicates?**
A duplicated product would be counted twice and distort averages and rankings.

**17. What does `pd.to_numeric(errors="coerce")` do?**
It converts text to numbers; anything that cannot be converted becomes NaN instead of crashing, and we handle those rows afterwards.

**18. How do you validate an uploaded CSV?**
We check that all required columns exist and show a clear error naming the missing ones; we also reject empty or unreadable files and files with no usable rows.

## D. Matplotlib
**19. What is Matplotlib?**
A library for drawing charts such as bar charts, histograms and scatter plots.

**20. Which charts did you draw and why?**
Bar charts for comparing categories, a histogram for the rating distribution, a scatter plot for price vs rating, and a pie chart for review share.

**21. Why do you use a log scale on the price axis?**
Prices range from ₹199 to ₹1,09,990. On a normal axis almost every point would be squeezed on the left; the log scale spreads them out.

**22. Why do you write chart functions instead of drawing directly in each page?**
To avoid repeated code. One function is reused on several pages and is easy to test.

## E. Recommendation systems
**23. What is a recommendation system?**
Software that suggests items a user is likely to want, using data about items, users or both.

**24. What types of recommenders exist?**
Popularity-based, content-based, collaborative filtering and hybrid. We implemented popularity, content-based and hybrid.

**25. Why not use collaborative filtering?**
It needs data about many users' ratings or purchases of the same products. Our dataset has only product information.

**26. What is the cold-start problem?**
A new user or a new product has no history, so collaborative filtering cannot work. Content-based methods avoid this because they only need product text.

**27. Why do you use a weighted rating?**
A 5.0 rating from 2 reviews is less reliable than 4.6 from 5,000 reviews. The weighted rating pulls products with few reviews toward the overall average.

**28. Explain the weighted rating formula.**
`WR = v/(v+m)·R + m/(v+m)·C`. R is the product rating, v its reviews, C the average rating of all products, m the minimum reviews needed. If v is small the score is close to C; if v is large it is close to R.

**29. How is the match score calculated?**
Hard filters first. Then four scores between 0 and 1 — preference, content similarity, weighted rating, popularity — are combined with weights 0.30, 0.30, 0.25, 0.15 and multiplied by 100.

**30. What if the customer gives no brand or features?**
That component is dropped and the other weights are rescaled to add up to 1, so the product is not penalised for information we never asked for.

**31. What happens when nothing matches?**
We show which criteria are too strict — for example the price range of the category or the best available rating — so the user knows what to change.

**32. Are your match scores random or hard-coded?**
No. They come from calculations on the data. The same input always gives the same output, and tests check that changing the input changes the ranking.

## F. Scikit-learn, TF-IDF and cosine similarity
**33. What is TF-IDF?**
Term Frequency × Inverse Document Frequency. It gives a word a high weight if it is frequent in one product's text but rare across all products.

**34. Why not just count words?**
Common words such as "the" or "and" would dominate. TF-IDF reduces their weight and boosts distinctive words like "gaming".

**35. What is cosine similarity?**
The cosine of the angle between two vectors. It is 1 for the same direction (very similar texts) and 0 when the texts share no words.

**36. Why cosine similarity and not Euclidean distance?**
Cosine ignores text length and compares only direction, so a short and a long description about the same thing still look similar.

**37. What is a stop word?**
A very common word such as "the" or "is" that carries little meaning; we remove them with `stop_words="english"`.

**38. What does `fit_transform` do?**
`fit` learns the vocabulary and IDF values from the product texts; `transform` converts texts into vectors. `fit_transform` does both. For a customer's text we only call `transform`, with the vocabulary already learned.

**39. Is TF-IDF machine learning?**
It is a text-vectorisation technique from machine learning (feature extraction). Our models are unsupervised and similarity-based; nothing is trained to predict labels.

## G. Evaluation and limitations
**40. What is the accuracy of your model?**
We do not report one. Accuracy needs ground truth such as what customers actually bought or rated. Our demo data has none, so any accuracy number would be invented. We only run a sanity check (text-similar products are usually in the same category) and clearly label it as not an accuracy metric.

## Bonus questions
**41. How could you evaluate it properly?** Collect user clicks or ratings, hide some, and measure precision@k and recall@k on the hidden ones.
**42. What are the limitations?** Demonstration data, no real interactions, hand-chosen weights, word-matching only (no synonyms).
**43. What would you improve?** Learn weights from data, add collaborative filtering, use word embeddings, connect a verified product source.
**44. How can a new dataset be used?** Upload a CSV with the required columns on the Home page; it is validated, cleaned, and every page and model switches to it automatically.
