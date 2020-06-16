For https://tianchi.aliyun.com/competition/entrance/231785/information

# KDD Cup 2020 Challenges for Modern E-Commerce Platform: Debiasing

Team build-success, former Team666.

Four team members





## 数据特征

The files are in CSV format, with UTF-8 encoding. The columns of the CSV files can be:

- item_id：the unique identifier of the item
- txt_vec：the item's text feature, which is a 128-dimensional real-valued vector produced by a pre-trained model
- img_vec：the item's image feature, which is a 128-dimensional real-valued vector produced by a pre-trained model
- user_id：the unique identifier of the user
- time：timestamp when the click event happens, i.e.,（unix_timestamp - random_number_1）/ random_number_2，，，These two random numbers are kept the same for all clicks. The time orders of the clicks are therefore preserved.
- user_age_level：the age group to which the user belongs
- user_gender：the gender of the user, which can be empty
- user_city_level：the tier to which the user's city belongs



## 重复数据

- **Q: Are there overlapped (duplicate) data between phase T and phase T+1? Should I remove the duplicate data?**

- A: Yes. Roughly 2/3 data will be the same. You may need to remove them if you need to use more than two phases when making predictions for the present phase.

  ## Metric相关

- **Q: How to win?**

- A: A team first need to be among the top 10% in terms of NDCG@50-full (aka. ndcg_50_full on the leaderboard) so as to be *qualified*. The winning teams will then be the *qualified* teams that achieve the best NDCG@50-rare **(aka. ndcg_50_half on the leaderboard).**

- **Q: If I score a very high NDCG@50-full that gets me into the top 10% but my NDCG@50-rare is pretty low. Then I make a trade-off b/w these 2 metrics and my new result scores a lower NDCG@50-FULL, which still gets me into the top 10%, but with a much higher NDCG@50-rare. Which one of these 2 results will be used as the final submission, since the leaderboard only shows the first one?** 

- A: We only look at the results shown on the leaderboard. We won't change the leaderboard so that it sorts the scores by ndcg_50_half directly, because this design of sorting by ndcg_50_full is intended as a way to encourage you to *only submit one time in total for phase 9*. If you still need to adjust your algorithm when we are already in phase 9, then you are probably just overfitting the dataset. P.S.: Tianchi doesn't support limiting the total submit times directly at the moment. It only supports limiting the submit times *for each day*.

  ## 提交格式

- repo已经按照说明页面做好文件目录
- user_data放中间结果
- 注意在本地加上data文件夹（记得ignore掉）