import { useState } from 'react'
import headerImage from "./assets/pexels-kindelmedia-8566449.jpg"
import katex from "katex";
import "katex/dist/katex.min.css";
import './App.css'
import { Brain, Database, Lambda, GitBranch, UsersRound } from 'lucide-react';

function BlockMath({ math }) {
  const html = katex.renderToString(math, { displayMode: true, throwOnError: false });
  return <div dangerouslySetInnerHTML={{ __html: html }} />;
}

const resultRows = [
  { name: "Always Predict “Thin”", accuracy: 0.647, thin: 1.0, fat: 0.0, obese: 0.0, macro: 0.162, balanced: 0.25, mcc: 0.0, color: "#e5737f" },
  { name: "Naive Bayes (All Features)", accuracy: 0.515, thin: 0.595, fat: 0.284, obese: 0.24, macro: 0.402, balanced: 0.389, mcc: 0.142, color: "#4f7cf6" },
  { name: "Naive Bayes (Last Name)", accuracy: 0.534, thin: 0.738, fat: 0.116, obese: 0.0, macro: 0.326, balanced: 0.264, mcc: 0.01, color: "#a78bfa" },
  { name: "K-Means Clustering", accuracy: 0.613, thin: 0.906, fat: 0.062, obese: 0.0, macro: 0.372, balanced: 0.265, mcc: 0.038, color: "#f0b062" },
  { name: "K-Nearest Neighbors", accuracy: 0.374, thin: 0.371, fat: 0.018, obese: 0.132, macro: 0.543, balanced: 0.294, mcc: 0.048, color: "#f6d94d" },
];

const resultColumns = [
  { key: "accuracy", label: "Accuracy" },
  { key: "thin", label: "Thin Recall" },
  { key: "fat", label: "Fat Recall" },
  { key: "obese", label: "Obese Recall" },
  { key: "macro", label: "Macro Precision" },
  { key: "balanced", label: "Balanced Accuracy" },
  { key: "mcc", label: "MCC" },
];

function ResultsChart() {
  const groupWidth = 120, barWidth = 18, top = 20, plotHeight = 260, left = 40;
  const ticks = [0, 0.25, 0.5, 0.75, 1];
  return (
    <svg viewBox="0 0 900 310" className="w-full" role="img"
      aria-label="Grouped bar chart comparing every method on accuracy, thin, fat and obese recall, macro precision, balanced accuracy and MCC">
      {ticks.map((t) => {
        const y = top + plotHeight - t * plotHeight;
        return (
          <g key={t}>
            <line x1={left} x2={880} y1={y} y2={y} stroke="#e2e8f0" />
            <text x={left - 6} y={y + 4} textAnchor="end" fontSize="11" fill="#64748b">{t.toFixed(2)}</text>
          </g>
        );
      })}
      {resultColumns.map((col, i) => {
        const x0 = left + 10 + i * groupWidth;
        return (
          <g key={col.key}>
            {resultRows.map((row, j) => {
              const h = row[col.key] * plotHeight;
              return (
                <rect key={row.name} x={x0 + j * barWidth} y={top + plotHeight - h}
                  width={barWidth - 2} height={h} fill={row.color}>
                  <title>{`${row.name}, ${col.label}: ${row[col.key].toFixed(3)}`}</title>
                </rect>
              );
            })}
            <text x={x0 + (resultRows.length * barWidth) / 2} y={top + plotHeight + 22}
              textAnchor="middle" fontSize="12" fill="#334155">{col.label}</text>
          </g>
        );
      })}
    </svg>
  );
}

function App() {
  const metrics = [
    {
      name: "Precision (Per Dynasty Type)",
      description: <p className='text-sm leading-7 text-slate-600 max-h-[300px] overflow-auto'>
        For the four types, the confusion matrix is a 2x2 table of true type against predicted type. Given a type k, precision is the share of rows we labeled as type k that really are type k. Macro-averaged precision is the average of the four per-type values, so a rare type such as obese counts as much as the common “none” type [8].
        <br /><br />
        <BlockMath
          math={String.raw`
              \text{Precision of Type } k =
              \frac{TP_k}{TP_k + FP_k}
            `}
        />
        <br />
        <BlockMath
          math={String.raw`
              \text{Macro Precision} =
              \frac{
                P_{\text{none}} +
                P_{\text{thin}} +
                P_{\text{fat}} +
                P_{\text{obese}}
              }{4}
            `}
        />
        <br /><br />
        Precision is our primary measure because calling someone part of a fat/obese dynasty when they are not is a false claim. We report it per type because the test rows are heavily imbalanced and so accuracy alone would reward a model that always answers “none”. We compare every method with that “none” baseline [8] [9].
      </p>
    },
    {
      name: "Recall",
      description: <p className='text-sm leading-7 text-slate-600 max-h-[300px] overflow-auto'>
        For a type k, it is the share of the real type k rows that we found: TP_k / (TP_k + FN_k). Balanced accuracy is the average of the four. It answers a question that precision does not which is: how many of the real thin, fat, and obese officials did the method really find?
        <br />
        <div className="rounded-xl bg-white p-6 text-center shadow-sm">
          <BlockMath
            math={String.raw`
            \text{Balanced Accuracy} =
            \frac{
              \frac{TP}{Total_{row1}} +
              \frac{TN}{Total_{row2}}
            }{2}
          `}
          />
        </div>
      </p>
    },
    {
      name: "F1-Score",
      description: <p className='text-sm leading-7 text-slate-600 max-h-[300px] overflow-auto'>
        The MCC summarizes the whole confusion matrix in one number. For K types, the <BlockMath
          math={String.raw`
            \text{MCC} =
            \frac{
              (TP \cdot TN) - (FP \cdot FN)
            }{
              \sqrt{
                (TP + FP)(TP + FN)(TN + FP)(TN + FN)
              }
            }
          `}
        />[8]. It ranges from -1 to 1, where 1 is perfect, 0 is random guessing, and negative is worse. Chicco and Jurman argued that the MCC is more reliable especially on imbalance data because it gives a high score only when all parts of the confusion matrix are good [4]. We use it as the single number for ranking the three methods.
        <br /><br />
        Together the three metrics cover what a wrong answer costs (precision), what is missed (balanced accuracy), and the overall agreement (MCC). All of them are possible to be applied for KNN, K-Means, and Naive Bayes because they use only the predicted types.

      </p>

    },
  ];

  return (
    <main className="min-h-screen  mx-auto bg-slate-100 flex flex-col">
      {/* header */}
      <div className='flex px-12 py-8 bg-slate-900 items-center'>
        <div className='flex gap-4 flex-col flex-grow text-white'>
          <div className='flex gap-2'><Brain />CMSC 170: Applying AI and Machine Learning Methods</div>
          <div className='flex gap-2'><Database />OpenHalalan: Philippine National and Local Election Dataset {" "}
            <a href="https://data.bettergov.ph/datasets/25" className='underline'>(https://data.bettergov.ph/datasets/25)</a>
          </div>
          <div className='flex gap-2'><Lambda /> Naive Bayes | K-Means Clustering | K-Nearest Neighbors</div>
          <div className='flex gap-2'><GitBranch /><a href="https://github.com/jaeson-cho/uplb-cmsc170-j.git" className='underline hover:font-blue-600'>https://github.com/jaeson-cho/uplb-cmsc170-j.git </a></div>
          <div className='flex gap-2'><UsersRound /> Group J - Cho | Licayan | Figuro | Osorio</div>
        </div>
        <div className='flex w-1/4 shrink-0 items-center justify-center'>
          <img
            src={headerImage}
            className='w-full max-w-[320px] object-contain shadow-2xl rounded-xl'
          />
        </div>

      </div>
      <div className='flex flex-col gap-4'>

        {/* Intro */}
        <section className='mx-2 bg-white p-8 shadow-sm rounded-b-2xl'>
          <h2 className='mb-6 text-3xl font-bold text-slate-900'>01. Introduction</h2>
          <p className='leading-8 text-slate-600'>
            Political Dynasties have long since been a significant aspect of Philippine politics which allows certain families to maintain their political influences across multiple generations. According to the 1987 Constitution, it states that the state shall prohibit political dynasties “as may be defined by law”, but no such definition has been enacted. According to Mendoza et al, they found that political dynasties make up over 70% of the Philippine Congress and that they have a worsening effect on poverty in provinces outside Luzon [1]. This leads to corruption, and failure for people seated in power to properly serve the Filipino people. Identifying locations where political dynasties occur can help provide a clearer understanding of the distribution of political power across provinces and cities in the Philippines. This helps us, the Filipino people, to properly scrutinize if the right people are being assigned to the right jobs.

            Identifying where political dynasties occur can help provide a clear picture and understanding of the political power that is distributed across the cities and provinces of the Philippines. This helps the Filipino people to properly scrutinize whether the right people are assigned to the right positions or jobs.

            This project aims to analyze OpenHalalan’s dataset containing records of politicians to detect the probability of political dynasties happening in a given location. Using Artificial Intelligence methods taught (and possibly beyond what was taught in class), the project seeks to identify patterns that indicate presence of political dynasties in specific areas. The methods will be proposed and discussed below, as well as the reasonings as to why the specific methods are selected.

          </p>
        </section>
        {/* Goals */}
        <section className='mx-2 bg-slate-300 p-8 pb-10 shadow-sm rounded-b-2xl'>
          <h2 className='mb-6 text-3xl font-bold text-slate-900'>02. Goals</h2>
          <p className='text-slate-600'>
            Studies have shown that a dynasty isn’t just a yes or no question. Studies separate “thin” dynasties (relatives succeed each other) from “fat” dynasties (relatives hold office at the same time) and “obese” dynasties (extreme fat cases) [7], and report that the fat kind is the one linked to worse poverty outcomes [1]. So our goal is to identify what kind of dynasty presence each local official shows and where it is concentrated. We can expect:
            <p className='ml-2 mt-2'>1) Dynasty type for local officials: Each mayor, vice mayor, and councilor is classified as none, thin, fat, or obese. This is a four-class task so each method must give four outcomes instead of two.
            </p>
            <p className='ml-2 mt-2'>2) See where dynasties are concentrated: Provinces and cities are ranked by the share of fat and obese officials, using the predicted types and compared with the types from our results.
            </p>
          </p>
        </section>
        {/* Definitions */}
        <section className='mx-2 bg-white p-8 shadow-sm rounded-b-2xl'>
          <h2 className='mb-6 text-3xl font-bold text-slate-900'>03. Definitions</h2>
          <p className='text-slate-600'>
            The dataset shows no family relationships so we approximate them using names with a shared Last Name as the sign that two officials are related. Following the evaluation of thin, fat, and obese [7], each city or municipality level winner gets one of the four types. Let n be the number of different people with different First Names with the same Last Name who won in the same province,  city, and year counting the official:
            <p className='ml-2 mt-2'>1) <strong>Obese</strong>: n = 4 (four or more relatives in office at the same time)</p>
            <p className='ml-2 mt-2'>2) <strong>Fat</strong>: n is 2 or 3 (relatives in office at the same time)</p>
            <p className='ml-2 mt-2'>3) <strong>Thin</strong>: n is 1 but someone with the same last name won in the same city in a previous election cycle</p>
            <p className='ml-2 mt-2 mb-2'>4) <strong>None</strong>: None of the above</p>

            This label is shared by all three methods so the comparison stays fair. Under this rule, the training rows (elections up to 2016) are about 41% none, 48% thin, 10.5% fat, and 0.6% obese. In the later election cycles used for testing (2019 to 2025), the shares are about 19.8% none, 64.7% thin, 14.3% fat, and 1.2% obese.


          </p>
        </section>
        {/* Methods */}
        <section className='mx-2 bg-slate-300 p-8 pb-10 shadow-sm rounded-b-2xl'>
          <h2 className='mb-6 text-3xl font-bold text-slate-900'>04. Methods</h2>
          <div className='grid grid-cols-3 gap-2'>
            {/* Naive */}
            <div className='rounded-xl border border-slate-200 bg-white p-6 shadow-md '>
              <h3 className='mb-3 text-xl font-semibold text-slate-900'>
                Naive Bayes
              </h3>
              <p className='text-sm leading-7 text-slate-600 max-h-[300px] overflow-auto'>
                &nbsp;&nbsp;&nbsp;&nbsp;This method will be used to classify each row of the record by calculating how probable it is that the row belongs to a Political Dynasty or not. It will use the Last Name, Middle Name, Province, City, and Position as features. The method will count how often each feature value, such as the surname in a given province appears in each class. It then uses Bayes’ Rule to give each row a probability of being part of a political dynasty. This allows the developers to see not only whether that given row is classified as a dynasty but also how confident the prediction is, which helps us decide if the percentage is accurate enough to claim a dynasty. Laplace Smoothing will be applied so that new values do not make output a probability of zero [2].
                <br /> <br /> &nbsp;&nbsp;&nbsp;&nbsp;Naive Bayes assumes that the features are independent given the class [2], which is not fully true in our data (Example: City and Province are related). Because of this, the probability it outputs will be treated as a confidence score rather than an exact probability. The probability output still lets us see how confident each prediction is, which helps in deciding whether the percentage is accurate enough to claim a dynasty.
              </p>
            </div>
            {/* Kmeans */}
            <div className='rounded-xl border border-slate-200 bg-white p-6 shadow-md'>
              <h3 className='mb-3 text-xl font-semibold text-slate-900'>
                K-Means Clustering
              </h3>
              <p className='text-sm leading-7 text-slate-600 max-h-[300px] overflow-auto'>
                &nbsp;&nbsp;&nbsp;&nbsp;This method will group similar political records without requiring a predefined set of training data. This allows us, the developers, to be able to let the AI find connections to the training data itself to analyze possible clusters that can be grouped as political dynasties or not. A record for example can be analyzed whether repeated surnames, with multiple positions are within the same area, across years. K-Means repeatedly assigns each record to its nearest centroid and then computes again the centroids until they stop changing. It is popular due to its simple implementation but it needs the number of clusters in advance, and is sensitive to the starting centroids and relying on Euclidean distance. To handle this, we will convert our features to numbers, and try several values of k and several random starts [6].
              </p>
            </div>
            {/* KNN */}
            <div className='rounded-xl border border-slate-200 bg-white p-6 shadow-md'>
              <h3 className='mb-3 text-xl font-semibold text-slate-900'>
                K-Nearest Neighbors
              </h3>
              <p className='text-sm leading-7 text-slate-600 max-h-[300px] overflow-auto'>
                &nbsp;&nbsp;&nbsp;&nbsp;This method will be used to classify each row of the record based on similarities to prior examples. It will reference Last Name and Middle Name used with Province, City, Year, and Position to identify patterns to check if Political Dynasties are probable in given locations. This will require a manual addition of rows that are verified as political dynasties as well as those verified not to be one. KNN will be used train the features and help classify succeeding rows that need to be tested
              </p>
            </div>
          </div>
        </section>
        {/* Evaluation Methods */}
        <section className='mx-2 bg-white p-8 shadow-sm rounded-b-2xl'>
          <h2 className='mb-6 text-3xl font-bold text-slate-900'>05. Evaluation Methods</h2>
          <p className='text-slate-600'>
            <div className="grid gap-5 md:grid-cols-3">
              {metrics.map((metric) => (
                <div
                  key={metric.name}
                  className="rounded-2xl border border-slate-200 bg-white p-7"
                >
                  <div className="mb-5 h-1 w-12 rounded-full bg-indigo-500" />
                  <h3 className="text-xl font-bold">{metric.name}</h3>
                  <p className="mt-3 leading-7 text-slate-600">
                    {metric.description}
                  </p>
                </div>
              ))}
            </div>
          </p>
        </section>
        {/* Results */}
        <section className='mx-2 bg-slate-300  p-8 pb-10  shadow-sm rounded-b-2xl'>
          <h2 className='mb-6 text-3xl font-bold text-slate-900'>06. Results</h2>
          <p className='leading-8 text-slate-600'>
            All methods were tested on the same 49,932 local election winners from the elections after 2016, using the same labels and the same metrics. The baseline always predicts the most common type, “Thin”.
          </p>
          <div className='mt-4 overflow-x-auto rounded-xl bg-white p-4 shadow-sm'>
            <table className='w-full text-left text-sm text-slate-700'>
              <thead>
                <tr className='border-b-2 border-slate-900'>
                  <th className='py-2 pr-3'>Methods</th>
                  {resultColumns.map((col) => (
                    <th key={col.key} className='px-2 py-2 text-right'>{col.label}</th>
                  ))}
                </tr>
              </thead>
              <tbody>
                {resultRows.map((row, r) => (
                  <tr key={row.name} className='border-b border-slate-200'>
                    <td className='py-2 pr-3 font-semibold'>{row.name}</td>
                    {resultColumns.map((col) => {
                      // bold the best value of each column, the baseline is not counted
                      const best = Math.max(...resultRows.slice(1).map((x) => x[col.key]));
                      const isBest = r > 0 && row[col.key] === best;
                      return (
                        <td key={col.key} className={`px-2 py-2 text-right ${isBest ? 'font-bold text-indigo-600' : ''}`}>
                          {row[col.key].toFixed(3)}
                        </td>
                      );
                    })}
                  </tr>
                ))}
              </tbody>
            </table>
          </div>
          <div className='mt-4 rounded-xl bg-white p-4 shadow-sm rounded-b-2xl'>
            <div className='mb-3 flex flex-wrap gap-x-5 gap-y-1 text-xs text-slate-600'>
              {resultRows.map((row) => (
                <span key={row.name} className='flex items-center gap-2'>
                  <span className='inline-block h-3 w-3 rounded-sm' style={{ background: row.color }} />
                  {row.name}
                </span>
              ))}
            </div>
            <ResultsChart />
          </div>
          <h3 className='mb-3 mt-8 text-xl font-semibold text-slate-900'>Interpreting the Table</h3>
          <div className='flex flex-col gap-3 leading-7 text-slate-600'>
            <p>
              <strong>Accuracy is misleading here.</strong> About 64.7% of the test rows are Thin, so the baseline that always answers “Thin” already gets the highest accuracy (0.647). K-Means (0.613) comes close only because it labels most rows as Thin (Thin recall 0.906). KNN has the lowest accuracy (0.374) because it often labels Thin rows as “None” (Thin recall 0.371). Since a high accuracy can come from guessing the biggest type, we judge the methods by the three metrics that treat all four types equally.
            </p>
            <p>
              <strong>Only Naive Bayes with all features finds the rare types.</strong> It recovers 28.4% of Fat and 24.0% of Obese officials. KNN finds only 1.8% of Fat and 13.2% of Obese, K-Means finds 6.2% of Fat and no Obese at all, and Naive Bayes with only the last name also finds no Obese.
            </p>
            <p>
              <strong>KNN has the highest macro precision (0.543), but that is not the whole story.</strong> When KNN does predict a rare type, it is often correct, but it predicts those types so rarely that it misses most of them. Precision alone does not show this, which is why it is read together with recall and balanced accuracy.
            </p>
            <p>
              <strong>Balanced accuracy and MCC show how much each method really learned.</strong> With four types, guessing gives a balanced accuracy of 0.25, which is exactly the baseline. Naive Bayes with all features (0.389) is clearly above it, KNN (0.294) is slightly above, and K-Means (0.265) and Naive Bayes with the last name only (0.264) are barely above it. The MCC tells the same story: Naive Bayes with all features is the best at 0.142, while the other methods score between 0.010 and 0.048. Since an MCC of 0 is guessing and 1 is perfect, even the best method is only weakly better than chance, so none of them predicts the dynasty type reliably yet.
            </p>
            <p>
              <strong>The other features matter.</strong> Adding the province, city, middle name, and position to the last name raises the Naive Bayes MCC from 0.010 to 0.142. The last name alone cannot recognize Obese officials, so the place and position give the model information that names do not.
            </p>
          </div>

          <h3 className='mb-3 mt-8 text-xl font-semibold text-slate-900'>Interpreting the Graph</h3>
          <div className='flex flex-col gap-3 leading-7 text-slate-600'>
            <p>
              The tallest bars are the Thin recall of the baseline (1.000) and K-Means (0.906). They are tall because Thin is the biggest group, not because those methods are skilled. The Fat recall and Obese recall bars are almost empty for every method except Naive Bayes with all features, which is the only one that stands out for the rare types.
            </p>
            <p>
              In the macro precision group, KNN is the tallest bar, followed by Naive Bayes with all features. In the balanced accuracy group, Naive Bayes with all features is the tallest and the other methods stay close to the 0.25 baseline. In the MCC group every bar is short, and only Naive Bayes with all features is clearly visible. Reading the graph from left to right, the methods look different on accuracy and recall, but on the fair metrics (balanced accuracy and MCC) the differences are small, and Naive Bayes with all features leads.
            </p>
          </div>
        </section>
        {/* Conclusion */}
        <section className='mx-2 bg-white p-8 shadow-sm rounded-b-2xl'>
          <h2 className='mb-6 text-3xl font-bold text-slate-900'>07. Conclusion</h2>
          <div className='flex flex-col gap-3 leading-8 text-slate-600'>
            <p>
              This project changed the question from “is there a political dynasty?” to “what kind of dynasty is it?” by classifying each local official as None, Thin, Fat, or Obese. We compared Naive Bayes, K-Means Clustering, and K-Nearest Neighbors, all written from scratch, using the same labels, the same train-test split (training up to 2016 and testing on the later elections), and the same metrics.
            </p>
            <p>
              Naive Bayes using all features performed best overall. It had the highest balanced accuracy (0.389) and MCC (0.142) and was the only method that found a meaningful share of the Fat and Obese officials. KNN was the most precise on the rare types (macro precision 0.543) but found very few of them. K-Means mostly predicted the majority type, so it behaved close to the baseline (MCC 0.038). The accuracy of the baseline (0.647) was higher than all three methods, which confirms that accuracy alone is not a fair way to judge this task.
            </p>
            <p>
              Even the best result is weak, so names and places alone are not enough to describe dynasties well. Our labels are also a proxy: they use a shared last name to guess family ties, which can wrongly match unrelated people with common surnames, miss relatives with different surnames, and count the same person who is re-elected as Thin. The rare types (Fat and Obese) have far fewer examples than Thin, which makes them harder to learn. Future work could use stricter family matching, add features such as party or number of terms served, give more weight to the rare types during training, and use clustering methods designed for categorical data.
            </p>
          </div>
        </section>
        {/* References */}
        <section className='mx-2 bg-slate-300  p-8 pb-10  shadow-sm rounded-b-2xl'>
          <h2 className='mb-6 text-3xl font-bold text-slate-900'>08. References</h2>
          <div className='text-slate-600 flex flex-col pl-2'>
            <div>
              [1] Mendoza, R. U., Beja, E. L., Venida, V. S., & Yap, D. B. (2016). Political dynasties and poverty: measurement and evidence of linkages in the Philippines. Oxford Development Studies, 44(2), 189–201. https://doi.org/10.1080/13600818.2016.1169264
            </div>
            <div>
              [2] Berrar, Daniel. (2018). Bayes’ Theorem and Naive Bayes Classifier. 10.1016/B978-0-12-809633-8.20473-1.
            </div>
            <div>
              [3] Tharwat, A. (2021). Classification assessment methods. Applied computing and informatics, 17(1), 168-192.
            </div>
            <div>
              [4] Chicco, D., & Jurman, G. (2020). The advantages of the Matthews correlation coefficient (MCC) over F1 score and accuracy in binary classification evaluation. BMC genomics, 21(1), 6.
            </div>
            <div>
              [5] Shahapure, K. R., & Nicholas, C. (2020, October). Cluster quality analysis using silhouette score. In 2020 IEEE 7th international conference on data science and advanced analytics (DSAA) (pp. 747-748). IEEE.
            </div>
            <div>
              [6] Ikotun, A. M., Ezugwu, A. E., Abualigah, L., Abuhaija, B., & Heming, J. (2023). K-means clustering algorithms: A comprehensive review, variants analysis, and advances in the era of big data. Information Sciences, 622, 178-210.
            </div>
            <div>
              [7] Mendoza, R. U., Jaminola, L. M., & Yap, J. (2019). From fat to obese: Political dynasties after the 2019 midterm elections. Ateneo School of Government working paper series, September, 19-013.
            </div>
            <div>
              [8] Grandini, M., Bagli, E., & Visani, G. (2020). Metrics for multi-class classification: an overview. arXiv preprint arXiv:2008.05756.
            </div>
            <div>
              [9] Tharwat, A. (2021). Classification assessment methods. Applied computing and informatics, 17(1), 168-192.
            </div>

          </div>
        </section>
        {/* AI Use */}
        <section className='mx-2 bg-white p-8 shadow-sm rounded-b-2xl'>
          <h2 className='mb-6 text-3xl font-bold text-slate-900'>09. AI Usage Disclosure</h2>
          AI tools were used during the development of this project for the following purposes:<br /><br />
          <div className='text-slate-600 flex flex-col px-4'>
            <div>
              1) <strong>Debugging and Troubleshooting</strong>: Assisting in identifying, understanding, and debugging errors encountered during implementation.
            </div>
            <div>
              2) <strong>Learning Purposes</strong>: Providing explanations to help supplement and understand concepts and how it can be applied to the project.
              This is to help the developers understand the relevant algorithms and how to modify it for the project's purposes.
            </div>
            <div>
              3) <strong>Elimination of tedious and repetitive code</strong>: For code that is already implemented but must be repeated throughout.
            </div>
            <div>
              4) <strong>Project Website enhancement</strong>: For color coordination, UI debugging, layout issues, and styling. All of which are from the developer's implementation and are further enhanced. Additionally, AI was used to help convert formula to BlockMath for better visuals.
            </div>
            <div>
              5) <strong>YML deployment file</strong>: Assisted in creating a file for deploying subfolder in monorepo as starting point for github pages
            </div>
          </div >
        </section>
      </div>
    </main>
  )
}

export default App
