import { useState } from 'react'
import headerImage from "./assets/pexels-kindelmedia-8566449.jpg"
import { BlockMath } from "react-katex";
import "katex/dist/katex.min.css";
import './App.css'
import { Brain, Database, Lambda, GitBranch, UserRoundGroup } from 'lucide-react';
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
    <main className="min-h-screen  mx-auto bg-slate-50 flex flex-col">
      {/* header */}
      <div className='flex px-12 py-8 bg-slate-900 items-center'>
        <div className='flex gap-4 flex-col flex-grow text-white'>
          <div className='flex gap-2'><Brain />CMSC 170: Applying AI and Machine Learning Methods</div>
          <div className='flex gap-2'><Database />OpenHalalan: Philippine National and Local Election Dataset {" "}
            <a href="https://data.bettergov.ph/datasets/25" className='underline'>(https://data.bettergov.ph/datasets/25)</a>
          </div>
          <div className='flex gap-2'><Lambda /> Naive Bayes | K-Means Clustering | K-Nearest Neighbors</div>
          <div className='flex gap-2'><GitBranch /><a href="https://github.com/jaeson-cho/uplb-cmsc170-j.git" className='underline hover:font-blue-600'>https://github.com/jaeson-cho/uplb-cmsc170-j.git </a></div>
          <div className='flex gap-2'><UserRoundGroup /> Group J - Cho | Licayan | Figuro | Osorio</div>
        </div>
        <div className='flex w-1/4 shrink-0 items-center justify-center'>
          <img
            src={headerImage}
            className='w-full max-w-[320px] object-contain shadow-2xl rounded-xl'
          />
        </div>

      </div>
      <div className='flex flex-col gap-2'>

        {/* Intro */}
        <section className='mx-2 bg-white p-8 shadow-sm'>
          <h2 className='mb-6 text-3xl font-bold text-slate-900'>01. Introduction</h2>
          <p className='leading-8 text-slate-600'>
            Political Dynasties have long since been a significant aspect of Philippine politics which allows certain families to maintain their political influences across multiple generations. According to the 1987 Constitution, it states that the state shall prohibit political dynasties “as may be defined by law”, but no such definition has been enacted. According to Mendoza et al, they found that political dynasties make up over 70% of the Philippine Congress and that they have a worsening effect on poverty in provinces outside Luzon [1]. This leads to corruption, and failure for people seated in power to properly serve the Filipino people. Identifying locations where political dynasties occur can help provide a clearer understanding of the distribution of political power across provinces and cities in the Philippines. This helps us, the Filipino people, to properly scrutinize if the right people are being assigned to the right jobs.

            Identifying where political dynasties occur can help provide a clear picture and understanding of the political power that is distributed across the cities and provinces of the Philippines. This helps the Filipino people to properly scrutinize whether the right people are assigned to the right positions or jobs.

            This project aims to analyze OpenHalalan’s dataset containing records of politicians to detect the probability of political dynasties happening in a given location. Using Artificial Intelligence methods taught (and possibly beyond what was taught in class), the project seeks to identify patterns that indicate presence of political dynasties in specific areas. The methods will be proposed and discussed below, as well as the reasonings as to why the specific methods are selected.

          </p>
        </section>
        {/* Goals */}
        <section className='mx-2 bg-slate-300 p-8 pb-10 shadow-sm'>
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
        <section className='mx-2 bg-white p-8 shadow-sm'>
          <h2 className='mb-6 text-3xl font-bold text-slate-900'>03. Definitions</h2>
          <p className='text-slate-600'>
            The dataset shows no family relationships so we approximate them using names with a shared Last Name as the sign that two officials are related. Following the evaluation of thin, fat, and obese [7], each city or municipality level winner gets one of the four types. Let n be the number of different people with different First Names with the same Last Name who won in the same province,  city, and year counting the official:
            <p className='ml-2 mt-2'>1) <strong>Obese</strong>: n = 4 (four or more relatives in office at the same time)</p>
            <p className='ml-2 mt-2'>2) <strong>Fat</strong>: n is 2 or 3 (relatives in office at the same time)</p>
            <p className='ml-2 mt-2'>3) <strong>Thin</strong>: n is 1 but someone with the same last name won in the same city in a previous election cycle</p>
            <p className='ml-2 mt-2 mb-2'>4) <strong>None</strong>: None of the above</p>

            The labeled rows will start from 2004. This label will be shared by all three methods so the comparison stays fair. Under this rule about 75% are roughly none, 12% are thin, 12% are fat, and 1% are obese. In the later election cycles, the shares are 70-15-14-1 in that order.


          </p>
        </section>
        {/* Methods */}
        <section className='mx-2 bg-slate-300 p-8 pb-10 shadow-sm'>
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
        <section className='mx-2 bg-white p-8 shadow-sm'>
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
        <section className='mx-2 bg-slate-300  p-8 pb-10  shadow-sm'>
          <h2 className='mb-6 text-3xl font-bold text-slate-900'>06. Results</h2>
          <p className='text-slate-600'></p>
        </section>
        {/* Conclusion */}
        <section className='mx-2 bg-white p-8 shadow-sm'>
          <h2 className='mb-6 text-3xl font-bold text-slate-900'>07. Conclusion</h2>
          <p className='text-slate-600'></p>
        </section>
        {/* References */}
        <section className='mx-2 bg-white p-8 shadow-sm'>
          <h2 className='mb-6 text-3xl font-bold text-slate-900'>08. References</h2>
          <p className='text-slate-600'></p>
        </section>
        {/* AI Use */}
        <section className='mx-2 bg-white p-8 shadow-sm'>
          <h2 className='mb-6 text-3xl font-bold text-slate-900'>09. AI Usage Disclosure</h2>
          <div className='text-slate-600 flex flex-col'>
            AI tools were used during the development of this project for the following purposes:<br /><br />
            <div>
              &nbsp;&nbsp;&nbsp;&nbsp; 1) <strong>Debugging and Troubleshooting</strong>: Assisting in identifying, understanding, and debugging errors encountered during implementation.
            </div>
            <div>
              &nbsp;&nbsp;&nbsp;&nbsp; 2) <strong>Learning Purposes</strong>: Providing explanations to help supplement and understand concepts and how it can be applied to the project.
              This is to help the developers understand the relevant algorithms and how to modify it for the project's purposes.
            </div>
            <div>
              &nbsp;&nbsp;&nbsp;&nbsp; 3) <strong>Elimination of tedious and repetitive code</strong>: For code that is already implemented but must be repeated throughout.
            </div>
            <div>
              &nbsp;&nbsp;&nbsp;&nbsp; 4) <strong>Project Website enhancement</strong>: For color coordination, UI debugging, layout issues, and styling. All of which are from the developer's implementation and are further enhanced.
            </div>
          </div >
        </section>
      </div>
    </main>
  )
}

export default App
