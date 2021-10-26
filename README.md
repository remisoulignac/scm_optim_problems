# SCX SCREAM
This is my  attempt using Mixed Integer Linear Programming to find the best solution for the SCREAM challenge http://scxscream.herokuapp.com/home/.

Let's put it right away : it does not give the best answer. Indeed, I found it extremely hard to understand and modelize properly the behaviour of the SCREAM simulator. There are few strange behaviours like :
 - even with DC, Plant and Supplier closed for a year and with no backup, you still make profit !
 - when you buy backup FG inventory upfront you only pay the conversion costs $20. You will pay $50 more for the WIP once you use this backup FG. But you don't need to ensure how these WIP will be delivered to you. These WIP are there even if your supplier or plant are closed. I don't understand how one can accumulate a "conversion inventory".

Moreover, the score of SCXScream is calculated as if each scenario had the same probability of occurence. Here, with the present solution, we try to find the best solution for a combinaison of random scenarios based on their probability of occurence. 

These limitations put aside, it's was a good interesting optimization problem. However, because SCREAM Simulator is a black box, MILP does not seem to be the best way to find the solution. A better solution could be a Monte Carlo approach which would directly target the REST API (https://scxscream.herokuapp.com/tests/) with various backup parameters and test scenarios.

# Usage
import and run the notebook into a https://colab.research.google.com/. The calculus may take several hours, depending on the number of scenarios.

# Problem analysis
The hard problem could be sum up on 
1) how do we estimate the risk probability for the various disruptions.
2) how do  we modelize the supply chain

For example, let's say,
 - we are 100% sure we will have full disruption of our DC each year. In that case, it is better to directly buy backup inventory of Finished Goods to cover our full yearly demand at the beginning of the year. We will endure holding cost but we are sure we will use this inventory.
 - but now, let's say we are only 20% sure our DC will fail each year. Are we going to buy a full year of backup FG and pay full holding cost even though we will not use it ? In that case, it may be worth paying an option for an additionnal DC.

Now let's say we have :
- 50% probability of no DC failure, 10% probability of 2 months DC failure, 5% probability 1 months DC failure
- 40% probability of no plant failure, 20% probability of 2 months Plant failure
What should be the optimal mix of WIP backup inventory, FG backup inventory, Supplier Backup Option, ...?

The question lies with how we estimate disruption probability risks of the supply chain of the proposed Use Case Textile Computation, Inc.
* The company is vertically-integrated => Good Control
* The supply chain is very simple => Low complexity
* Waco, Texas Factory has no union force => Lower risks of strike
* Texas : what are the risk probability disaster in Texas ? Here is a gold mine to help estimate this : https://hazards.fema.gov/nri/map. Dallas is "Relatively Moderate". One can find raw figures here https://hazards.fema.gov/nri/data-resources#csvDownload. We see that for county of McLennan (the one of the Waco Factory) that the "the number of occurrences per year of various hazard occurrence " are
  - Drought : 28 events/year
  - Earthquake : 0 events/year
  - Hail : 4 events/year
  - Landslide : 80 events/year
  - Strong Wind : 2 events/year
These figures relate to the entire county. So the probability and the level of disruption to apply to the factory of Waco should be adjusted accordingly.

Then, we keep on this analysis for the other parts of the supply chain.

Now, another challenge is how to convert these qualitative and statistics into a probability distribution of event occurences and durations ? Here is a excerpt of a few publications regarding this question :
 - [Natural_Disasters_Casualties_and_Power_Laws_A_Comparative_Analysis_with_Armed_Conflict](https://www.researchgate.net/publication/228576670_Natural_Disasters_Casualties_and_Power_Laws_A_Comparative_Analysis_with_Armed_Conflict)
 - [Ensemble Model Development for the Prediction of a Disaster Index in Water Treatment Systems](https://www.mdpi.com/2073-4441/12/11/3195)
 - https://public.wmo.int/en/resources/bulletin/quantifying-risk-disasters-occur-hazard-information-probabilistic-risk-assessment
 - [Fat Tailed Distributions for Deaths in Conflicts and Disasters](https://www.researchgate.net/publication/305983595_Fat_Tailed_Distributions_for_Deaths_in_Conflicts_and_Disasters)

These publications highlight the Power Law continous distribution to estimate the probability of disasters.
The discrete distribution matching the continuous Power Law distribution is the Zipf distribution. It is a very wild distribution. The mean of a draw is very variable. If you target a mean, let's say of 3 disrupted weeks per year over 10 years, you may have to draw one hundred times to get a serie matching the mean.

Once we have finished identifying the various probabiliy distributions of disasters, we apply these scenarios to a modelized supply chain. The goal is to find the most resilient supply chain, that is to say the supply chain which will statistically provide the best expectation of Profit and Item Fill Rate. We play with a penalty applied to missed sales to favor higher Item Fill Rate.

MILP is very aggressive at optimizing : it will tend to accumulate inventory before the disruptions, up to the full capacity of the facilities. That is not realistic because normally, one doesn't know that disruption occurs. To force MILP to not anticipate, we apply a limit on the flow between the facilities : the cumulative sum of the flows cannot be greater than the demand over the same period.

