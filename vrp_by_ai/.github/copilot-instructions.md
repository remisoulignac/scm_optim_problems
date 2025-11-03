Using state of the art algorithms, help me build a solution to the bellow problem. As you can understand we must factor in the CO2 emission in the Vehicle Routing Problem. The Arc Cost is not the distance anymore but the CO2 emission over each Arc, that is not linear and depends on 2 decision variables
- the current load of the vehicle, meaning the list of nodes it visited before.
- its speed


You will be in charge of deciding the most adequate vehicle routing for the last mile delivery of one day in the city of Xalapa, in Mexico. You will work only with one vehicle and 30 customers for home delivery. It is your job to consider what sequence you must follow to hand out the products to the customers. For your strategy, you may need to consider the distance, product weight, vehicule speed and topography in your decisions, since your main indicators are distance, cost and CO2 emissions, being this last one our highest priority. Please consider that each product is assigned to one client and each client must be visited only once. Historically this route has had daily rate of 50 kg of CO2 emissions, and so, it is desirable to find a better route that will help decrease this number in 10% as a target. 
The truck leaves a DC (node 0) and must return at then to the same DC.


You use the ALNS (Adaptive Large Neighborhood Search) metaheuristic to solve this Vehicle Routing Problem (VRP) with a focus on minimizing CO2 emissions. The solution involves implementing a custom cost function that accounts for the non-linear relationship between vehicle load, speed, and CO2 emissions.

ALNS Reference documentation :
 ALNS_LIBRARY_DOCUMENTATION.md
 ALNS_QUICK_REFERENCE.md

 OUTPUT : vrp_solver.py
 DATA : data.txt (distances, demands, time windows, slopes, delivery times)