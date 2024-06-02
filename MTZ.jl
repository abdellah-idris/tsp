using JuMP, Cbc
using Random, LinearAlgebra, Dates


# Fonction pour générer des instances TSP
function generate_tsp_instance(n::Int)
    points = [(rand(0:100), rand(0:100)) for _ in 1:n]
    distance_matrix = round.(Int, [sqrt((points[i][1] - points[j][1])^2 + (points[i][2] - points[j][2])^2) for i in 1:n, j in 1:n])
    return points, distance_matrix
end

# Générer les instances
instances = Dict()
sizes = [10, 20, 30, 50, 100]

for size in sizes
    instances[size] = [generate_tsp_instance(size) for _ in 1:5]
end


# Fonction pour extraire la séquence des villes à partir de la solution
function extract_tour(x)
    n = size(x, 1)
    tour = Int[]
    visited = falses(n)
    current_city = 1

    while length(tour) < n
        push!(tour, current_city)
        visited[current_city] = true
        for next_city in 1:n
            if x[current_city, next_city] > 0.5
                current_city = next_city
                break
            end
        end
    end

    return tour
end


# Fonction pour résoudre le TSP avec la formulation MTZ
function solve_tsp_mtz(distance_matrix; time_limit=300)
    n = size(distance_matrix, 1)
    model = Model(Cbc.Optimizer)
    set_optimizer_attribute(model, "seconds", time_limit)
    set_optimizer_attribute(model, "logLevel", 0)  # Suppress solver output

    @variable(model, x[1:n, 1:n], Bin)
    @variable(model, u[2:n])

    @objective(model, Min, sum(distance_matrix[i, j] * x[i, j] for i in 1:n, j in 1:n))

    @constraint(model, [i in 1:n], sum(x[i, j] for j in 1:n if i != j) == 1)
    @constraint(model, [j in 1:n], sum(x[i, j] for i in 1:n if i != j) == 1)
    @constraint(model, [i in 2:n, j in 2:n; i != j], u[i] - u[j] + n * x[i, j] <= n - 1)

    start_time = now()
    optimize!(model)
    end_time = now()
    elapsed_time = end_time - start_time

    optimal = termination_status(model) == MOI.OPTIMAL
    objective_value_result = objective_value(model)
    solution = value.(x)
    tour = extract_tour(solution)
    return objective_value_result, tour, elapsed_time, optimal
end

function perturb_solution(solution)
    n = length(solution)
    i, j = sort(rand(1:n, 2))
    new_solution = copy(solution)
    new_solution[i:j] = reverse(solution[i:j])
    return new_solution
end

function tour_cost(solution, distance_matrix)
    return sum(distance_matrix[solution[i], solution[i+1]] for i in 1:length(solution)-1) + distance_matrix[solution[end], solution[1]]
end 

function simulated_annealing(distance_matrix, max_iterations=1000, initial_temperature=100.0, cooling_rate=0.95)
    n = size(distance_matrix, 1)
    current_solution = randperm(n)
    current_cost = tour_cost(current_solution, distance_matrix)
    best_solution = copy(current_solution)
    best_cost = current_cost
    temperature = initial_temperature

    for iter in 1:max_iterations
        new_solution = perturb_solution(current_solution)
        new_cost = tour_cost(new_solution, distance_matrix)

        if new_cost < current_cost || exp((current_cost - new_cost) / temperature) > rand()
            current_solution = new_solution
            current_cost = new_cost

            if new_cost < best_cost
                best_solution = new_solution
                best_cost = new_cost
            end
        end

        temperature *= cooling_rate
    end

    return best_solution, best_cost
end

# Résoudre les instances avec le recuit simulé
results_sa = Dict()
for size in sizes
    results_sa[size] = []
    for instance in instances[size]
        points, distance_matrix = instance
        best_solution, best_cost = simulated_annealing(distance_matrix)
        push!(results_sa[size], (best_solution, best_cost))
    end
end


# Résoudre les instances générées
results_mtz = Dict()
results_sa = Dict()
for size in sizes
    results_mtz[size] = []
    results_sa[size] = []
    for (index, instance) in enumerate(instances[size])
        points, distance_matrix = instance
        obj_val, solution, elapsed_time, optimal = solve_tsp_mtz(distance_matrix)
        best_solution, best_cost = simulated_annealing(distance_matrix)
        println("Instance $index de taille $size:")
        println("Solution MTZ: ", solution)
        println("Valeur de l'objectif MTZ: ", obj_val)
        println("Temps écoulé: ", elapsed_time)
        println("Est optimale: ", optimal)
        println("Meilleure solution Recuit simulé: ", best_solution)
        println("Coût de la meilleure solution Recuit simulé: ", best_cost)
        push!(results_sa[size], (best_solution, best_cost))
        push!(results_mtz[size], (obj_val, solution, elapsed_time, optimal))
        println("############################################")
    end
    println(" ")
    println("NEW SIZE")
end
