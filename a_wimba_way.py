import numpy as np

# Name: Huy Anh Nguyen
# Section: 002
# Student Number: 20287429
# Email: 20ahan@queensu.ca

# I confirm that this assignment solution is my own work and conforms to
# Queen's standards of Academic Integrity.

# Note to self: Remember the encoding = "utf-8"?


class KMeansAlgo:
    """
    A class for separating data into clusters using the k_means clustering algorithm.
    This will output two results: One using the Manhattan Index Measurement and the other using
    the Euclidean Distance Measurement.
    """
    def __init__(self):
        """Initialize the attributes and constants"""
        self.num_clusters = 7
        self.__extract_data()

        self.__initialize_and_clear()

        # Store which distancing methods to call in a dictionary.
        self.distance_measurements = {"manhattan": self.__manhattan_index, "euclidean": self.__euclidean_value}

        # CONSTANTS
        # For checking if none of the clusters move by more than a small amount
        self.max_difference = 0.1
        self.max_iterations = 100
        self.num_algo_execute = 50

    def __initialize_and_clear(self):
        """
        Reset everything back to when the class is first called so as to use another distancing measurement
        Can also serve as a part of initializing several variables.
        """
        # Store the frequency of results of the algorithm using the same distance measurement method
        self.pattern_dict = {}
        # Store the indices of the cluster centers if one of them happen to be an animal...
        self.cluster_indices = np.zeros(shape=self.num_clusters)
        self.old_cluster_indices = np.zeros(shape=self.num_clusters)

        # Store the group indices into lists corresponding to keys which are the group number.
        self.group_dict = {}

        for i in range(self.num_clusters):
            self.group_dict[i] = []

        # Result of the algorithm using a specific distance measurement method
        self.resultant_groups = ()

        # Var for tracking the number of iterations as part of stability check.
        self.num_iterations = 0

    def __extract_data(self):
        """Extract the necessary data out of the zoo_2.txt. And initialize dependent variables too"""
        with open("zoo_2.txt", encoding="utf-8") as f:
            self.n_cols = len(f.readline().split())

        # For this assignment, assume that the first column is the name one.
        self.animals_data = np.loadtxt("zoo_2.txt", skiprows=1, usecols=range(1, self.n_cols))

        self.n_rows = self.animals_data.shape[0]
        self.animals_name = np.loadtxt("zoo_2.txt", skiprows=1, usecols=0, dtype=str)

        # Preallocate memory for faster access and indexing when working with cluster.
        self.cluster_centers = np.zeros(shape=(self.num_clusters, self.n_cols-1))

        # The ndarray of cluster centers for the next iteration of the algorithm.
        self.next_centers = np.zeros(shape=(self.num_clusters, self.n_cols-1))

        # Ndarray for storing the number of elements per group in the current iteration
        self.current_num_members = np.zeros(shape=self.num_clusters)

        # Ndarray for storing the number of elements per group in the prev iteration
        self.prev_num_members = np.zeros(shape=self.num_clusters)

    def start_algorithm(self):
        """
        Start the algorithm by running through two versions of it:
            - One using the Manhattan Index Formula.
            - One using the Euclidean Distance Formula
        """
        for method in self.distance_measurements:
            print(f"K-means using {method.title()} method:")
            self.__executing_algorithm(measurement=self.distance_measurements[method])
            # Clear variables for the next version of the algorithm that uses
            # a different distance measurement method.
            self.__initialize_and_clear()
            print("")

    def __executing_algorithm(self, measurement):
        """Execute the algorithm for a certain (hardcoded) number of times"""
        # Big loop of number of times the algorithm will run
        for i in range(self.num_algo_execute):
            self.__random_start()
            while True:
                self.__iterating_data(measurement)
                # Numpy converts int to float64 when copy stuff, which is weird...
                np.copyto(src=self.cluster_indices, dst=self.old_cluster_indices)
                self.__create_new_centers()
                self.num_iterations += 1
                if self.__check_stability():
                    break
                self.__prepare_next_iteration()

            # post-While-loop here:
            self.__pattern_formulation()
            self.num_iterations = 0

        # Get the pattern that occurs the most in self.pattern_dict
        self.__get_most_common_pattern()
        # Output the result
        self.__finalized_output()

    def __random_start(self):
        """
        Randomly choose 7 centers index and extract the animal data
        with those indexes at the start for the algorithm
        """
        # Ranging from 0 to n_rows - 1
        self.cluster_indices = np.random.randint(low=0, high=self.n_rows, size=self.num_clusters, dtype=int)

        # Should be an ndarry with shapes (num_clusters, n_cols), or (7, 15) in this case
        self.cluster_centers = self.animals_data[self.cluster_indices]

    def __iterating_data(self, measurement):
        """
        Iterating through the data and apply the 'farness' calculation method on
        each data point that are not the centers to all the current cluster centers
        """
        for i in range(self.n_rows):
            # Prevent duplication...
            if i not in self.cluster_indices:
                group_num = measurement(self.animals_data[i])
                self.group_dict[group_num].append(i)
                # Increment the number of members in the corresponding ndarray column tracking the
                # number of members per group
                self.current_num_members[group_num] += 1

    def __manhattan_index(self, animal: np.ndarray) -> int:
        """
        Calculate the Manhattan index value for each data point to the cluster centres
        Returns the group that corresponds to the centre with lowest Manhattan index.
        """
        manhattan_list = []
        for i in range(self.num_clusters):
            man_metric = np.sum(np.abs(np.subtract(animal, self.cluster_centers[i])))
            # Adding the Manhattan Metric along with the group_number.
            manhattan_list.append((man_metric, i))

        # Get the min man_metric value out of the list
        min_group = min(manhattan_list, key=lambda x: x[0])
        return min_group[1]

    def __euclidean_value(self, animal: np.ndarray) -> int:
        """
        Calculate the Euclidean distance for each data point to the cluster centres.
        Return the group index corresponding to the cluster centre with lowest Euclidean distance.
        """
        euclidean_list = []
        for i in range(self.num_clusters):
            euclid_dist = np.sqrt(np.sum(np.subtract(animal, self.cluster_centers[i]) ** 2))
            # Adding the Euclidean Distance value along with the group_number.
            euclidean_list.append((euclid_dist, i))

        # Get the min euclid_dist value out of the list
        min_group = min(euclidean_list, key=lambda x: x[0])
        return min_group[1]

    def __check_stability(self):
        """
        Check if the clustering algorithm has reached stability by checking on three criteria
        :return: True if the algorithm meets one of the three criteria.
                    False if the algorithm does not meet any of the three criteria.
        """
        if self.__check_num_iterations() or \
                self.__check_centers_difference()\
                or self.__check_points_move():
            return True
        return False

    def __check_points_move(self):
        """
        Check if no items switch to a different cluster after the distances are computed.
        :return: False for there exists difference in members count per group in current iteration vs prev one
                 True for otherwise.
        """
        # Check if each group's member count still stays the same in this iteration when compared with
        # the previous version.
        if (self.current_num_members == self.prev_num_members).all():
            # print(f"No animals jump group!")
            return True
        else:
            return False

    def __check_centers_difference(self):
        """
        Check if every cluster center moves by over a certain limit using Euclidean Distance Formula
        :return True for stability and False for not meeting the conditions for stability
        """
        # Use ndarray of new_centers and the cluster_centers
        for i in range(self.num_clusters):
            center_distance = np.sqrt(np.sum(np.subtract(self.next_centers[i], self.cluster_centers[i]) ** 2))
            # print(f"Their Euclidean distance: {center_distance}\n")
            if center_distance > self.max_difference:
                return False
        # print(f"No cluster center moves by over {self.max_difference}")
        return True

    def __create_new_centers(self):
        """
        Create new temp cluster_centers for this iteration.
        """
        # Prevent from accidentally matching the first value 0 if I do fill(0)
        self.cluster_indices.fill(-1)
        # Calculate new centers and put them into self.new_centers
        for i in range(self.num_clusters):
            members_indices = self.group_dict[i]
            # Including the center too.
            num_members = len(members_indices) + 1

            # Duct-tape solution because for some reason some of the lists in self.group_dict
            # have the first element as a float instead of an integer.
            # This will forcibly turn the entire list into ints.
            members_indices = [int(x) for x in members_indices]

            # Add the cluster center separately, because the members_indices only contain the indices of
            # members that are not the cluster center.
            sum_group = np.sum(self.animals_data[members_indices], axis=0) + self.cluster_centers[i]
            new_center = sum_group / num_members
            self.next_centers[i] = new_center

            # Get the index of the new_center index in the self.animals_data
            if_animal_index = self.__check_if_animal(array_to_check=new_center)
            # -1 is for when the average is not in the self.animals_data
            if if_animal_index != -1:
                self.cluster_indices[i] = if_animal_index

    def __check_if_animal(self, array_to_check) -> int:
        """
        Check if an array is a part of the larger self.animals_data
        :return The index of the array_to_check in self.animals_data
                    OR -1 if that is not the case.
        """
        # This should output the list of indices of array matching the array_to_check
        index = np.where(np.all(self.animals_data == array_to_check, axis=1))[0]
        if len(index) != 0:
            return index[0]
        return -1

    def __check_num_iterations(self):
        """
        Check whether the current run of the algorithm has reached the maximum number of iterations.
        :return False for not running long enough and True for running long enough...
        """
        if self.num_iterations < self.max_iterations:
            return False
        else:
            print(f"Number of iterations reaches max {self.max_iterations}!")
            return True

    def __prepare_next_iteration(self):
        """
        Prepare for the next iteration of the K-means Algorithm
        """
        # Copy data from the future next_centers to the current cluster_centers at the start of the
        # new iteration.
        np.copyto(src=self.next_centers, dst=self.cluster_centers)
        # Set the prev's iteration num_members and reset the current_num_member for the new iteration.
        np.copyto(src=self.current_num_members, dst=self.prev_num_members)
        self.current_num_members.fill(0)

        # Clear all the lists in group_dict.
        for group_num in range(self.num_clusters):
            self.group_dict[group_num].clear()

    def __pattern_formulation(self):
        """
        Create a nested tuple out of self.group_dict after algorithm reaches stability
        """
        # Check if current_center's is an animal...
        # Add pattern to the pattern_dict...
        # Go through self.cluster_indices to add the index of animal acting as cluster center into their
        # respective group.

        # This should be a list of tuples
        groups_members_indices = []
        for i in range(self.num_clusters):
            # Get the member of self.cluster_indices and append those that are not -1 to their respective group
            center_index = self.old_cluster_indices[i]
            members_indices = self.group_dict[i]
            if center_index != -1:
                members_indices.append(center_index)
            members_indices.sort()
            groups_members_indices.append(tuple(members_indices))

        # Make the whole thing a tuple so as to be qualified for being a dictionary key.
        groups_members_indices.sort()
        groups_members_indices = tuple(groups_members_indices)

        # Add count to the number of time a specific frequency happen.
        self.pattern_dict[groups_members_indices] = self.pattern_dict.get(groups_members_indices, 0) + 1

    def __get_most_common_pattern(self):
        self.resultant_groups = max(self.pattern_dict, key=lambda x: self.pattern_dict[x])

    def __finalized_output(self):
        """
        Output the group with names of the animals
        """
        for group_num in range(self.num_clusters):
            animal_name_indices = np.array(self.resultant_groups[group_num])
            num_members = len(animal_name_indices)
            animal_name_indices = animal_name_indices.astype(dtype=int)

            animal_name_group = self.animals_name[animal_name_indices]
            result = f"Group {group_num + 1} ({num_members} members): {' '.join(animal_name_group)}"
            print(result)


if __name__ == "__main__":
    algorithm = KMeansAlgo()
    algorithm.start_algorithm()
