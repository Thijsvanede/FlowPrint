import numpy as np

class Burst(object):

    def split(self, packets, threshold=1):
        """Split packets in bursts based on given threshold.
            A burst is defined as a period of inactivity specified by treshold.

            Parameters
            ----------
            packets : np.array of shape=(n_samples, n_features)
                Where the first (0-th) feature is the timestamp.

            threshold : float, default=1
                Burst threshold in seconds.

            Returns
            -------
            result : list
                List of np.array, where each list entry are the packets in a
                burst.
            """
        # Initialise result
        result = list()

        # In case of no packets return empty list
        if not packets.shape[0]:
            return result

        # Compute difference between packets
        diff = np.diff(packets[:, 0])

        # Select indices where difference is greater than threshold
        indices_split = np.argwhere(diff > threshold)
        # Add 0 as start and length as end index
        indices_split = [0] + list(indices_split.flatten()) + [packets.shape[0]]
        for start, end in zip(indices_split, indices_split[1:]):
            result.append(packets[start+1:end+1])

        # Return result
        return result
