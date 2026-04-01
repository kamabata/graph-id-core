from __future__ import annotations

from typing import Any

import numpy as np
from pymatgen.analysis.local_env import NearNeighbors, VoronoiNN, CrystalNN
from pymatgen.core import IStructure, Structure
from sklearn.cluster import DBSCAN


def _get_original_site(structure, site):
    """Find the original site index from a periodic neighbor site.

    Parameters
    ----------
    structure : Structure or IStructure
        The structure containing the sites.
    site : Site
        A neighbor site (possibly in a periodic image).

    Returns
    -------
    int
        The index of the corresponding site in the original structure.

    Raises
    ------
    Exception
        If the site cannot be found in the structure.
    """
    if isinstance(structure, IStructure | Structure):
        site_fcoords = site.frac_coords
        strc_fcoords = structure.frac_coords
        tol = 1e-8  # threshold in Site.is_periodic_image
        # sort to reduce the iteration
        nearest_i = np.argsort(-(np.abs(strc_fcoords - site_fcoords) < tol).sum(axis=1))

        for i in nearest_i:
            if site.is_periodic_image(structure[i]):
                return i
    else:
        for i, s in enumerate(structure):
            if site == s:
                return i
    raise Exception("Site not found!")  # noqa: TRY002, TRY003, EM101


class DistanceClusteringNN(NearNeighbors):

    """Neighbor detection using DBSCAN clustering on interatomic distances.

    This class identifies neighbors by clustering the distribution of
    interatomic distances using the DBSCAN algorithm. This allows for
    automatic detection of distinct bond length populations, which is
    useful for structures with multiple bond types or unusual bonding.

    The algorithm:

    1. Computes all pairwise distances within a cutoff
    2. Applies DBSCAN clustering (eps=0.5, min_samples=2)
    3. Each cluster represents a distinct bond length population
    4. Neighbors are assigned to clusters by their distance

    Examples
    --------
    >>> from graph_id.analysis.local_env import DistanceClusteringNN
    >>> nn = DistanceClusteringNN()
    >>> neighbors = nn.get_nn_info(structure, site_index=0, rank_k=0)

    See Also
    --------
    DistanceClusteringGraphID : Graph ID generator using this class
    """

    def __init__(self) -> None:
        """Initialize the DistanceClusteringNN neighbor finder."""

    @property
    def structures_allowed(self) -> bool:
        """Check if this neighbor finder can be used with Structure objects.

        Returns
        -------
        bool
            Always True for this class.
        """
        return True

    def get_nn_info(self, structure: Structure, n: int, rank_k: int, cutoff: float = 6.0) -> list[dict[str, Any]]:
        """Get neighbor information for a specific site and distance cluster.

        Parameters
        ----------
        structure : Structure
            The input pymatgen Structure.
        n : int
            Index of the site to find neighbors for.
        rank_k : int
            The distance cluster index (0-based). Cluster 0 contains the
            shortest bonds, cluster 1 the next shortest, etc.
        cutoff : float, default 6.0
            Maximum distance cutoff in Angstroms.

        Returns
        -------
        list of dict
            List of neighbor information dictionaries, each containing:

            - ``site``: The neighbor Site object
            - ``image``: Periodic image indices (i, j, k)
            - ``weight``: The bond distance (rounded to 3 decimals)
            - ``site_index``: Index of the neighbor in the structure
            - ``edge_properties``: Dict with ``cluster_idx`` key
        """
        site = structure[n]
        cutoff_cluster_list = self.get_cutoff_cluster(structure, n, cutoff)
        if len(cutoff_cluster_list) <= rank_k:
            return []

        neighs_dists = structure.get_neighbors(site, cutoff_cluster_list[rank_k])
        max_weight = round(cutoff_cluster_list[rank_k], 3)
        # is_periodic = isinstance(structure, Structure | IStructure) # Python 3.10 以降でのみサポート
        is_periodic = isinstance(structure, (IStructure, Structure))
        siw = []

        for nn in neighs_dists:
            weight = round(nn.nn_distance, 3)
            if (rank_k > 0 and weight <= max_weight and weight > round(cutoff_cluster_list[rank_k - 1], 3)) or (
                rank_k == 0 and weight <= max_weight
            ):
                siw.append(
                    {
                        "site": nn,
                        "image": self._get_image(structure, nn) if is_periodic else None,
                        "weight": weight,
                        "site_index": self._get_original_site(structure, nn),
                        "edge_properties": {"cluster_idx": rank_k + 1},
                    },
                )

        return siw

    def get_cutoff_cluster(self, structure: Structure, n: int, cutoff: float = 6.0) -> list:
        """Get distance cluster cutoffs using DBSCAN clustering.

        Computes all interatomic distances from site n within the cutoff,
        then clusters them using DBSCAN. Returns the maximum distance in
        each cluster as cutoff thresholds.

        Parameters
        ----------
        structure : Structure
            The input pymatgen Structure.
        n : int
            Index of the central site.
        cutoff : float, default 6.0
            Maximum distance to consider in Angstroms.

        Returns
        -------
        list of float
            Sorted list of maximum distances for each cluster.
            The i-th element is the cutoff for cluster i.

        Notes
        -----
        Uses DBSCAN with eps=0.5 and min_samples=2 to cluster distances.
        Distances that don't fit into any cluster are ignored.
        """
        # # スーパーセルを作成し、6.0angまでの結合長を数え上げる
        # copy_structure = structure.copy()
        # supercell = copy_structure.make_supercell([3, 3, 3])
        # site_i = structure[n]

        # site_index = None
        # for idx, site in enumerate(supercell):
        #     # Siteのdistanceメソッドを使うとなぜか正しく距離が計算されない
        #     if float(np.linalg.norm(site_i.coords - site.coords)) < 0.01:
        #         site_index = idx
        #         break

        distance_list = []
        neighbors = structure.get_sites_in_sphere(structure[n].coords, cutoff)
        for neighbor in neighbors:
            dist = neighbor.nn_distance
            distance_list.append([dist, 0])

        dbscan = DBSCAN(eps=0.5, min_samples=2)
        dbscan.fit(distance_list)
        labels = dbscan.labels_

        max_dist_list = [0 for _ in range(max(labels) + 1)]
        for label_number in range(max(labels) + 1):
            max_dist = 0
            for label, distance in zip(labels, distance_list, strict=False):
                if label == label_number:
                    max_dist = max(max_dist, distance[0])

            max_dist_list[label_number] = max_dist

        return sorted(max_dist_list)


class BondClusteringNN(CrystalNN):

    def __init__(
        self,
        weighted_cn=False,
        cation_anion=False,
        distance_cutoffs=(0.5, 1),
        x_diff_weight=3.0,
        porous_adjustment=True,
        search_cutoff=10,
        fingerprint_length=None,
    ) -> None:
        """
        Initialize CrystalNN with desired parameters. Default parameters assume
        "chemical bond" type behavior is desired. For geometric neighbor
        finding (e.g., structural framework), set (i) distance_cutoffs=None,
        (ii) x_diff_weight=0 and (optionally) (iii) porous_adjustment=False
        which will disregard the atomic identities and perform best for a purely
        geometric match.

        Args:
            weighted_cn (bool): if set to True, will return fractional weights
                for each potential near neighbor.
            cation_anion (bool): if set True, will restrict bonding targets to
                sites with opposite or zero charge. Requires an oxidation states
                on all sites in the structure.
            distance_cutoffs ([float, float]): - if not None, penalizes neighbor
                distances greater than sum of covalent radii plus
                distance_cutoffs[0]. Distances greater than covalent radii sum
                plus distance_cutoffs[1] are enforced to have zero weight.
            x_diff_weight (float): - if multiple types of neighbor elements are
                possible, this sets preferences for targets with higher
                electronegativity difference.
            porous_adjustment (bool): - if True, readjusts Voronoi weights to
                better describe layered / porous structures
            search_cutoff (float): cutoff in Angstroms for initial neighbor
                search; this will be adjusted if needed internally
            fingerprint_length (int): if a fixed_length CN "fingerprint" is
                desired from get_nn_data(), set this parameter
        """
        super().__init__(
            weighted_cn,
            cation_anion,
            distance_cutoffs,
            x_diff_weight,
            porous_adjustment,
            search_cutoff,
            fingerprint_length
        )

    def get_nn_info(self, structure: Structure, n: int, cutoff: float = 10.0) -> list[dict[str, Any]]:
        """
        Args:
            structure (Structure): input structure.
            n (int): index of site for which to determine near
                neighbors.
            cutoff (float): distance cutoff parameter.
        Returns:
            siw (list[dict]): dicts with (Site, array, float) each one of which represents a
                neighbor site, its image location, and its weight.
        """

        site = structure[n]
        nn_data = self.get_nn_data(structure, n)

        max_key = max(nn_data.cn_weights, key=lambda k: nn_data.cn_weights[k])
        nn = nn_data.cn_nninfo[max_key]

        weight_list = []
        for entry in nn_data.all_nninfo:
            weight = 0
            for cn in nn_data.cn_nninfo:
                for cn_entry in nn_data.cn_nninfo[cn]:
                    if entry["site"] == cn_entry["site"]:
                        weight += nn_data.cn_weights[cn]

            # entry["weight"] = weight
            weight_list.append([weight, 0])

        if weight_list == []:
            return nn_data.all_nninfo
            
        dbscan = DBSCAN(eps=0.5, min_samples=2)
        dbscan.fit(weight_list)
        labels = dbscan.labels_

        for entry, label in zip(nn_data.all_nninfo, labels):
            entry["weight"] = label + 1 # weightが0だとStructureGraphのedgeにweightを持たせてくれない
        
        return nn_data.all_nninfo
