from pathlib import Path
from unittest import TestCase

import functools
import networkx as nx
from networkx.algorithms.distance_measures import diameter
from pymatgen.core import Structure

from graph_id.analysis.graphs import StructureGraph
from graph_id.analysis.compositional_sequence import CompositionalSequence
from graph_id.analysis.local_env import BondClusteringNN
from graph_id.core.bond_clustering_graph_id import BondClusteringGraphID

TEST_FILES = str(Path(__file__).resolve().parent / "test_files")


class TestBondClusteringGraphID(TestCase):
    def test_small(self):
        """
        Single site structures test
        """
        s = Structure.from_file(f"{TEST_FILES}/mp-36.cif")

        bcgid = BondClusteringGraphID(nn=BondClusteringNN())

        id_1 = bcgid.get_id(s)

        assert id_1 == "Sc-3D-4b650d014ff2479e"

    def test_jt_bond(self):
        """
        Test for bonds with Jahn-Teller distortion
        """

        lacoo_monoclinic = Structure.from_file(f"{TEST_FILES}/LaCoO3_monoclinic.cif")
        lacoo_trigonal = Structure.from_file(f"{TEST_FILES}/LaCoO3_trigonal.cif")

        sg_monoclinic = StructureGraph.from_local_env_strategy(lacoo_monoclinic, BondClusteringNN(), weights=True)
        sg_trigonal = StructureGraph.from_local_env_strategy(lacoo_trigonal, BondClusteringNN(), weights=True)
        # monoclinic: ['O2-_O2--Co3+(a)2La3+(a)4O2-(a)9', 'O2-_O2--Co3+(a)2La3+(a)4O2-(a)8', 'Co3+_Co3+-O2-(a)6', 'La3+_La3+-O2-(a)12']
        # trigonal: ['O2-_O2--Co3+(a)2La3+(a)4O2-(a)9', 'La3+_La3+-O2-(a)12', 'Co3+_Co3+-O2-(a)6', 'O2-_O2--Co3+(a)2La3+(a)4O2-(a)10']

        additional_depth = 1
        diameter_factor = 0
        hash_cs = False
        
        node_cs_list = []
        for sg in [sg_monoclinic, sg_trigonal]: 
            use_previous_cs = False

            compound = sg.structure
            prev_num_uniq = len(compound.composition)
            
            sg.set_elemental_labels()

            node_attributes = {}
            sg.cc_cs = []
            get_connected_sites_light = functools.lru_cache(maxsize=None)(sg.get_connected_sites_light)

            ug = sg.graph.to_undirected()

            for cc in nx.connected_components(ug):
                cs_list = []

                d = diameter(ug.subgraph(cc))

                for focused_site_i in cc:
                    depth = diameter_factor * d + additional_depth

                    cs = CompositionalSequence(
                        focused_site_i=focused_site_i,
                        starting_labels=sg.starting_labels,
                        hash_cs=hash_cs,
                        use_previous_cs=use_previous_cs,
                    )

                    for _ in range(depth):
                        for c_site in cs.get_current_starting_sites():
                            nsites = get_connected_sites_light(c_site[0], c_site[1])
                            cs.count_composition_for_neighbors(nsites)
                            # print(f"composition_counter: {cs.composition_counter}")

                        cs.finalize_this_depth()

                    this_cs = str(cs)

                    node_attributes[focused_site_i] = sg.starting_labels[focused_site_i] + "_" + this_cs
                    cs_list.append(this_cs)

                sg.cc_cs.append({"site_i": cc, "cs_list": cs_list})

            nx.set_node_attributes(sg.graph, values=node_attributes, name="compositional_sequence")
            node_cs_list.append(list(set(nx.get_node_attributes(sg.graph, "compositional_sequence").values())))

        assert sorted(node_cs_list[0]) == ['Co3+_Co3+-O2-(a)6', 'La3+_La3+-O2-(a)12', 'O2-_O2--Co3+(a)2La3+(a)4O2-(a)8', 'O2-_O2--Co3+(a)2La3+(a)4O2-(a)9']
        assert sorted(node_cs_list[1]) == ['Co3+_Co3+-O2-(a)6', 'La3+_La3+-O2-(a)12', 'O2-_O2--Co3+(a)2La3+(a)4O2-(a)10', 'O2-_O2--Co3+(a)2La3+(a)4O2-(a)9']


    def test_clay_bond(self):
        """
        Test for bonds with Jahn-Teller distortion
        """

        clay_before = Structure.from_file(f"{TEST_FILES}/mp-720262_0.cif")
        clay_after = Structure.from_file(f"{TEST_FILES}/mp-720262_12.cif")

        sg_clay_before = StructureGraph.from_local_env_strategy(clay_before, BondClusteringNN(), weights=True)
        sg_clay_after = StructureGraph.from_local_env_strategy(clay_after, BondClusteringNN(), weights=True)
        # monoclinic: ['O2-_O2--Co3+(a)2La3+(a)4O2-(a)9', 'O2-_O2--Co3+(a)2La3+(a)4O2-(a)8', 'Co3+_Co3+-O2-(a)6', 'La3+_La3+-O2-(a)12']
        # trigonal: ['O2-_O2--Co3+(a)2La3+(a)4O2-(a)9', 'La3+_La3+-O2-(a)12', 'Co3+_Co3+-O2-(a)6', 'O2-_O2--Co3+(a)2La3+(a)4O2-(a)10']

        additional_depth = 1
        diameter_factor = 0
        hash_cs = False
        
        node_cs_list = []
        for sg in [sg_clay_before, sg_clay_after]: 
            use_previous_cs = False

            compound = sg.structure
            prev_num_uniq = len(compound.composition)
            
            sg.set_elemental_labels()

            node_attributes = {}
            sg.cc_cs = []
            get_connected_sites_light = functools.lru_cache(maxsize=None)(sg.get_connected_sites_light)

            ug = sg.graph.to_undirected()

            for cc in nx.connected_components(ug):
                cs_list = []

                d = diameter(ug.subgraph(cc))

                for focused_site_i in cc:
                    depth = diameter_factor * d + additional_depth

                    cs = CompositionalSequence(
                        focused_site_i=focused_site_i,
                        starting_labels=sg.starting_labels,
                        hash_cs=hash_cs,
                        use_previous_cs=use_previous_cs,
                    )

                    for _ in range(depth):
                        for c_site in cs.get_current_starting_sites():
                            nsites = get_connected_sites_light(c_site[0], c_site[1])
                            cs.count_composition_for_neighbors(nsites)
                            # print(f"composition_counter: {cs.composition_counter}")

                        cs.finalize_this_depth()

                    this_cs = str(cs)

                    node_attributes[focused_site_i] = sg.starting_labels[focused_site_i] + "_" + this_cs
                    cs_list.append(this_cs)

                sg.cc_cs.append({"site_i": cc, "cs_list": cs_list})

            nx.set_node_attributes(sg.graph, values=node_attributes, name="compositional_sequence")
            node_cs_list.append(list(set(nx.get_node_attributes(sg.graph, "compositional_sequence").values())))

        assert sorted(node_cs_list[0]) == ['Al_Al-H(b)2O(a)6', 'H_H-Al(b)1O1', 'H_H-Al(b)2O1', 'H_H-O1', 'O_O-Al(a)2H1', 'O_O-Al(a)2Si(a)1', 'O_O-Si(a)2', 'Si_Si-O(a)4']
        assert sorted(node_cs_list[1]) == ['Al_Al-O(a)5', 'Al_Al-O1O(a)4', 'H_H-', 'H_H-H1', 'H_H-O1O(a)1Si1', 'O_O-Al(a)1H(a)1Si(a)1', 'O_O-Al(a)1O(a)1', 'O_O-Al(a)2', 'O_O-Al(a)2O(a)1', 'O_O-Al1Al(a)1H1Si(a)1', 'O_O-Si(a)2', 'Si_Si-H1O(a)4', 'Si_Si-O(a)4']
