/*=======================================================
 * The fast-pcst aglorithm, including pairing_heap.h, pcs
 * t_fast.cc, pcst_fast.h and priority_queue.h was implem
 * ented by Ludwig Schmidt. You can check his original co
 * de at the following url:
 *      https://github.com/ludwigschmidt/pcst-fast
=========================================================*/
#ifndef __PCST_FAST_H__
#define __PCST_FAST_H__

#include <string>
#include <utility>
#include <vector>
#include <cstdlib>
#include <map>
#include <set>


namespace cluster_approx {

    template<typename ValueType, typename IndexType>
    class PriorityQueue {
    public:
        PriorityQueue() {}

        bool is_empty() {
            return sorted_set.empty();
        }

        bool get_min(ValueType *value, IndexType *index) {
            if (sorted_set.empty()) {
                return false;
            }
            *value = sorted_set.begin()->first;
            *index = sorted_set.begin()->second;
            return true;
        }

        bool delete_min(ValueType *value, IndexType *index) {
            if (sorted_set.empty()) {
                return false;
            }
            *value = sorted_set.begin()->first;
            *index = sorted_set.begin()->second;
            sorted_set.erase(sorted_set.begin());
            return true;
        }

        void insert(ValueType value, IndexType index) {
            if (index >= static_cast<int>(index_to_iterator.size())) {
                index_to_iterator.resize(index + 1);
            }
            index_to_iterator[index] =
                    sorted_set.insert(std::make_pair(value, index)).first;
        }

        void decrease_key(ValueType new_value, IndexType index) {
            sorted_set.erase(index_to_iterator[index]);
            index_to_iterator[index] =
                    sorted_set.insert(std::make_pair(new_value, index)).first;
        }

        void delete_element(IndexType index) {
            sorted_set.erase(index_to_iterator[index]);
        }

    private:
        std::set<std::pair<ValueType, IndexType> > sorted_set;
        std::vector<typename std::set<std::pair<ValueType, IndexType> >::iterator>
                index_to_iterator;
    };


    template<typename ValueType, typename PayloadType>
    class PairingHeap {
    private:
        struct Node {
            Node *sibling;
            Node *child;
            Node *left_up;
            ValueType value;
            ValueType child_offset;
            PayloadType payload;
        };

    public:
        typedef Node *ItemHandle;

        PairingHeap(std::vector<ItemHandle> *shared_buffer) : root(NULL) {
            buffer = shared_buffer;
        }

        void release_memory() {
            // Delete heap nodes
            buffer->resize(0);
            if (root != NULL) {
                buffer->push_back(root);
            }
            size_t curi = 0;
            while (curi < buffer->size()) {
                Node *cur_node = (*buffer)[curi];
                if (cur_node->child != NULL) {
                    buffer->push_back(cur_node->child);
                }
                if (cur_node->sibling != NULL) {
                    buffer->push_back(cur_node->sibling);
                }
                curi += 1;
            }
            for (size_t ii = 0; ii < buffer->size(); ++ii) {
                delete (*buffer)[ii];
            }
        }

        bool is_empty() {
            return root == NULL;
        }

        bool get_min(ValueType *value, PayloadType *payload) {
            if (root != NULL) {
                //printf("In get_min, current root: %x\n", root);
                //fflush(stdout);
                *value = root->value;
                *payload = root->payload;
                return true;
            } else {
                return false;
            }
        }

        ItemHandle insert(ValueType value, PayloadType payload) {
            Node *new_node = new Node();
            new_node->sibling = NULL;
            new_node->child = NULL;
            new_node->left_up = NULL;
            new_node->value = value;
            new_node->payload = payload;
            new_node->child_offset = 0;
            root = link(root, new_node);
            return new_node;
        }

        void add_to_heap(ValueType value) {
            if (root != NULL) {
                root->value += value;
                root->child_offset += value;
            }
        }

        void decrease_key(ItemHandle node, ValueType from_value, ValueType to_value) {
            ValueType additional_offset = from_value - node->value;
            node->child_offset += additional_offset;
            node->value = to_value;
            if (node->left_up != NULL) {
                if (node->left_up->child == node) {
                    node->left_up->child = node->sibling;
                } else {
                    node->left_up->sibling = node->sibling;
                }
                if (node->sibling != NULL) {
                    node->sibling->left_up = node->left_up;
                }
                node->left_up = NULL;
                node->sibling = NULL;
                root = link(root, node);
            }
        }

        bool delete_min(ValueType *value, PayloadType *payload) {
            if (root == NULL) {
                return false;
            }
            //printf("In delete_min, root is %x (payload %d)\n", root, root->payload);
            //fflush(stdout);
            Node *result = root;
            buffer->resize(0);
            Node *cur_child = root->child;
            Node *next_child = NULL;
            while (cur_child != NULL) {
                buffer->push_back(cur_child);
                //printf("In delete_min, added child %x to buffer\n", cur_child);
                //fflush(stdout);
                next_child = cur_child->sibling;
                cur_child->left_up = NULL;
                cur_child->sibling = NULL;
                cur_child->value += result->child_offset;
                cur_child->child_offset += result->child_offset;
                cur_child = next_child;
            }

            //printf("In delete_min, root hat %lu children\n", buffer->size());
            //fflush(stdout);

            size_t merged_children = 0;
            while (merged_children + 2 <= buffer->size()) {
                (*buffer)[merged_children / 2] = link(
                        (*buffer)[merged_children], (*buffer)[merged_children + 1]);
                merged_children += 2;
            }
            if (merged_children != buffer->size()) {
                (*buffer)[merged_children / 2] = (*buffer)[merged_children];
                buffer->resize(merged_children / 2 + 1);
            } else {
                buffer->resize(merged_children / 2);
            }

            if (buffer->size() > 0) {
                root = (*buffer)[buffer->size() - 1];
                for (int ii = buffer->size() - 2; ii >= 0; --ii) {
                    root = link(root, (*buffer)[ii]);
                }
            } else {
                root = NULL;
            }

            *value = result->value;
            *payload = result->payload;
            //printf("In delete_min, deleting %x\n", result);
            //printf("In delete_min, new root: %x\n", root);
            //fflush(stdout);
            delete result;
            return true;
        }

        static PairingHeap meld(PairingHeap *heap1, PairingHeap *heap2) {
            PairingHeap result(heap1->buffer);
            result.root = link(heap1->root, heap2->root);
            heap1->root = NULL;
            heap2->root = NULL;
            return result;
        }

    private:
        Node *root;
        std::vector<ItemHandle> *buffer;

        static Node *link(Node *node1, Node *node2) {
            if (node1 == NULL) {
                return node2;
            }
            if (node2 == NULL) {
                return node1;
            }
            Node *smaller_node = node2;
            Node *larger_node = node1;
            if (node1->value < node2->value) {
                smaller_node = node1;
                larger_node = node2;
            }
            //printf("Linking %x (smaller node) and %x (larger node)\n",
            //    smaller_node, larger_node);
            //fflush(stdout);
            larger_node->sibling = smaller_node->child;
            if (larger_node->sibling != NULL) {
                larger_node->sibling->left_up = larger_node;
            }
            larger_node->left_up = smaller_node;
            smaller_node->child = larger_node;
            larger_node->value -= smaller_node->child_offset;
            larger_node->child_offset -= smaller_node->child_offset;
            return smaller_node;
        }
    };

    class PCSTFast {
    public:
        enum PruningMethod {
            kNoPruning = 0,
            kSimplePruning,
            kGWPruning,
            kStrongPruning,
            kUnknownPruning,
        };

        struct Statistics {
            long long total_num_edge_events;
            long long num_deleted_edge_events;
            long long num_merged_edge_events;
            long long total_num_merge_events;
            long long num_active_active_merge_events;
            long long num_active_inactive_merge_events;
            long long total_num_edge_growth_events;
            long long num_active_active_edge_growth_events;
            long long num_active_inactive_edge_growth_events;
            long long num_cluster_events;

            Statistics();
        };

        const static int kNoRoot = -1;

        static PruningMethod parse_pruning_method(const std::string &input);


        PCSTFast(const std::vector<std::pair<int, int> > &edges_,
                 const std::vector<double> &prizes_,
                 const std::vector<double> &costs_,
                 int root_,
                 int target_num_active_clusters_,
                 PruningMethod pruning_,
                 int verbosity_level_,
                 void (*output_function_)(const char *));

        ~PCSTFast();

        bool run(std::vector<int> *result_nodes,
                 std::vector<int> *result_edges);

        void get_statistics(Statistics *s);


    private:
        typedef PairingHeap<double, int> PairingHeapType;
        typedef PriorityQueue<double, int> PriorityQueueType;

        struct EdgeInfo {
            int inactive_merge_event;
        };

        struct EdgePart {
            double next_event_val;
            bool deleted;
            PairingHeapType::ItemHandle heap_node;
        };

        struct InactiveMergeEvent {
            int active_cluster_index;
            int inactive_cluster_index;
            int active_cluster_node;
            int inactive_cluster_node;
        };

        struct Cluster {
            PairingHeapType edge_parts;
            bool active;
            double active_start_time;
            double active_end_time;
            int merged_into;
            double prize_sum;
            double subcluster_moat_sum;
            double moat;
            bool contains_root;
            int skip_up;
            double skip_up_sum;
            int merged_along;
            int child_cluster_1;
            int child_cluster_2;
            bool necessary;

            Cluster(std::vector<PairingHeapType::ItemHandle> *heap_buffer)
                    : edge_parts(heap_buffer) {}
        };


        const std::vector<std::pair<int, int> > &edges;
        const std::vector<double> &prizes;
        const std::vector<double> &costs;
        int root;
        int target_num_active_clusters;
        PruningMethod pruning;
        int verbosity_level;

        void (*output_function)(const char *);

        Statistics stats;

        std::vector<PairingHeapType::ItemHandle> pairing_heap_buffer;
        std::vector<EdgePart> edge_parts;
        std::vector<EdgeInfo> edge_info;
        std::vector<Cluster> clusters;
        std::vector<InactiveMergeEvent> inactive_merge_events;
        PriorityQueueType clusters_deactivation;
        PriorityQueueType clusters_next_edge_event;
        double current_time;
        double eps;
        // marks whether a node survives simple pruning
        std::vector<bool> node_good;
        std::vector<bool> node_deleted;
        std::vector<int> phase2_result;

        std::vector<std::pair<int, double> > path_compression_visited;
        std::vector<int> cluster_queue;
        std::vector<std::vector<std::pair<int, double> > > phase3_neighbors;

        // for strong pruning
        std::vector<int> final_component_label;
        std::vector<std::vector<int> > final_components;
        int root_component_index;
        std::vector<std::pair<int, double> > strong_pruning_parent;
        std::vector<double> strong_pruning_payoff;
        std::vector<std::pair<bool, int> > stack;
        std::vector<int> stack2;


        const static int kOutputBufferSize = 10000;
        char output_buffer[kOutputBufferSize];

        void get_next_edge_event(double *next_time,
                                 int *next_cluster_index,
                                 int *next_edge_part_index);

        void remove_next_edge_event(int next_cluster_index);

        void get_next_cluster_event(double *next_time, int *next_cluster_index);

        void remove_next_cluster_event();

        void get_sum_on_edge_part(int edge_part_index,
                                  double *total_sum,
                                  double *finished_moat_sum,
                                  int *cur_cluser_index);

        void mark_nodes_as_good(int start_cluster_index);

        void mark_clusters_as_necessary(int start_cluster_index);

        void mark_nodes_as_deleted(int start_node_index, int parent_node_index);

        void label_final_component(int start_node_index, int new_component_index);

        void strong_pruning_from(int start_node_index, bool mark_as_deleted);

        int find_best_component_root(int component_index);

        void build_phase1_node_set(const std::vector<int> &edge_set,
                                   std::vector<int> *node_set);

        void build_phase3_node_set(std::vector<int> *node_set);

        void build_phase2_node_set(std::vector<int> *node_set);


        int get_other_edge_part_index(int edge_part_index) {
            if (edge_part_index % 2 == 0) {
                return edge_part_index + 1;
            } else {
                return edge_part_index - 1;
            }
        }
    };

}  // namespace cluster_approx


#endif
