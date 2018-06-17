//
// Created by baojian on 6/3/18.
//
#include <stdlib.h>
#include <stdio.h>
#include <string.h>

#define left(i)   ((i) << 1)
#define right(i)  (((i) << 1) + 1)
#define parent(i) ((i) >> 1)


typedef struct node_t {
    double pri;
    int val;
    size_t pos;
} node_t;


/** debug callback function to print a entry */
typedef void (*pqueue_print_entry_f)(FILE *out, void *a);

/** the priority queue handle */
typedef struct pqueue_t {
    size_t size; // number of elements in this queue
    size_t avail; // slots available in this queue
    size_t step; // growth stepping setting
    node_t **d; // the actually queue in binary heap form
} pqueue_t;


pqueue_t *pqueue_init(size_t n) {
    pqueue_t *q;
    if (!(q = (pqueue_t *) malloc(sizeof(pqueue_t)))) { return NULL; }
    // Need to allocate n+1 elements since element 0 isn't used.
    if (!(q->d = (node_t **) malloc((n + 1) * sizeof(void *)))) {
        free(q);
        return NULL;
    }
    q->size = 1;
    q->avail = q->step = (n + 1); //see comment above about n+1
    return q;
}

void pqueue_free(pqueue_t *q) {
    free(q->d);
    free(q);
}

size_t pqueue_size(pqueue_t *q) {
    // queue element 0 exists but doesn't count since it isn't used.
    return (q->size - 1);
}


static void bubble_up(pqueue_t *q, size_t i) {
    size_t parent_node;
    node_t *moving_node = q->d[i];
    double moving_pri = moving_node->pri;
    for (parent_node = parent(i);
         ((i > 1) && (q->d[parent_node]->pri < moving_pri));
         i = parent_node, parent_node = parent(i)) {
        q->d[i] = q->d[parent_node];
        q->d[i]->pos = i;
    }
    q->d[i] = moving_node;
    moving_node->pos = i;
}


static size_t maxchild(pqueue_t *q, size_t i) {
    size_t child_node = left(i);
    if (child_node >= q->size) { return 0; }
    if ((child_node + 1) < q->size &&
        (q->d[child_node]->pri < q->d[child_node + 1]->pri))
        child_node++; /* use right child instead of left */

    return child_node;
}


static void percolate_down(pqueue_t *q, size_t i) {
    size_t child_node;
    node_t *moving_node = q->d[i];
    double moving_pri = moving_node->pri;
    while ((child_node = maxchild(q, i)) &&
           (moving_pri < q->d[child_node]->pri)) {
        q->d[i] = q->d[child_node];
        q->d[i]->pos = i;
        i = child_node;
    }
    q->d[i] = moving_node;
    moving_node->pos = i;
}


int pqueue_insert(pqueue_t *q, void *d) {
    void *tmp;
    size_t i;
    size_t newsize;
    if (!q) return 1;
    /* allocate more memory if necessary */
    if (q->size >= q->avail) {
        newsize = q->size + q->step;
        if (!(tmp = realloc(q->d, sizeof(void *) * newsize)))
            return 1;
        q->d = tmp;
        q->avail = newsize;
    }
    /* insert item */
    i = q->size++;
    q->d[i] = d;
    bubble_up(q, i);
    return 0;
}


void pqueue_change_priority(pqueue_t *q, double new_pri, node_t *d) {
    size_t posn;
    double old_pri = d->pri;
    d->pri = new_pri;
    posn = d->pos;
    if (old_pri < new_pri)
        bubble_up(q, posn);
    else
        percolate_down(q, posn);
}


int pqueue_remove(pqueue_t *q, node_t *d) {
    size_t posn = d->pos;
    q->d[posn] = q->d[--q->size];
    if (d->pri < q->d[posn]->pri)
        bubble_up(q, posn);
    else
        percolate_down(q, posn);
    return 0;
}

void *pqueue_pop(pqueue_t *q) {
    void *head;
    if (!q || q->size == 1)
        return NULL;
    head = q->d[1];
    q->d[1] = q->d[--q->size];
    percolate_down(q, 1);
    return head;
}

void *pqueue_peek(pqueue_t *q) {
    void *d;
    if (!q || q->size == 1)
        return NULL;
    d = q->d[1];
    return d;
}


void pqueue_dump(pqueue_t *q, FILE *out, pqueue_print_entry_f print) {
    int i;
    fprintf(stdout, "posn\tleft\tright\tparent\tmaxchild\t...\n");
    for (i = 1; i < q->size; i++) {
        fprintf(stdout, "%d\t%d\t%d\t%d\t%ul\t",
                i, left(i), right(i), parent(i),
                (unsigned int) maxchild(q, i));
        print(out, q->d[i]);
    }
}

void pqueue_print(pqueue_t *q, FILE *out, pqueue_print_entry_f print) {
    pqueue_t *dup;
    void *e;
    dup = pqueue_init(q->size);
    dup->size = q->size;
    dup->avail = q->avail;
    dup->step = q->step;
    memcpy(dup->d, q->d, (q->size * sizeof(void *)));
    while ((e = pqueue_pop(dup))) {
        print(out, e);
    }
    pqueue_free(dup);
}


static int subtree_is_valid(pqueue_t *q, int pos) {
    if (left(pos) < q->size) {
        /* has a left child */
        if (q->d[pos]->pri < q->d[left(pos)]->pri)
            return 0;
        if (!subtree_is_valid(q, left(pos)))
            return 0;
    }
    if (right(pos) < q->size) {
        /* has a right child */
        if (q->d[pos]->pri < q->d[right(pos)]->pri)
            return 0;
        if (!subtree_is_valid(q, right(pos)))
            return 0;
    }
    return 1;
}


int pqueue_is_valid(pqueue_t *q) {
    return subtree_is_valid(q, 1);
}


int main(void) {
    pqueue_t *pq;
    node_t *ns;
    node_t *n;
    ns = malloc(10 * sizeof(node_t));
    pq = pqueue_init(10);
    if (!(ns && pq)) return 1;
    ns[0].pri = 5;
    ns[0].val = -5;
    pqueue_insert(pq, &ns[0]);
    ns[1].pri = 4;
    ns[1].val = -4;
    pqueue_insert(pq, &ns[1]);
    ns[2].pri = 2;
    ns[2].val = -2;
    pqueue_insert(pq, &ns[2]);
    ns[3].pri = 6;
    ns[3].val = -6;
    pqueue_insert(pq, &ns[3]);
    ns[4].pri = 1;
    ns[4].val = -1;
    pqueue_insert(pq, &ns[4]);
    pqueue_remove(pq, &ns[4]);
    printf("%d\n", pqueue_is_valid(pq));
    n = pqueue_peek(pq);
    printf("peek: %lf [%d]\n", n->pri, n->val);
    pqueue_change_priority(pq, 8, &ns[4]);
    pqueue_change_priority(pq, 7, &ns[2]);
    while ((n = pqueue_pop(pq)))
        printf("pop: %lf [%d]\n", n->pri, n->val);

    pqueue_free(pq);
    free(ns);

    return 0;
}
