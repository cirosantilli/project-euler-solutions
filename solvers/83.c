#include <assert.h>
#include <inttypes.h>
#include <stdbool.h>
#include <stddef.h>
#include <stdint.h>
#include <stdio.h>
#include <stdlib.h>

#include "../data/project-euler-statements/data/c/common.h"

static size_t neighbors(size_t index, size_t rows, size_t cols, size_t *out) {
    size_t r = index / cols, c = index % cols, count = 0;
    if (r)
        out[count++] = index - cols;
    if (r + 1 < rows)
        out[count++] = index + cols;
    if (c)
        out[count++] = index - 1;
    if (c + 1 < cols)
        out[count++] = index + 1;
    return count;
}

typedef struct {
    int64_t distance;
    size_t vertex;
} HeapEntry;
static void push(HeapEntry *heap, size_t *size, HeapEntry value) {
    size_t i = (*size)++;
    while (i && heap[(i - 1) / 2].distance > value.distance) {
        heap[i] = heap[(i - 1) / 2];
        i = (i - 1) / 2;
    }
    heap[i] = value;
}

static HeapEntry pop(HeapEntry *heap, size_t *size) {
    HeapEntry result = heap[0], last = heap[--*size];
    size_t i = 0;
    while (2 * i + 1 < *size) {
        size_t j = 2 * i + 1;
        if (j + 1 < *size && heap[j + 1].distance < heap[j].distance)
            ++j;
        if (heap[j].distance >= last.distance)
            break;
        heap[i] = heap[j];
        i = j;
    }
    heap[i] = last;
    return result;
}

int64_t solve(const int64_t *matrix, size_t rows, size_t cols) {
    if (!rows || !cols)
        return 0;
    size_t count = rows * cols, size = 0;
    int64_t infinity = 1;
    for (size_t i = 0; i < count; ++i) {
        assert(matrix[i] >= 0);
        infinity += matrix[i];
    }
    int64_t *dist = euler_alloc(count, sizeof(*dist));
    bool *visited = euler_alloc(count, sizeof(*visited));
    /* Each directed edge can cause at most one insertion after its source is settled.
     */
    HeapEntry *heap = euler_alloc(4 * count + 1, sizeof(*heap));
    for (size_t i = 0; i < count; ++i)
        dist[i] = infinity;
    dist[0] = matrix[0];
    push(heap, &size, (HeapEntry){dist[0], 0});
    while (size) {
        HeapEntry current = pop(heap, &size);
        size_t u = current.vertex;
        if (visited[u])
            continue;
        visited[u] = true;
        if (u == count - 1)
            break;
        size_t adjacent[4], length = neighbors(u, rows, cols, adjacent);
        for (size_t i = 0; i < length; ++i) {
            size_t v = adjacent[i];
            int64_t candidate = current.distance + matrix[v];
            if (candidate < dist[v]) {
                dist[v] = candidate;
                push(heap, &size, (HeapEntry){candidate, v});
            }
        }
    }
    int64_t answer = dist[count - 1];
    free(dist);
    free(visited);
    free(heap);
    return answer;
}

int main(void) {
    const int64_t sample[] = {131, 673, 234, 103, 18,  201, 96,  342, 965,
                              150, 630, 803, 746, 422, 111, 537, 699, 497,
                              121, 956, 805, 732, 524, 37,  331};
    assert(solve(sample, 5, 5) == 2297);
    size_t count;
    int64_t *matrix = euler_read_numbers("0083_matrix.txt", &count);
    size_t side = (size_t)euler_isqrt((int64_t)count);
    assert(side * side == count);
    printf("%" PRId64 "\n", solve(matrix, side, side));
    free(matrix);
    return 0;
}
