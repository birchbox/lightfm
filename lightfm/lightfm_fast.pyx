#!python
#cython: boundscheck=False, wraparound=False, cdivision=True, initializedcheck=False

import numpy as np
from cython.parallel import parallel, prange
cimport cython.operator.dereference as deref
from libc.stdlib cimport free, malloc
cimport openmp


ctypedef float flt


cdef extern from "math.h" nogil:
    double sqrt(double)
    double exp(double)
    double log(double)
    double floor(double)


cdef extern from "stdlib.h" nogil:
    int rand_r(unsigned int*)
    ctypedef void const_void "const void"
    void qsort(void *base, int nmemb, int size,
               int(*compar)(const_void *, const_void *)) nogil
    void* bsearch(const void *key, void *base, int nmemb, int size,
                  int(*compar)(const_void *, const_void *)) nogil


cdef int sample_range(int min_val, int max_val, unsigned int *seed) nogil:

    cdef int val_range

    val_range = max_val - min_val

    return min_val + (rand_r(seed) % val_range)


cdef int int_min(int x, int y) nogil:

    if x < y:
        return x
    else:
        return y


cdef struct Pair:
    int idx
    flt val


cdef int reverse_pair_compare(const_void *a, const_void *b) nogil:

    cdef flt diff

    diff = ((<Pair*>a)).val - ((<Pair*>b)).val
    if diff < 0:
        return 1
    else:
        return -1


cdef int int_compare(const_void *a, const_void *b) nogil:

    if deref(<int*>a) - deref(<int*>b) > 0:
        return 1
    elif deref(<int*>a) - deref(<int*>b) < 0:
        return -1
    else:
        return 0


cdef class CSRMatrix:
    """
    Utility class for accessing elements
    of a CSR matrix.
    """

    cdef int[::1] indices
    cdef int[::1] indptr
    cdef flt[::1] data

    cdef int rows
    cdef int cols
    cdef int nnz

    def __init__(self, csr_matrix):

        self.indices = csr_matrix.indices
        self.indptr = csr_matrix.indptr
        self.data = csr_matrix.data

        self.rows, self.cols = csr_matrix.shape
        self.nnz = len(self.data)

    cdef int get_row_start(self, int row) nogil:
        """
        Return the pointer to the start of the
        data for row.
        """

        return self.indptr[row]

    cdef int get_row_end(self, int row) nogil:
        """
        Return the pointer to the end of the
        data for row.
        """

        return self.indptr[row + 1]


cdef class FastLightFM:
    """
    Class holding all the model state.
    """

    cdef flt[:, ::1] item_features
    cdef flt[:, ::1] item_feature_gradients
    cdef flt[:, ::1] item_feature_momentum

    cdef flt[::1] item_biases
    cdef flt[::1] item_bias_gradients
    cdef flt[::1] item_bias_momentum

    cdef flt[:, ::1] user_features
    cdef flt[:, ::1] user_feature_gradients
    cdef flt[:, ::1] user_feature_momentum

    cdef flt[::1] user_biases
    cdef flt[::1] user_bias_gradients
    cdef flt[::1] user_bias_momentum

    cdef int no_components
    cdef int adadelta
    cdef flt learning_rate
    cdef flt rho
    cdef flt eps

    cdef double item_scale
    cdef double user_scale

    def __init__(self,
                 flt[:, ::1] item_features,
                 flt[:, ::1] item_feature_gradients,
                 flt[:, ::1] item_feature_momentum,
                 flt[::1] item_biases,
                 flt[::1] item_bias_gradients,
                 flt[::1] item_bias_momentum,
                 flt[:, ::1] user_features,
                 flt[:, ::1] user_feature_gradients,
                 flt[:, ::1] user_feature_momentum,
                 flt[::1] user_biases,
                 flt[::1] user_bias_gradients,
                 flt[::1] user_bias_momentum,
                 int no_components,
                 int adadelta,
                 flt learning_rate,
                 flt rho,
                 flt epsilon):

        self.item_features = item_features
        self.item_feature_gradients = item_feature_gradients
        self.item_feature_momentum = item_feature_momentum
        self.item_biases = item_biases
        self.item_bias_gradients = item_bias_gradients
        self.item_bias_momentum = item_bias_momentum
        self.user_features = user_features
        self.user_feature_gradients = user_feature_gradients
        self.user_feature_momentum = user_feature_momentum
        self.user_biases = user_biases
        self.user_bias_gradients = user_bias_gradients
        self.user_bias_momentum = user_bias_momentum

        self.no_components = no_components
        self.learning_rate = learning_rate
        self.rho = rho
        self.eps = epsilon

        self.item_scale = 1.0
        self.user_scale = 1.0

        self.adadelta = adadelta


cdef inline flt sigmoid(flt v) nogil:
    """
    Compute the sigmoid of v.
    """

    return 1.0 / (1.0 + exp(-v))


cdef inline int in_positives(int item_id, int user_id, CSRMatrix interactions) nogil:

    cdef int i, start_idx, stop_idx

    start_idx = interactions.get_row_start(user_id)
    stop_idx = interactions.get_row_end(user_id)

    if bsearch(&item_id,
               &interactions.indices[start_idx],
               stop_idx - start_idx,
               sizeof(int),
               int_compare) == NULL:
        return 0
    else:
        return 1


cdef inline int find_item_index(int user_id, int item_id, CSRMatrix interactions) nogil:
    """Given a user_id and item_id, grab the interactions index.
    Used to find the data value for that user_id + item_id combo.
    Note: I don't really know what I'm doing, but this seems to work
    """

    cdef void *idx
    cdef int *i
    cdef int return_i
    cdef int start_idx, stop_idx

    start_idx = interactions.get_row_start(user_id)
    stop_idx = interactions.get_row_end(user_id)
    idx = bsearch(&item_id,
                &interactions.indices[start_idx],
                stop_idx - start_idx,
                sizeof(int),
                int_compare)

    if not idx:
        return -1
    else:
        i = <int*> idx
        return_i = <int> (i - &interactions.indices[start_idx])
        return return_i + start_idx


cdef inline void compute_representation(CSRMatrix features,
                                        flt[:, ::1] feature_embeddings,
                                        flt[::1] feature_biases,
                                        FastLightFM lightfm,
                                        int row_id,
                                        double scale,
                                        flt *representation) nogil:
    """
    Compute latent representation for row_id.
    The last element of the representation is the bias.
    """

    cdef int i, j, start_index, stop_index, feature
    cdef flt feature_weight

    start_index = features.get_row_start(row_id)
    stop_index = features.get_row_end(row_id)

    for i in range(lightfm.no_components + 1):
        representation[i] = 0.0

    for i in range(start_index, stop_index):

        feature = features.indices[i]
        feature_weight = features.data[i] * scale

        for j in range(lightfm.no_components):

            representation[j] += feature_weight * feature_embeddings[feature, j]

        representation[lightfm.no_components] += feature_weight * feature_biases[feature]


cdef inline flt compute_prediction_from_repr(flt *repr_1,
                                             flt *repr_2,
                                             int no_components) nogil:
    """Add biases and take latent factor dot product for two representations"""

    cdef int i
    cdef flt result

    # Biases
    result = repr_1[no_components] + repr_2[no_components]

    # Latent factor dot product
    for i in range(no_components):
        result += repr_1[i] * repr_2[i]

    return result


cdef inline flt compute_prediction_from_repr_nobias(flt *repr_1,
                                                    flt *repr_2,
                                                    int no_components) nogil:
    """Don't include bias term when calculating prediction."""

    cdef int i
    cdef flt result

    # Latent factor dot product
    for i in range(no_components):
        result += repr_1[i] * repr_2[i]

    return result


cdef double update_biases(CSRMatrix feature_indices,
                          int start,
                          int stop,
                          flt[::1] biases,
                          flt[::1] gradients,
                          flt[::1] momentum,
                          double gradient,
                          int adadelta,
                          double learning_rate,
                          double alpha,
                          flt rho,
                          flt eps) nogil:
    """
    Perform a SGD update of the bias terms.
    """

    cdef int i, feature
    cdef double feature_weight, local_learning_rate, sum_learning_rate, update

    sum_learning_rate = 0.0

    if adadelta:
        for i in range(start, stop):

            feature = feature_indices.indices[i]
            feature_weight = feature_indices.data[i]

            gradients[feature] = rho * gradients[feature] + (1 - rho) * (feature_weight * gradient) ** 2
            local_learning_rate = sqrt(momentum[feature] + eps) / sqrt(gradients[feature] + eps)
            update = local_learning_rate * gradient * feature_weight
            momentum[feature] = rho * momentum[feature] + (1 - rho) * update ** 2
            biases[feature] -= update

            # Lazy regularization: scale up by the regularization
            # parameter.
            biases[feature] *= (1.0 + alpha * local_learning_rate)

            sum_learning_rate += local_learning_rate
    else:
        for i in range(start, stop):

            feature = feature_indices.indices[i]
            feature_weight = feature_indices.data[i]

            local_learning_rate = learning_rate / sqrt(gradients[feature])
            biases[feature] -= local_learning_rate * feature_weight * gradient
            gradients[feature] += gradient ** 2

            # Lazy regularization: scale up by the regularization
            # parameter.
            biases[feature] *= (1.0 + alpha * local_learning_rate)

            sum_learning_rate += local_learning_rate

    return sum_learning_rate


cdef double update_biases_simplereg(CSRMatrix feature_indices,
                          int start,
                          int stop,
                          flt[::1] biases,
                          flt[::1] gradients,
                          double loss,
                          double learning_rate,
                          double alpha) nogil:
    """
    Perform a SGD update of the bias terms.
    """

    cdef int i, feature
    cdef double feature_weight, local_learning_rate, sum_learning_rate, gradient

    sum_learning_rate = 0.0

    for i in range(start, stop):

        feature = feature_indices.indices[i]
        feature_weight = feature_indices.data[i]

        local_learning_rate = learning_rate / sqrt(gradients[feature])
        gradient = loss - feature_weight * alpha
        biases[feature] -= local_learning_rate * gradient
        gradients[feature] += gradient ** 2

        sum_learning_rate += local_learning_rate

    return sum_learning_rate


cdef inline double update_features(CSRMatrix feature_indices,
                                   flt[:, ::1] features,
                                   flt[:, ::1] gradients,
                                   flt[:, ::1] momentum,
                                   int component,
                                   int start,
                                   int stop,
                                   double gradient,
                                   int adadelta,
                                   double learning_rate,
                                   double alpha,
                                   flt rho,
                                   flt eps) nogil:
    """
    Update feature vectors.
    """

    cdef int i, feature,
    cdef double feature_weight, local_learning_rate, sum_learning_rate, update

    sum_learning_rate = 0.0

    if adadelta:
        for i in range(start, stop):

            feature = feature_indices.indices[i]
            feature_weight = feature_indices.data[i]

            gradients[feature, component] = (rho * gradients[feature, component]
                                             + (1 - rho) * (feature_weight * gradient) ** 2)
            local_learning_rate = (sqrt(momentum[feature, component] + eps)
                                   / sqrt(gradients[feature, component] + eps))
            update = local_learning_rate * gradient * feature_weight
            momentum[feature, component] = rho * momentum[feature, component] + (1 - rho) * update ** 2
            features[feature, component] -= update

            # Lazy regularization: scale up by the regularization
            # parameter.
            features[feature, component] *= (1.0 + alpha * local_learning_rate)

            sum_learning_rate += local_learning_rate
    else:
        for i in range(start, stop):

            feature = feature_indices.indices[i]
            feature_weight = feature_indices.data[i]

            local_learning_rate = learning_rate / sqrt(gradients[feature, component])
            features[feature, component] -= local_learning_rate * feature_weight * gradient
            gradients[feature, component] += gradient ** 2

            sum_learning_rate += local_learning_rate

    return sum_learning_rate


cdef inline double update_item_features(CSRMatrix feature_indices,
                                   flt[:, ::1] features,
                                   flt[:, ::1] gradients,
                                   int no_components,
                                   int start,
                                   int stop,
                                   double loss,
                                   flt* user_repr,
                                   double learning_rate,
                                   double alpha) nogil:
    """
    Update item feature vectors with max norm constraint.
    """

    cdef int i, j, feature
    cdef double feature_weight, local_learning_rate, sum_learning_rate, norm

    sum_learning_rate = 0.0


    for i in range(start, stop):
        feature = feature_indices.indices[i]
        feature_weight = feature_indices.data[i]
        norm = 0.0
        for j in range(no_components):
            local_learning_rate = learning_rate / sqrt(gradients[feature, j])
            gradient = loss * user_repr[j]
            # I think we ignore feature_weight here. Should already be included in gradient?
            # features[feature, j] -= local_learning_rate * feature_weight * gradient
            features[feature, j] -= local_learning_rate * gradient
            gradients[feature, j] += gradient ** 2
            norm += features[feature, j]**2
            sum_learning_rate += local_learning_rate
        if norm > alpha:
            # If norm lies outside L2 ball of radius alpha,
            # project feature back onto shell of the ball.
            for j in range(no_components):
                features[feature, j] = features[feature, j] * alpha / norm


    return sum_learning_rate

cdef inline double update_user_features(CSRMatrix feature_indices,
                                   flt[:, ::1] features,
                                   flt[:, ::1] gradients,
                                   int no_components,
                                   int start,
                                   int stop,
                                   double loss,
                                   flt* pos_it_repr,
                                   flt* neg_it_repr,
                                   double learning_rate,
                                   double alpha) nogil:
    """
    Update user feature vectors with max norm constraint.
    """

    cdef int i, j, feature
    cdef double feature_weight, local_learning_rate, sum_learning_rate, norm

    sum_learning_rate = 0.0


    for i in range(start, stop):
        feature = feature_indices.indices[i]
        feature_weight = feature_indices.data[i]
        norm = 0.0
        for j in range(no_components):
            local_learning_rate = learning_rate / sqrt(gradients[feature, j])
            gradient = loss * (neg_it_repr[j] - pos_it_repr[j])
            features[feature, j] -= local_learning_rate * feature_weight * gradient
            gradients[feature, j] += gradient ** 2
            norm += features[feature, j]**2
            sum_learning_rate += local_learning_rate
        if norm > alpha:
            # If norm lies outside L2 ball of radius alpha,
            # project feature back onto shell of the ball.
            for j in range(no_components):
                features[feature, j] = features[feature, j] * alpha / norm


    return sum_learning_rate

cdef inline void update(double loss,
                        CSRMatrix item_features,
                        CSRMatrix user_features,
                        int user_id,
                        int item_id,
                        flt *user_repr,
                        flt *it_repr,
                        FastLightFM lightfm,
                        double item_alpha,
                        double user_alpha) nogil:
    """
    Apply the gradient step.
    """

    cdef int i, j, item_start_index, item_stop_index, user_start_index, user_stop_index
    cdef double avg_learning_rate
    cdef flt item_component, user_component

    avg_learning_rate = 0.0

    # Get the iteration ranges for features
    # for this training example.
    item_start_index = item_features.get_row_start(item_id)
    item_stop_index = item_features.get_row_end(item_id)

    user_start_index = user_features.get_row_start(user_id)
    user_stop_index = user_features.get_row_end(user_id)

    avg_learning_rate += update_biases(item_features, item_start_index, item_stop_index,
                                       lightfm.item_biases, lightfm.item_bias_gradients,
                                       lightfm.item_bias_momentum,
                                       loss,
                                       lightfm.adadelta,
                                       lightfm.learning_rate,
                                       item_alpha,
                                       lightfm.rho,
                                       lightfm.eps)
    avg_learning_rate += update_biases(user_features, user_start_index, user_stop_index,
                                       lightfm.user_biases, lightfm.user_bias_gradients,
                                       lightfm.user_bias_momentum,
                                       loss,
                                       lightfm.adadelta,
                                       lightfm.learning_rate,
                                       user_alpha,
                                       lightfm.rho,
                                       lightfm.eps)

    # Update latent representations.
    for i in range(lightfm.no_components):

        user_component = user_repr[i]
        item_component = it_repr[i]

        avg_learning_rate += update_features(item_features, lightfm.item_features,
                                             lightfm.item_feature_gradients,
                                             lightfm.item_feature_momentum,
                                             i, item_start_index, item_stop_index,
                                             loss * user_component,
                                             lightfm.adadelta,
                                             lightfm.learning_rate,
                                             item_alpha,
                                             lightfm.rho,
                                             lightfm.eps)
        avg_learning_rate += update_features(user_features, lightfm.user_features,
                                             lightfm.user_feature_gradients,
                                             lightfm.user_feature_momentum,
                                             i, user_start_index, user_stop_index,
                                             loss * item_component,
                                             lightfm.adadelta,
                                             lightfm.learning_rate,
                                             user_alpha,
                                             lightfm.rho,
                                             lightfm.eps)

    avg_learning_rate /= ((lightfm.no_components + 1) * (user_stop_index - user_start_index)
                          + (lightfm.no_components + 1) * (item_stop_index - item_start_index))

    # Update the scaling factors for lazy regularization, using the average learning rate
    # of features updated for this example.
    lightfm.item_scale *= (1 - item_alpha * avg_learning_rate)
    lightfm.user_scale *= (1 - user_alpha * avg_learning_rate)


cdef void warp_update(double loss,
                      CSRMatrix item_features,
                      CSRMatrix user_features,
                      int user_id,
                      int positive_item_id,
                      int negative_item_id,
                      flt *user_repr,
                      flt *pos_it_repr,
                      flt *neg_it_repr,
                      FastLightFM lightfm,
                      double item_alpha,
                      double user_alpha) nogil:
    """
    Apply the gradient step.
    """

    cdef int i, j, positive_item_start_index, positive_item_stop_index
    cdef int  user_start_index, user_stop_index, negative_item_start_index, negative_item_stop_index
    cdef double avg_learning_rate
    cdef flt positive_item_component, negative_item_component, user_component

    avg_learning_rate = 0.0

    # Get the iteration ranges for features
    # for this training example.
    positive_item_start_index = item_features.get_row_start(positive_item_id)
    positive_item_stop_index = item_features.get_row_end(positive_item_id)

    negative_item_start_index = item_features.get_row_start(negative_item_id)
    negative_item_stop_index = item_features.get_row_end(negative_item_id)

    user_start_index = user_features.get_row_start(user_id)
    user_stop_index = user_features.get_row_end(user_id)


    avg_learning_rate += update_biases_simplereg(item_features,
                                       positive_item_start_index,
                                       positive_item_stop_index,
                                       lightfm.item_biases, lightfm.item_bias_gradients,
                                       -loss,
                                       lightfm.learning_rate,
                                       item_alpha)

    avg_learning_rate += update_biases_simplereg(item_features, negative_item_start_index,
                                       negative_item_stop_index,
                                       lightfm.item_biases, lightfm.item_bias_gradients,
                                       loss,
                                       lightfm.learning_rate,
                                       item_alpha)
    avg_learning_rate += update_biases_simplereg(user_features, user_start_index, user_stop_index,
                                       lightfm.user_biases, lightfm.user_bias_gradients,
                                       loss,
                                       lightfm.learning_rate,
                                       user_alpha)

    # Update latent representations.

    avg_learning_rate += update_item_features(item_features, lightfm.item_features,
                                         lightfm.item_feature_gradients,
                                         lightfm.no_components,
                                         positive_item_start_index, positive_item_stop_index,
                                         -loss,
                                         user_repr,
                                         lightfm.learning_rate,
                                         item_alpha)
    avg_learning_rate += update_item_features(item_features, lightfm.item_features,
                                         lightfm.item_feature_gradients,
                                         lightfm.no_components,
                                         negative_item_start_index, negative_item_stop_index,
                                         loss,
                                         user_repr,
                                         lightfm.learning_rate,
                                         item_alpha)
    avg_learning_rate += update_user_features(user_features, lightfm.user_features,
                                         lightfm.user_feature_gradients,
                                         lightfm.no_components,
                                         user_start_index, user_stop_index,
                                         loss,
                                         pos_it_repr,
                                         neg_it_repr,
                                         lightfm.learning_rate,
                                         user_alpha)


    avg_learning_rate /= ((lightfm.no_components + 1) * (user_stop_index - user_start_index)
                          + (lightfm.no_components + 1) *
                          (positive_item_stop_index - positive_item_start_index)
                          + (lightfm.no_components + 1)
                          * (negative_item_stop_index - negative_item_start_index))

cdef void warp_kos_update(double loss,
                      CSRMatrix item_features,
                      CSRMatrix user_features,
                      int user_id,
                      int positive_item_id,
                      int negative_item_id,
                      flt *user_repr,
                      flt *pos_it_repr,
                      flt *neg_it_repr,
                      FastLightFM lightfm,
                      double item_alpha,
                      double user_alpha) nogil:
    """
    Apply the gradient step.
    """

    cdef int i, j, positive_item_start_index, positive_item_stop_index
    cdef int  user_start_index, user_stop_index, negative_item_start_index, negative_item_stop_index
    cdef double avg_learning_rate
    cdef flt positive_item_component, negative_item_component, user_component

    avg_learning_rate = 0.0

    # Get the iteration ranges for features
    # for this training example.
    positive_item_start_index = item_features.get_row_start(positive_item_id)
    positive_item_stop_index = item_features.get_row_end(positive_item_id)

    negative_item_start_index = item_features.get_row_start(negative_item_id)
    negative_item_stop_index = item_features.get_row_end(negative_item_id)

    user_start_index = user_features.get_row_start(user_id)
    user_stop_index = user_features.get_row_end(user_id)

    avg_learning_rate += update_biases(item_features, positive_item_start_index,
                                       positive_item_stop_index,
                                       lightfm.item_biases, lightfm.item_bias_gradients,
                                       lightfm.item_bias_momentum,
                                       -loss,
                                       lightfm.adadelta,
                                       lightfm.learning_rate,
                                       item_alpha,
                                       lightfm.rho,
                                       lightfm.eps)
    avg_learning_rate += update_biases(item_features, negative_item_start_index,
                                       negative_item_stop_index,
                                       lightfm.item_biases, lightfm.item_bias_gradients,
                                       lightfm.item_bias_momentum,
                                       loss,
                                       lightfm.adadelta,
                                       lightfm.learning_rate,
                                       item_alpha,
                                       lightfm.rho,
                                       lightfm.eps)
    avg_learning_rate += update_biases(user_features, user_start_index, user_stop_index,
                                       lightfm.user_biases, lightfm.user_bias_gradients,
                                       lightfm.user_bias_momentum,
                                       loss,
                                       lightfm.adadelta,
                                       lightfm.learning_rate,
                                       user_alpha,
                                       lightfm.rho,
                                       lightfm.eps)

    # Update latent representations.

    for i in range(lightfm.no_components):

        user_component = user_repr[i]
        positive_item_component = pos_it_repr[i]
        negative_item_component = neg_it_repr[i]

        avg_learning_rate += update_features(item_features, lightfm.item_features,
                                             lightfm.item_feature_gradients,
                                             lightfm.item_feature_momentum,
                                             i, positive_item_start_index, positive_item_stop_index,
                                             -loss * user_component,
                                             lightfm.adadelta,
                                             lightfm.learning_rate,
                                             item_alpha,
                                             lightfm.rho,
                                             lightfm.eps)
        avg_learning_rate += update_features(item_features, lightfm.item_features,
                                             lightfm.item_feature_gradients,
                                             lightfm.item_feature_momentum,
                                             i, negative_item_start_index, negative_item_stop_index,
                                             loss * user_component,
                                             lightfm.adadelta,
                                             lightfm.learning_rate,
                                             item_alpha,
                                             lightfm.rho,
                                             lightfm.eps)
        avg_learning_rate += update_features(user_features, lightfm.user_features,
                                             lightfm.user_feature_gradients,
                                             lightfm.user_feature_momentum,
                                             i, user_start_index, user_stop_index,
                                             loss * (negative_item_component -
                                                     positive_item_component),
                                             lightfm.adadelta,
                                             lightfm.learning_rate,
                                             user_alpha,
                                             lightfm.rho,
                                             lightfm.eps)


    avg_learning_rate /= ((lightfm.no_components + 1) * (user_stop_index - user_start_index)
                          + (lightfm.no_components + 1) *
                          (positive_item_stop_index - positive_item_start_index)
                          + (lightfm.no_components + 1)
                          * (negative_item_stop_index - negative_item_start_index))

    # Update the scaling factors for lazy regularization, using the average learning rate
    # of features updated for this example.
    lightfm.item_scale *= (1 - item_alpha * avg_learning_rate)
    lightfm.user_scale *= (1 - user_alpha * avg_learning_rate)


cdef void regularize(FastLightFM lightfm,
                     double item_alpha,
                     double user_alpha) nogil:
    """
    Apply accumulated L2 regularization to all features.
    """

    cdef int i, j
    cdef int no_features = lightfm.item_features.shape[0]
    cdef int no_users = lightfm.user_features.shape[0]

    for i in range(no_features):
        for j in range(lightfm.no_components):
            lightfm.item_features[i, j] *= lightfm.item_scale

        lightfm.item_biases[i] *= lightfm.item_scale

    for i in range(no_users):
        for j in range(lightfm.no_components):
            lightfm.user_features[i, j] *= lightfm.user_scale
        lightfm.user_biases[i] *= lightfm.user_scale

    lightfm.item_scale = 1.0
    lightfm.user_scale = 1.0


def fit_logistic(CSRMatrix item_features,
                 CSRMatrix user_features,
                 int[::1] user_ids,
                 int[::1] item_ids,
                 flt[::1] Y,
                 int[::1] shuffle_indices,
                 FastLightFM lightfm,
                 double learning_rate,
                 double item_alpha,
                 double user_alpha,
                 int num_threads):
    """
    Fit the LightFM model.
    """

    cdef int i, no_examples, user_id, item_id, row
    cdef double prediction, loss
    cdef int y
    cdef flt y_row
    cdef flt *user_repr
    cdef flt *it_repr

    no_examples = Y.shape[0]

    with nogil, parallel(num_threads=num_threads):

        user_repr = <flt *>malloc(sizeof(flt) * (lightfm.no_components + 1))
        it_repr = <flt *>malloc(sizeof(flt) * (lightfm.no_components + 1))

        for i in prange(no_examples):

            row = shuffle_indices[i]

            user_id = user_ids[row]
            item_id = item_ids[row]

            compute_representation(user_features,
                                   lightfm.user_features,
                                   lightfm.user_biases,
                                   lightfm,
                                   user_id,
                                   lightfm.user_scale,
                                   user_repr)
            compute_representation(item_features,
                                   lightfm.item_features,
                                   lightfm.item_biases,
                                   lightfm,
                                   item_id,
                                   lightfm.item_scale,
                                   it_repr)

            prediction = sigmoid(compute_prediction_from_repr(user_repr,
                                                              it_repr,
                                                              lightfm.no_components))

            # Any value less or equal to zero
            # is a negative interaction.
            y_row = Y[row]
            if y_row <= 0:
                y = 0
            else:
                y = 1

            loss = (prediction - y)
            update(loss,
                   item_features,
                   user_features,
                   user_id,
                   item_id,
                   user_repr,
                   it_repr,
                   lightfm,
                   item_alpha,
                   user_alpha)

        free(user_repr)
        free(it_repr)

    regularize(lightfm,
               item_alpha,
               user_alpha)


def fit_warp(CSRMatrix item_features,
             CSRMatrix user_features,
             CSRMatrix interactions,
             int[::1] user_ids,
             int[::1] item_ids,
             flt[::1] Y,
             int[::1] shuffle_indices,
             FastLightFM lightfm,
             double learning_rate,
             double item_alpha,
             double user_alpha,
             int num_threads):
    """
    Fit the model using the WARP loss.
    """

    cdef int i, no_examples, user_id, positive_item_id, gamma, max_sampled
    cdef int negative_item_id, sampled, row, negative_item_index
    cdef double positive_prediction, negative_prediction
    cdef double loss, MAX_LOSS
    cdef flt *user_repr
    cdef flt *pos_it_repr
    cdef flt *neg_it_repr
    cdef unsigned int[::1] random_states

    random_states = np.random.randint(0,
                                      np.iinfo(np.int32).max,
                                      size=num_threads).astype(np.uint32)

    no_examples = Y.shape[0]
    gamma = 10
    MAX_LOSS = 10.0

    max_sampled = item_features.rows / gamma

    with nogil, parallel(num_threads=num_threads):

        user_repr = <flt *>malloc(sizeof(flt) * (lightfm.no_components + 1))
        pos_it_repr = <flt *>malloc(sizeof(flt) * (lightfm.no_components + 1))
        neg_it_repr = <flt *>malloc(sizeof(flt) * (lightfm.no_components + 1))

        for i in prange(no_examples):
            row = shuffle_indices[i]

            user_id = user_ids[row]
            positive_item_id = item_ids[row]

            if not Y[row] > 0:
                continue

            compute_representation(user_features,
                                   lightfm.user_features,
                                   lightfm.user_biases,
                                   lightfm,
                                   user_id,
                                   lightfm.user_scale,
                                   user_repr)
            compute_representation(item_features,
                                   lightfm.item_features,
                                   lightfm.item_biases,
                                   lightfm,
                                   positive_item_id,
                                   lightfm.item_scale,
                                   pos_it_repr)

            positive_prediction = compute_prediction_from_repr(user_repr,
                                                               pos_it_repr,
                                                               lightfm.no_components)

            sampled = 0

            while sampled < max_sampled:

                sampled = sampled + 1
                negative_item_id = (rand_r(&random_states[openmp.omp_get_thread_num()])
                                    % item_features.rows)

                # Sample again if interaction counts for negative item are greater than or
                # equal to positive item
                negative_item_index = find_item_index(user_id, negative_item_id, interactions)
                if negative_item_index != -1:
                    if Y[negative_item_index] >= Y[row]:
                        continue

                compute_representation(item_features,
                                       lightfm.item_features,
                                       lightfm.item_biases,
                                       lightfm,
                                       negative_item_id,
                                       lightfm.item_scale,
                                       neg_it_repr)

                negative_prediction = compute_prediction_from_repr(user_repr,
                                                                   neg_it_repr,
                                                                   lightfm.no_components)

                if negative_prediction > positive_prediction - 1:

                    loss = log(floor((item_features.rows - 1) / sampled))

                    # Clip gradients for numerical stability.
                    if loss > MAX_LOSS:
                        loss = MAX_LOSS

                    warp_update(loss,
                                item_features,
                                user_features,
                                user_id,
                                positive_item_id,
                                negative_item_id,
                                user_repr,
                                pos_it_repr,
                                neg_it_repr,
                                lightfm,
                                item_alpha,
                                user_alpha)
                    break

        free(user_repr)
        free(pos_it_repr)
        free(neg_it_repr)



def fit_warp_kos(CSRMatrix item_features,
                 CSRMatrix user_features,
                 CSRMatrix data,
                 int[::1] user_ids,
                 int[::1] item_ids,
                 flt[::1] Y,
                 int[::1] shuffle_indices,
                 FastLightFM lightfm,
                 double learning_rate,
                 double item_alpha,
                 double user_alpha,
                 int k,
                 int n,
                 int num_threads):
    """
    Fit the model using the WARP loss.
    """

    cdef int i, j, no_examples, user_id, positive_item_id, gamma, max_sampled
    cdef int negative_item_id, sampled, row, sampled_positive_item_id, negative_item_index
    cdef int user_pids_start, user_pids_stop, no_positives, POS_SAMPLES
    cdef double positive_prediction, negative_prediction
    cdef double loss, MAX_LOSS, sampled_positive_prediction
    cdef flt *user_repr
    cdef flt *pos_it_repr
    cdef flt *neg_it_repr
    cdef Pair *pos_pairs
    cdef unsigned int[::1] random_states

    random_states = np.random.randint(0,
                                      np.iinfo(np.int32).max,
                                      size=num_threads).astype(np.uint32)

    no_examples = user_ids.shape[0]
    gamma = 10
    MAX_LOSS = 10.0

    max_sampled = item_features.rows / gamma

    with nogil, parallel(num_threads=num_threads):

        user_repr = <flt *>malloc(sizeof(flt) * (lightfm.no_components + 1))
        pos_it_repr = <flt *>malloc(sizeof(flt) * (lightfm.no_components + 1))
        neg_it_repr = <flt *>malloc(sizeof(flt) * (lightfm.no_components + 1))
        pos_pairs = <Pair*>malloc(sizeof(Pair) * n)

        for i in prange(no_examples):
            row = shuffle_indices[i]
            user_id = user_ids[row]

            compute_representation(user_features,
                                   lightfm.user_features,
                                   lightfm.user_biases,
                                   lightfm,
                                   user_id,
                                   lightfm.user_scale,
                                   user_repr)

            user_pids_start = data.get_row_start(user_id)
            user_pids_stop = data.get_row_end(user_id)

            if user_pids_stop == user_pids_start:
                continue

            # Sample k-th positive item
            no_positives = int_min(n, user_pids_stop - user_pids_start)
            for j in range(no_positives):
                sampled_positive_item_id = data.indices[sample_range(user_pids_start,
                                                                     user_pids_stop,
                                                                     &random_states[openmp.omp_get_thread_num()])]

                compute_representation(item_features,
                                       lightfm.item_features,
                                       lightfm.item_biases,
                                       lightfm,
                                       sampled_positive_item_id,
                                       lightfm.item_scale,
                                       pos_it_repr)

                sampled_positive_prediction = compute_prediction_from_repr(user_repr,
                                                                           pos_it_repr,
                                                                           lightfm.no_components)

                pos_pairs[j].idx = sampled_positive_item_id
                pos_pairs[j].val = sampled_positive_prediction

            qsort(pos_pairs,
                  no_positives,
                  sizeof(Pair),
                  reverse_pair_compare)

            positive_item_id = pos_pairs[int_min(k, no_positives) - 1].idx
            positive_prediction = pos_pairs[int_min(k, no_positives) - 1].val

            compute_representation(item_features,
                                   lightfm.item_features,
                                   lightfm.item_biases,
                                   lightfm,
                                   positive_item_id,
                                   lightfm.item_scale,
                                   pos_it_repr)

            # Move on to the WARP step
            sampled = 0

            while sampled < max_sampled:

                sampled = sampled + 1
                negative_item_id = (rand_r(&random_states[openmp.omp_get_thread_num()])
                                    % item_features.rows)

                # Sample again if interaction counts for negative item are greater than or
                # equal to positive item
                negative_item_index = find_item_index(user_id, negative_item_id, data)
                if negative_item_index != -1:
                    if Y[negative_item_index] >= Y[row]:
                        continue

                compute_representation(item_features,
                                       lightfm.item_features,
                                       lightfm.item_biases,
                                       lightfm,
                                       negative_item_id,
                                       lightfm.item_scale,
                                       neg_it_repr)

                negative_prediction = compute_prediction_from_repr(user_repr,
                                                                   neg_it_repr,
                                                                   lightfm.no_components)

                if negative_prediction > positive_prediction - 1:

                    loss = log(floor((item_features.rows - 1) / sampled))

                    # Clip gradients for numerical stability.
                    if loss > MAX_LOSS:
                        loss = MAX_LOSS

                    warp_kos_update(loss,
                                item_features,
                                user_features,
                                user_id,
                                positive_item_id,
                                negative_item_id,
                                user_repr,
                                pos_it_repr,
                                neg_it_repr,
                                lightfm,
                                item_alpha,
                                user_alpha)
                    break

        free(user_repr)
        free(pos_it_repr)
        free(neg_it_repr)
        free(pos_pairs)

    regularize(lightfm,
               item_alpha,
               user_alpha)


def fit_bpr(CSRMatrix item_features,
            CSRMatrix user_features,
            CSRMatrix interactions,
            int[::1] user_ids,
            int[::1] item_ids,
            flt[::1] Y,
            int[::1] shuffle_indices,
            FastLightFM lightfm,
            double learning_rate,
            double item_alpha,
            double user_alpha,
            int num_threads):
    """
    Fit the model using the BPR loss.
    """

    cdef int i, no_examples, user_id, positive_item_id
    cdef int negative_item_id, sampled, row, negative_item_index
    cdef double positive_prediction, negative_prediction
    cdef unsigned int[::1] random_states
    cdef flt *user_repr
    cdef flt *pos_it_repr
    cdef flt *neg_it_repr

    random_states = np.random.randint(0,
                                      np.iinfo(np.int32).max,
                                      size=num_threads).astype(np.uint32)

    no_examples = Y.shape[0]

    with nogil, parallel(num_threads=num_threads):

        user_repr = <flt *>malloc(sizeof(flt) * (lightfm.no_components + 1))
        pos_it_repr = <flt *>malloc(sizeof(flt) * (lightfm.no_components + 1))
        neg_it_repr = <flt *>malloc(sizeof(flt) * (lightfm.no_components + 1))

        for i in prange(no_examples):
            row = shuffle_indices[i]

            if not Y[row] > 0:
                continue

            user_id = user_ids[row]
            positive_item_id = item_ids[row]

            while True:
                negative_item_id = (rand_r(&random_states[openmp.omp_get_thread_num()])
                                    % item_features.rows)

                negative_item_index = find_item_index(user_id, negative_item_id, interactions)
                if negative_item_index == -1 or Y[negative_item_index] < Y[row]:
                    # Negative item is truly negative compared to positive item.
                    break


            compute_representation(user_features,
                                   lightfm.user_features,
                                   lightfm.user_biases,
                                   lightfm,
                                   user_id,
                                   lightfm.user_scale,
                                   user_repr)
            compute_representation(item_features,
                                   lightfm.item_features,
                                   lightfm.item_biases,
                                   lightfm,
                                   positive_item_id,
                                   lightfm.item_scale,
                                   pos_it_repr)
            compute_representation(item_features,
                                   lightfm.item_features,
                                   lightfm.item_biases,
                                   lightfm,
                                   negative_item_id,
                                   lightfm.item_scale,
                                   neg_it_repr)

            positive_prediction = compute_prediction_from_repr(user_repr,
                                                               pos_it_repr,
                                                               lightfm.no_components)
            negative_prediction = compute_prediction_from_repr(user_repr,
                                                               neg_it_repr,
                                                               lightfm.no_components)

            warp_kos_update(sigmoid(positive_prediction - negative_prediction),
                        item_features,
                        user_features,
                        user_id,
                        positive_item_id,
                        negative_item_id,
                        user_repr,
                        pos_it_repr,
                        neg_it_repr,
                        lightfm,
                        item_alpha,
                        user_alpha)

        free(user_repr)
        free(pos_it_repr)
        free(neg_it_repr)

    regularize(lightfm,
               item_alpha,
               user_alpha)


def predict_lightfm(CSRMatrix item_features,
                    CSRMatrix user_features,
                    int[::1] user_ids,
                    int[::1] item_ids,
                    double[::1] predictions,
                    FastLightFM lightfm,
                    int num_threads,
                    int bias):
    """
    Generate predictions.
    """

    cdef int i, no_examples
    cdef flt *user_repr
    cdef flt *it_repr

    no_examples = predictions.shape[0]

    with nogil, parallel(num_threads=num_threads):

        user_repr = <flt *>malloc(sizeof(flt) * (lightfm.no_components + 1))
        it_repr = <flt *>malloc(sizeof(flt) * (lightfm.no_components + 1))

        for i in prange(no_examples):

            compute_representation(user_features,
                                   lightfm.user_features,
                                   lightfm.user_biases,
                                   lightfm,
                                   user_ids[i],
                                   lightfm.user_scale,
                                   user_repr)
            compute_representation(item_features,
                                   lightfm.item_features,
                                   lightfm.item_biases,
                                   lightfm,
                                   item_ids[i],
                                   lightfm.item_scale,
                                   it_repr)

            if bias == 1:
                predictions[i] = compute_prediction_from_repr(user_repr,
                                                              it_repr,
                                                              lightfm.no_components)
            else:
                predictions[i] = compute_prediction_from_repr_nobias(user_repr,
                                                                     it_repr,
                                                                     lightfm.no_components)


def item_to_item_lightfm(CSRMatrix item_features,
                         int[::1] item_ids,
                         double[:,:] predictions,
                         FastLightFM lightfm,
                         int bias):
    """
    Generate item-to-item recommendaitons.
    """

    cdef int i, no_examples
    cdef flt *it_i_repr
    cdef flt *it_j_repr

    no_examples = predictions.shape[0]

    # with nogil, parallel(num_threads=num_threads):

    it_i_repr = <flt *>malloc(sizeof(flt) * (lightfm.no_components + 1))
    it_j_repr = <flt *>malloc(sizeof(flt) * (lightfm.no_components + 1))

    for i in range(no_examples):
        for j in range(i, no_examples):
            compute_representation(item_features,
                                   lightfm.item_features,
                                   lightfm.item_biases,
                                   lightfm,
                                   item_ids[i],
                                   lightfm.item_scale,
                                   it_i_repr)
            compute_representation(item_features,
                                   lightfm.item_features,
                                   lightfm.item_biases,
                                   lightfm,
                                   item_ids[j],
                                   lightfm.item_scale,
                                   it_j_repr)

            if bias == 1:
                predictions[i, j] = compute_prediction_from_repr(it_i_repr,
                                                                 it_j_repr,
                                                                 lightfm.no_components)
            else:
                predictions[i, j] = compute_prediction_from_repr_nobias(it_i_repr,
                                                                        it_j_repr,
                                                                        lightfm.no_components)


def user_norms_lightfm(CSRMatrix user_features,
                         int[::1] user_ids,
                         double[::1] user_norms,
                         FastLightFM lightfm,
                         int num_threads):
    """
    Generate full user and item vectors
    """

    cdef int i, no_examples
    cdef flt *user_repr
    cdef flt *it_repr

    no_users = user_norms.shape[0]

    with nogil, parallel(num_threads=num_threads):

        user_repr = <flt *>malloc(sizeof(flt) * (lightfm.no_components + 1))

        for i in prange(no_users):
            compute_representation(user_features,
                                   lightfm.user_features,
                                   lightfm.user_biases,
                                   lightfm,
                                   user_ids[i],
                                   lightfm.user_scale,
                                   user_repr)

            user_norms[i] = compute_prediction_from_repr_nobias(user_repr,
                                                                user_repr,
                                                                lightfm.no_components)


def item_norms_lightfm(CSRMatrix item_features,
                         int[::1] item_ids,
                         double[::1] item_norms,
                         FastLightFM lightfm,
                         int num_threads):
    """
    Generate norms of item vectors
    """

    cdef int i, no_items
    cdef flt *it_repr

    no_items = item_norms.shape[0]

    with nogil, parallel(num_threads=num_threads):

        it_repr = <flt *>malloc(sizeof(flt) * (lightfm.no_components + 1))

        for i in prange(no_items):
            compute_representation(item_features,
                                   lightfm.item_features,
                                   lightfm.item_biases,
                                   lightfm,
                                   item_ids[i],
                                   lightfm.item_scale,
                                   it_repr)

            item_norms[i] = compute_prediction_from_repr_nobias(it_repr,
                                                                it_repr,
                                                                lightfm.no_components)


def __test_find_func(CSRMatrix item_features,
                 CSRMatrix user_features,
                 CSRMatrix interactions,
                 int[::1] user_ids,
                 int[::1] item_ids,
                 flt[::1] Y,
                 int[::1] shuffle_indices,
                 FastLightFM lightfm,
                 double learning_rate,
                 double item_alpha,
                 double user_alpha,
                 int num_threads,
                 int user_id,
                 int positive_item_index,
                 int negative_item_id):

    cdef int negative_item_index

    negative_item_index = find_item_index(user_id, negative_item_id, interactions)

    print 'Negative item index: %d' % negative_item_index
    print 'Positive item index: %d' % positive_item_index
    print 'Negative item counts: %d' % Y[negative_item_index]
    print 'Positive item counts: %d' % Y[positive_item_index]


# Expose test functions
def __test_in_positives(int row, int col, CSRMatrix mat):

    if in_positives(col, row, mat):
        return True
    else:
        return False
