from sklearn.svm import SVC

def svc(C, kernel, gamma = None, degree = None):
    if kernel == "rbf":
        return SVC(C = C, kernel = kernel, gamma = gamma)
    if kernel == "poly":
        return SVC(C = C, kernel = kernel, degree = degree)
    return SVC(C = C, kernel = kernel)

