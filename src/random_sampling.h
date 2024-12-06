class RandomSampler {
    public:

    double gamma_sample(const double a, const double b);
    double beta_sample(const double a, const double b);
    double gaussian_sample(const double sigma);
    void multinomial_sample(int K, int N, double p[], int n[]);
    double uniform_sample();
    double beta_pdf_val(const double x, const double a, const double b);
};
