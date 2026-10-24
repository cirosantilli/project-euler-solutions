import ProjectEulerSolutions.Lib.SpiralPrimeSegments
import ProjectEulerSolutions.Lib.SpiralPrimeCertificates.All
import ProjectEulerSolutions.Lib.MillerRabin
import ProjectEulerSolutions.Lib.SpiralMillerAgreement
/-! Finite spiral row and counter certificates. Python execution is checked separately. -/
namespace ProjectEulerSolutions.P58Numerical
open SpiralPrimes
set_option Elab.async false
def row_1921 : RowCertificate 1921 where
  first := true
  first_correct := by
    have prime : Nat.Prime (corner 1921 1) := by change Nat.Prime 14764807; exact ProjectEulerSolutions.P58PrimeNodes.p_14764807.prime
    exact decide_eq_true prime
  second := false
  second_correct := by
    have composite : ¬ Nat.Prime (corner 1921 2) := by change ¬ Nat.Prime 14760965; exact PrattCertificates.composite_of_product (factor := 5) (cofactor := 2952193) rfl (by decide) (by decide)
    exact decide_eq_false composite
  third := false
  third_correct := by
    have composite : ¬ Nat.Prime (corner 1921 3) := by change ¬ Nat.Prime 14757123; exact PrattCertificates.composite_of_product (factor := 3) (cofactor := 4919041) rfl (by decide) (by decide)
    exact decide_eq_false composite
theorem row_count_1921 : row 1921 = 1 := (row_1921.checked).trans rfl
theorem model_count_1921 : modelRow 1921 = 1 := by decide
theorem modelrow_agrees_1921 : modelRow 1921 = row 1921 := model_count_1921.trans row_count_1921.symm
def row_1922 : RowCertificate 1922 where
  first := false
  first_correct := by
    have composite : ¬ Nat.Prime (corner 1922 1) := by change ¬ Nat.Prime 14780181; exact PrattCertificates.composite_of_product (factor := 3) (cofactor := 4926727) rfl (by decide) (by decide)
    exact decide_eq_false composite
  second := false
  second_correct := by
    have composite : ¬ Nat.Prime (corner 1922 2) := by change ¬ Nat.Prime 14776337; exact PrattCertificates.composite_of_product (factor := 761) (cofactor := 19417) rfl (by decide) (by decide)
    exact decide_eq_false composite
  third := true
  third_correct := by
    have prime : Nat.Prime (corner 1922 3) := by change Nat.Prime 14772493; exact ProjectEulerSolutions.P58PrimeNodes.p_14772493.prime
    exact decide_eq_true prime
theorem row_count_1922 : row 1922 = 1 := (row_1922.checked).trans rfl
theorem model_count_1922 : modelRow 1922 = 1 := by decide
theorem modelrow_agrees_1922 : modelRow 1922 = row 1922 := model_count_1922.trans row_count_1922.symm
def row_1923 : RowCertificate 1923 where
  first := true
  first_correct := by
    have prime : Nat.Prime (corner 1923 1) := by change Nat.Prime 14795563; exact ProjectEulerSolutions.P58PrimeNodes.p_14795563.prime
    exact decide_eq_true prime
  second := false
  second_correct := by
    have composite : ¬ Nat.Prime (corner 1923 2) := by change ¬ Nat.Prime 14791717; exact PrattCertificates.composite_of_product (factor := 17) (cofactor := 870101) rfl (by decide) (by decide)
    exact decide_eq_false composite
  third := false
  third_correct := by
    have composite : ¬ Nat.Prime (corner 1923 3) := by change ¬ Nat.Prime 14787871; exact PrattCertificates.composite_of_product (factor := 7) (cofactor := 2112553) rfl (by decide) (by decide)
    exact decide_eq_false composite
theorem row_count_1923 : row 1923 = 1 := (row_1923.checked).trans rfl
theorem model_count_1923 : modelRow 1923 = 1 := by decide
theorem modelrow_agrees_1923 : modelRow 1923 = row 1923 := model_count_1923.trans row_count_1923.symm
def row_1924 : RowCertificate 1924 where
  first := false
  first_correct := by
    have composite : ¬ Nat.Prime (corner 1924 1) := by change ¬ Nat.Prime 14810953; exact PrattCertificates.composite_of_product (factor := 67) (cofactor := 221059) rfl (by decide) (by decide)
    exact decide_eq_false composite
  second := false
  second_correct := by
    have composite : ¬ Nat.Prime (corner 1924 2) := by change ¬ Nat.Prime 14807105; exact PrattCertificates.composite_of_product (factor := 5) (cofactor := 2961421) rfl (by decide) (by decide)
    exact decide_eq_false composite
  third := false
  third_correct := by
    have composite : ¬ Nat.Prime (corner 1924 3) := by change ¬ Nat.Prime 14803257; exact PrattCertificates.composite_of_product (factor := 3) (cofactor := 4934419) rfl (by decide) (by decide)
    exact decide_eq_false composite
theorem row_count_1924 : row 1924 = 0 := (row_1924.checked).trans rfl
theorem model_count_1924 : modelRow 1924 = 0 := by decide
theorem modelrow_agrees_1924 : modelRow 1924 = row 1924 := model_count_1924.trans row_count_1924.symm
def row_1925 : RowCertificate 1925 where
  first := false
  first_correct := by
    have composite : ¬ Nat.Prime (corner 1925 1) := by change ¬ Nat.Prime 14826351; exact PrattCertificates.composite_of_product (factor := 3) (cofactor := 4942117) rfl (by decide) (by decide)
    exact decide_eq_false composite
  second := true
  second_correct := by
    have prime : Nat.Prime (corner 1925 2) := by change Nat.Prime 14822501; exact ProjectEulerSolutions.P58PrimeNodes.p_14822501.prime
    exact decide_eq_true prime
  third := false
  third_correct := by
    have composite : ¬ Nat.Prime (corner 1925 3) := by change ¬ Nat.Prime 14818651; exact PrattCertificates.composite_of_product (factor := 19) (cofactor := 779929) rfl (by decide) (by decide)
    exact decide_eq_false composite
theorem row_count_1925 : row 1925 = 1 := (row_1925.checked).trans rfl
theorem model_count_1925 : modelRow 1925 = 1 := by decide
theorem modelrow_agrees_1925 : modelRow 1925 = row 1925 := model_count_1925.trans row_count_1925.symm
def row_1926 : RowCertificate 1926 where
  first := false
  first_correct := by
    have composite : ¬ Nat.Prime (corner 1926 1) := by change ¬ Nat.Prime 14841757; exact PrattCertificates.composite_of_product (factor := 7) (cofactor := 2120251) rfl (by decide) (by decide)
    exact decide_eq_false composite
  second := false
  second_correct := by
    have composite : ¬ Nat.Prime (corner 1926 2) := by change ¬ Nat.Prime 14837905; exact PrattCertificates.composite_of_product (factor := 5) (cofactor := 2967581) rfl (by decide) (by decide)
    exact decide_eq_false composite
  third := false
  third_correct := by
    have composite : ¬ Nat.Prime (corner 1926 3) := by change ¬ Nat.Prime 14834053; exact PrattCertificates.composite_of_product (factor := 13) (cofactor := 1141081) rfl (by decide) (by decide)
    exact decide_eq_false composite
theorem row_count_1926 : row 1926 = 0 := (row_1926.checked).trans rfl
theorem model_count_1926 : modelRow 1926 = 0 := by decide
theorem modelrow_agrees_1926 : modelRow 1926 = row 1926 := model_count_1926.trans row_count_1926.symm
def row_1927 : RowCertificate 1927 where
  first := false
  first_correct := by
    have composite : ¬ Nat.Prime (corner 1927 1) := by change ¬ Nat.Prime 14857171; exact PrattCertificates.composite_of_product (factor := 7) (cofactor := 2122453) rfl (by decide) (by decide)
    exact decide_eq_false composite
  second := false
  second_correct := by
    have composite : ¬ Nat.Prime (corner 1927 2) := by change ¬ Nat.Prime 14853317; exact PrattCertificates.composite_of_product (factor := 37) (cofactor := 401441) rfl (by decide) (by decide)
    exact decide_eq_false composite
  third := false
  third_correct := by
    have composite : ¬ Nat.Prime (corner 1927 3) := by change ¬ Nat.Prime 14849463; exact PrattCertificates.composite_of_product (factor := 3) (cofactor := 4949821) rfl (by decide) (by decide)
    exact decide_eq_false composite
theorem row_count_1927 : row 1927 = 0 := (row_1927.checked).trans rfl
theorem model_count_1927 : modelRow 1927 = 0 := by decide
theorem modelrow_agrees_1927 : modelRow 1927 = row 1927 := model_count_1927.trans row_count_1927.symm
def row_1928 : RowCertificate 1928 where
  first := false
  first_correct := by
    have composite : ¬ Nat.Prime (corner 1928 1) := by change ¬ Nat.Prime 14872593; exact PrattCertificates.composite_of_product (factor := 3) (cofactor := 4957531) rfl (by decide) (by decide)
    exact decide_eq_false composite
  second := false
  second_correct := by
    have composite : ¬ Nat.Prime (corner 1928 2) := by change ¬ Nat.Prime 14868737; exact PrattCertificates.composite_of_product (factor := 13) (cofactor := 1143749) rfl (by decide) (by decide)
    exact decide_eq_false composite
  third := true
  third_correct := by
    have prime : Nat.Prime (corner 1928 3) := by change Nat.Prime 14864881; exact ProjectEulerSolutions.P58PrimeNodes.p_14864881.prime
    exact decide_eq_true prime
theorem row_count_1928 : row 1928 = 1 := (row_1928.checked).trans rfl
theorem model_count_1928 : modelRow 1928 = 1 := by decide
theorem modelrow_agrees_1928 : modelRow 1928 = row 1928 := model_count_1928.trans row_count_1928.symm
def row_1929 : RowCertificate 1929 where
  first := false
  first_correct := by
    have composite : ¬ Nat.Prime (corner 1929 1) := by change ¬ Nat.Prime 14888023; exact PrattCertificates.composite_of_product (factor := 37) (cofactor := 402379) rfl (by decide) (by decide)
    exact decide_eq_false composite
  second := false
  second_correct := by
    have composite : ¬ Nat.Prime (corner 1929 2) := by change ¬ Nat.Prime 14884165; exact PrattCertificates.composite_of_product (factor := 5) (cofactor := 2976833) rfl (by decide) (by decide)
    exact decide_eq_false composite
  third := false
  third_correct := by
    have composite : ¬ Nat.Prime (corner 1929 3) := by change ¬ Nat.Prime 14880307; exact PrattCertificates.composite_of_product (factor := 13) (cofactor := 1144639) rfl (by decide) (by decide)
    exact decide_eq_false composite
theorem row_count_1929 : row 1929 = 0 := (row_1929.checked).trans rfl
theorem model_count_1929 : modelRow 1929 = 0 := by decide
theorem modelrow_agrees_1929 : modelRow 1929 = row 1929 := model_count_1929.trans row_count_1929.symm
def row_1930 : RowCertificate 1930 where
  first := false
  first_correct := by
    have composite : ¬ Nat.Prime (corner 1930 1) := by change ¬ Nat.Prime 14903461; exact PrattCertificates.composite_of_product (factor := 73) (cofactor := 204157) rfl (by decide) (by decide)
    exact decide_eq_false composite
  second := false
  second_correct := by
    have composite : ¬ Nat.Prime (corner 1930 2) := by change ¬ Nat.Prime 14899601; exact PrattCertificates.composite_of_product (factor := 661) (cofactor := 22541) rfl (by decide) (by decide)
    exact decide_eq_false composite
  third := false
  third_correct := by
    have composite : ¬ Nat.Prime (corner 1930 3) := by change ¬ Nat.Prime 14895741; exact PrattCertificates.composite_of_product (factor := 3) (cofactor := 4965247) rfl (by decide) (by decide)
    exact decide_eq_false composite
theorem row_count_1930 : row 1930 = 0 := (row_1930.checked).trans rfl
theorem model_count_1930 : modelRow 1930 = 0 := by decide
theorem modelrow_agrees_1930 : modelRow 1930 = row 1930 := model_count_1930.trans row_count_1930.symm
def row_1931 : RowCertificate 1931 where
  first := false
  first_correct := by
    have composite : ¬ Nat.Prime (corner 1931 1) := by change ¬ Nat.Prime 14918907; exact PrattCertificates.composite_of_product (factor := 3) (cofactor := 4972969) rfl (by decide) (by decide)
    exact decide_eq_false composite
  second := false
  second_correct := by
    have composite : ¬ Nat.Prime (corner 1931 2) := by change ¬ Nat.Prime 14915045; exact PrattCertificates.composite_of_product (factor := 5) (cofactor := 2983009) rfl (by decide) (by decide)
    exact decide_eq_false composite
  third := false
  third_correct := by
    have composite : ¬ Nat.Prime (corner 1931 3) := by change ¬ Nat.Prime 14911183; exact PrattCertificates.composite_of_product (factor := 7) (cofactor := 2130169) rfl (by decide) (by decide)
    exact decide_eq_false composite
theorem row_count_1931 : row 1931 = 0 := (row_1931.checked).trans rfl
theorem model_count_1931 : modelRow 1931 = 0 := by decide
theorem modelrow_agrees_1931 : modelRow 1931 = row 1931 := model_count_1931.trans row_count_1931.symm
def row_1932 : RowCertificate 1932 where
  first := false
  first_correct := by
    have composite : ¬ Nat.Prime (corner 1932 1) := by change ¬ Nat.Prime 14934361; exact PrattCertificates.composite_of_product (factor := 13) (cofactor := 1148797) rfl (by decide) (by decide)
    exact decide_eq_false composite
  second := false
  second_correct := by
    have composite : ¬ Nat.Prime (corner 1932 2) := by change ¬ Nat.Prime 14930497; exact PrattCertificates.composite_of_product (factor := 509) (cofactor := 29333) rfl (by decide) (by decide)
    exact decide_eq_false composite
  third := false
  third_correct := by
    have composite : ¬ Nat.Prime (corner 1932 3) := by change ¬ Nat.Prime 14926633; exact PrattCertificates.composite_of_product (factor := 43) (cofactor := 347131) rfl (by decide) (by decide)
    exact decide_eq_false composite
theorem row_count_1932 : row 1932 = 0 := (row_1932.checked).trans rfl
theorem model_count_1932 : modelRow 1932 = 0 := by decide
theorem modelrow_agrees_1932 : modelRow 1932 = row 1932 := model_count_1932.trans row_count_1932.symm
def row_1933 : RowCertificate 1933 where
  first := false
  first_correct := by
    have composite : ¬ Nat.Prime (corner 1933 1) := by change ¬ Nat.Prime 14949823; exact PrattCertificates.composite_of_product (factor := 7) (cofactor := 2135689) rfl (by decide) (by decide)
    exact decide_eq_false composite
  second := false
  second_correct := by
    have composite : ¬ Nat.Prime (corner 1933 2) := by change ¬ Nat.Prime 14945957; exact PrattCertificates.composite_of_product (factor := 13) (cofactor := 1149689) rfl (by decide) (by decide)
    exact decide_eq_false composite
  third := false
  third_correct := by
    have composite : ¬ Nat.Prime (corner 1933 3) := by change ¬ Nat.Prime 14942091; exact PrattCertificates.composite_of_product (factor := 3) (cofactor := 4980697) rfl (by decide) (by decide)
    exact decide_eq_false composite
theorem row_count_1933 : row 1933 = 0 := (row_1933.checked).trans rfl
theorem model_count_1933 : modelRow 1933 = 0 := by decide
theorem modelrow_agrees_1933 : modelRow 1933 = row 1933 := model_count_1933.trans row_count_1933.symm
def row_1934 : RowCertificate 1934 where
  first := false
  first_correct := by
    have composite : ¬ Nat.Prime (corner 1934 1) := by change ¬ Nat.Prime 14965293; exact PrattCertificates.composite_of_product (factor := 3) (cofactor := 4988431) rfl (by decide) (by decide)
    exact decide_eq_false composite
  second := false
  second_correct := by
    have composite : ¬ Nat.Prime (corner 1934 2) := by change ¬ Nat.Prime 14961425; exact PrattCertificates.composite_of_product (factor := 5) (cofactor := 2992285) rfl (by decide) (by decide)
    exact decide_eq_false composite
  third := false
  third_correct := by
    have composite : ¬ Nat.Prime (corner 1934 3) := by change ¬ Nat.Prime 14957557; exact PrattCertificates.composite_of_product (factor := 103) (cofactor := 145219) rfl (by decide) (by decide)
    exact decide_eq_false composite
theorem row_count_1934 : row 1934 = 0 := (row_1934.checked).trans rfl
theorem model_count_1934 : modelRow 1934 = 0 := by decide
theorem modelrow_agrees_1934 : modelRow 1934 = row 1934 := model_count_1934.trans row_count_1934.symm
def row_1935 : RowCertificate 1935 where
  first := false
  first_correct := by
    have composite : ¬ Nat.Prime (corner 1935 1) := by change ¬ Nat.Prime 14980771; exact PrattCertificates.composite_of_product (factor := 13) (cofactor := 1152367) rfl (by decide) (by decide)
    exact decide_eq_false composite
  second := true
  second_correct := by
    have prime : Nat.Prime (corner 1935 2) := by change Nat.Prime 14976901; exact ProjectEulerSolutions.P58PrimeNodes.p_14976901.prime
    exact decide_eq_true prime
  third := false
  third_correct := by
    have composite : ¬ Nat.Prime (corner 1935 3) := by change ¬ Nat.Prime 14973031; exact PrattCertificates.composite_of_product (factor := 31) (cofactor := 483001) rfl (by decide) (by decide)
    exact decide_eq_false composite
theorem row_count_1935 : row 1935 = 1 := (row_1935.checked).trans rfl
theorem model_count_1935 : modelRow 1935 = 1 := by decide
theorem modelrow_agrees_1935 : modelRow 1935 = row 1935 := model_count_1935.trans row_count_1935.symm
def row_1936 : RowCertificate 1936 where
  first := true
  first_correct := by
    have prime : Nat.Prime (corner 1936 1) := by change Nat.Prime 14996257; exact ProjectEulerSolutions.P58PrimeNodes.p_14996257.prime
    exact decide_eq_true prime
  second := false
  second_correct := by
    have composite : ¬ Nat.Prime (corner 1936 2) := by change ¬ Nat.Prime 14992385; exact PrattCertificates.composite_of_product (factor := 5) (cofactor := 2998477) rfl (by decide) (by decide)
    exact decide_eq_false composite
  third := false
  third_correct := by
    have composite : ¬ Nat.Prime (corner 1936 3) := by change ¬ Nat.Prime 14988513; exact PrattCertificates.composite_of_product (factor := 3) (cofactor := 4996171) rfl (by decide) (by decide)
    exact decide_eq_false composite
theorem row_count_1936 : row 1936 = 1 := (row_1936.checked).trans rfl
theorem model_count_1936 : modelRow 1936 = 1 := by decide
theorem modelrow_agrees_1936 : modelRow 1936 = row 1936 := model_count_1936.trans row_count_1936.symm
def row_1937 : RowCertificate 1937 where
  first := false
  first_correct := by
    have composite : ¬ Nat.Prime (corner 1937 1) := by change ¬ Nat.Prime 15011751; exact PrattCertificates.composite_of_product (factor := 3) (cofactor := 5003917) rfl (by decide) (by decide)
    exact decide_eq_false composite
  second := false
  second_correct := by
    have composite : ¬ Nat.Prime (corner 1937 2) := by change ¬ Nat.Prime 15007877; exact PrattCertificates.composite_of_product (factor := 29) (cofactor := 517513) rfl (by decide) (by decide)
    exact decide_eq_false composite
  third := false
  third_correct := by
    have composite : ¬ Nat.Prime (corner 1937 3) := by change ¬ Nat.Prime 15004003; exact PrattCertificates.composite_of_product (factor := 7) (cofactor := 2143429) rfl (by decide) (by decide)
    exact decide_eq_false composite
theorem row_count_1937 : row 1937 = 0 := (row_1937.checked).trans rfl
theorem model_count_1937 : modelRow 1937 = 0 := by decide
theorem modelrow_agrees_1937 : modelRow 1937 = row 1937 := model_count_1937.trans row_count_1937.symm
def row_1938 : RowCertificate 1938 where
  first := false
  first_correct := by
    have composite : ¬ Nat.Prime (corner 1938 1) := by change ¬ Nat.Prime 15027253; exact PrattCertificates.composite_of_product (factor := 43) (cofactor := 349471) rfl (by decide) (by decide)
    exact decide_eq_false composite
  second := false
  second_correct := by
    have composite : ¬ Nat.Prime (corner 1938 2) := by change ¬ Nat.Prime 15023377; exact PrattCertificates.composite_of_product (factor := 1733) (cofactor := 8669) rfl (by decide) (by decide)
    exact decide_eq_false composite
  third := false
  third_correct := by
    have composite : ¬ Nat.Prime (corner 1938 3) := by change ¬ Nat.Prime 15019501; exact PrattCertificates.composite_of_product (factor := 7) (cofactor := 2145643) rfl (by decide) (by decide)
    exact decide_eq_false composite
theorem row_count_1938 : row 1938 = 0 := (row_1938.checked).trans rfl
theorem model_count_1938 : modelRow 1938 = 0 := by decide
theorem modelrow_agrees_1938 : modelRow 1938 = row 1938 := model_count_1938.trans row_count_1938.symm
def row_1939 : RowCertificate 1939 where
  first := false
  first_correct := by
    have composite : ¬ Nat.Prime (corner 1939 1) := by change ¬ Nat.Prime 15042763; exact PrattCertificates.composite_of_product (factor := 109) (cofactor := 138007) rfl (by decide) (by decide)
    exact decide_eq_false composite
  second := false
  second_correct := by
    have composite : ¬ Nat.Prime (corner 1939 2) := by change ¬ Nat.Prime 15038885; exact PrattCertificates.composite_of_product (factor := 5) (cofactor := 3007777) rfl (by decide) (by decide)
    exact decide_eq_false composite
  third := false
  third_correct := by
    have composite : ¬ Nat.Prime (corner 1939 3) := by change ¬ Nat.Prime 15035007; exact PrattCertificates.composite_of_product (factor := 3) (cofactor := 5011669) rfl (by decide) (by decide)
    exact decide_eq_false composite
theorem row_count_1939 : row 1939 = 0 := (row_1939.checked).trans rfl
theorem model_count_1939 : modelRow 1939 = 0 := by decide
theorem modelrow_agrees_1939 : modelRow 1939 = row 1939 := model_count_1939.trans row_count_1939.symm
def row_1940 : RowCertificate 1940 where
  first := false
  first_correct := by
    have composite : ¬ Nat.Prime (corner 1940 1) := by change ¬ Nat.Prime 15058281; exact PrattCertificates.composite_of_product (factor := 3) (cofactor := 5019427) rfl (by decide) (by decide)
    exact decide_eq_false composite
  second := false
  second_correct := by
    have composite : ¬ Nat.Prime (corner 1940 2) := by change ¬ Nat.Prime 15054401; exact PrattCertificates.composite_of_product (factor := 17) (cofactor := 885553) rfl (by decide) (by decide)
    exact decide_eq_false composite
  third := true
  third_correct := by
    have prime : Nat.Prime (corner 1940 3) := by change Nat.Prime 15050521; exact ProjectEulerSolutions.P58PrimeNodes.p_15050521.prime
    exact decide_eq_true prime
theorem row_count_1940 : row 1940 = 1 := (row_1940.checked).trans rfl
theorem model_count_1940 : modelRow 1940 = 1 := by decide
theorem modelrow_agrees_1940 : modelRow 1940 = row 1940 := model_count_1940.trans row_count_1940.symm
def row_1941 : RowCertificate 1941 where
  first := false
  first_correct := by
    have composite : ¬ Nat.Prime (corner 1941 1) := by change ¬ Nat.Prime 15073807; exact PrattCertificates.composite_of_product (factor := 7) (cofactor := 2153401) rfl (by decide) (by decide)
    exact decide_eq_false composite
  second := false
  second_correct := by
    have composite : ¬ Nat.Prime (corner 1941 2) := by change ¬ Nat.Prime 15069925; exact PrattCertificates.composite_of_product (factor := 5) (cofactor := 3013985) rfl (by decide) (by decide)
    exact decide_eq_false composite
  third := true
  third_correct := by
    have prime : Nat.Prime (corner 1941 3) := by change Nat.Prime 15066043; exact ProjectEulerSolutions.P58PrimeNodes.p_15066043.prime
    exact decide_eq_true prime
theorem row_count_1941 : row 1941 = 1 := (row_1941.checked).trans rfl
theorem model_count_1941 : modelRow 1941 = 1 := by decide
theorem modelrow_agrees_1941 : modelRow 1941 = row 1941 := model_count_1941.trans row_count_1941.symm
def row_1942 : RowCertificate 1942 where
  first := true
  first_correct := by
    have prime : Nat.Prime (corner 1942 1) := by change Nat.Prime 15089341; exact ProjectEulerSolutions.P58PrimeNodes.p_15089341.prime
    exact decide_eq_true prime
  second := true
  second_correct := by
    have prime : Nat.Prime (corner 1942 2) := by change Nat.Prime 15085457; exact ProjectEulerSolutions.P58PrimeNodes.p_15085457.prime
    exact decide_eq_true prime
  third := false
  third_correct := by
    have composite : ¬ Nat.Prime (corner 1942 3) := by change ¬ Nat.Prime 15081573; exact PrattCertificates.composite_of_product (factor := 3) (cofactor := 5027191) rfl (by decide) (by decide)
    exact decide_eq_false composite
theorem row_count_1942 : row 1942 = 2 := (row_1942.checked).trans rfl
theorem model_count_1942 : modelRow 1942 = 2 := by decide
theorem modelrow_agrees_1942 : modelRow 1942 = row 1942 := model_count_1942.trans row_count_1942.symm
def row_1943 : RowCertificate 1943 where
  first := false
  first_correct := by
    have composite : ¬ Nat.Prime (corner 1943 1) := by change ¬ Nat.Prime 15104883; exact PrattCertificates.composite_of_product (factor := 3) (cofactor := 5034961) rfl (by decide) (by decide)
    exact decide_eq_false composite
  second := false
  second_correct := by
    have composite : ¬ Nat.Prime (corner 1943 2) := by change ¬ Nat.Prime 15100997; exact PrattCertificates.composite_of_product (factor := 41) (cofactor := 368317) rfl (by decide) (by decide)
    exact decide_eq_false composite
  third := true
  third_correct := by
    have prime : Nat.Prime (corner 1943 3) := by change Nat.Prime 15097111; exact ProjectEulerSolutions.P58PrimeNodes.p_15097111.prime
    exact decide_eq_true prime
theorem row_count_1943 : row 1943 = 1 := (row_1943.checked).trans rfl
theorem model_count_1943 : modelRow 1943 = 1 := by decide
theorem modelrow_agrees_1943 : modelRow 1943 = row 1943 := model_count_1943.trans row_count_1943.symm
def row_1944 : RowCertificate 1944 where
  first := true
  first_correct := by
    have prime : Nat.Prime (corner 1944 1) := by change Nat.Prime 15120433; exact ProjectEulerSolutions.P58PrimeNodes.p_15120433.prime
    exact decide_eq_true prime
  second := false
  second_correct := by
    have composite : ¬ Nat.Prime (corner 1944 2) := by change ¬ Nat.Prime 15116545; exact PrattCertificates.composite_of_product (factor := 5) (cofactor := 3023309) rfl (by decide) (by decide)
    exact decide_eq_false composite
  third := false
  third_correct := by
    have composite : ¬ Nat.Prime (corner 1944 3) := by change ¬ Nat.Prime 15112657; exact PrattCertificates.composite_of_product (factor := 7) (cofactor := 2158951) rfl (by decide) (by decide)
    exact decide_eq_false composite
theorem row_count_1944 : row 1944 = 1 := (row_1944.checked).trans rfl
theorem model_count_1944 : modelRow 1944 = 1 := by decide
theorem modelrow_agrees_1944 : modelRow 1944 = row 1944 := model_count_1944.trans row_count_1944.symm
def row_1945 : RowCertificate 1945 where
  first := false
  first_correct := by
    have composite : ¬ Nat.Prime (corner 1945 1) := by change ¬ Nat.Prime 15135991; exact PrattCertificates.composite_of_product (factor := 13) (cofactor := 1164307) rfl (by decide) (by decide)
    exact decide_eq_false composite
  second := true
  second_correct := by
    have prime : Nat.Prime (corner 1945 2) := by change Nat.Prime 15132101; exact ProjectEulerSolutions.P58PrimeNodes.p_15132101.prime
    exact decide_eq_true prime
  third := false
  third_correct := by
    have composite : ¬ Nat.Prime (corner 1945 3) := by change ¬ Nat.Prime 15128211; exact PrattCertificates.composite_of_product (factor := 3) (cofactor := 5042737) rfl (by decide) (by decide)
    exact decide_eq_false composite
theorem row_count_1945 : row 1945 = 1 := (row_1945.checked).trans rfl
theorem model_count_1945 : modelRow 1945 = 1 := by decide
theorem modelrow_agrees_1945 : modelRow 1945 = row 1945 := model_count_1945.trans row_count_1945.symm
def row_1946 : RowCertificate 1946 where
  first := false
  first_correct := by
    have composite : ¬ Nat.Prime (corner 1946 1) := by change ¬ Nat.Prime 15151557; exact PrattCertificates.composite_of_product (factor := 3) (cofactor := 5050519) rfl (by decide) (by decide)
    exact decide_eq_false composite
  second := false
  second_correct := by
    have composite : ¬ Nat.Prime (corner 1946 2) := by change ¬ Nat.Prime 15147665; exact PrattCertificates.composite_of_product (factor := 5) (cofactor := 3029533) rfl (by decide) (by decide)
    exact decide_eq_false composite
  third := true
  third_correct := by
    have prime : Nat.Prime (corner 1946 3) := by change Nat.Prime 15143773; exact ProjectEulerSolutions.P58PrimeNodes.p_15143773.prime
    exact decide_eq_true prime
theorem row_count_1946 : row 1946 = 1 := (row_1946.checked).trans rfl
theorem model_count_1946 : modelRow 1946 = 1 := by decide
theorem modelrow_agrees_1946 : modelRow 1946 = row 1946 := model_count_1946.trans row_count_1946.symm
def row_1947 : RowCertificate 1947 where
  first := false
  first_correct := by
    have composite : ¬ Nat.Prime (corner 1947 1) := by change ¬ Nat.Prime 15167131; exact PrattCertificates.composite_of_product (factor := 7) (cofactor := 2166733) rfl (by decide) (by decide)
    exact decide_eq_false composite
  second := true
  second_correct := by
    have prime : Nat.Prime (corner 1947 2) := by change Nat.Prime 15163237; exact ProjectEulerSolutions.P58PrimeNodes.p_15163237.prime
    exact decide_eq_true prime
  third := false
  third_correct := by
    have composite : ¬ Nat.Prime (corner 1947 3) := by change ¬ Nat.Prime 15159343; exact PrattCertificates.composite_of_product (factor := 151) (cofactor := 100393) rfl (by decide) (by decide)
    exact decide_eq_false composite
theorem row_count_1947 : row 1947 = 1 := (row_1947.checked).trans rfl
theorem model_count_1947 : modelRow 1947 = 1 := by decide
theorem modelrow_agrees_1947 : modelRow 1947 = row 1947 := model_count_1947.trans row_count_1947.symm
def row_1948 : RowCertificate 1948 where
  first := false
  first_correct := by
    have composite : ¬ Nat.Prime (corner 1948 1) := by change ¬ Nat.Prime 15182713; exact PrattCertificates.composite_of_product (factor := 7) (cofactor := 2168959) rfl (by decide) (by decide)
    exact decide_eq_false composite
  second := false
  second_correct := by
    have composite : ¬ Nat.Prime (corner 1948 2) := by change ¬ Nat.Prime 15178817; exact PrattCertificates.composite_of_product (factor := 73) (cofactor := 207929) rfl (by decide) (by decide)
    exact decide_eq_false composite
  third := false
  third_correct := by
    have composite : ¬ Nat.Prime (corner 1948 3) := by change ¬ Nat.Prime 15174921; exact PrattCertificates.composite_of_product (factor := 3) (cofactor := 5058307) rfl (by decide) (by decide)
    exact decide_eq_false composite
theorem row_count_1948 : row 1948 = 0 := (row_1948.checked).trans rfl
theorem model_count_1948 : modelRow 1948 = 0 := by decide
theorem modelrow_agrees_1948 : modelRow 1948 = row 1948 := model_count_1948.trans row_count_1948.symm
def row_1949 : RowCertificate 1949 where
  first := false
  first_correct := by
    have composite : ¬ Nat.Prime (corner 1949 1) := by change ¬ Nat.Prime 15198303; exact PrattCertificates.composite_of_product (factor := 3) (cofactor := 5066101) rfl (by decide) (by decide)
    exact decide_eq_false composite
  second := false
  second_correct := by
    have composite : ¬ Nat.Prime (corner 1949 2) := by change ¬ Nat.Prime 15194405; exact PrattCertificates.composite_of_product (factor := 5) (cofactor := 3038881) rfl (by decide) (by decide)
    exact decide_eq_false composite
  third := false
  third_correct := by
    have composite : ¬ Nat.Prime (corner 1949 3) := by change ¬ Nat.Prime 15190507; exact PrattCertificates.composite_of_product (factor := 2797) (cofactor := 5431) rfl (by decide) (by decide)
    exact decide_eq_false composite
theorem row_count_1949 : row 1949 = 0 := (row_1949.checked).trans rfl
theorem model_count_1949 : modelRow 1949 = 0 := by decide
theorem modelrow_agrees_1949 : modelRow 1949 = row 1949 := model_count_1949.trans row_count_1949.symm
def row_1950 : RowCertificate 1950 where
  first := false
  first_correct := by
    have composite : ¬ Nat.Prime (corner 1950 1) := by change ¬ Nat.Prime 15213901; exact PrattCertificates.composite_of_product (factor := 31) (cofactor := 490771) rfl (by decide) (by decide)
    exact decide_eq_false composite
  second := true
  second_correct := by
    have prime : Nat.Prime (corner 1950 2) := by change Nat.Prime 15210001; exact ProjectEulerSolutions.P58PrimeNodes.p_15210001.prime
    exact decide_eq_true prime
  third := true
  third_correct := by
    have prime : Nat.Prime (corner 1950 3) := by change Nat.Prime 15206101; exact ProjectEulerSolutions.P58PrimeNodes.p_15206101.prime
    exact decide_eq_true prime
theorem row_count_1950 : row 1950 = 2 := (row_1950.checked).trans rfl
theorem model_count_1950 : modelRow 1950 = 2 := by decide
theorem modelrow_agrees_1950 : modelRow 1950 = row 1950 := model_count_1950.trans row_count_1950.symm
def row_1951 : RowCertificate 1951 where
  first := false
  first_correct := by
    have composite : ¬ Nat.Prime (corner 1951 1) := by change ¬ Nat.Prime 15229507; exact PrattCertificates.composite_of_product (factor := 19) (cofactor := 801553) rfl (by decide) (by decide)
    exact decide_eq_false composite
  second := false
  second_correct := by
    have composite : ¬ Nat.Prime (corner 1951 2) := by change ¬ Nat.Prime 15225605; exact PrattCertificates.composite_of_product (factor := 5) (cofactor := 3045121) rfl (by decide) (by decide)
    exact decide_eq_false composite
  third := false
  third_correct := by
    have composite : ¬ Nat.Prime (corner 1951 3) := by change ¬ Nat.Prime 15221703; exact PrattCertificates.composite_of_product (factor := 3) (cofactor := 5073901) rfl (by decide) (by decide)
    exact decide_eq_false composite
theorem row_count_1951 : row 1951 = 0 := (row_1951.checked).trans rfl
theorem model_count_1951 : modelRow 1951 = 0 := by decide
theorem modelrow_agrees_1951 : modelRow 1951 = row 1951 := model_count_1951.trans row_count_1951.symm
def row_1952 : RowCertificate 1952 where
  first := false
  first_correct := by
    have composite : ¬ Nat.Prime (corner 1952 1) := by change ¬ Nat.Prime 15245121; exact PrattCertificates.composite_of_product (factor := 3) (cofactor := 5081707) rfl (by decide) (by decide)
    exact decide_eq_false composite
  second := false
  second_correct := by
    have composite : ¬ Nat.Prime (corner 1952 2) := by change ¬ Nat.Prime 15241217; exact PrattCertificates.composite_of_product (factor := 41) (cofactor := 371737) rfl (by decide) (by decide)
    exact decide_eq_false composite
  third := false
  third_correct := by
    have composite : ¬ Nat.Prime (corner 1952 3) := by change ¬ Nat.Prime 15237313; exact PrattCertificates.composite_of_product (factor := 7) (cofactor := 2176759) rfl (by decide) (by decide)
    exact decide_eq_false composite
theorem row_count_1952 : row 1952 = 0 := (row_1952.checked).trans rfl
theorem model_count_1952 : modelRow 1952 = 0 := by decide
theorem modelrow_agrees_1952 : modelRow 1952 = row 1952 := model_count_1952.trans row_count_1952.symm
def row_1953 : RowCertificate 1953 where
  first := false
  first_correct := by
    have composite : ¬ Nat.Prime (corner 1953 1) := by change ¬ Nat.Prime 15260743; exact PrattCertificates.composite_of_product (factor := 19) (cofactor := 803197) rfl (by decide) (by decide)
    exact decide_eq_false composite
  second := false
  second_correct := by
    have composite : ¬ Nat.Prime (corner 1953 2) := by change ¬ Nat.Prime 15256837; exact PrattCertificates.composite_of_product (factor := 17) (cofactor := 897461) rfl (by decide) (by decide)
    exact decide_eq_false composite
  third := true
  third_correct := by
    have prime : Nat.Prime (corner 1953 3) := by change Nat.Prime 15252931; exact ProjectEulerSolutions.P58PrimeNodes.p_15252931.prime
    exact decide_eq_true prime
theorem row_count_1953 : row 1953 = 1 := (row_1953.checked).trans rfl
theorem model_count_1953 : modelRow 1953 = 1 := by decide
theorem modelrow_agrees_1953 : modelRow 1953 = row 1953 := model_count_1953.trans row_count_1953.symm
def row_1954 : RowCertificate 1954 where
  first := false
  first_correct := by
    have composite : ¬ Nat.Prime (corner 1954 1) := by change ¬ Nat.Prime 15276373; exact PrattCertificates.composite_of_product (factor := 7) (cofactor := 2182339) rfl (by decide) (by decide)
    exact decide_eq_false composite
  second := false
  second_correct := by
    have composite : ¬ Nat.Prime (corner 1954 2) := by change ¬ Nat.Prime 15272465; exact PrattCertificates.composite_of_product (factor := 5) (cofactor := 3054493) rfl (by decide) (by decide)
    exact decide_eq_false composite
  third := false
  third_correct := by
    have composite : ¬ Nat.Prime (corner 1954 3) := by change ¬ Nat.Prime 15268557; exact PrattCertificates.composite_of_product (factor := 3) (cofactor := 5089519) rfl (by decide) (by decide)
    exact decide_eq_false composite
theorem row_count_1954 : row 1954 = 0 := (row_1954.checked).trans rfl
theorem model_count_1954 : modelRow 1954 = 0 := by decide
theorem modelrow_agrees_1954 : modelRow 1954 = row 1954 := model_count_1954.trans row_count_1954.symm
def row_1955 : RowCertificate 1955 where
  first := false
  first_correct := by
    have composite : ¬ Nat.Prime (corner 1955 1) := by change ¬ Nat.Prime 15292011; exact PrattCertificates.composite_of_product (factor := 3) (cofactor := 5097337) rfl (by decide) (by decide)
    exact decide_eq_false composite
  second := true
  second_correct := by
    have prime : Nat.Prime (corner 1955 2) := by change Nat.Prime 15288101; exact ProjectEulerSolutions.P58PrimeNodes.p_15288101.prime
    exact decide_eq_true prime
  third := false
  third_correct := by
    have composite : ¬ Nat.Prime (corner 1955 3) := by change ¬ Nat.Prime 15284191; exact PrattCertificates.composite_of_product (factor := 13) (cofactor := 1175707) rfl (by decide) (by decide)
    exact decide_eq_false composite
theorem row_count_1955 : row 1955 = 1 := (row_1955.checked).trans rfl
theorem model_count_1955 : modelRow 1955 = 1 := by decide
theorem modelrow_agrees_1955 : modelRow 1955 = row 1955 := model_count_1955.trans row_count_1955.symm
def row_1956 : RowCertificate 1956 where
  first := false
  first_correct := by
    have composite : ¬ Nat.Prime (corner 1956 1) := by change ¬ Nat.Prime 15307657; exact PrattCertificates.composite_of_product (factor := 157) (cofactor := 97501) rfl (by decide) (by decide)
    exact decide_eq_false composite
  second := false
  second_correct := by
    have composite : ¬ Nat.Prime (corner 1956 2) := by change ¬ Nat.Prime 15303745; exact PrattCertificates.composite_of_product (factor := 5) (cofactor := 3060749) rfl (by decide) (by decide)
    exact decide_eq_false composite
  third := false
  third_correct := by
    have composite : ¬ Nat.Prime (corner 1956 3) := by change ¬ Nat.Prime 15299833; exact PrattCertificates.composite_of_product (factor := 31) (cofactor := 493543) rfl (by decide) (by decide)
    exact decide_eq_false composite
theorem row_count_1956 : row 1956 = 0 := (row_1956.checked).trans rfl
theorem model_count_1956 : modelRow 1956 = 0 := by decide
theorem modelrow_agrees_1956 : modelRow 1956 = row 1956 := model_count_1956.trans row_count_1956.symm
def row_1957 : RowCertificate 1957 where
  first := true
  first_correct := by
    have prime : Nat.Prime (corner 1957 1) := by change Nat.Prime 15323311; exact ProjectEulerSolutions.P58PrimeNodes.p_15323311.prime
    exact decide_eq_true prime
  second := false
  second_correct := by
    have composite : ¬ Nat.Prime (corner 1957 2) := by change ¬ Nat.Prime 15319397; exact PrattCertificates.composite_of_product (factor := 17) (cofactor := 901141) rfl (by decide) (by decide)
    exact decide_eq_false composite
  third := false
  third_correct := by
    have composite : ¬ Nat.Prime (corner 1957 3) := by change ¬ Nat.Prime 15315483; exact PrattCertificates.composite_of_product (factor := 3) (cofactor := 5105161) rfl (by decide) (by decide)
    exact decide_eq_false composite
theorem row_count_1957 : row 1957 = 1 := (row_1957.checked).trans rfl
theorem model_count_1957 : modelRow 1957 = 1 := by decide
theorem modelrow_agrees_1957 : modelRow 1957 = row 1957 := model_count_1957.trans row_count_1957.symm
def row_1958 : RowCertificate 1958 where
  first := false
  first_correct := by
    have composite : ¬ Nat.Prime (corner 1958 1) := by change ¬ Nat.Prime 15338973; exact PrattCertificates.composite_of_product (factor := 3) (cofactor := 5112991) rfl (by decide) (by decide)
    exact decide_eq_false composite
  second := false
  second_correct := by
    have composite : ¬ Nat.Prime (corner 1958 2) := by change ¬ Nat.Prime 15335057; exact PrattCertificates.composite_of_product (factor := 37) (cofactor := 414461) rfl (by decide) (by decide)
    exact decide_eq_false composite
  third := false
  third_correct := by
    have composite : ¬ Nat.Prime (corner 1958 3) := by change ¬ Nat.Prime 15331141; exact PrattCertificates.composite_of_product (factor := 7) (cofactor := 2190163) rfl (by decide) (by decide)
    exact decide_eq_false composite
theorem row_count_1958 : row 1958 = 0 := (row_1958.checked).trans rfl
theorem model_count_1958 : modelRow 1958 = 0 := by decide
theorem modelrow_agrees_1958 : modelRow 1958 = row 1958 := model_count_1958.trans row_count_1958.symm
def row_1959 : RowCertificate 1959 where
  first := true
  first_correct := by
    have prime : Nat.Prime (corner 1959 1) := by change Nat.Prime 15354643; exact ProjectEulerSolutions.P58PrimeNodes.p_15354643.prime
    exact decide_eq_true prime
  second := false
  second_correct := by
    have composite : ¬ Nat.Prime (corner 1959 2) := by change ¬ Nat.Prime 15350725; exact PrattCertificates.composite_of_product (factor := 5) (cofactor := 3070145) rfl (by decide) (by decide)
    exact decide_eq_false composite
  third := false
  third_correct := by
    have composite : ¬ Nat.Prime (corner 1959 3) := by change ¬ Nat.Prime 15346807; exact PrattCertificates.composite_of_product (factor := 7) (cofactor := 2192401) rfl (by decide) (by decide)
    exact decide_eq_false composite
theorem row_count_1959 : row 1959 = 1 := (row_1959.checked).trans rfl
theorem model_count_1959 : modelRow 1959 = 1 := by decide
theorem modelrow_agrees_1959 : modelRow 1959 = row 1959 := model_count_1959.trans row_count_1959.symm
def row_1960 : RowCertificate 1960 where
  first := true
  first_correct := by
    have prime : Nat.Prime (corner 1960 1) := by change Nat.Prime 15370321; exact ProjectEulerSolutions.P58PrimeNodes.p_15370321.prime
    exact decide_eq_true prime
  second := false
  second_correct := by
    have composite : ¬ Nat.Prime (corner 1960 2) := by change ¬ Nat.Prime 15366401; exact PrattCertificates.composite_of_product (factor := 241) (cofactor := 63761) rfl (by decide) (by decide)
    exact decide_eq_false composite
  third := false
  third_correct := by
    have composite : ¬ Nat.Prime (corner 1960 3) := by change ¬ Nat.Prime 15362481; exact PrattCertificates.composite_of_product (factor := 3) (cofactor := 5120827) rfl (by decide) (by decide)
    exact decide_eq_false composite
theorem row_count_1960 : row 1960 = 1 := (row_1960.checked).trans rfl
theorem model_count_1960 : modelRow 1960 = 1 := by decide
theorem modelrow_agrees_1960 : modelRow 1960 = row 1960 := model_count_1960.trans row_count_1960.symm
def row_1961 : RowCertificate 1961 where
  first := false
  first_correct := by
    have composite : ¬ Nat.Prime (corner 1961 1) := by change ¬ Nat.Prime 15386007; exact PrattCertificates.composite_of_product (factor := 3) (cofactor := 5128669) rfl (by decide) (by decide)
    exact decide_eq_false composite
  second := false
  second_correct := by
    have composite : ¬ Nat.Prime (corner 1961 2) := by change ¬ Nat.Prime 15382085; exact PrattCertificates.composite_of_product (factor := 5) (cofactor := 3076417) rfl (by decide) (by decide)
    exact decide_eq_false composite
  third := false
  third_correct := by
    have composite : ¬ Nat.Prime (corner 1961 3) := by change ¬ Nat.Prime 15378163; exact PrattCertificates.composite_of_product (factor := 19) (cofactor := 809377) rfl (by decide) (by decide)
    exact decide_eq_false composite
theorem row_count_1961 : row 1961 = 0 := (row_1961.checked).trans rfl
theorem model_count_1961 : modelRow 1961 = 0 := by decide
theorem modelrow_agrees_1961 : modelRow 1961 = row 1961 := model_count_1961.trans row_count_1961.symm
def row_1962 : RowCertificate 1962 where
  first := false
  first_correct := by
    have composite : ¬ Nat.Prime (corner 1962 1) := by change ¬ Nat.Prime 15401701; exact PrattCertificates.composite_of_product (factor := 7) (cofactor := 2200243) rfl (by decide) (by decide)
    exact decide_eq_false composite
  second := true
  second_correct := by
    have prime : Nat.Prime (corner 1962 2) := by change Nat.Prime 15397777; exact ProjectEulerSolutions.P58PrimeNodes.p_15397777.prime
    exact decide_eq_true prime
  third := false
  third_correct := by
    have composite : ¬ Nat.Prime (corner 1962 3) := by change ¬ Nat.Prime 15393853; exact PrattCertificates.composite_of_product (factor := 67) (cofactor := 229759) rfl (by decide) (by decide)
    exact decide_eq_false composite
theorem row_count_1962 : row 1962 = 1 := (row_1962.checked).trans rfl
theorem model_count_1962 : modelRow 1962 = 1 := by decide
theorem modelrow_agrees_1962 : modelRow 1962 = row 1962 := model_count_1962.trans row_count_1962.symm
def row_1963 : RowCertificate 1963 where
  first := false
  first_correct := by
    have composite : ¬ Nat.Prime (corner 1963 1) := by change ¬ Nat.Prime 15417403; exact PrattCertificates.composite_of_product (factor := 79) (cofactor := 195157) rfl (by decide) (by decide)
    exact decide_eq_false composite
  second := false
  second_correct := by
    have composite : ¬ Nat.Prime (corner 1963 2) := by change ¬ Nat.Prime 15413477; exact PrattCertificates.composite_of_product (factor := 197) (cofactor := 78241) rfl (by decide) (by decide)
    exact decide_eq_false composite
  third := false
  third_correct := by
    have composite : ¬ Nat.Prime (corner 1963 3) := by change ¬ Nat.Prime 15409551; exact PrattCertificates.composite_of_product (factor := 3) (cofactor := 5136517) rfl (by decide) (by decide)
    exact decide_eq_false composite
theorem row_count_1963 : row 1963 = 0 := (row_1963.checked).trans rfl
theorem model_count_1963 : modelRow 1963 = 0 := by decide
theorem modelrow_agrees_1963 : modelRow 1963 = row 1963 := model_count_1963.trans row_count_1963.symm
def row_1964 : RowCertificate 1964 where
  first := false
  first_correct := by
    have composite : ¬ Nat.Prime (corner 1964 1) := by change ¬ Nat.Prime 15433113; exact PrattCertificates.composite_of_product (factor := 3) (cofactor := 5144371) rfl (by decide) (by decide)
    exact decide_eq_false composite
  second := false
  second_correct := by
    have composite : ¬ Nat.Prime (corner 1964 2) := by change ¬ Nat.Prime 15429185; exact PrattCertificates.composite_of_product (factor := 5) (cofactor := 3085837) rfl (by decide) (by decide)
    exact decide_eq_false composite
  third := true
  third_correct := by
    have prime : Nat.Prime (corner 1964 3) := by change Nat.Prime 15425257; exact ProjectEulerSolutions.P58PrimeNodes.p_15425257.prime
    exact decide_eq_true prime
theorem row_count_1964 : row 1964 = 1 := (row_1964.checked).trans rfl
theorem model_count_1964 : modelRow 1964 = 1 := by decide
theorem modelrow_agrees_1964 : modelRow 1964 = row 1964 := model_count_1964.trans row_count_1964.symm
def row_1965 : RowCertificate 1965 where
  first := false
  first_correct := by
    have composite : ¬ Nat.Prime (corner 1965 1) := by change ¬ Nat.Prime 15448831; exact PrattCertificates.composite_of_product (factor := 2473) (cofactor := 6247) rfl (by decide) (by decide)
    exact decide_eq_false composite
  second := false
  second_correct := by
    have composite : ¬ Nat.Prime (corner 1965 2) := by change ¬ Nat.Prime 15444901; exact PrattCertificates.composite_of_product (factor := 2309) (cofactor := 6689) rfl (by decide) (by decide)
    exact decide_eq_false composite
  third := false
  third_correct := by
    have composite : ¬ Nat.Prime (corner 1965 3) := by change ¬ Nat.Prime 15440971; exact PrattCertificates.composite_of_product (factor := 7) (cofactor := 2205853) rfl (by decide) (by decide)
    exact decide_eq_false composite
theorem row_count_1965 : row 1965 = 0 := (row_1965.checked).trans rfl
theorem model_count_1965 : modelRow 1965 = 0 := by decide
theorem modelrow_agrees_1965 : modelRow 1965 = row 1965 := model_count_1965.trans row_count_1965.symm
def row_1966 : RowCertificate 1966 where
  first := false
  first_correct := by
    have composite : ¬ Nat.Prime (corner 1966 1) := by change ¬ Nat.Prime 15464557; exact PrattCertificates.composite_of_product (factor := 37) (cofactor := 417961) rfl (by decide) (by decide)
    exact decide_eq_false composite
  second := false
  second_correct := by
    have composite : ¬ Nat.Prime (corner 1966 2) := by change ¬ Nat.Prime 15460625; exact PrattCertificates.composite_of_product (factor := 5) (cofactor := 3092125) rfl (by decide) (by decide)
    exact decide_eq_false composite
  third := false
  third_correct := by
    have composite : ¬ Nat.Prime (corner 1966 3) := by change ¬ Nat.Prime 15456693; exact PrattCertificates.composite_of_product (factor := 3) (cofactor := 5152231) rfl (by decide) (by decide)
    exact decide_eq_false composite
theorem row_count_1966 : row 1966 = 0 := (row_1966.checked).trans rfl
theorem model_count_1966 : modelRow 1966 = 0 := by decide
theorem modelrow_agrees_1966 : modelRow 1966 = row 1966 := model_count_1966.trans row_count_1966.symm
def row_1967 : RowCertificate 1967 where
  first := false
  first_correct := by
    have composite : ¬ Nat.Prime (corner 1967 1) := by change ¬ Nat.Prime 15480291; exact PrattCertificates.composite_of_product (factor := 3) (cofactor := 5160097) rfl (by decide) (by decide)
    exact decide_eq_false composite
  second := false
  second_correct := by
    have composite : ¬ Nat.Prime (corner 1967 2) := by change ¬ Nat.Prime 15476357; exact PrattCertificates.composite_of_product (factor := 13) (cofactor := 1190489) rfl (by decide) (by decide)
    exact decide_eq_false composite
  third := false
  third_correct := by
    have composite : ¬ Nat.Prime (corner 1967 3) := by change ¬ Nat.Prime 15472423; exact PrattCertificates.composite_of_product (factor := 73) (cofactor := 211951) rfl (by decide) (by decide)
    exact decide_eq_false composite
theorem row_count_1967 : row 1967 = 0 := (row_1967.checked).trans rfl
theorem model_count_1967 : modelRow 1967 = 0 := by decide
theorem modelrow_agrees_1967 : modelRow 1967 = row 1967 := model_count_1967.trans row_count_1967.symm
def row_1968 : RowCertificate 1968 where
  first := false
  first_correct := by
    have composite : ¬ Nat.Prime (corner 1968 1) := by change ¬ Nat.Prime 15496033; exact PrattCertificates.composite_of_product (factor := 7) (cofactor := 2213719) rfl (by decide) (by decide)
    exact decide_eq_false composite
  second := false
  second_correct := by
    have composite : ¬ Nat.Prime (corner 1968 2) := by change ¬ Nat.Prime 15492097; exact PrattCertificates.composite_of_product (factor := 137) (cofactor := 113081) rfl (by decide) (by decide)
    exact decide_eq_false composite
  third := false
  third_correct := by
    have composite : ¬ Nat.Prime (corner 1968 3) := by change ¬ Nat.Prime 15488161; exact PrattCertificates.composite_of_product (factor := 13) (cofactor := 1191397) rfl (by decide) (by decide)
    exact decide_eq_false composite
theorem row_count_1968 : row 1968 = 0 := (row_1968.checked).trans rfl
theorem model_count_1968 : modelRow 1968 = 0 := by decide
theorem modelrow_agrees_1968 : modelRow 1968 = row 1968 := model_count_1968.trans row_count_1968.symm
def row_1969 : RowCertificate 1969 where
  first := false
  first_correct := by
    have composite : ¬ Nat.Prime (corner 1969 1) := by change ¬ Nat.Prime 15511783; exact PrattCertificates.composite_of_product (factor := 7) (cofactor := 2215969) rfl (by decide) (by decide)
    exact decide_eq_false composite
  second := false
  second_correct := by
    have composite : ¬ Nat.Prime (corner 1969 2) := by change ¬ Nat.Prime 15507845; exact PrattCertificates.composite_of_product (factor := 5) (cofactor := 3101569) rfl (by decide) (by decide)
    exact decide_eq_false composite
  third := false
  third_correct := by
    have composite : ¬ Nat.Prime (corner 1969 3) := by change ¬ Nat.Prime 15503907; exact PrattCertificates.composite_of_product (factor := 3) (cofactor := 5167969) rfl (by decide) (by decide)
    exact decide_eq_false composite
theorem row_count_1969 : row 1969 = 0 := (row_1969.checked).trans rfl
theorem model_count_1969 : modelRow 1969 = 0 := by decide
theorem modelrow_agrees_1969 : modelRow 1969 = row 1969 := model_count_1969.trans row_count_1969.symm
def row_1970 : RowCertificate 1970 where
  first := false
  first_correct := by
    have composite : ¬ Nat.Prime (corner 1970 1) := by change ¬ Nat.Prime 15527541; exact PrattCertificates.composite_of_product (factor := 3) (cofactor := 5175847) rfl (by decide) (by decide)
    exact decide_eq_false composite
  second := false
  second_correct := by
    have composite : ¬ Nat.Prime (corner 1970 2) := by change ¬ Nat.Prime 15523601; exact PrattCertificates.composite_of_product (factor := 17) (cofactor := 913153) rfl (by decide) (by decide)
    exact decide_eq_false composite
  third := true
  third_correct := by
    have prime : Nat.Prime (corner 1970 3) := by change Nat.Prime 15519661; exact ProjectEulerSolutions.P58PrimeNodes.p_15519661.prime
    exact decide_eq_true prime
theorem row_count_1970 : row 1970 = 1 := (row_1970.checked).trans rfl
theorem model_count_1970 : modelRow 1970 = 1 := by decide
theorem modelrow_agrees_1970 : modelRow 1970 = row 1970 := model_count_1970.trans row_count_1970.symm
def row_1971 : RowCertificate 1971 where
  first := false
  first_correct := by
    have composite : ¬ Nat.Prime (corner 1971 1) := by change ¬ Nat.Prime 15543307; exact PrattCertificates.composite_of_product (factor := 13) (cofactor := 1195639) rfl (by decide) (by decide)
    exact decide_eq_false composite
  second := false
  second_correct := by
    have composite : ¬ Nat.Prime (corner 1971 2) := by change ¬ Nat.Prime 15539365; exact PrattCertificates.composite_of_product (factor := 5) (cofactor := 3107873) rfl (by decide) (by decide)
    exact decide_eq_false composite
  third := false
  third_correct := by
    have composite : ¬ Nat.Prime (corner 1971 3) := by change ¬ Nat.Prime 15535423; exact PrattCertificates.composite_of_product (factor := 97) (cofactor := 160159) rfl (by decide) (by decide)
    exact decide_eq_false composite
theorem row_count_1971 : row 1971 = 0 := (row_1971.checked).trans rfl
theorem model_count_1971 : modelRow 1971 = 0 := by decide
theorem modelrow_agrees_1971 : modelRow 1971 = row 1971 := model_count_1971.trans row_count_1971.symm
def row_1972 : RowCertificate 1972 where
  first := false
  first_correct := by
    have composite : ¬ Nat.Prime (corner 1972 1) := by change ¬ Nat.Prime 15559081; exact PrattCertificates.composite_of_product (factor := 19) (cofactor := 818899) rfl (by decide) (by decide)
    exact decide_eq_false composite
  second := false
  second_correct := by
    have composite : ¬ Nat.Prime (corner 1972 2) := by change ¬ Nat.Prime 15555137; exact PrattCertificates.composite_of_product (factor := 13) (cofactor := 1196549) rfl (by decide) (by decide)
    exact decide_eq_false composite
  third := false
  third_correct := by
    have composite : ¬ Nat.Prime (corner 1972 3) := by change ¬ Nat.Prime 15551193; exact PrattCertificates.composite_of_product (factor := 3) (cofactor := 5183731) rfl (by decide) (by decide)
    exact decide_eq_false composite
theorem row_count_1972 : row 1972 = 0 := (row_1972.checked).trans rfl
theorem model_count_1972 : modelRow 1972 = 0 := by decide
theorem modelrow_agrees_1972 : modelRow 1972 = row 1972 := model_count_1972.trans row_count_1972.symm
def row_1973 : RowCertificate 1973 where
  first := false
  first_correct := by
    have composite : ¬ Nat.Prime (corner 1973 1) := by change ¬ Nat.Prime 15574863; exact PrattCertificates.composite_of_product (factor := 3) (cofactor := 5191621) rfl (by decide) (by decide)
    exact decide_eq_false composite
  second := true
  second_correct := by
    have prime : Nat.Prime (corner 1973 2) := by change Nat.Prime 15570917; exact ProjectEulerSolutions.P58PrimeNodes.p_15570917.prime
    exact decide_eq_true prime
  third := false
  third_correct := by
    have composite : ¬ Nat.Prime (corner 1973 3) := by change ¬ Nat.Prime 15566971; exact PrattCertificates.composite_of_product (factor := 7) (cofactor := 2223853) rfl (by decide) (by decide)
    exact decide_eq_false composite
theorem row_count_1973 : row 1973 = 1 := (row_1973.checked).trans rfl
theorem model_count_1973 : modelRow 1973 = 1 := by decide
theorem modelrow_agrees_1973 : modelRow 1973 = row 1973 := model_count_1973.trans row_count_1973.symm
def row_1974 : RowCertificate 1974 where
  first := false
  first_correct := by
    have composite : ¬ Nat.Prime (corner 1974 1) := by change ¬ Nat.Prime 15590653; exact PrattCertificates.composite_of_product (factor := 13) (cofactor := 1199281) rfl (by decide) (by decide)
    exact decide_eq_false composite
  second := false
  second_correct := by
    have composite : ¬ Nat.Prime (corner 1974 2) := by change ¬ Nat.Prime 15586705; exact PrattCertificates.composite_of_product (factor := 5) (cofactor := 3117341) rfl (by decide) (by decide)
    exact decide_eq_false composite
  third := false
  third_correct := by
    have composite : ¬ Nat.Prime (corner 1974 3) := by change ¬ Nat.Prime 15582757; exact PrattCertificates.composite_of_product (factor := 1543) (cofactor := 10099) rfl (by decide) (by decide)
    exact decide_eq_false composite
theorem row_count_1974 : row 1974 = 0 := (row_1974.checked).trans rfl
theorem model_count_1974 : modelRow 1974 = 0 := by decide
theorem modelrow_agrees_1974 : modelRow 1974 = row 1974 := model_count_1974.trans row_count_1974.symm
def row_1975 : RowCertificate 1975 where
  first := false
  first_correct := by
    have composite : ¬ Nat.Prime (corner 1975 1) := by change ¬ Nat.Prime 15606451; exact PrattCertificates.composite_of_product (factor := 7) (cofactor := 2229493) rfl (by decide) (by decide)
    exact decide_eq_false composite
  second := false
  second_correct := by
    have composite : ¬ Nat.Prime (corner 1975 2) := by change ¬ Nat.Prime 15602501; exact PrattCertificates.composite_of_product (factor := 89) (cofactor := 175309) rfl (by decide) (by decide)
    exact decide_eq_false composite
  third := false
  third_correct := by
    have composite : ¬ Nat.Prime (corner 1975 3) := by change ¬ Nat.Prime 15598551; exact PrattCertificates.composite_of_product (factor := 3) (cofactor := 5199517) rfl (by decide) (by decide)
    exact decide_eq_false composite
theorem row_count_1975 : row 1975 = 0 := (row_1975.checked).trans rfl
theorem model_count_1975 : modelRow 1975 = 0 := by decide
theorem modelrow_agrees_1975 : modelRow 1975 = row 1975 := model_count_1975.trans row_count_1975.symm
def row_1976 : RowCertificate 1976 where
  first := false
  first_correct := by
    have composite : ¬ Nat.Prime (corner 1976 1) := by change ¬ Nat.Prime 15622257; exact PrattCertificates.composite_of_product (factor := 3) (cofactor := 5207419) rfl (by decide) (by decide)
    exact decide_eq_false composite
  second := false
  second_correct := by
    have composite : ¬ Nat.Prime (corner 1976 2) := by change ¬ Nat.Prime 15618305; exact PrattCertificates.composite_of_product (factor := 5) (cofactor := 3123661) rfl (by decide) (by decide)
    exact decide_eq_false composite
  third := false
  third_correct := by
    have composite : ¬ Nat.Prime (corner 1976 3) := by change ¬ Nat.Prime 15614353; exact PrattCertificates.composite_of_product (factor := 61) (cofactor := 255973) rfl (by decide) (by decide)
    exact decide_eq_false composite
theorem row_count_1976 : row 1976 = 0 := (row_1976.checked).trans rfl
theorem model_count_1976 : modelRow 1976 = 0 := by decide
theorem modelrow_agrees_1976 : modelRow 1976 = row 1976 := model_count_1976.trans row_count_1976.symm
def row_1977 : RowCertificate 1977 where
  first := true
  first_correct := by
    have prime : Nat.Prime (corner 1977 1) := by change Nat.Prime 15638071; exact ProjectEulerSolutions.P58PrimeNodes.p_15638071.prime
    exact decide_eq_true prime
  second := false
  second_correct := by
    have composite : ¬ Nat.Prime (corner 1977 2) := by change ¬ Nat.Prime 15634117; exact PrattCertificates.composite_of_product (factor := 61) (cofactor := 256297) rfl (by decide) (by decide)
    exact decide_eq_false composite
  third := false
  third_correct := by
    have composite : ¬ Nat.Prime (corner 1977 3) := by change ¬ Nat.Prime 15630163; exact PrattCertificates.composite_of_product (factor := 367) (cofactor := 42589) rfl (by decide) (by decide)
    exact decide_eq_false composite
theorem row_count_1977 : row 1977 = 1 := (row_1977.checked).trans rfl
theorem model_count_1977 : modelRow 1977 = 1 := by decide
theorem modelrow_agrees_1977 : modelRow 1977 = row 1977 := model_count_1977.trans row_count_1977.symm
def row_1978 : RowCertificate 1978 where
  first := false
  first_correct := by
    have composite : ¬ Nat.Prime (corner 1978 1) := by change ¬ Nat.Prime 15653893; exact PrattCertificates.composite_of_product (factor := 127) (cofactor := 123259) rfl (by decide) (by decide)
    exact decide_eq_false composite
  second := false
  second_correct := by
    have composite : ¬ Nat.Prime (corner 1978 2) := by change ¬ Nat.Prime 15649937; exact PrattCertificates.composite_of_product (factor := 29) (cofactor := 539653) rfl (by decide) (by decide)
    exact decide_eq_false composite
  third := false
  third_correct := by
    have composite : ¬ Nat.Prime (corner 1978 3) := by change ¬ Nat.Prime 15645981; exact PrattCertificates.composite_of_product (factor := 3) (cofactor := 5215327) rfl (by decide) (by decide)
    exact decide_eq_false composite
theorem row_count_1978 : row 1978 = 0 := (row_1978.checked).trans rfl
theorem model_count_1978 : modelRow 1978 = 0 := by decide
theorem modelrow_agrees_1978 : modelRow 1978 = row 1978 := model_count_1978.trans row_count_1978.symm
def row_1979 : RowCertificate 1979 where
  first := false
  first_correct := by
    have composite : ¬ Nat.Prime (corner 1979 1) := by change ¬ Nat.Prime 15669723; exact PrattCertificates.composite_of_product (factor := 3) (cofactor := 5223241) rfl (by decide) (by decide)
    exact decide_eq_false composite
  second := false
  second_correct := by
    have composite : ¬ Nat.Prime (corner 1979 2) := by change ¬ Nat.Prime 15665765; exact PrattCertificates.composite_of_product (factor := 5) (cofactor := 3133153) rfl (by decide) (by decide)
    exact decide_eq_false composite
  third := false
  third_correct := by
    have composite : ¬ Nat.Prime (corner 1979 3) := by change ¬ Nat.Prime 15661807; exact PrattCertificates.composite_of_product (factor := 7) (cofactor := 2237401) rfl (by decide) (by decide)
    exact decide_eq_false composite
theorem row_count_1979 : row 1979 = 0 := (row_1979.checked).trans rfl
theorem model_count_1979 : modelRow 1979 = 0 := by decide
theorem modelrow_agrees_1979 : modelRow 1979 = row 1979 := model_count_1979.trans row_count_1979.symm
def row_1980 : RowCertificate 1980 where
  first := false
  first_correct := by
    have composite : ¬ Nat.Prime (corner 1980 1) := by change ¬ Nat.Prime 15685561; exact PrattCertificates.composite_of_product (factor := 103) (cofactor := 152287) rfl (by decide) (by decide)
    exact decide_eq_false composite
  second := false
  second_correct := by
    have composite : ¬ Nat.Prime (corner 1980 2) := by change ¬ Nat.Prime 15681601; exact PrattCertificates.composite_of_product (factor := 13) (cofactor := 1206277) rfl (by decide) (by decide)
    exact decide_eq_false composite
  third := false
  third_correct := by
    have composite : ¬ Nat.Prime (corner 1980 3) := by change ¬ Nat.Prime 15677641; exact PrattCertificates.composite_of_product (factor := 7) (cofactor := 2239663) rfl (by decide) (by decide)
    exact decide_eq_false composite
theorem row_count_1980 : row 1980 = 0 := (row_1980.checked).trans rfl
theorem model_count_1980 : modelRow 1980 = 0 := by decide
theorem modelrow_agrees_1980 : modelRow 1980 = row 1980 := model_count_1980.trans row_count_1980.symm
def row_1981 : RowCertificate 1981 where
  first := false
  first_correct := by
    have composite : ¬ Nat.Prime (corner 1981 1) := by change ¬ Nat.Prime 15701407; exact PrattCertificates.composite_of_product (factor := 31) (cofactor := 506497) rfl (by decide) (by decide)
    exact decide_eq_false composite
  second := false
  second_correct := by
    have composite : ¬ Nat.Prime (corner 1981 2) := by change ¬ Nat.Prime 15697445; exact PrattCertificates.composite_of_product (factor := 5) (cofactor := 3139489) rfl (by decide) (by decide)
    exact decide_eq_false composite
  third := false
  third_correct := by
    have composite : ¬ Nat.Prime (corner 1981 3) := by change ¬ Nat.Prime 15693483; exact PrattCertificates.composite_of_product (factor := 3) (cofactor := 5231161) rfl (by decide) (by decide)
    exact decide_eq_false composite
theorem row_count_1981 : row 1981 = 0 := (row_1981.checked).trans rfl
theorem model_count_1981 : modelRow 1981 = 0 := by decide
theorem modelrow_agrees_1981 : modelRow 1981 = row 1981 := model_count_1981.trans row_count_1981.symm
def row_1982 : RowCertificate 1982 where
  first := false
  first_correct := by
    have composite : ¬ Nat.Prime (corner 1982 1) := by change ¬ Nat.Prime 15717261; exact PrattCertificates.composite_of_product (factor := 3) (cofactor := 5239087) rfl (by decide) (by decide)
    exact decide_eq_false composite
  second := false
  second_correct := by
    have composite : ¬ Nat.Prime (corner 1982 2) := by change ¬ Nat.Prime 15713297; exact PrattCertificates.composite_of_product (factor := 293) (cofactor := 53629) rfl (by decide) (by decide)
    exact decide_eq_false composite
  third := false
  third_correct := by
    have composite : ¬ Nat.Prime (corner 1982 3) := by change ¬ Nat.Prime 15709333; exact PrattCertificates.composite_of_product (factor := 19) (cofactor := 826807) rfl (by decide) (by decide)
    exact decide_eq_false composite
theorem row_count_1982 : row 1982 = 0 := (row_1982.checked).trans rfl
theorem model_count_1982 : modelRow 1982 = 0 := by decide
theorem modelrow_agrees_1982 : modelRow 1982 = row 1982 := model_count_1982.trans row_count_1982.symm
def row_1983 : RowCertificate 1983 where
  first := false
  first_correct := by
    have composite : ¬ Nat.Prime (corner 1983 1) := by change ¬ Nat.Prime 15733123; exact PrattCertificates.composite_of_product (factor := 7) (cofactor := 2247589) rfl (by decide) (by decide)
    exact decide_eq_false composite
  second := true
  second_correct := by
    have prime : Nat.Prime (corner 1983 2) := by change Nat.Prime 15729157; exact ProjectEulerSolutions.P58PrimeNodes.p_15729157.prime
    exact decide_eq_true prime
  third := true
  third_correct := by
    have prime : Nat.Prime (corner 1983 3) := by change Nat.Prime 15725191; exact ProjectEulerSolutions.P58PrimeNodes.p_15725191.prime
    exact decide_eq_true prime
theorem row_count_1983 : row 1983 = 2 := (row_1983.checked).trans rfl
theorem model_count_1983 : modelRow 1983 = 2 := by decide
theorem modelrow_agrees_1983 : modelRow 1983 = row 1983 := model_count_1983.trans row_count_1983.symm
def row_1984 : RowCertificate 1984 where
  first := false
  first_correct := by
    have composite : ¬ Nat.Prime (corner 1984 1) := by change ¬ Nat.Prime 15748993; exact PrattCertificates.composite_of_product (factor := 13) (cofactor := 1211461) rfl (by decide) (by decide)
    exact decide_eq_false composite
  second := false
  second_correct := by
    have composite : ¬ Nat.Prime (corner 1984 2) := by change ¬ Nat.Prime 15745025; exact PrattCertificates.composite_of_product (factor := 5) (cofactor := 3149005) rfl (by decide) (by decide)
    exact decide_eq_false composite
  third := false
  third_correct := by
    have composite : ¬ Nat.Prime (corner 1984 3) := by change ¬ Nat.Prime 15741057; exact PrattCertificates.composite_of_product (factor := 3) (cofactor := 5247019) rfl (by decide) (by decide)
    exact decide_eq_false composite
theorem row_count_1984 : row 1984 = 0 := (row_1984.checked).trans rfl
theorem model_count_1984 : modelRow 1984 = 0 := by decide
theorem modelrow_agrees_1984 : modelRow 1984 = row 1984 := model_count_1984.trans row_count_1984.symm
def row_1985 : RowCertificate 1985 where
  first := false
  first_correct := by
    have composite : ¬ Nat.Prime (corner 1985 1) := by change ¬ Nat.Prime 15764871; exact PrattCertificates.composite_of_product (factor := 3) (cofactor := 5254957) rfl (by decide) (by decide)
    exact decide_eq_false composite
  second := false
  second_correct := by
    have composite : ¬ Nat.Prime (corner 1985 2) := by change ¬ Nat.Prime 15760901; exact PrattCertificates.composite_of_product (factor := 13) (cofactor := 1212377) rfl (by decide) (by decide)
    exact decide_eq_false composite
  third := false
  third_correct := by
    have composite : ¬ Nat.Prime (corner 1985 3) := by change ¬ Nat.Prime 15756931; exact PrattCertificates.composite_of_product (factor := 37) (cofactor := 425863) rfl (by decide) (by decide)
    exact decide_eq_false composite
theorem row_count_1985 : row 1985 = 0 := (row_1985.checked).trans rfl
theorem model_count_1985 : modelRow 1985 = 0 := by decide
theorem modelrow_agrees_1985 : modelRow 1985 = row 1985 := model_count_1985.trans row_count_1985.symm
def row_1986 : RowCertificate 1986 where
  first := false
  first_correct := by
    have composite : ¬ Nat.Prime (corner 1986 1) := by change ¬ Nat.Prime 15780757; exact PrattCertificates.composite_of_product (factor := 733) (cofactor := 21529) rfl (by decide) (by decide)
    exact decide_eq_false composite
  second := false
  second_correct := by
    have composite : ¬ Nat.Prime (corner 1986 2) := by change ¬ Nat.Prime 15776785; exact PrattCertificates.composite_of_product (factor := 5) (cofactor := 3155357) rfl (by decide) (by decide)
    exact decide_eq_false composite
  third := false
  third_correct := by
    have composite : ¬ Nat.Prime (corner 1986 3) := by change ¬ Nat.Prime 15772813; exact PrattCertificates.composite_of_product (factor := 7) (cofactor := 2253259) rfl (by decide) (by decide)
    exact decide_eq_false composite
theorem row_count_1986 : row 1986 = 0 := (row_1986.checked).trans rfl
theorem model_count_1986 : modelRow 1986 = 0 := by decide
theorem modelrow_agrees_1986 : modelRow 1986 = row 1986 := model_count_1986.trans row_count_1986.symm
def row_1987 : RowCertificate 1987 where
  first := false
  first_correct := by
    have composite : ¬ Nat.Prime (corner 1987 1) := by change ¬ Nat.Prime 15796651; exact PrattCertificates.composite_of_product (factor := 13) (cofactor := 1215127) rfl (by decide) (by decide)
    exact decide_eq_false composite
  second := false
  second_correct := by
    have composite : ¬ Nat.Prime (corner 1987 2) := by change ¬ Nat.Prime 15792677; exact PrattCertificates.composite_of_product (factor := 17) (cofactor := 928981) rfl (by decide) (by decide)
    exact decide_eq_false composite
  third := false
  third_correct := by
    have composite : ¬ Nat.Prime (corner 1987 3) := by change ¬ Nat.Prime 15788703; exact PrattCertificates.composite_of_product (factor := 3) (cofactor := 5262901) rfl (by decide) (by decide)
    exact decide_eq_false composite
theorem row_count_1987 : row 1987 = 0 := (row_1987.checked).trans rfl
theorem model_count_1987 : modelRow 1987 = 0 := by decide
theorem modelrow_agrees_1987 : modelRow 1987 = row 1987 := model_count_1987.trans row_count_1987.symm
def row_1988 : RowCertificate 1988 where
  first := false
  first_correct := by
    have composite : ¬ Nat.Prime (corner 1988 1) := by change ¬ Nat.Prime 15812553; exact PrattCertificates.composite_of_product (factor := 3) (cofactor := 5270851) rfl (by decide) (by decide)
    exact decide_eq_false composite
  second := false
  second_correct := by
    have composite : ¬ Nat.Prime (corner 1988 2) := by change ¬ Nat.Prime 15808577; exact PrattCertificates.composite_of_product (factor := 61) (cofactor := 259157) rfl (by decide) (by decide)
    exact decide_eq_false composite
  third := true
  third_correct := by
    have prime : Nat.Prime (corner 1988 3) := by change Nat.Prime 15804601; exact ProjectEulerSolutions.P58PrimeNodes.p_15804601.prime
    exact decide_eq_true prime
theorem row_count_1988 : row 1988 = 1 := (row_1988.checked).trans rfl
theorem model_count_1988 : modelRow 1988 = 1 := by decide
theorem modelrow_agrees_1988 : modelRow 1988 = row 1988 := model_count_1988.trans row_count_1988.symm
def row_1989 : RowCertificate 1989 where
  first := false
  first_correct := by
    have composite : ¬ Nat.Prime (corner 1989 1) := by change ¬ Nat.Prime 15828463; exact PrattCertificates.composite_of_product (factor := 7) (cofactor := 2261209) rfl (by decide) (by decide)
    exact decide_eq_false composite
  second := false
  second_correct := by
    have composite : ¬ Nat.Prime (corner 1989 2) := by change ¬ Nat.Prime 15824485; exact PrattCertificates.composite_of_product (factor := 5) (cofactor := 3164897) rfl (by decide) (by decide)
    exact decide_eq_false composite
  third := false
  third_correct := by
    have composite : ¬ Nat.Prime (corner 1989 3) := by change ¬ Nat.Prime 15820507; exact PrattCertificates.composite_of_product (factor := 3529) (cofactor := 4483) rfl (by decide) (by decide)
    exact decide_eq_false composite
theorem row_count_1989 : row 1989 = 0 := (row_1989.checked).trans rfl
theorem model_count_1989 : modelRow 1989 = 0 := by decide
theorem modelrow_agrees_1989 : modelRow 1989 = row 1989 := model_count_1989.trans row_count_1989.symm
def row_1990 : RowCertificate 1990 where
  first := false
  first_correct := by
    have composite : ¬ Nat.Prime (corner 1990 1) := by change ¬ Nat.Prime 15844381; exact PrattCertificates.composite_of_product (factor := 7) (cofactor := 2263483) rfl (by decide) (by decide)
    exact decide_eq_false composite
  second := false
  second_correct := by
    have composite : ¬ Nat.Prime (corner 1990 2) := by change ¬ Nat.Prime 15840401; exact PrattCertificates.composite_of_product (factor := 673) (cofactor := 23537) rfl (by decide) (by decide)
    exact decide_eq_false composite
  third := false
  third_correct := by
    have composite : ¬ Nat.Prime (corner 1990 3) := by change ¬ Nat.Prime 15836421; exact PrattCertificates.composite_of_product (factor := 3) (cofactor := 5278807) rfl (by decide) (by decide)
    exact decide_eq_false composite
theorem row_count_1990 : row 1990 = 0 := (row_1990.checked).trans rfl
theorem model_count_1990 : modelRow 1990 = 0 := by decide
theorem modelrow_agrees_1990 : modelRow 1990 = row 1990 := model_count_1990.trans row_count_1990.symm
def row_1991 : RowCertificate 1991 where
  first := false
  first_correct := by
    have composite : ¬ Nat.Prime (corner 1991 1) := by change ¬ Nat.Prime 15860307; exact PrattCertificates.composite_of_product (factor := 3) (cofactor := 5286769) rfl (by decide) (by decide)
    exact decide_eq_false composite
  second := false
  second_correct := by
    have composite : ¬ Nat.Prime (corner 1991 2) := by change ¬ Nat.Prime 15856325; exact PrattCertificates.composite_of_product (factor := 5) (cofactor := 3171265) rfl (by decide) (by decide)
    exact decide_eq_false composite
  third := false
  third_correct := by
    have composite : ¬ Nat.Prime (corner 1991 3) := by change ¬ Nat.Prime 15852343; exact PrattCertificates.composite_of_product (factor := 13) (cofactor := 1219411) rfl (by decide) (by decide)
    exact decide_eq_false composite
theorem row_count_1991 : row 1991 = 0 := (row_1991.checked).trans rfl
theorem model_count_1991 : modelRow 1991 = 0 := by decide
theorem modelrow_agrees_1991 : modelRow 1991 = row 1991 := model_count_1991.trans row_count_1991.symm
def row_1992 : RowCertificate 1992 where
  first := false
  first_correct := by
    have composite : ¬ Nat.Prime (corner 1992 1) := by change ¬ Nat.Prime 15876241; exact PrattCertificates.composite_of_product (factor := 2719) (cofactor := 5839) rfl (by decide) (by decide)
    exact decide_eq_false composite
  second := true
  second_correct := by
    have prime : Nat.Prime (corner 1992 2) := by change Nat.Prime 15872257; exact ProjectEulerSolutions.P58PrimeNodes.p_15872257.prime
    exact decide_eq_true prime
  third := false
  third_correct := by
    have composite : ¬ Nat.Prime (corner 1992 3) := by change ¬ Nat.Prime 15868273; exact PrattCertificates.composite_of_product (factor := 1489) (cofactor := 10657) rfl (by decide) (by decide)
    exact decide_eq_false composite
theorem row_count_1992 : row 1992 = 1 := (row_1992.checked).trans rfl
theorem model_count_1992 : modelRow 1992 = 1 := by decide
theorem modelrow_agrees_1992 : modelRow 1992 = row 1992 := model_count_1992.trans row_count_1992.symm
def row_1993 : RowCertificate 1993 where
  first := true
  first_correct := by
    have prime : Nat.Prime (corner 1993 1) := by change Nat.Prime 15892183; exact ProjectEulerSolutions.P58PrimeNodes.p_15892183.prime
    exact decide_eq_true prime
  second := false
  second_correct := by
    have composite : ¬ Nat.Prime (corner 1993 2) := by change ¬ Nat.Prime 15888197; exact PrattCertificates.composite_of_product (factor := 13) (cofactor := 1222169) rfl (by decide) (by decide)
    exact decide_eq_false composite
  third := false
  third_correct := by
    have composite : ¬ Nat.Prime (corner 1993 3) := by change ¬ Nat.Prime 15884211; exact PrattCertificates.composite_of_product (factor := 3) (cofactor := 5294737) rfl (by decide) (by decide)
    exact decide_eq_false composite
theorem row_count_1993 : row 1993 = 1 := (row_1993.checked).trans rfl
theorem model_count_1993 : modelRow 1993 = 1 := by decide
theorem modelrow_agrees_1993 : modelRow 1993 = row 1993 := model_count_1993.trans row_count_1993.symm
def row_1994 : RowCertificate 1994 where
  first := false
  first_correct := by
    have composite : ¬ Nat.Prime (corner 1994 1) := by change ¬ Nat.Prime 15908133; exact PrattCertificates.composite_of_product (factor := 3) (cofactor := 5302711) rfl (by decide) (by decide)
    exact decide_eq_false composite
  second := false
  second_correct := by
    have composite : ¬ Nat.Prime (corner 1994 2) := by change ¬ Nat.Prime 15904145; exact PrattCertificates.composite_of_product (factor := 5) (cofactor := 3180829) rfl (by decide) (by decide)
    exact decide_eq_false composite
  third := false
  third_correct := by
    have composite : ¬ Nat.Prime (corner 1994 3) := by change ¬ Nat.Prime 15900157; exact PrattCertificates.composite_of_product (factor := 7) (cofactor := 2271451) rfl (by decide) (by decide)
    exact decide_eq_false composite
theorem row_count_1994 : row 1994 = 0 := (row_1994.checked).trans rfl
theorem model_count_1994 : modelRow 1994 = 0 := by decide
theorem modelrow_agrees_1994 : modelRow 1994 = row 1994 := model_count_1994.trans row_count_1994.symm
def row_1995 : RowCertificate 1995 where
  first := false
  first_correct := by
    have composite : ¬ Nat.Prime (corner 1995 1) := by change ¬ Nat.Prime 15924091; exact PrattCertificates.composite_of_product (factor := 67) (cofactor := 237673) rfl (by decide) (by decide)
    exact decide_eq_false composite
  second := false
  second_correct := by
    have composite : ¬ Nat.Prime (corner 1995 2) := by change ¬ Nat.Prime 15920101; exact PrattCertificates.composite_of_product (factor := 29) (cofactor := 548969) rfl (by decide) (by decide)
    exact decide_eq_false composite
  third := true
  third_correct := by
    have prime : Nat.Prime (corner 1995 3) := by change Nat.Prime 15916111; exact ProjectEulerSolutions.P58PrimeNodes.p_15916111.prime
    exact decide_eq_true prime
theorem row_count_1995 : row 1995 = 1 := (row_1995.checked).trans rfl
theorem model_count_1995 : modelRow 1995 = 1 := by decide
theorem modelrow_agrees_1995 : modelRow 1995 = row 1995 := model_count_1995.trans row_count_1995.symm
def row_1996 : RowCertificate 1996 where
  first := false
  first_correct := by
    have composite : ¬ Nat.Prime (corner 1996 1) := by change ¬ Nat.Prime 15940057; exact PrattCertificates.composite_of_product (factor := 7) (cofactor := 2277151) rfl (by decide) (by decide)
    exact decide_eq_false composite
  second := false
  second_correct := by
    have composite : ¬ Nat.Prime (corner 1996 2) := by change ¬ Nat.Prime 15936065; exact PrattCertificates.composite_of_product (factor := 5) (cofactor := 3187213) rfl (by decide) (by decide)
    exact decide_eq_false composite
  third := false
  third_correct := by
    have composite : ¬ Nat.Prime (corner 1996 3) := by change ¬ Nat.Prime 15932073; exact PrattCertificates.composite_of_product (factor := 3) (cofactor := 5310691) rfl (by decide) (by decide)
    exact decide_eq_false composite
theorem row_count_1996 : row 1996 = 0 := (row_1996.checked).trans rfl
theorem model_count_1996 : modelRow 1996 = 0 := by decide
theorem modelrow_agrees_1996 : modelRow 1996 = row 1996 := model_count_1996.trans row_count_1996.symm
def row_1997 : RowCertificate 1997 where
  first := false
  first_correct := by
    have composite : ¬ Nat.Prime (corner 1997 1) := by change ¬ Nat.Prime 15956031; exact PrattCertificates.composite_of_product (factor := 3) (cofactor := 5318677) rfl (by decide) (by decide)
    exact decide_eq_false composite
  second := true
  second_correct := by
    have prime : Nat.Prime (corner 1997 2) := by change Nat.Prime 15952037; exact ProjectEulerSolutions.P58PrimeNodes.p_15952037.prime
    exact decide_eq_true prime
  third := false
  third_correct := by
    have composite : ¬ Nat.Prime (corner 1997 3) := by change ¬ Nat.Prime 15948043; exact PrattCertificates.composite_of_product (factor := 31) (cofactor := 514453) rfl (by decide) (by decide)
    exact decide_eq_false composite
theorem row_count_1997 : row 1997 = 1 := (row_1997.checked).trans rfl
theorem model_count_1997 : modelRow 1997 = 1 := by decide
theorem modelrow_agrees_1997 : modelRow 1997 = row 1997 := model_count_1997.trans row_count_1997.symm
def row_1998 : RowCertificate 1998 where
  first := true
  first_correct := by
    have prime : Nat.Prime (corner 1998 1) := by change Nat.Prime 15972013; exact ProjectEulerSolutions.P58PrimeNodes.p_15972013.prime
    exact decide_eq_true prime
  second := false
  second_correct := by
    have composite : ¬ Nat.Prime (corner 1998 2) := by change ¬ Nat.Prime 15968017; exact PrattCertificates.composite_of_product (factor := 13) (cofactor := 1228309) rfl (by decide) (by decide)
    exact decide_eq_false composite
  third := true
  third_correct := by
    have prime : Nat.Prime (corner 1998 3) := by change Nat.Prime 15964021; exact ProjectEulerSolutions.P58PrimeNodes.p_15964021.prime
    exact decide_eq_true prime
theorem row_count_1998 : row 1998 = 2 := (row_1998.checked).trans rfl
theorem model_count_1998 : modelRow 1998 = 2 := by decide
theorem modelrow_agrees_1998 : modelRow 1998 = row 1998 := model_count_1998.trans row_count_1998.symm
def row_1999 : RowCertificate 1999 where
  first := true
  first_correct := by
    have prime : Nat.Prime (corner 1999 1) := by change Nat.Prime 15988003; exact ProjectEulerSolutions.P58PrimeNodes.p_15988003.prime
    exact decide_eq_true prime
  second := false
  second_correct := by
    have composite : ¬ Nat.Prime (corner 1999 2) := by change ¬ Nat.Prime 15984005; exact PrattCertificates.composite_of_product (factor := 5) (cofactor := 3196801) rfl (by decide) (by decide)
    exact decide_eq_false composite
  third := false
  third_correct := by
    have composite : ¬ Nat.Prime (corner 1999 3) := by change ¬ Nat.Prime 15980007; exact PrattCertificates.composite_of_product (factor := 3) (cofactor := 5326669) rfl (by decide) (by decide)
    exact decide_eq_false composite
theorem row_count_1999 : row 1999 = 1 := (row_1999.checked).trans rfl
theorem model_count_1999 : modelRow 1999 = 1 := by decide
theorem modelrow_agrees_1999 : modelRow 1999 = row 1999 := model_count_1999.trans row_count_1999.symm
def row_2000 : RowCertificate 2000 where
  first := false
  first_correct := by
    have composite : ¬ Nat.Prime (corner 2000 1) := by change ¬ Nat.Prime 16004001; exact PrattCertificates.composite_of_product (factor := 3) (cofactor := 5334667) rfl (by decide) (by decide)
    exact decide_eq_false composite
  second := false
  second_correct := by
    have composite : ¬ Nat.Prime (corner 2000 2) := by change ¬ Nat.Prime 16000001; exact PrattCertificates.composite_of_product (factor := 109) (cofactor := 146789) rfl (by decide) (by decide)
    exact decide_eq_false composite
  third := false
  third_correct := by
    have composite : ¬ Nat.Prime (corner 2000 3) := by change ¬ Nat.Prime 15996001; exact PrattCertificates.composite_of_product (factor := 7) (cofactor := 2285143) rfl (by decide) (by decide)
    exact decide_eq_false composite
theorem row_count_2000 : row 2000 = 0 := (row_2000.checked).trans rfl
theorem model_count_2000 : modelRow 2000 = 0 := by decide
theorem modelrow_agrees_2000 : modelRow 2000 = row 2000 := model_count_2000.trans row_count_2000.symm
def row_2001 : RowCertificate 2001 where
  first := true
  first_correct := by
    have prime : Nat.Prime (corner 2001 1) := by change Nat.Prime 16020007; exact ProjectEulerSolutions.P58PrimeNodes.p_16020007.prime
    exact decide_eq_true prime
  second := false
  second_correct := by
    have composite : ¬ Nat.Prime (corner 2001 2) := by change ¬ Nat.Prime 16016005; exact PrattCertificates.composite_of_product (factor := 5) (cofactor := 3203201) rfl (by decide) (by decide)
    exact decide_eq_false composite
  third := false
  third_correct := by
    have composite : ¬ Nat.Prime (corner 2001 3) := by change ¬ Nat.Prime 16012003; exact PrattCertificates.composite_of_product (factor := 7) (cofactor := 2287429) rfl (by decide) (by decide)
    exact decide_eq_false composite
theorem row_count_2001 : row 2001 = 1 := (row_2001.checked).trans rfl
theorem model_count_2001 : modelRow 2001 = 1 := by decide
theorem modelrow_agrees_2001 : modelRow 2001 = row 2001 := model_count_2001.trans row_count_2001.symm
def row_2002 : RowCertificate 2002 where
  first := false
  first_correct := by
    have composite : ¬ Nat.Prime (corner 2002 1) := by change ¬ Nat.Prime 16036021; exact PrattCertificates.composite_of_product (factor := 31) (cofactor := 517291) rfl (by decide) (by decide)
    exact decide_eq_false composite
  second := false
  second_correct := by
    have composite : ¬ Nat.Prime (corner 2002 2) := by change ¬ Nat.Prime 16032017; exact PrattCertificates.composite_of_product (factor := 457) (cofactor := 35081) rfl (by decide) (by decide)
    exact decide_eq_false composite
  third := false
  third_correct := by
    have composite : ¬ Nat.Prime (corner 2002 3) := by change ¬ Nat.Prime 16028013; exact PrattCertificates.composite_of_product (factor := 3) (cofactor := 5342671) rfl (by decide) (by decide)
    exact decide_eq_false composite
theorem row_count_2002 : row 2002 = 0 := (row_2002.checked).trans rfl
theorem model_count_2002 : modelRow 2002 = 0 := by decide
theorem modelrow_agrees_2002 : modelRow 2002 = row 2002 := model_count_2002.trans row_count_2002.symm
def row_2003 : RowCertificate 2003 where
  first := false
  first_correct := by
    have composite : ¬ Nat.Prime (corner 2003 1) := by change ¬ Nat.Prime 16052043; exact PrattCertificates.composite_of_product (factor := 3) (cofactor := 5350681) rfl (by decide) (by decide)
    exact decide_eq_false composite
  second := true
  second_correct := by
    have prime : Nat.Prime (corner 2003 2) := by change Nat.Prime 16048037; exact ProjectEulerSolutions.P58PrimeNodes.p_16048037.prime
    exact decide_eq_true prime
  third := false
  third_correct := by
    have composite : ¬ Nat.Prime (corner 2003 3) := by change ¬ Nat.Prime 16044031; exact PrattCertificates.composite_of_product (factor := 43) (cofactor := 373117) rfl (by decide) (by decide)
    exact decide_eq_false composite
theorem row_count_2003 : row 2003 = 1 := (row_2003.checked).trans rfl
theorem model_count_2003 : modelRow 2003 = 1 := by decide
theorem modelrow_agrees_2003 : modelRow 2003 = row 2003 := model_count_2003.trans row_count_2003.symm
def row_2004 : RowCertificate 2004 where
  first := false
  first_correct := by
    have composite : ¬ Nat.Prime (corner 2004 1) := by change ¬ Nat.Prime 16068073; exact PrattCertificates.composite_of_product (factor := 7) (cofactor := 2295439) rfl (by decide) (by decide)
    exact decide_eq_false composite
  second := false
  second_correct := by
    have composite : ¬ Nat.Prime (corner 2004 2) := by change ¬ Nat.Prime 16064065; exact PrattCertificates.composite_of_product (factor := 5) (cofactor := 3212813) rfl (by decide) (by decide)
    exact decide_eq_false composite
  third := false
  third_correct := by
    have composite : ¬ Nat.Prime (corner 2004 3) := by change ¬ Nat.Prime 16060057; exact PrattCertificates.composite_of_product (factor := 13) (cofactor := 1235389) rfl (by decide) (by decide)
    exact decide_eq_false composite
theorem row_count_2004 : row 2004 = 0 := (row_2004.checked).trans rfl
theorem model_count_2004 : modelRow 2004 = 0 := by decide
theorem modelrow_agrees_2004 : modelRow 2004 = row 2004 := model_count_2004.trans row_count_2004.symm
def row_2005 : RowCertificate 2005 where
  first := false
  first_correct := by
    have composite : ¬ Nat.Prime (corner 2005 1) := by change ¬ Nat.Prime 16084111; exact PrattCertificates.composite_of_product (factor := 967) (cofactor := 16633) rfl (by decide) (by decide)
    exact decide_eq_false composite
  second := false
  second_correct := by
    have composite : ¬ Nat.Prime (corner 2005 2) := by change ¬ Nat.Prime 16080101; exact PrattCertificates.composite_of_product (factor := 137) (cofactor := 117373) rfl (by decide) (by decide)
    exact decide_eq_false composite
  third := false
  third_correct := by
    have composite : ¬ Nat.Prime (corner 2005 3) := by change ¬ Nat.Prime 16076091; exact PrattCertificates.composite_of_product (factor := 3) (cofactor := 5358697) rfl (by decide) (by decide)
    exact decide_eq_false composite
theorem row_count_2005 : row 2005 = 0 := (row_2005.checked).trans rfl
theorem model_count_2005 : modelRow 2005 = 0 := by decide
theorem modelrow_agrees_2005 : modelRow 2005 = row 2005 := model_count_2005.trans row_count_2005.symm
def row_2006 : RowCertificate 2006 where
  first := false
  first_correct := by
    have composite : ¬ Nat.Prime (corner 2006 1) := by change ¬ Nat.Prime 16100157; exact PrattCertificates.composite_of_product (factor := 3) (cofactor := 5366719) rfl (by decide) (by decide)
    exact decide_eq_false composite
  second := false
  second_correct := by
    have composite : ¬ Nat.Prime (corner 2006 2) := by change ¬ Nat.Prime 16096145; exact PrattCertificates.composite_of_product (factor := 5) (cofactor := 3219229) rfl (by decide) (by decide)
    exact decide_eq_false composite
  third := false
  third_correct := by
    have composite : ¬ Nat.Prime (corner 2006 3) := by change ¬ Nat.Prime 16092133; exact PrattCertificates.composite_of_product (factor := 1237) (cofactor := 13009) rfl (by decide) (by decide)
    exact decide_eq_false composite
theorem row_count_2006 : row 2006 = 0 := (row_2006.checked).trans rfl
theorem model_count_2006 : modelRow 2006 = 0 := by decide
theorem modelrow_agrees_2006 : modelRow 2006 = row 2006 := model_count_2006.trans row_count_2006.symm
def row_2007 : RowCertificate 2007 where
  first := false
  first_correct := by
    have composite : ¬ Nat.Prime (corner 2007 1) := by change ¬ Nat.Prime 16116211; exact PrattCertificates.composite_of_product (factor := 373) (cofactor := 43207) rfl (by decide) (by decide)
    exact decide_eq_false composite
  second := false
  second_correct := by
    have composite : ¬ Nat.Prime (corner 2007 2) := by change ¬ Nat.Prime 16112197; exact PrattCertificates.composite_of_product (factor := 29) (cofactor := 555593) rfl (by decide) (by decide)
    exact decide_eq_false composite
  third := false
  third_correct := by
    have composite : ¬ Nat.Prime (corner 2007 3) := by change ¬ Nat.Prime 16108183; exact PrattCertificates.composite_of_product (factor := 7) (cofactor := 2301169) rfl (by decide) (by decide)
    exact decide_eq_false composite
theorem row_count_2007 : row 2007 = 0 := (row_2007.checked).trans rfl
theorem model_count_2007 : modelRow 2007 = 0 := by decide
theorem modelrow_agrees_2007 : modelRow 2007 = row 2007 := model_count_2007.trans row_count_2007.symm
def row_2008 : RowCertificate 2008 where
  first := false
  first_correct := by
    have composite : ¬ Nat.Prime (corner 2008 1) := by change ¬ Nat.Prime 16132273; exact PrattCertificates.composite_of_product (factor := 19) (cofactor := 849067) rfl (by decide) (by decide)
    exact decide_eq_false composite
  second := false
  second_correct := by
    have composite : ¬ Nat.Prime (corner 2008 2) := by change ¬ Nat.Prime 16128257; exact PrattCertificates.composite_of_product (factor := 17) (cofactor := 948721) rfl (by decide) (by decide)
    exact decide_eq_false composite
  third := false
  third_correct := by
    have composite : ¬ Nat.Prime (corner 2008 3) := by change ¬ Nat.Prime 16124241; exact PrattCertificates.composite_of_product (factor := 3) (cofactor := 5374747) rfl (by decide) (by decide)
    exact decide_eq_false composite
theorem row_count_2008 : row 2008 = 0 := (row_2008.checked).trans rfl
theorem model_count_2008 : modelRow 2008 = 0 := by decide
theorem modelrow_agrees_2008 : modelRow 2008 = row 2008 := model_count_2008.trans row_count_2008.symm
def row_2009 : RowCertificate 2009 where
  first := false
  first_correct := by
    have composite : ¬ Nat.Prime (corner 2009 1) := by change ¬ Nat.Prime 16148343; exact PrattCertificates.composite_of_product (factor := 3) (cofactor := 5382781) rfl (by decide) (by decide)
    exact decide_eq_false composite
  second := false
  second_correct := by
    have composite : ¬ Nat.Prime (corner 2009 2) := by change ¬ Nat.Prime 16144325; exact PrattCertificates.composite_of_product (factor := 5) (cofactor := 3228865) rfl (by decide) (by decide)
    exact decide_eq_false composite
  third := true
  third_correct := by
    have prime : Nat.Prime (corner 2009 3) := by change Nat.Prime 16140307; exact ProjectEulerSolutions.P58PrimeNodes.p_16140307.prime
    exact decide_eq_true prime
theorem row_count_2009 : row 2009 = 1 := (row_2009.checked).trans rfl
theorem model_count_2009 : modelRow 2009 = 1 := by decide
theorem modelrow_agrees_2009 : modelRow 2009 = row 2009 := model_count_2009.trans row_count_2009.symm
def row_2010 : RowCertificate 2010 where
  first := false
  first_correct := by
    have composite : ¬ Nat.Prime (corner 2010 1) := by change ¬ Nat.Prime 16164421; exact PrattCertificates.composite_of_product (factor := 7) (cofactor := 2309203) rfl (by decide) (by decide)
    exact decide_eq_false composite
  second := false
  second_correct := by
    have composite : ¬ Nat.Prime (corner 2010 2) := by change ¬ Nat.Prime 16160401; exact PrattCertificates.composite_of_product (factor := 1901) (cofactor := 8501) rfl (by decide) (by decide)
    exact decide_eq_false composite
  third := false
  third_correct := by
    have composite : ¬ Nat.Prime (corner 2010 3) := by change ¬ Nat.Prime 16156381; exact PrattCertificates.composite_of_product (factor := 829) (cofactor := 19489) rfl (by decide) (by decide)
    exact decide_eq_false composite
theorem row_count_2010 : row 2010 = 0 := (row_2010.checked).trans rfl
theorem model_count_2010 : modelRow 2010 = 0 := by decide
theorem modelrow_agrees_2010 : modelRow 2010 = row 2010 := model_count_2010.trans row_count_2010.symm
def row_2011 : RowCertificate 2011 where
  first := false
  first_correct := by
    have composite : ¬ Nat.Prime (corner 2011 1) := by change ¬ Nat.Prime 16180507; exact PrattCertificates.composite_of_product (factor := 7) (cofactor := 2311501) rfl (by decide) (by decide)
    exact decide_eq_false composite
  second := false
  second_correct := by
    have composite : ¬ Nat.Prime (corner 2011 2) := by change ¬ Nat.Prime 16176485; exact PrattCertificates.composite_of_product (factor := 5) (cofactor := 3235297) rfl (by decide) (by decide)
    exact decide_eq_false composite
  third := false
  third_correct := by
    have composite : ¬ Nat.Prime (corner 2011 3) := by change ¬ Nat.Prime 16172463; exact PrattCertificates.composite_of_product (factor := 3) (cofactor := 5390821) rfl (by decide) (by decide)
    exact decide_eq_false composite
theorem row_count_2011 : row 2011 = 0 := (row_2011.checked).trans rfl
theorem model_count_2011 : modelRow 2011 = 0 := by decide
theorem modelrow_agrees_2011 : modelRow 2011 = row 2011 := model_count_2011.trans row_count_2011.symm
def row_2012 : RowCertificate 2012 where
  first := false
  first_correct := by
    have composite : ¬ Nat.Prime (corner 2012 1) := by change ¬ Nat.Prime 16196601; exact PrattCertificates.composite_of_product (factor := 3) (cofactor := 5398867) rfl (by decide) (by decide)
    exact decide_eq_false composite
  second := true
  second_correct := by
    have prime : Nat.Prime (corner 2012 2) := by change Nat.Prime 16192577; exact ProjectEulerSolutions.P58PrimeNodes.p_16192577.prime
    exact decide_eq_true prime
  third := false
  third_correct := by
    have composite : ¬ Nat.Prime (corner 2012 3) := by change ¬ Nat.Prime 16188553; exact PrattCertificates.composite_of_product (factor := 73) (cofactor := 221761) rfl (by decide) (by decide)
    exact decide_eq_false composite
theorem row_count_2012 : row 2012 = 1 := (row_2012.checked).trans rfl
theorem model_count_2012 : modelRow 2012 = 1 := by decide
theorem modelrow_agrees_2012 : modelRow 2012 = row 2012 := model_count_2012.trans row_count_2012.symm
def row_2013 : RowCertificate 2013 where
  first := false
  first_correct := by
    have composite : ¬ Nat.Prime (corner 2013 1) := by change ¬ Nat.Prime 16212703; exact PrattCertificates.composite_of_product (factor := 13) (cofactor := 1247131) rfl (by decide) (by decide)
    exact decide_eq_false composite
  second := true
  second_correct := by
    have prime : Nat.Prime (corner 2013 2) := by change Nat.Prime 16208677; exact ProjectEulerSolutions.P58PrimeNodes.p_16208677.prime
    exact decide_eq_true prime
  third := true
  third_correct := by
    have prime : Nat.Prime (corner 2013 3) := by change Nat.Prime 16204651; exact ProjectEulerSolutions.P58PrimeNodes.p_16204651.prime
    exact decide_eq_true prime
theorem row_count_2013 : row 2013 = 2 := (row_2013.checked).trans rfl
theorem model_count_2013 : modelRow 2013 = 2 := by decide
theorem modelrow_agrees_2013 : modelRow 2013 = row 2013 := model_count_2013.trans row_count_2013.symm
def row_2014 : RowCertificate 2014 where
  first := true
  first_correct := by
    have prime : Nat.Prime (corner 2014 1) := by change Nat.Prime 16228813; exact ProjectEulerSolutions.P58PrimeNodes.p_16228813.prime
    exact decide_eq_true prime
  second := false
  second_correct := by
    have composite : ¬ Nat.Prime (corner 2014 2) := by change ¬ Nat.Prime 16224785; exact PrattCertificates.composite_of_product (factor := 5) (cofactor := 3244957) rfl (by decide) (by decide)
    exact decide_eq_false composite
  third := false
  third_correct := by
    have composite : ¬ Nat.Prime (corner 2014 3) := by change ¬ Nat.Prime 16220757; exact PrattCertificates.composite_of_product (factor := 3) (cofactor := 5406919) rfl (by decide) (by decide)
    exact decide_eq_false composite
theorem row_count_2014 : row 2014 = 1 := (row_2014.checked).trans rfl
theorem model_count_2014 : modelRow 2014 = 1 := by decide
theorem modelrow_agrees_2014 : modelRow 2014 = row 2014 := model_count_2014.trans row_count_2014.symm
def row_2015 : RowCertificate 2015 where
  first := false
  first_correct := by
    have composite : ¬ Nat.Prime (corner 2015 1) := by change ¬ Nat.Prime 16244931; exact PrattCertificates.composite_of_product (factor := 3) (cofactor := 5414977) rfl (by decide) (by decide)
    exact decide_eq_false composite
  second := false
  second_correct := by
    have composite : ¬ Nat.Prime (corner 2015 2) := by change ¬ Nat.Prime 16240901; exact PrattCertificates.composite_of_product (factor := 101) (cofactor := 160801) rfl (by decide) (by decide)
    exact decide_eq_false composite
  third := false
  third_correct := by
    have composite : ¬ Nat.Prime (corner 2015 3) := by change ¬ Nat.Prime 16236871; exact PrattCertificates.composite_of_product (factor := 7) (cofactor := 2319553) rfl (by decide) (by decide)
    exact decide_eq_false composite
theorem row_count_2015 : row 2015 = 0 := (row_2015.checked).trans rfl
theorem model_count_2015 : modelRow 2015 = 0 := by decide
theorem modelrow_agrees_2015 : modelRow 2015 = row 2015 := model_count_2015.trans row_count_2015.symm
def row_2016 : RowCertificate 2016 where
  first := false
  first_correct := by
    have composite : ¬ Nat.Prime (corner 2016 1) := by change ¬ Nat.Prime 16261057; exact PrattCertificates.composite_of_product (factor := 2377) (cofactor := 6841) rfl (by decide) (by decide)
    exact decide_eq_false composite
  second := false
  second_correct := by
    have composite : ¬ Nat.Prime (corner 2016 2) := by change ¬ Nat.Prime 16257025; exact PrattCertificates.composite_of_product (factor := 5) (cofactor := 3251405) rfl (by decide) (by decide)
    exact decide_eq_false composite
  third := true
  third_correct := by
    have prime : Nat.Prime (corner 2016 3) := by change Nat.Prime 16252993; exact ProjectEulerSolutions.P58PrimeNodes.p_16252993.prime
    exact decide_eq_true prime
theorem row_count_2016 : row 2016 = 1 := (row_2016.checked).trans rfl
theorem model_count_2016 : modelRow 2016 = 1 := by decide
theorem modelrow_agrees_2016 : modelRow 2016 = row 2016 := model_count_2016.trans row_count_2016.symm
def row_2017 : RowCertificate 2017 where
  first := false
  first_correct := by
    have composite : ¬ Nat.Prime (corner 2017 1) := by change ¬ Nat.Prime 16277191; exact PrattCertificates.composite_of_product (factor := 7) (cofactor := 2325313) rfl (by decide) (by decide)
    exact decide_eq_false composite
  second := true
  second_correct := by
    have prime : Nat.Prime (corner 2017 2) := by change Nat.Prime 16273157; exact ProjectEulerSolutions.P58PrimeNodes.p_16273157.prime
    exact decide_eq_true prime
  third := false
  third_correct := by
    have composite : ¬ Nat.Prime (corner 2017 3) := by change ¬ Nat.Prime 16269123; exact PrattCertificates.composite_of_product (factor := 3) (cofactor := 5423041) rfl (by decide) (by decide)
    exact decide_eq_false composite
theorem row_count_2017 : row 2017 = 1 := (row_2017.checked).trans rfl
theorem model_count_2017 : modelRow 2017 = 1 := by decide
theorem modelrow_agrees_2017 : modelRow 2017 = row 2017 := model_count_2017.trans row_count_2017.symm
def row_2018 : RowCertificate 2018 where
  first := false
  first_correct := by
    have composite : ¬ Nat.Prime (corner 2018 1) := by change ¬ Nat.Prime 16293333; exact PrattCertificates.composite_of_product (factor := 3) (cofactor := 5431111) rfl (by decide) (by decide)
    exact decide_eq_false composite
  second := false
  second_correct := by
    have composite : ¬ Nat.Prime (corner 2018 2) := by change ¬ Nat.Prime 16289297; exact PrattCertificates.composite_of_product (factor := 677) (cofactor := 24061) rfl (by decide) (by decide)
    exact decide_eq_false composite
  third := false
  third_correct := by
    have composite : ¬ Nat.Prime (corner 2018 3) := by change ¬ Nat.Prime 16285261; exact PrattCertificates.composite_of_product (factor := 19) (cofactor := 857119) rfl (by decide) (by decide)
    exact decide_eq_false composite
theorem row_count_2018 : row 2018 = 0 := (row_2018.checked).trans rfl
theorem model_count_2018 : modelRow 2018 = 0 := by decide
theorem modelrow_agrees_2018 : modelRow 2018 = row 2018 := model_count_2018.trans row_count_2018.symm
def row_2019 : RowCertificate 2019 where
  first := false
  first_correct := by
    have composite : ¬ Nat.Prime (corner 2019 1) := by change ¬ Nat.Prime 16309483; exact PrattCertificates.composite_of_product (factor := 97) (cofactor := 168139) rfl (by decide) (by decide)
    exact decide_eq_false composite
  second := false
  second_correct := by
    have composite : ¬ Nat.Prime (corner 2019 2) := by change ¬ Nat.Prime 16305445; exact PrattCertificates.composite_of_product (factor := 5) (cofactor := 3261089) rfl (by decide) (by decide)
    exact decide_eq_false composite
  third := false
  third_correct := by
    have composite : ¬ Nat.Prime (corner 2019 3) := by change ¬ Nat.Prime 16301407; exact PrattCertificates.composite_of_product (factor := 1459) (cofactor := 11173) rfl (by decide) (by decide)
    exact decide_eq_false composite
theorem row_count_2019 : row 2019 = 0 := (row_2019.checked).trans rfl
theorem model_count_2019 : modelRow 2019 = 0 := by decide
theorem modelrow_agrees_2019 : modelRow 2019 = row 2019 := model_count_2019.trans row_count_2019.symm
def row_2020 : RowCertificate 2020 where
  first := false
  first_correct := by
    have composite : ¬ Nat.Prime (corner 2020 1) := by change ¬ Nat.Prime 16325641; exact PrattCertificates.composite_of_product (factor := 3067) (cofactor := 5323) rfl (by decide) (by decide)
    exact decide_eq_false composite
  second := false
  second_correct := by
    have composite : ¬ Nat.Prime (corner 2020 2) := by change ¬ Nat.Prime 16321601; exact PrattCertificates.composite_of_product (factor := 929) (cofactor := 17569) rfl (by decide) (by decide)
    exact decide_eq_false composite
  third := false
  third_correct := by
    have composite : ¬ Nat.Prime (corner 2020 3) := by change ¬ Nat.Prime 16317561; exact PrattCertificates.composite_of_product (factor := 3) (cofactor := 5439187) rfl (by decide) (by decide)
    exact decide_eq_false composite
theorem row_count_2020 : row 2020 = 0 := (row_2020.checked).trans rfl
theorem model_count_2020 : modelRow 2020 = 0 := by decide
theorem modelrow_agrees_2020 : modelRow 2020 = row 2020 := model_count_2020.trans row_count_2020.symm
def row_2021 : RowCertificate 2021 where
  first := false
  first_correct := by
    have composite : ¬ Nat.Prime (corner 2021 1) := by change ¬ Nat.Prime 16341807; exact PrattCertificates.composite_of_product (factor := 3) (cofactor := 5447269) rfl (by decide) (by decide)
    exact decide_eq_false composite
  second := false
  second_correct := by
    have composite : ¬ Nat.Prime (corner 2021 2) := by change ¬ Nat.Prime 16337765; exact PrattCertificates.composite_of_product (factor := 5) (cofactor := 3267553) rfl (by decide) (by decide)
    exact decide_eq_false composite
  third := false
  third_correct := by
    have composite : ¬ Nat.Prime (corner 2021 3) := by change ¬ Nat.Prime 16333723; exact PrattCertificates.composite_of_product (factor := 7) (cofactor := 2333389) rfl (by decide) (by decide)
    exact decide_eq_false composite
theorem row_count_2021 : row 2021 = 0 := (row_2021.checked).trans rfl
theorem model_count_2021 : modelRow 2021 = 0 := by decide
theorem modelrow_agrees_2021 : modelRow 2021 = row 2021 := model_count_2021.trans row_count_2021.symm
def row_2022 : RowCertificate 2022 where
  first := false
  first_correct := by
    have composite : ¬ Nat.Prime (corner 2022 1) := by change ¬ Nat.Prime 16357981; exact PrattCertificates.composite_of_product (factor := 127) (cofactor := 128803) rfl (by decide) (by decide)
    exact decide_eq_false composite
  second := false
  second_correct := by
    have composite : ¬ Nat.Prime (corner 2022 2) := by change ¬ Nat.Prime 16353937; exact PrattCertificates.composite_of_product (factor := 313) (cofactor := 52249) rfl (by decide) (by decide)
    exact decide_eq_false composite
  third := false
  third_correct := by
    have composite : ¬ Nat.Prime (corner 2022 3) := by change ¬ Nat.Prime 16349893; exact PrattCertificates.composite_of_product (factor := 7) (cofactor := 2335699) rfl (by decide) (by decide)
    exact decide_eq_false composite
theorem row_count_2022 : row 2022 = 0 := (row_2022.checked).trans rfl
theorem model_count_2022 : modelRow 2022 = 0 := by decide
theorem modelrow_agrees_2022 : modelRow 2022 = row 2022 := model_count_2022.trans row_count_2022.symm
def row_2023 : RowCertificate 2023 where
  first := false
  first_correct := by
    have composite : ¬ Nat.Prime (corner 2023 1) := by change ¬ Nat.Prime 16374163; exact PrattCertificates.composite_of_product (factor := 13) (cofactor := 1259551) rfl (by decide) (by decide)
    exact decide_eq_false composite
  second := true
  second_correct := by
    have prime : Nat.Prime (corner 2023 2) := by change Nat.Prime 16370117; exact ProjectEulerSolutions.P58PrimeNodes.p_16370117.prime
    exact decide_eq_true prime
  third := false
  third_correct := by
    have composite : ¬ Nat.Prime (corner 2023 3) := by change ¬ Nat.Prime 16366071; exact PrattCertificates.composite_of_product (factor := 3) (cofactor := 5455357) rfl (by decide) (by decide)
    exact decide_eq_false composite
theorem row_count_2023 : row 2023 = 1 := (row_2023.checked).trans rfl
theorem model_count_2023 : modelRow 2023 = 1 := by decide
theorem modelrow_agrees_2023 : modelRow 2023 = row 2023 := model_count_2023.trans row_count_2023.symm
def row_2024 : RowCertificate 2024 where
  first := false
  first_correct := by
    have composite : ¬ Nat.Prime (corner 2024 1) := by change ¬ Nat.Prime 16390353; exact PrattCertificates.composite_of_product (factor := 3) (cofactor := 5463451) rfl (by decide) (by decide)
    exact decide_eq_false composite
  second := false
  second_correct := by
    have composite : ¬ Nat.Prime (corner 2024 2) := by change ¬ Nat.Prime 16386305; exact PrattCertificates.composite_of_product (factor := 5) (cofactor := 3277261) rfl (by decide) (by decide)
    exact decide_eq_false composite
  third := true
  third_correct := by
    have prime : Nat.Prime (corner 2024 3) := by change Nat.Prime 16382257; exact ProjectEulerSolutions.P58PrimeNodes.p_16382257.prime
    exact decide_eq_true prime
theorem row_count_2024 : row 2024 = 1 := (row_2024.checked).trans rfl
theorem model_count_2024 : modelRow 2024 = 1 := by decide
theorem modelrow_agrees_2024 : modelRow 2024 = row 2024 := model_count_2024.trans row_count_2024.symm
def row_2025 : RowCertificate 2025 where
  first := false
  first_correct := by
    have composite : ¬ Nat.Prime (corner 2025 1) := by change ¬ Nat.Prime 16406551; exact PrattCertificates.composite_of_product (factor := 7) (cofactor := 2343793) rfl (by decide) (by decide)
    exact decide_eq_false composite
  second := false
  second_correct := by
    have composite : ¬ Nat.Prime (corner 2025 2) := by change ¬ Nat.Prime 16402501; exact PrattCertificates.composite_of_product (factor := 17) (cofactor := 964853) rfl (by decide) (by decide)
    exact decide_eq_false composite
  third := false
  third_correct := by
    have composite : ¬ Nat.Prime (corner 2025 3) := by change ¬ Nat.Prime 16398451; exact PrattCertificates.composite_of_product (factor := 67) (cofactor := 244753) rfl (by decide) (by decide)
    exact decide_eq_false composite
theorem row_count_2025 : row 2025 = 0 := (row_2025.checked).trans rfl
theorem model_count_2025 : modelRow 2025 = 0 := by decide
theorem modelrow_agrees_2025 : modelRow 2025 = row 2025 := model_count_2025.trans row_count_2025.symm
def row_2026 : RowCertificate 2026 where
  first := false
  first_correct := by
    have composite : ¬ Nat.Prime (corner 2026 1) := by change ¬ Nat.Prime 16422757; exact PrattCertificates.composite_of_product (factor := 13) (cofactor := 1263289) rfl (by decide) (by decide)
    exact decide_eq_false composite
  second := false
  second_correct := by
    have composite : ¬ Nat.Prime (corner 2026 2) := by change ¬ Nat.Prime 16418705; exact PrattCertificates.composite_of_product (factor := 5) (cofactor := 3283741) rfl (by decide) (by decide)
    exact decide_eq_false composite
  third := false
  third_correct := by
    have composite : ¬ Nat.Prime (corner 2026 3) := by change ¬ Nat.Prime 16414653; exact PrattCertificates.composite_of_product (factor := 3) (cofactor := 5471551) rfl (by decide) (by decide)
    exact decide_eq_false composite
theorem row_count_2026 : row 2026 = 0 := (row_2026.checked).trans rfl
theorem model_count_2026 : modelRow 2026 = 0 := by decide
theorem modelrow_agrees_2026 : modelRow 2026 = row 2026 := model_count_2026.trans row_count_2026.symm
def row_2027 : RowCertificate 2027 where
  first := false
  first_correct := by
    have composite : ¬ Nat.Prime (corner 2027 1) := by change ¬ Nat.Prime 16438971; exact PrattCertificates.composite_of_product (factor := 3) (cofactor := 5479657) rfl (by decide) (by decide)
    exact decide_eq_false composite
  second := false
  second_correct := by
    have composite : ¬ Nat.Prime (corner 2027 2) := by change ¬ Nat.Prime 16434917; exact PrattCertificates.composite_of_product (factor := 157) (cofactor := 104681) rfl (by decide) (by decide)
    exact decide_eq_false composite
  third := false
  third_correct := by
    have composite : ¬ Nat.Prime (corner 2027 3) := by change ¬ Nat.Prime 16430863; exact PrattCertificates.composite_of_product (factor := 223) (cofactor := 73681) rfl (by decide) (by decide)
    exact decide_eq_false composite
theorem row_count_2027 : row 2027 = 0 := (row_2027.checked).trans rfl
theorem model_count_2027 : modelRow 2027 = 0 := by decide
theorem modelrow_agrees_2027 : modelRow 2027 = row 2027 := model_count_2027.trans row_count_2027.symm
def row_2028 : RowCertificate 2028 where
  first := true
  first_correct := by
    have prime : Nat.Prime (corner 2028 1) := by change Nat.Prime 16455193; exact ProjectEulerSolutions.P58PrimeNodes.p_16455193.prime
    exact decide_eq_true prime
  second := true
  second_correct := by
    have prime : Nat.Prime (corner 2028 2) := by change Nat.Prime 16451137; exact ProjectEulerSolutions.P58PrimeNodes.p_16451137.prime
    exact decide_eq_true prime
  third := false
  third_correct := by
    have composite : ¬ Nat.Prime (corner 2028 3) := by change ¬ Nat.Prime 16447081; exact PrattCertificates.composite_of_product (factor := 7) (cofactor := 2349583) rfl (by decide) (by decide)
    exact decide_eq_false composite
theorem row_count_2028 : row 2028 = 2 := (row_2028.checked).trans rfl
theorem model_count_2028 : modelRow 2028 = 2 := by decide
theorem modelrow_agrees_2028 : modelRow 2028 = row 2028 := model_count_2028.trans row_count_2028.symm
def row_2029 : RowCertificate 2029 where
  first := false
  first_correct := by
    have composite : ¬ Nat.Prime (corner 2029 1) := by change ¬ Nat.Prime 16471423; exact PrattCertificates.composite_of_product (factor := 19) (cofactor := 866917) rfl (by decide) (by decide)
    exact decide_eq_false composite
  second := false
  second_correct := by
    have composite : ¬ Nat.Prime (corner 2029 2) := by change ¬ Nat.Prime 16467365; exact PrattCertificates.composite_of_product (factor := 5) (cofactor := 3293473) rfl (by decide) (by decide)
    exact decide_eq_false composite
  third := false
  third_correct := by
    have composite : ¬ Nat.Prime (corner 2029 3) := by change ¬ Nat.Prime 16463307; exact PrattCertificates.composite_of_product (factor := 3) (cofactor := 5487769) rfl (by decide) (by decide)
    exact decide_eq_false composite
theorem row_count_2029 : row 2029 = 0 := (row_2029.checked).trans rfl
theorem model_count_2029 : modelRow 2029 = 0 := by decide
theorem modelrow_agrees_2029 : modelRow 2029 = row 2029 := model_count_2029.trans row_count_2029.symm
def row_2030 : RowCertificate 2030 where
  first := false
  first_correct := by
    have composite : ¬ Nat.Prime (corner 2030 1) := by change ¬ Nat.Prime 16487661; exact PrattCertificates.composite_of_product (factor := 3) (cofactor := 5495887) rfl (by decide) (by decide)
    exact decide_eq_false composite
  second := false
  second_correct := by
    have composite : ¬ Nat.Prime (corner 2030 2) := by change ¬ Nat.Prime 16483601; exact PrattCertificates.composite_of_product (factor := 89) (cofactor := 185209) rfl (by decide) (by decide)
    exact decide_eq_false composite
  third := false
  third_correct := by
    have composite : ¬ Nat.Prime (corner 2030 3) := by change ¬ Nat.Prime 16479541; exact PrattCertificates.composite_of_product (factor := 13) (cofactor := 1267657) rfl (by decide) (by decide)
    exact decide_eq_false composite
theorem row_count_2030 : row 2030 = 0 := (row_2030.checked).trans rfl
theorem model_count_2030 : modelRow 2030 = 0 := by decide
theorem modelrow_agrees_2030 : modelRow 2030 = row 2030 := model_count_2030.trans row_count_2030.symm
def row_2031 : RowCertificate 2031 where
  first := false
  first_correct := by
    have composite : ¬ Nat.Prime (corner 2031 1) := by change ¬ Nat.Prime 16503907; exact PrattCertificates.composite_of_product (factor := 7) (cofactor := 2357701) rfl (by decide) (by decide)
    exact decide_eq_false composite
  second := false
  second_correct := by
    have composite : ¬ Nat.Prime (corner 2031 2) := by change ¬ Nat.Prime 16499845; exact PrattCertificates.composite_of_product (factor := 5) (cofactor := 3299969) rfl (by decide) (by decide)
    exact decide_eq_false composite
  third := true
  third_correct := by
    have prime : Nat.Prime (corner 2031 3) := by change Nat.Prime 16495783; exact ProjectEulerSolutions.P58PrimeNodes.p_16495783.prime
    exact decide_eq_true prime
theorem row_count_2031 : row 2031 = 1 := (row_2031.checked).trans rfl
theorem model_count_2031 : modelRow 2031 = 1 := by decide
theorem modelrow_agrees_2031 : modelRow 2031 = row 2031 := model_count_2031.trans row_count_2031.symm
def row_2032 : RowCertificate 2032 where
  first := false
  first_correct := by
    have composite : ¬ Nat.Prime (corner 2032 1) := by change ¬ Nat.Prime 16520161; exact PrattCertificates.composite_of_product (factor := 7) (cofactor := 2360023) rfl (by decide) (by decide)
    exact decide_eq_false composite
  second := false
  second_correct := by
    have composite : ¬ Nat.Prime (corner 2032 2) := by change ¬ Nat.Prime 16516097; exact PrattCertificates.composite_of_product (factor := 13) (cofactor := 1270469) rfl (by decide) (by decide)
    exact decide_eq_false composite
  third := false
  third_correct := by
    have composite : ¬ Nat.Prime (corner 2032 3) := by change ¬ Nat.Prime 16512033; exact PrattCertificates.composite_of_product (factor := 3) (cofactor := 5504011) rfl (by decide) (by decide)
    exact decide_eq_false composite
theorem row_count_2032 : row 2032 = 0 := (row_2032.checked).trans rfl
theorem model_count_2032 : modelRow 2032 = 0 := by decide
theorem modelrow_agrees_2032 : modelRow 2032 = row 2032 := model_count_2032.trans row_count_2032.symm
def row_2033 : RowCertificate 2033 where
  first := false
  first_correct := by
    have composite : ¬ Nat.Prime (corner 2033 1) := by change ¬ Nat.Prime 16536423; exact PrattCertificates.composite_of_product (factor := 3) (cofactor := 5512141) rfl (by decide) (by decide)
    exact decide_eq_false composite
  second := false
  second_correct := by
    have composite : ¬ Nat.Prime (corner 2033 2) := by change ¬ Nat.Prime 16532357; exact PrattCertificates.composite_of_product (factor := 109) (cofactor := 151673) rfl (by decide) (by decide)
    exact decide_eq_false composite
  third := false
  third_correct := by
    have composite : ¬ Nat.Prime (corner 2033 3) := by change ¬ Nat.Prime 16528291; exact PrattCertificates.composite_of_product (factor := 13) (cofactor := 1271407) rfl (by decide) (by decide)
    exact decide_eq_false composite
theorem row_count_2033 : row 2033 = 0 := (row_2033.checked).trans rfl
theorem model_count_2033 : modelRow 2033 = 0 := by decide
theorem modelrow_agrees_2033 : modelRow 2033 = row 2033 := model_count_2033.trans row_count_2033.symm
def row_2034 : RowCertificate 2034 where
  first := true
  first_correct := by
    have prime : Nat.Prime (corner 2034 1) := by change Nat.Prime 16552693; exact ProjectEulerSolutions.P58PrimeNodes.p_16552693.prime
    exact decide_eq_true prime
  second := false
  second_correct := by
    have composite : ¬ Nat.Prime (corner 2034 2) := by change ¬ Nat.Prime 16548625; exact PrattCertificates.composite_of_product (factor := 5) (cofactor := 3309725) rfl (by decide) (by decide)
    exact decide_eq_false composite
  third := true
  third_correct := by
    have prime : Nat.Prime (corner 2034 3) := by change Nat.Prime 16544557; exact ProjectEulerSolutions.P58PrimeNodes.p_16544557.prime
    exact decide_eq_true prime
theorem row_count_2034 : row 2034 = 2 := (row_2034.checked).trans rfl
theorem model_count_2034 : modelRow 2034 = 2 := by decide
theorem modelrow_agrees_2034 : modelRow 2034 = row 2034 := model_count_2034.trans row_count_2034.symm
def row_2035 : RowCertificate 2035 where
  first := false
  first_correct := by
    have composite : ¬ Nat.Prime (corner 2035 1) := by change ¬ Nat.Prime 16568971; exact PrattCertificates.composite_of_product (factor := 937) (cofactor := 17683) rfl (by decide) (by decide)
    exact decide_eq_false composite
  second := true
  second_correct := by
    have prime : Nat.Prime (corner 2035 2) := by change Nat.Prime 16564901; exact ProjectEulerSolutions.P58PrimeNodes.p_16564901.prime
    exact decide_eq_true prime
  third := false
  third_correct := by
    have composite : ¬ Nat.Prime (corner 2035 3) := by change ¬ Nat.Prime 16560831; exact PrattCertificates.composite_of_product (factor := 3) (cofactor := 5520277) rfl (by decide) (by decide)
    exact decide_eq_false composite
theorem row_count_2035 : row 2035 = 1 := (row_2035.checked).trans rfl
theorem model_count_2035 : modelRow 2035 = 1 := by decide
theorem modelrow_agrees_2035 : modelRow 2035 = row 2035 := model_count_2035.trans row_count_2035.symm
def row_2036 : RowCertificate 2036 where
  first := false
  first_correct := by
    have composite : ¬ Nat.Prime (corner 2036 1) := by change ¬ Nat.Prime 16585257; exact PrattCertificates.composite_of_product (factor := 3) (cofactor := 5528419) rfl (by decide) (by decide)
    exact decide_eq_false composite
  second := false
  second_correct := by
    have composite : ¬ Nat.Prime (corner 2036 2) := by change ¬ Nat.Prime 16581185; exact PrattCertificates.composite_of_product (factor := 5) (cofactor := 3316237) rfl (by decide) (by decide)
    exact decide_eq_false composite
  third := false
  third_correct := by
    have composite : ¬ Nat.Prime (corner 2036 3) := by change ¬ Nat.Prime 16577113; exact PrattCertificates.composite_of_product (factor := 7) (cofactor := 2368159) rfl (by decide) (by decide)
    exact decide_eq_false composite
theorem row_count_2036 : row 2036 = 0 := (row_2036.checked).trans rfl
theorem model_count_2036 : modelRow 2036 = 0 := by decide
theorem modelrow_agrees_2036 : modelRow 2036 = row 2036 := model_count_2036.trans row_count_2036.symm
def row_2037 : RowCertificate 2037 where
  first := true
  first_correct := by
    have prime : Nat.Prime (corner 2037 1) := by change Nat.Prime 16601551; exact ProjectEulerSolutions.P58PrimeNodes.p_16601551.prime
    exact decide_eq_true prime
  second := false
  second_correct := by
    have composite : ¬ Nat.Prime (corner 2037 2) := by change ¬ Nat.Prime 16597477; exact PrattCertificates.composite_of_product (factor := 13) (cofactor := 1276729) rfl (by decide) (by decide)
    exact decide_eq_false composite
  third := false
  third_correct := by
    have composite : ¬ Nat.Prime (corner 2037 3) := by change ¬ Nat.Prime 16593403; exact PrattCertificates.composite_of_product (factor := 19) (cofactor := 873337) rfl (by decide) (by decide)
    exact decide_eq_false composite
theorem row_count_2037 : row 2037 = 1 := (row_2037.checked).trans rfl
theorem model_count_2037 : modelRow 2037 = 1 := by decide
theorem modelrow_agrees_2037 : modelRow 2037 = row 2037 := model_count_2037.trans row_count_2037.symm
def row_2038 : RowCertificate 2038 where
  first := false
  first_correct := by
    have composite : ¬ Nat.Prime (corner 2038 1) := by change ¬ Nat.Prime 16617853; exact PrattCertificates.composite_of_product (factor := 7) (cofactor := 2373979) rfl (by decide) (by decide)
    exact decide_eq_false composite
  second := false
  second_correct := by
    have composite : ¬ Nat.Prime (corner 2038 2) := by change ¬ Nat.Prime 16613777; exact PrattCertificates.composite_of_product (factor := 17) (cofactor := 977281) rfl (by decide) (by decide)
    exact decide_eq_false composite
  third := false
  third_correct := by
    have composite : ¬ Nat.Prime (corner 2038 3) := by change ¬ Nat.Prime 16609701; exact PrattCertificates.composite_of_product (factor := 3) (cofactor := 5536567) rfl (by decide) (by decide)
    exact decide_eq_false composite
theorem row_count_2038 : row 2038 = 0 := (row_2038.checked).trans rfl
theorem model_count_2038 : modelRow 2038 = 0 := by decide
theorem modelrow_agrees_2038 : modelRow 2038 = row 2038 := model_count_2038.trans row_count_2038.symm
def row_2039 : RowCertificate 2039 where
  first := false
  first_correct := by
    have composite : ¬ Nat.Prime (corner 2039 1) := by change ¬ Nat.Prime 16634163; exact PrattCertificates.composite_of_product (factor := 3) (cofactor := 5544721) rfl (by decide) (by decide)
    exact decide_eq_false composite
  second := false
  second_correct := by
    have composite : ¬ Nat.Prime (corner 2039 2) := by change ¬ Nat.Prime 16630085; exact PrattCertificates.composite_of_product (factor := 5) (cofactor := 3326017) rfl (by decide) (by decide)
    exact decide_eq_false composite
  third := false
  third_correct := by
    have composite : ¬ Nat.Prime (corner 2039 3) := by change ¬ Nat.Prime 16626007; exact PrattCertificates.composite_of_product (factor := 19) (cofactor := 875053) rfl (by decide) (by decide)
    exact decide_eq_false composite
theorem row_count_2039 : row 2039 = 0 := (row_2039.checked).trans rfl
theorem model_count_2039 : modelRow 2039 = 0 := by decide
theorem modelrow_agrees_2039 : modelRow 2039 = row 2039 := model_count_2039.trans row_count_2039.symm
def row_2040 : RowCertificate 2040 where
  first := false
  first_correct := by
    have composite : ¬ Nat.Prime (corner 2040 1) := by change ¬ Nat.Prime 16650481; exact PrattCertificates.composite_of_product (factor := 37) (cofactor := 450013) rfl (by decide) (by decide)
    exact decide_eq_false composite
  second := true
  second_correct := by
    have prime : Nat.Prime (corner 2040 2) := by change Nat.Prime 16646401; exact ProjectEulerSolutions.P58PrimeNodes.p_16646401.prime
    exact decide_eq_true prime
  third := false
  third_correct := by
    have composite : ¬ Nat.Prime (corner 2040 3) := by change ¬ Nat.Prime 16642321; exact PrattCertificates.composite_of_product (factor := 73) (cofactor := 227977) rfl (by decide) (by decide)
    exact decide_eq_false composite
theorem row_count_2040 : row 2040 = 1 := (row_2040.checked).trans rfl
theorem model_count_2040 : modelRow 2040 = 1 := by decide
theorem modelrow_agrees_2040 : modelRow 2040 = row 2040 := model_count_2040.trans row_count_2040.symm
def row_2041 : RowCertificate 2041 where
  first := true
  first_correct := by
    have prime : Nat.Prime (corner 2041 1) := by change Nat.Prime 16666807; exact ProjectEulerSolutions.P58PrimeNodes.p_16666807.prime
    exact decide_eq_true prime
  second := false
  second_correct := by
    have composite : ¬ Nat.Prime (corner 2041 2) := by change ¬ Nat.Prime 16662725; exact PrattCertificates.composite_of_product (factor := 5) (cofactor := 3332545) rfl (by decide) (by decide)
    exact decide_eq_false composite
  third := false
  third_correct := by
    have composite : ¬ Nat.Prime (corner 2041 3) := by change ¬ Nat.Prime 16658643; exact PrattCertificates.composite_of_product (factor := 3) (cofactor := 5552881) rfl (by decide) (by decide)
    exact decide_eq_false composite
theorem row_count_2041 : row 2041 = 1 := (row_2041.checked).trans rfl
theorem model_count_2041 : modelRow 2041 = 1 := by decide
theorem modelrow_agrees_2041 : modelRow 2041 = row 2041 := model_count_2041.trans row_count_2041.symm
def row_2042 : RowCertificate 2042 where
  first := false
  first_correct := by
    have composite : ¬ Nat.Prime (corner 2042 1) := by change ¬ Nat.Prime 16683141; exact PrattCertificates.composite_of_product (factor := 3) (cofactor := 5561047) rfl (by decide) (by decide)
    exact decide_eq_false composite
  second := false
  second_correct := by
    have composite : ¬ Nat.Prime (corner 2042 2) := by change ¬ Nat.Prime 16679057; exact PrattCertificates.composite_of_product (factor := 17) (cofactor := 981121) rfl (by decide) (by decide)
    exact decide_eq_false composite
  third := false
  third_correct := by
    have composite : ¬ Nat.Prime (corner 2042 3) := by change ¬ Nat.Prime 16674973; exact PrattCertificates.composite_of_product (factor := 7) (cofactor := 2382139) rfl (by decide) (by decide)
    exact decide_eq_false composite
theorem row_count_2042 : row 2042 = 0 := (row_2042.checked).trans rfl
theorem model_count_2042 : modelRow 2042 = 0 := by decide
theorem modelrow_agrees_2042 : modelRow 2042 = row 2042 := model_count_2042.trans row_count_2042.symm
def row_2043 : RowCertificate 2043 where
  first := false
  first_correct := by
    have composite : ¬ Nat.Prime (corner 2043 1) := by change ¬ Nat.Prime 16699483; exact PrattCertificates.composite_of_product (factor := 31) (cofactor := 538693) rfl (by decide) (by decide)
    exact decide_eq_false composite
  second := true
  second_correct := by
    have prime : Nat.Prime (corner 2043 2) := by change Nat.Prime 16695397; exact ProjectEulerSolutions.P58PrimeNodes.p_16695397.prime
    exact decide_eq_true prime
  third := false
  third_correct := by
    have composite : ¬ Nat.Prime (corner 2043 3) := by change ¬ Nat.Prime 16691311; exact PrattCertificates.composite_of_product (factor := 7) (cofactor := 2384473) rfl (by decide) (by decide)
    exact decide_eq_false composite
theorem row_count_2043 : row 2043 = 1 := (row_2043.checked).trans rfl
theorem model_count_2043 : modelRow 2043 = 1 := by decide
theorem modelrow_agrees_2043 : modelRow 2043 = row 2043 := model_count_2043.trans row_count_2043.symm
def row_2044 : RowCertificate 2044 where
  first := false
  first_correct := by
    have composite : ¬ Nat.Prime (corner 2044 1) := by change ¬ Nat.Prime 16715833; exact PrattCertificates.composite_of_product (factor := 2389) (cofactor := 6997) rfl (by decide) (by decide)
    exact decide_eq_false composite
  second := false
  second_correct := by
    have composite : ¬ Nat.Prime (corner 2044 2) := by change ¬ Nat.Prime 16711745; exact PrattCertificates.composite_of_product (factor := 5) (cofactor := 3342349) rfl (by decide) (by decide)
    exact decide_eq_false composite
  third := false
  third_correct := by
    have composite : ¬ Nat.Prime (corner 2044 3) := by change ¬ Nat.Prime 16707657; exact PrattCertificates.composite_of_product (factor := 3) (cofactor := 5569219) rfl (by decide) (by decide)
    exact decide_eq_false composite
theorem row_count_2044 : row 2044 = 0 := (row_2044.checked).trans rfl
theorem model_count_2044 : modelRow 2044 = 0 := by decide
theorem modelrow_agrees_2044 : modelRow 2044 = row 2044 := model_count_2044.trans row_count_2044.symm
def row_2045 : RowCertificate 2045 where
  first := false
  first_correct := by
    have composite : ¬ Nat.Prime (corner 2045 1) := by change ¬ Nat.Prime 16732191; exact PrattCertificates.composite_of_product (factor := 3) (cofactor := 5577397) rfl (by decide) (by decide)
    exact decide_eq_false composite
  second := false
  second_correct := by
    have composite : ¬ Nat.Prime (corner 2045 2) := by change ¬ Nat.Prime 16728101; exact PrattCertificates.composite_of_product (factor := 13) (cofactor := 1286777) rfl (by decide) (by decide)
    exact decide_eq_false composite
  third := true
  third_correct := by
    have prime : Nat.Prime (corner 2045 3) := by change Nat.Prime 16724011; exact ProjectEulerSolutions.P58PrimeNodes.p_16724011.prime
    exact decide_eq_true prime
theorem row_count_2045 : row 2045 = 1 := (row_2045.checked).trans rfl
theorem model_count_2045 : modelRow 2045 = 1 := by decide
theorem modelrow_agrees_2045 : modelRow 2045 = row 2045 := model_count_2045.trans row_count_2045.symm
def row_2046 : RowCertificate 2046 where
  first := false
  first_correct := by
    have composite : ¬ Nat.Prime (corner 2046 1) := by change ¬ Nat.Prime 16748557; exact PrattCertificates.composite_of_product (factor := 7) (cofactor := 2392651) rfl (by decide) (by decide)
    exact decide_eq_false composite
  second := false
  second_correct := by
    have composite : ¬ Nat.Prime (corner 2046 2) := by change ¬ Nat.Prime 16744465; exact PrattCertificates.composite_of_product (factor := 5) (cofactor := 3348893) rfl (by decide) (by decide)
    exact decide_eq_false composite
  third := false
  third_correct := by
    have composite : ¬ Nat.Prime (corner 2046 3) := by change ¬ Nat.Prime 16740373; exact PrattCertificates.composite_of_product (factor := 13) (cofactor := 1287721) rfl (by decide) (by decide)
    exact decide_eq_false composite
theorem row_count_2046 : row 2046 = 0 := (row_2046.checked).trans rfl
theorem model_count_2046 : modelRow 2046 = 0 := by decide
theorem modelrow_agrees_2046 : modelRow 2046 = row 2046 := model_count_2046.trans row_count_2046.symm
def row_2047 : RowCertificate 2047 where
  first := false
  first_correct := by
    have composite : ¬ Nat.Prime (corner 2047 1) := by change ¬ Nat.Prime 16764931; exact PrattCertificates.composite_of_product (factor := 157) (cofactor := 106783) rfl (by decide) (by decide)
    exact decide_eq_false composite
  second := false
  second_correct := by
    have composite : ¬ Nat.Prime (corner 2047 2) := by change ¬ Nat.Prime 16760837; exact PrattCertificates.composite_of_product (factor := 313) (cofactor := 53549) rfl (by decide) (by decide)
    exact decide_eq_false composite
  third := false
  third_correct := by
    have composite : ¬ Nat.Prime (corner 2047 3) := by change ¬ Nat.Prime 16756743; exact PrattCertificates.composite_of_product (factor := 3) (cofactor := 5585581) rfl (by decide) (by decide)
    exact decide_eq_false composite
theorem row_count_2047 : row 2047 = 0 := (row_2047.checked).trans rfl
theorem model_count_2047 : modelRow 2047 = 0 := by decide
theorem modelrow_agrees_2047 : modelRow 2047 = row 2047 := model_count_2047.trans row_count_2047.symm
def row_2048 : RowCertificate 2048 where
  first := false
  first_correct := by
    have composite : ¬ Nat.Prime (corner 2048 1) := by change ¬ Nat.Prime 16781313; exact PrattCertificates.composite_of_product (factor := 3) (cofactor := 5593771) rfl (by decide) (by decide)
    exact decide_eq_false composite
  second := false
  second_correct := by
    have composite : ¬ Nat.Prime (corner 2048 2) := by change ¬ Nat.Prime 16777217; exact PrattCertificates.composite_of_product (factor := 97) (cofactor := 172961) rfl (by decide) (by decide)
    exact decide_eq_false composite
  third := false
  third_correct := by
    have composite : ¬ Nat.Prime (corner 2048 3) := by change ¬ Nat.Prime 16773121; exact PrattCertificates.composite_of_product (factor := 433) (cofactor := 38737) rfl (by decide) (by decide)
    exact decide_eq_false composite
theorem row_count_2048 : row 2048 = 0 := (row_2048.checked).trans rfl
theorem model_count_2048 : modelRow 2048 = 0 := by decide
theorem modelrow_agrees_2048 : modelRow 2048 = row 2048 := model_count_2048.trans row_count_2048.symm
private def atomic_1921 : CountSegment 1 10 13120 1920 1921 986 987 := CountSegment.single (first := 1920) (initial := 986) row_count_1921 (by decide)
private theorem atomic_1921_agreement : RowAgreement 1920 1921 := RowAgreement.single (first := 1920) modelrow_agrees_1921
private def atomic_1922 : CountSegment 1 10 13120 1921 1922 987 988 := CountSegment.single (first := 1921) (initial := 987) row_count_1922 (by decide)
private theorem atomic_1922_agreement : RowAgreement 1921 1922 := RowAgreement.single (first := 1921) modelrow_agrees_1922
private def atomic_1923 : CountSegment 1 10 13120 1922 1923 988 989 := CountSegment.single (first := 1922) (initial := 988) row_count_1923 (by decide)
private theorem atomic_1923_agreement : RowAgreement 1922 1923 := RowAgreement.single (first := 1922) modelrow_agrees_1923
private def atomic_1924 : CountSegment 1 10 13120 1923 1924 989 989 := CountSegment.single (first := 1923) (initial := 989) row_count_1924 (by decide)
private theorem atomic_1924_agreement : RowAgreement 1923 1924 := RowAgreement.single (first := 1923) modelrow_agrees_1924
private def atomic_1925 : CountSegment 1 10 13120 1924 1925 989 990 := CountSegment.single (first := 1924) (initial := 989) row_count_1925 (by decide)
private theorem atomic_1925_agreement : RowAgreement 1924 1925 := RowAgreement.single (first := 1924) modelrow_agrees_1925
private def atomic_1926 : CountSegment 1 10 13120 1925 1926 990 990 := CountSegment.single (first := 1925) (initial := 990) row_count_1926 (by decide)
private theorem atomic_1926_agreement : RowAgreement 1925 1926 := RowAgreement.single (first := 1925) modelrow_agrees_1926
private def atomic_1927 : CountSegment 1 10 13120 1926 1927 990 990 := CountSegment.single (first := 1926) (initial := 990) row_count_1927 (by decide)
private theorem atomic_1927_agreement : RowAgreement 1926 1927 := RowAgreement.single (first := 1926) modelrow_agrees_1927
private def atomic_1928 : CountSegment 1 10 13120 1927 1928 990 991 := CountSegment.single (first := 1927) (initial := 990) row_count_1928 (by decide)
private theorem atomic_1928_agreement : RowAgreement 1927 1928 := RowAgreement.single (first := 1927) modelrow_agrees_1928
private def atomic_1929 : CountSegment 1 10 13120 1928 1929 991 991 := CountSegment.single (first := 1928) (initial := 991) row_count_1929 (by decide)
private theorem atomic_1929_agreement : RowAgreement 1928 1929 := RowAgreement.single (first := 1928) modelrow_agrees_1929
private def atomic_1930 : CountSegment 1 10 13120 1929 1930 991 991 := CountSegment.single (first := 1929) (initial := 991) row_count_1930 (by decide)
private theorem atomic_1930_agreement : RowAgreement 1929 1930 := RowAgreement.single (first := 1929) modelrow_agrees_1930
private def atomic_1931 : CountSegment 1 10 13120 1930 1931 991 991 := CountSegment.single (first := 1930) (initial := 991) row_count_1931 (by decide)
private theorem atomic_1931_agreement : RowAgreement 1930 1931 := RowAgreement.single (first := 1930) modelrow_agrees_1931
private def atomic_1932 : CountSegment 1 10 13120 1931 1932 991 991 := CountSegment.single (first := 1931) (initial := 991) row_count_1932 (by decide)
private theorem atomic_1932_agreement : RowAgreement 1931 1932 := RowAgreement.single (first := 1931) modelrow_agrees_1932
private def atomic_1933 : CountSegment 1 10 13120 1932 1933 991 991 := CountSegment.single (first := 1932) (initial := 991) row_count_1933 (by decide)
private theorem atomic_1933_agreement : RowAgreement 1932 1933 := RowAgreement.single (first := 1932) modelrow_agrees_1933
private def atomic_1934 : CountSegment 1 10 13120 1933 1934 991 991 := CountSegment.single (first := 1933) (initial := 991) row_count_1934 (by decide)
private theorem atomic_1934_agreement : RowAgreement 1933 1934 := RowAgreement.single (first := 1933) modelrow_agrees_1934
private def atomic_1935 : CountSegment 1 10 13120 1934 1935 991 992 := CountSegment.single (first := 1934) (initial := 991) row_count_1935 (by decide)
private theorem atomic_1935_agreement : RowAgreement 1934 1935 := RowAgreement.single (first := 1934) modelrow_agrees_1935
private def atomic_1936 : CountSegment 1 10 13120 1935 1936 992 993 := CountSegment.single (first := 1935) (initial := 992) row_count_1936 (by decide)
private theorem atomic_1936_agreement : RowAgreement 1935 1936 := RowAgreement.single (first := 1935) modelrow_agrees_1936
private def atomic_1937 : CountSegment 1 10 13120 1936 1937 993 993 := CountSegment.single (first := 1936) (initial := 993) row_count_1937 (by decide)
private theorem atomic_1937_agreement : RowAgreement 1936 1937 := RowAgreement.single (first := 1936) modelrow_agrees_1937
private def atomic_1938 : CountSegment 1 10 13120 1937 1938 993 993 := CountSegment.single (first := 1937) (initial := 993) row_count_1938 (by decide)
private theorem atomic_1938_agreement : RowAgreement 1937 1938 := RowAgreement.single (first := 1937) modelrow_agrees_1938
private def atomic_1939 : CountSegment 1 10 13120 1938 1939 993 993 := CountSegment.single (first := 1938) (initial := 993) row_count_1939 (by decide)
private theorem atomic_1939_agreement : RowAgreement 1938 1939 := RowAgreement.single (first := 1938) modelrow_agrees_1939
private def atomic_1940 : CountSegment 1 10 13120 1939 1940 993 994 := CountSegment.single (first := 1939) (initial := 993) row_count_1940 (by decide)
private theorem atomic_1940_agreement : RowAgreement 1939 1940 := RowAgreement.single (first := 1939) modelrow_agrees_1940
private def atomic_1941 : CountSegment 1 10 13120 1940 1941 994 995 := CountSegment.single (first := 1940) (initial := 994) row_count_1941 (by decide)
private theorem atomic_1941_agreement : RowAgreement 1940 1941 := RowAgreement.single (first := 1940) modelrow_agrees_1941
private def atomic_1942 : CountSegment 1 10 13120 1941 1942 995 997 := CountSegment.single (first := 1941) (initial := 995) row_count_1942 (by decide)
private theorem atomic_1942_agreement : RowAgreement 1941 1942 := RowAgreement.single (first := 1941) modelrow_agrees_1942
private def atomic_1943 : CountSegment 1 10 13120 1942 1943 997 998 := CountSegment.single (first := 1942) (initial := 997) row_count_1943 (by decide)
private theorem atomic_1943_agreement : RowAgreement 1942 1943 := RowAgreement.single (first := 1942) modelrow_agrees_1943
private def atomic_1944 : CountSegment 1 10 13120 1943 1944 998 999 := CountSegment.single (first := 1943) (initial := 998) row_count_1944 (by decide)
private theorem atomic_1944_agreement : RowAgreement 1943 1944 := RowAgreement.single (first := 1943) modelrow_agrees_1944
private def atomic_1945 : CountSegment 1 10 13120 1944 1945 999 1000 := CountSegment.single (first := 1944) (initial := 999) row_count_1945 (by decide)
private theorem atomic_1945_agreement : RowAgreement 1944 1945 := RowAgreement.single (first := 1944) modelrow_agrees_1945
private def atomic_1946 : CountSegment 1 10 13120 1945 1946 1000 1001 := CountSegment.single (first := 1945) (initial := 1000) row_count_1946 (by decide)
private theorem atomic_1946_agreement : RowAgreement 1945 1946 := RowAgreement.single (first := 1945) modelrow_agrees_1946
private def atomic_1947 : CountSegment 1 10 13120 1946 1947 1001 1002 := CountSegment.single (first := 1946) (initial := 1001) row_count_1947 (by decide)
private theorem atomic_1947_agreement : RowAgreement 1946 1947 := RowAgreement.single (first := 1946) modelrow_agrees_1947
private def atomic_1948 : CountSegment 1 10 13120 1947 1948 1002 1002 := CountSegment.single (first := 1947) (initial := 1002) row_count_1948 (by decide)
private theorem atomic_1948_agreement : RowAgreement 1947 1948 := RowAgreement.single (first := 1947) modelrow_agrees_1948
private def atomic_1949 : CountSegment 1 10 13120 1948 1949 1002 1002 := CountSegment.single (first := 1948) (initial := 1002) row_count_1949 (by decide)
private theorem atomic_1949_agreement : RowAgreement 1948 1949 := RowAgreement.single (first := 1948) modelrow_agrees_1949
private def atomic_1950 : CountSegment 1 10 13120 1949 1950 1002 1004 := CountSegment.single (first := 1949) (initial := 1002) row_count_1950 (by decide)
private theorem atomic_1950_agreement : RowAgreement 1949 1950 := RowAgreement.single (first := 1949) modelrow_agrees_1950
private def atomic_1951 : CountSegment 1 10 13120 1950 1951 1004 1004 := CountSegment.single (first := 1950) (initial := 1004) row_count_1951 (by decide)
private theorem atomic_1951_agreement : RowAgreement 1950 1951 := RowAgreement.single (first := 1950) modelrow_agrees_1951
private def atomic_1952 : CountSegment 1 10 13120 1951 1952 1004 1004 := CountSegment.single (first := 1951) (initial := 1004) row_count_1952 (by decide)
private theorem atomic_1952_agreement : RowAgreement 1951 1952 := RowAgreement.single (first := 1951) modelrow_agrees_1952
private def atomic_1953 : CountSegment 1 10 13120 1952 1953 1004 1005 := CountSegment.single (first := 1952) (initial := 1004) row_count_1953 (by decide)
private theorem atomic_1953_agreement : RowAgreement 1952 1953 := RowAgreement.single (first := 1952) modelrow_agrees_1953
private def atomic_1954 : CountSegment 1 10 13120 1953 1954 1005 1005 := CountSegment.single (first := 1953) (initial := 1005) row_count_1954 (by decide)
private theorem atomic_1954_agreement : RowAgreement 1953 1954 := RowAgreement.single (first := 1953) modelrow_agrees_1954
private def atomic_1955 : CountSegment 1 10 13120 1954 1955 1005 1006 := CountSegment.single (first := 1954) (initial := 1005) row_count_1955 (by decide)
private theorem atomic_1955_agreement : RowAgreement 1954 1955 := RowAgreement.single (first := 1954) modelrow_agrees_1955
private def atomic_1956 : CountSegment 1 10 13120 1955 1956 1006 1006 := CountSegment.single (first := 1955) (initial := 1006) row_count_1956 (by decide)
private theorem atomic_1956_agreement : RowAgreement 1955 1956 := RowAgreement.single (first := 1955) modelrow_agrees_1956
private def atomic_1957 : CountSegment 1 10 13120 1956 1957 1006 1007 := CountSegment.single (first := 1956) (initial := 1006) row_count_1957 (by decide)
private theorem atomic_1957_agreement : RowAgreement 1956 1957 := RowAgreement.single (first := 1956) modelrow_agrees_1957
private def atomic_1958 : CountSegment 1 10 13120 1957 1958 1007 1007 := CountSegment.single (first := 1957) (initial := 1007) row_count_1958 (by decide)
private theorem atomic_1958_agreement : RowAgreement 1957 1958 := RowAgreement.single (first := 1957) modelrow_agrees_1958
private def atomic_1959 : CountSegment 1 10 13120 1958 1959 1007 1008 := CountSegment.single (first := 1958) (initial := 1007) row_count_1959 (by decide)
private theorem atomic_1959_agreement : RowAgreement 1958 1959 := RowAgreement.single (first := 1958) modelrow_agrees_1959
private def atomic_1960 : CountSegment 1 10 13120 1959 1960 1008 1009 := CountSegment.single (first := 1959) (initial := 1008) row_count_1960 (by decide)
private theorem atomic_1960_agreement : RowAgreement 1959 1960 := RowAgreement.single (first := 1959) modelrow_agrees_1960
private def atomic_1961 : CountSegment 1 10 13120 1960 1961 1009 1009 := CountSegment.single (first := 1960) (initial := 1009) row_count_1961 (by decide)
private theorem atomic_1961_agreement : RowAgreement 1960 1961 := RowAgreement.single (first := 1960) modelrow_agrees_1961
private def atomic_1962 : CountSegment 1 10 13120 1961 1962 1009 1010 := CountSegment.single (first := 1961) (initial := 1009) row_count_1962 (by decide)
private theorem atomic_1962_agreement : RowAgreement 1961 1962 := RowAgreement.single (first := 1961) modelrow_agrees_1962
private def atomic_1963 : CountSegment 1 10 13120 1962 1963 1010 1010 := CountSegment.single (first := 1962) (initial := 1010) row_count_1963 (by decide)
private theorem atomic_1963_agreement : RowAgreement 1962 1963 := RowAgreement.single (first := 1962) modelrow_agrees_1963
private def atomic_1964 : CountSegment 1 10 13120 1963 1964 1010 1011 := CountSegment.single (first := 1963) (initial := 1010) row_count_1964 (by decide)
private theorem atomic_1964_agreement : RowAgreement 1963 1964 := RowAgreement.single (first := 1963) modelrow_agrees_1964
private def atomic_1965 : CountSegment 1 10 13120 1964 1965 1011 1011 := CountSegment.single (first := 1964) (initial := 1011) row_count_1965 (by decide)
private theorem atomic_1965_agreement : RowAgreement 1964 1965 := RowAgreement.single (first := 1964) modelrow_agrees_1965
private def atomic_1966 : CountSegment 1 10 13120 1965 1966 1011 1011 := CountSegment.single (first := 1965) (initial := 1011) row_count_1966 (by decide)
private theorem atomic_1966_agreement : RowAgreement 1965 1966 := RowAgreement.single (first := 1965) modelrow_agrees_1966
private def atomic_1967 : CountSegment 1 10 13120 1966 1967 1011 1011 := CountSegment.single (first := 1966) (initial := 1011) row_count_1967 (by decide)
private theorem atomic_1967_agreement : RowAgreement 1966 1967 := RowAgreement.single (first := 1966) modelrow_agrees_1967
private def atomic_1968 : CountSegment 1 10 13120 1967 1968 1011 1011 := CountSegment.single (first := 1967) (initial := 1011) row_count_1968 (by decide)
private theorem atomic_1968_agreement : RowAgreement 1967 1968 := RowAgreement.single (first := 1967) modelrow_agrees_1968
private def atomic_1969 : CountSegment 1 10 13120 1968 1969 1011 1011 := CountSegment.single (first := 1968) (initial := 1011) row_count_1969 (by decide)
private theorem atomic_1969_agreement : RowAgreement 1968 1969 := RowAgreement.single (first := 1968) modelrow_agrees_1969
private def atomic_1970 : CountSegment 1 10 13120 1969 1970 1011 1012 := CountSegment.single (first := 1969) (initial := 1011) row_count_1970 (by decide)
private theorem atomic_1970_agreement : RowAgreement 1969 1970 := RowAgreement.single (first := 1969) modelrow_agrees_1970
private def atomic_1971 : CountSegment 1 10 13120 1970 1971 1012 1012 := CountSegment.single (first := 1970) (initial := 1012) row_count_1971 (by decide)
private theorem atomic_1971_agreement : RowAgreement 1970 1971 := RowAgreement.single (first := 1970) modelrow_agrees_1971
private def atomic_1972 : CountSegment 1 10 13120 1971 1972 1012 1012 := CountSegment.single (first := 1971) (initial := 1012) row_count_1972 (by decide)
private theorem atomic_1972_agreement : RowAgreement 1971 1972 := RowAgreement.single (first := 1971) modelrow_agrees_1972
private def atomic_1973 : CountSegment 1 10 13120 1972 1973 1012 1013 := CountSegment.single (first := 1972) (initial := 1012) row_count_1973 (by decide)
private theorem atomic_1973_agreement : RowAgreement 1972 1973 := RowAgreement.single (first := 1972) modelrow_agrees_1973
private def atomic_1974 : CountSegment 1 10 13120 1973 1974 1013 1013 := CountSegment.single (first := 1973) (initial := 1013) row_count_1974 (by decide)
private theorem atomic_1974_agreement : RowAgreement 1973 1974 := RowAgreement.single (first := 1973) modelrow_agrees_1974
private def atomic_1975 : CountSegment 1 10 13120 1974 1975 1013 1013 := CountSegment.single (first := 1974) (initial := 1013) row_count_1975 (by decide)
private theorem atomic_1975_agreement : RowAgreement 1974 1975 := RowAgreement.single (first := 1974) modelrow_agrees_1975
private def atomic_1976 : CountSegment 1 10 13120 1975 1976 1013 1013 := CountSegment.single (first := 1975) (initial := 1013) row_count_1976 (by decide)
private theorem atomic_1976_agreement : RowAgreement 1975 1976 := RowAgreement.single (first := 1975) modelrow_agrees_1976
private def atomic_1977 : CountSegment 1 10 13120 1976 1977 1013 1014 := CountSegment.single (first := 1976) (initial := 1013) row_count_1977 (by decide)
private theorem atomic_1977_agreement : RowAgreement 1976 1977 := RowAgreement.single (first := 1976) modelrow_agrees_1977
private def atomic_1978 : CountSegment 1 10 13120 1977 1978 1014 1014 := CountSegment.single (first := 1977) (initial := 1014) row_count_1978 (by decide)
private theorem atomic_1978_agreement : RowAgreement 1977 1978 := RowAgreement.single (first := 1977) modelrow_agrees_1978
private def atomic_1979 : CountSegment 1 10 13120 1978 1979 1014 1014 := CountSegment.single (first := 1978) (initial := 1014) row_count_1979 (by decide)
private theorem atomic_1979_agreement : RowAgreement 1978 1979 := RowAgreement.single (first := 1978) modelrow_agrees_1979
private def atomic_1980 : CountSegment 1 10 13120 1979 1980 1014 1014 := CountSegment.single (first := 1979) (initial := 1014) row_count_1980 (by decide)
private theorem atomic_1980_agreement : RowAgreement 1979 1980 := RowAgreement.single (first := 1979) modelrow_agrees_1980
private def atomic_1981 : CountSegment 1 10 13120 1980 1981 1014 1014 := CountSegment.single (first := 1980) (initial := 1014) row_count_1981 (by decide)
private theorem atomic_1981_agreement : RowAgreement 1980 1981 := RowAgreement.single (first := 1980) modelrow_agrees_1981
private def atomic_1982 : CountSegment 1 10 13120 1981 1982 1014 1014 := CountSegment.single (first := 1981) (initial := 1014) row_count_1982 (by decide)
private theorem atomic_1982_agreement : RowAgreement 1981 1982 := RowAgreement.single (first := 1981) modelrow_agrees_1982
private def atomic_1983 : CountSegment 1 10 13120 1982 1983 1014 1016 := CountSegment.single (first := 1982) (initial := 1014) row_count_1983 (by decide)
private theorem atomic_1983_agreement : RowAgreement 1982 1983 := RowAgreement.single (first := 1982) modelrow_agrees_1983
private def atomic_1984 : CountSegment 1 10 13120 1983 1984 1016 1016 := CountSegment.single (first := 1983) (initial := 1016) row_count_1984 (by decide)
private theorem atomic_1984_agreement : RowAgreement 1983 1984 := RowAgreement.single (first := 1983) modelrow_agrees_1984
private def atomic_1985 : CountSegment 1 10 13120 1984 1985 1016 1016 := CountSegment.single (first := 1984) (initial := 1016) row_count_1985 (by decide)
private theorem atomic_1985_agreement : RowAgreement 1984 1985 := RowAgreement.single (first := 1984) modelrow_agrees_1985
private def atomic_1986 : CountSegment 1 10 13120 1985 1986 1016 1016 := CountSegment.single (first := 1985) (initial := 1016) row_count_1986 (by decide)
private theorem atomic_1986_agreement : RowAgreement 1985 1986 := RowAgreement.single (first := 1985) modelrow_agrees_1986
private def atomic_1987 : CountSegment 1 10 13120 1986 1987 1016 1016 := CountSegment.single (first := 1986) (initial := 1016) row_count_1987 (by decide)
private theorem atomic_1987_agreement : RowAgreement 1986 1987 := RowAgreement.single (first := 1986) modelrow_agrees_1987
private def atomic_1988 : CountSegment 1 10 13120 1987 1988 1016 1017 := CountSegment.single (first := 1987) (initial := 1016) row_count_1988 (by decide)
private theorem atomic_1988_agreement : RowAgreement 1987 1988 := RowAgreement.single (first := 1987) modelrow_agrees_1988
private def atomic_1989 : CountSegment 1 10 13120 1988 1989 1017 1017 := CountSegment.single (first := 1988) (initial := 1017) row_count_1989 (by decide)
private theorem atomic_1989_agreement : RowAgreement 1988 1989 := RowAgreement.single (first := 1988) modelrow_agrees_1989
private def atomic_1990 : CountSegment 1 10 13120 1989 1990 1017 1017 := CountSegment.single (first := 1989) (initial := 1017) row_count_1990 (by decide)
private theorem atomic_1990_agreement : RowAgreement 1989 1990 := RowAgreement.single (first := 1989) modelrow_agrees_1990
private def atomic_1991 : CountSegment 1 10 13120 1990 1991 1017 1017 := CountSegment.single (first := 1990) (initial := 1017) row_count_1991 (by decide)
private theorem atomic_1991_agreement : RowAgreement 1990 1991 := RowAgreement.single (first := 1990) modelrow_agrees_1991
private def atomic_1992 : CountSegment 1 10 13120 1991 1992 1017 1018 := CountSegment.single (first := 1991) (initial := 1017) row_count_1992 (by decide)
private theorem atomic_1992_agreement : RowAgreement 1991 1992 := RowAgreement.single (first := 1991) modelrow_agrees_1992
private def atomic_1993 : CountSegment 1 10 13120 1992 1993 1018 1019 := CountSegment.single (first := 1992) (initial := 1018) row_count_1993 (by decide)
private theorem atomic_1993_agreement : RowAgreement 1992 1993 := RowAgreement.single (first := 1992) modelrow_agrees_1993
private def atomic_1994 : CountSegment 1 10 13120 1993 1994 1019 1019 := CountSegment.single (first := 1993) (initial := 1019) row_count_1994 (by decide)
private theorem atomic_1994_agreement : RowAgreement 1993 1994 := RowAgreement.single (first := 1993) modelrow_agrees_1994
private def atomic_1995 : CountSegment 1 10 13120 1994 1995 1019 1020 := CountSegment.single (first := 1994) (initial := 1019) row_count_1995 (by decide)
private theorem atomic_1995_agreement : RowAgreement 1994 1995 := RowAgreement.single (first := 1994) modelrow_agrees_1995
private def atomic_1996 : CountSegment 1 10 13120 1995 1996 1020 1020 := CountSegment.single (first := 1995) (initial := 1020) row_count_1996 (by decide)
private theorem atomic_1996_agreement : RowAgreement 1995 1996 := RowAgreement.single (first := 1995) modelrow_agrees_1996
private def atomic_1997 : CountSegment 1 10 13120 1996 1997 1020 1021 := CountSegment.single (first := 1996) (initial := 1020) row_count_1997 (by decide)
private theorem atomic_1997_agreement : RowAgreement 1996 1997 := RowAgreement.single (first := 1996) modelrow_agrees_1997
private def atomic_1998 : CountSegment 1 10 13120 1997 1998 1021 1023 := CountSegment.single (first := 1997) (initial := 1021) row_count_1998 (by decide)
private theorem atomic_1998_agreement : RowAgreement 1997 1998 := RowAgreement.single (first := 1997) modelrow_agrees_1998
private def atomic_1999 : CountSegment 1 10 13120 1998 1999 1023 1024 := CountSegment.single (first := 1998) (initial := 1023) row_count_1999 (by decide)
private theorem atomic_1999_agreement : RowAgreement 1998 1999 := RowAgreement.single (first := 1998) modelrow_agrees_1999
private def atomic_2000 : CountSegment 1 10 13120 1999 2000 1024 1024 := CountSegment.single (first := 1999) (initial := 1024) row_count_2000 (by decide)
private theorem atomic_2000_agreement : RowAgreement 1999 2000 := RowAgreement.single (first := 1999) modelrow_agrees_2000
private def atomic_2001 : CountSegment 1 10 13120 2000 2001 1024 1025 := CountSegment.single (first := 2000) (initial := 1024) row_count_2001 (by decide)
private theorem atomic_2001_agreement : RowAgreement 2000 2001 := RowAgreement.single (first := 2000) modelrow_agrees_2001
private def atomic_2002 : CountSegment 1 10 13120 2001 2002 1025 1025 := CountSegment.single (first := 2001) (initial := 1025) row_count_2002 (by decide)
private theorem atomic_2002_agreement : RowAgreement 2001 2002 := RowAgreement.single (first := 2001) modelrow_agrees_2002
private def atomic_2003 : CountSegment 1 10 13120 2002 2003 1025 1026 := CountSegment.single (first := 2002) (initial := 1025) row_count_2003 (by decide)
private theorem atomic_2003_agreement : RowAgreement 2002 2003 := RowAgreement.single (first := 2002) modelrow_agrees_2003
private def atomic_2004 : CountSegment 1 10 13120 2003 2004 1026 1026 := CountSegment.single (first := 2003) (initial := 1026) row_count_2004 (by decide)
private theorem atomic_2004_agreement : RowAgreement 2003 2004 := RowAgreement.single (first := 2003) modelrow_agrees_2004
private def atomic_2005 : CountSegment 1 10 13120 2004 2005 1026 1026 := CountSegment.single (first := 2004) (initial := 1026) row_count_2005 (by decide)
private theorem atomic_2005_agreement : RowAgreement 2004 2005 := RowAgreement.single (first := 2004) modelrow_agrees_2005
private def atomic_2006 : CountSegment 1 10 13120 2005 2006 1026 1026 := CountSegment.single (first := 2005) (initial := 1026) row_count_2006 (by decide)
private theorem atomic_2006_agreement : RowAgreement 2005 2006 := RowAgreement.single (first := 2005) modelrow_agrees_2006
private def atomic_2007 : CountSegment 1 10 13120 2006 2007 1026 1026 := CountSegment.single (first := 2006) (initial := 1026) row_count_2007 (by decide)
private theorem atomic_2007_agreement : RowAgreement 2006 2007 := RowAgreement.single (first := 2006) modelrow_agrees_2007
private def atomic_2008 : CountSegment 1 10 13120 2007 2008 1026 1026 := CountSegment.single (first := 2007) (initial := 1026) row_count_2008 (by decide)
private theorem atomic_2008_agreement : RowAgreement 2007 2008 := RowAgreement.single (first := 2007) modelrow_agrees_2008
private def atomic_2009 : CountSegment 1 10 13120 2008 2009 1026 1027 := CountSegment.single (first := 2008) (initial := 1026) row_count_2009 (by decide)
private theorem atomic_2009_agreement : RowAgreement 2008 2009 := RowAgreement.single (first := 2008) modelrow_agrees_2009
private def atomic_2010 : CountSegment 1 10 13120 2009 2010 1027 1027 := CountSegment.single (first := 2009) (initial := 1027) row_count_2010 (by decide)
private theorem atomic_2010_agreement : RowAgreement 2009 2010 := RowAgreement.single (first := 2009) modelrow_agrees_2010
private def atomic_2011 : CountSegment 1 10 13120 2010 2011 1027 1027 := CountSegment.single (first := 2010) (initial := 1027) row_count_2011 (by decide)
private theorem atomic_2011_agreement : RowAgreement 2010 2011 := RowAgreement.single (first := 2010) modelrow_agrees_2011
private def atomic_2012 : CountSegment 1 10 13120 2011 2012 1027 1028 := CountSegment.single (first := 2011) (initial := 1027) row_count_2012 (by decide)
private theorem atomic_2012_agreement : RowAgreement 2011 2012 := RowAgreement.single (first := 2011) modelrow_agrees_2012
private def atomic_2013 : CountSegment 1 10 13120 2012 2013 1028 1030 := CountSegment.single (first := 2012) (initial := 1028) row_count_2013 (by decide)
private theorem atomic_2013_agreement : RowAgreement 2012 2013 := RowAgreement.single (first := 2012) modelrow_agrees_2013
private def atomic_2014 : CountSegment 1 10 13120 2013 2014 1030 1031 := CountSegment.single (first := 2013) (initial := 1030) row_count_2014 (by decide)
private theorem atomic_2014_agreement : RowAgreement 2013 2014 := RowAgreement.single (first := 2013) modelrow_agrees_2014
private def atomic_2015 : CountSegment 1 10 13120 2014 2015 1031 1031 := CountSegment.single (first := 2014) (initial := 1031) row_count_2015 (by decide)
private theorem atomic_2015_agreement : RowAgreement 2014 2015 := RowAgreement.single (first := 2014) modelrow_agrees_2015
private def atomic_2016 : CountSegment 1 10 13120 2015 2016 1031 1032 := CountSegment.single (first := 2015) (initial := 1031) row_count_2016 (by decide)
private theorem atomic_2016_agreement : RowAgreement 2015 2016 := RowAgreement.single (first := 2015) modelrow_agrees_2016
private def atomic_2017 : CountSegment 1 10 13120 2016 2017 1032 1033 := CountSegment.single (first := 2016) (initial := 1032) row_count_2017 (by decide)
private theorem atomic_2017_agreement : RowAgreement 2016 2017 := RowAgreement.single (first := 2016) modelrow_agrees_2017
private def atomic_2018 : CountSegment 1 10 13120 2017 2018 1033 1033 := CountSegment.single (first := 2017) (initial := 1033) row_count_2018 (by decide)
private theorem atomic_2018_agreement : RowAgreement 2017 2018 := RowAgreement.single (first := 2017) modelrow_agrees_2018
private def atomic_2019 : CountSegment 1 10 13120 2018 2019 1033 1033 := CountSegment.single (first := 2018) (initial := 1033) row_count_2019 (by decide)
private theorem atomic_2019_agreement : RowAgreement 2018 2019 := RowAgreement.single (first := 2018) modelrow_agrees_2019
private def atomic_2020 : CountSegment 1 10 13120 2019 2020 1033 1033 := CountSegment.single (first := 2019) (initial := 1033) row_count_2020 (by decide)
private theorem atomic_2020_agreement : RowAgreement 2019 2020 := RowAgreement.single (first := 2019) modelrow_agrees_2020
private def atomic_2021 : CountSegment 1 10 13120 2020 2021 1033 1033 := CountSegment.single (first := 2020) (initial := 1033) row_count_2021 (by decide)
private theorem atomic_2021_agreement : RowAgreement 2020 2021 := RowAgreement.single (first := 2020) modelrow_agrees_2021
private def atomic_2022 : CountSegment 1 10 13120 2021 2022 1033 1033 := CountSegment.single (first := 2021) (initial := 1033) row_count_2022 (by decide)
private theorem atomic_2022_agreement : RowAgreement 2021 2022 := RowAgreement.single (first := 2021) modelrow_agrees_2022
private def atomic_2023 : CountSegment 1 10 13120 2022 2023 1033 1034 := CountSegment.single (first := 2022) (initial := 1033) row_count_2023 (by decide)
private theorem atomic_2023_agreement : RowAgreement 2022 2023 := RowAgreement.single (first := 2022) modelrow_agrees_2023
private def atomic_2024 : CountSegment 1 10 13120 2023 2024 1034 1035 := CountSegment.single (first := 2023) (initial := 1034) row_count_2024 (by decide)
private theorem atomic_2024_agreement : RowAgreement 2023 2024 := RowAgreement.single (first := 2023) modelrow_agrees_2024
private def atomic_2025 : CountSegment 1 10 13120 2024 2025 1035 1035 := CountSegment.single (first := 2024) (initial := 1035) row_count_2025 (by decide)
private theorem atomic_2025_agreement : RowAgreement 2024 2025 := RowAgreement.single (first := 2024) modelrow_agrees_2025
private def atomic_2026 : CountSegment 1 10 13120 2025 2026 1035 1035 := CountSegment.single (first := 2025) (initial := 1035) row_count_2026 (by decide)
private theorem atomic_2026_agreement : RowAgreement 2025 2026 := RowAgreement.single (first := 2025) modelrow_agrees_2026
private def atomic_2027 : CountSegment 1 10 13120 2026 2027 1035 1035 := CountSegment.single (first := 2026) (initial := 1035) row_count_2027 (by decide)
private theorem atomic_2027_agreement : RowAgreement 2026 2027 := RowAgreement.single (first := 2026) modelrow_agrees_2027
private def atomic_2028 : CountSegment 1 10 13120 2027 2028 1035 1037 := CountSegment.single (first := 2027) (initial := 1035) row_count_2028 (by decide)
private theorem atomic_2028_agreement : RowAgreement 2027 2028 := RowAgreement.single (first := 2027) modelrow_agrees_2028
private def atomic_2029 : CountSegment 1 10 13120 2028 2029 1037 1037 := CountSegment.single (first := 2028) (initial := 1037) row_count_2029 (by decide)
private theorem atomic_2029_agreement : RowAgreement 2028 2029 := RowAgreement.single (first := 2028) modelrow_agrees_2029
private def atomic_2030 : CountSegment 1 10 13120 2029 2030 1037 1037 := CountSegment.single (first := 2029) (initial := 1037) row_count_2030 (by decide)
private theorem atomic_2030_agreement : RowAgreement 2029 2030 := RowAgreement.single (first := 2029) modelrow_agrees_2030
private def atomic_2031 : CountSegment 1 10 13120 2030 2031 1037 1038 := CountSegment.single (first := 2030) (initial := 1037) row_count_2031 (by decide)
private theorem atomic_2031_agreement : RowAgreement 2030 2031 := RowAgreement.single (first := 2030) modelrow_agrees_2031
private def atomic_2032 : CountSegment 1 10 13120 2031 2032 1038 1038 := CountSegment.single (first := 2031) (initial := 1038) row_count_2032 (by decide)
private theorem atomic_2032_agreement : RowAgreement 2031 2032 := RowAgreement.single (first := 2031) modelrow_agrees_2032
private def atomic_2033 : CountSegment 1 10 13120 2032 2033 1038 1038 := CountSegment.single (first := 2032) (initial := 1038) row_count_2033 (by decide)
private theorem atomic_2033_agreement : RowAgreement 2032 2033 := RowAgreement.single (first := 2032) modelrow_agrees_2033
private def atomic_2034 : CountSegment 1 10 13120 2033 2034 1038 1040 := CountSegment.single (first := 2033) (initial := 1038) row_count_2034 (by decide)
private theorem atomic_2034_agreement : RowAgreement 2033 2034 := RowAgreement.single (first := 2033) modelrow_agrees_2034
private def atomic_2035 : CountSegment 1 10 13120 2034 2035 1040 1041 := CountSegment.single (first := 2034) (initial := 1040) row_count_2035 (by decide)
private theorem atomic_2035_agreement : RowAgreement 2034 2035 := RowAgreement.single (first := 2034) modelrow_agrees_2035
private def atomic_2036 : CountSegment 1 10 13120 2035 2036 1041 1041 := CountSegment.single (first := 2035) (initial := 1041) row_count_2036 (by decide)
private theorem atomic_2036_agreement : RowAgreement 2035 2036 := RowAgreement.single (first := 2035) modelrow_agrees_2036
private def atomic_2037 : CountSegment 1 10 13120 2036 2037 1041 1042 := CountSegment.single (first := 2036) (initial := 1041) row_count_2037 (by decide)
private theorem atomic_2037_agreement : RowAgreement 2036 2037 := RowAgreement.single (first := 2036) modelrow_agrees_2037
private def atomic_2038 : CountSegment 1 10 13120 2037 2038 1042 1042 := CountSegment.single (first := 2037) (initial := 1042) row_count_2038 (by decide)
private theorem atomic_2038_agreement : RowAgreement 2037 2038 := RowAgreement.single (first := 2037) modelrow_agrees_2038
private def atomic_2039 : CountSegment 1 10 13120 2038 2039 1042 1042 := CountSegment.single (first := 2038) (initial := 1042) row_count_2039 (by decide)
private theorem atomic_2039_agreement : RowAgreement 2038 2039 := RowAgreement.single (first := 2038) modelrow_agrees_2039
private def atomic_2040 : CountSegment 1 10 13120 2039 2040 1042 1043 := CountSegment.single (first := 2039) (initial := 1042) row_count_2040 (by decide)
private theorem atomic_2040_agreement : RowAgreement 2039 2040 := RowAgreement.single (first := 2039) modelrow_agrees_2040
private def atomic_2041 : CountSegment 1 10 13120 2040 2041 1043 1044 := CountSegment.single (first := 2040) (initial := 1043) row_count_2041 (by decide)
private theorem atomic_2041_agreement : RowAgreement 2040 2041 := RowAgreement.single (first := 2040) modelrow_agrees_2041
private def atomic_2042 : CountSegment 1 10 13120 2041 2042 1044 1044 := CountSegment.single (first := 2041) (initial := 1044) row_count_2042 (by decide)
private theorem atomic_2042_agreement : RowAgreement 2041 2042 := RowAgreement.single (first := 2041) modelrow_agrees_2042
private def atomic_2043 : CountSegment 1 10 13120 2042 2043 1044 1045 := CountSegment.single (first := 2042) (initial := 1044) row_count_2043 (by decide)
private theorem atomic_2043_agreement : RowAgreement 2042 2043 := RowAgreement.single (first := 2042) modelrow_agrees_2043
private def atomic_2044 : CountSegment 1 10 13120 2043 2044 1045 1045 := CountSegment.single (first := 2043) (initial := 1045) row_count_2044 (by decide)
private theorem atomic_2044_agreement : RowAgreement 2043 2044 := RowAgreement.single (first := 2043) modelrow_agrees_2044
private def atomic_2045 : CountSegment 1 10 13120 2044 2045 1045 1046 := CountSegment.single (first := 2044) (initial := 1045) row_count_2045 (by decide)
private theorem atomic_2045_agreement : RowAgreement 2044 2045 := RowAgreement.single (first := 2044) modelrow_agrees_2045
private def atomic_2046 : CountSegment 1 10 13120 2045 2046 1046 1046 := CountSegment.single (first := 2045) (initial := 1046) row_count_2046 (by decide)
private theorem atomic_2046_agreement : RowAgreement 2045 2046 := RowAgreement.single (first := 2045) modelrow_agrees_2046
private def atomic_2047 : CountSegment 1 10 13120 2046 2047 1046 1046 := CountSegment.single (first := 2046) (initial := 1046) row_count_2047 (by decide)
private theorem atomic_2047_agreement : RowAgreement 2046 2047 := RowAgreement.single (first := 2046) modelrow_agrees_2047
private def atomic_2048 : CountSegment 1 10 13120 2047 2048 1046 1046 := CountSegment.single (first := 2047) (initial := 1046) row_count_2048 (by decide)
private theorem atomic_2048_agreement : RowAgreement 2047 2048 := RowAgreement.single (first := 2047) modelrow_agrees_2048
private def within_015_000 : CountSegment 1 10 13120 1920 1922 986 988 := CountSegment.join atomic_1921 atomic_1922
private theorem within_015_000_agreement : RowAgreement 1920 1922 := RowAgreement.join atomic_1921_agreement atomic_1922_agreement
private def within_015_001 : CountSegment 1 10 13120 1922 1924 988 989 := CountSegment.join atomic_1923 atomic_1924
private theorem within_015_001_agreement : RowAgreement 1922 1924 := RowAgreement.join atomic_1923_agreement atomic_1924_agreement
private def within_015_002 : CountSegment 1 10 13120 1920 1924 986 989 := CountSegment.join within_015_000 within_015_001
private theorem within_015_002_agreement : RowAgreement 1920 1924 := RowAgreement.join within_015_000_agreement within_015_001_agreement
private def within_015_003 : CountSegment 1 10 13120 1924 1926 989 990 := CountSegment.join atomic_1925 atomic_1926
private theorem within_015_003_agreement : RowAgreement 1924 1926 := RowAgreement.join atomic_1925_agreement atomic_1926_agreement
private def within_015_004 : CountSegment 1 10 13120 1926 1928 990 991 := CountSegment.join atomic_1927 atomic_1928
private theorem within_015_004_agreement : RowAgreement 1926 1928 := RowAgreement.join atomic_1927_agreement atomic_1928_agreement
private def within_015_005 : CountSegment 1 10 13120 1924 1928 989 991 := CountSegment.join within_015_003 within_015_004
private theorem within_015_005_agreement : RowAgreement 1924 1928 := RowAgreement.join within_015_003_agreement within_015_004_agreement
private def within_015_006 : CountSegment 1 10 13120 1920 1928 986 991 := CountSegment.join within_015_002 within_015_005
private theorem within_015_006_agreement : RowAgreement 1920 1928 := RowAgreement.join within_015_002_agreement within_015_005_agreement
private def within_015_007 : CountSegment 1 10 13120 1928 1930 991 991 := CountSegment.join atomic_1929 atomic_1930
private theorem within_015_007_agreement : RowAgreement 1928 1930 := RowAgreement.join atomic_1929_agreement atomic_1930_agreement
private def within_015_008 : CountSegment 1 10 13120 1930 1932 991 991 := CountSegment.join atomic_1931 atomic_1932
private theorem within_015_008_agreement : RowAgreement 1930 1932 := RowAgreement.join atomic_1931_agreement atomic_1932_agreement
private def within_015_009 : CountSegment 1 10 13120 1928 1932 991 991 := CountSegment.join within_015_007 within_015_008
private theorem within_015_009_agreement : RowAgreement 1928 1932 := RowAgreement.join within_015_007_agreement within_015_008_agreement
private def within_015_010 : CountSegment 1 10 13120 1932 1934 991 991 := CountSegment.join atomic_1933 atomic_1934
private theorem within_015_010_agreement : RowAgreement 1932 1934 := RowAgreement.join atomic_1933_agreement atomic_1934_agreement
private def within_015_011 : CountSegment 1 10 13120 1934 1936 991 993 := CountSegment.join atomic_1935 atomic_1936
private theorem within_015_011_agreement : RowAgreement 1934 1936 := RowAgreement.join atomic_1935_agreement atomic_1936_agreement
private def within_015_012 : CountSegment 1 10 13120 1932 1936 991 993 := CountSegment.join within_015_010 within_015_011
private theorem within_015_012_agreement : RowAgreement 1932 1936 := RowAgreement.join within_015_010_agreement within_015_011_agreement
private def within_015_013 : CountSegment 1 10 13120 1928 1936 991 993 := CountSegment.join within_015_009 within_015_012
private theorem within_015_013_agreement : RowAgreement 1928 1936 := RowAgreement.join within_015_009_agreement within_015_012_agreement
private def within_015_014 : CountSegment 1 10 13120 1920 1936 986 993 := CountSegment.join within_015_006 within_015_013
private theorem within_015_014_agreement : RowAgreement 1920 1936 := RowAgreement.join within_015_006_agreement within_015_013_agreement
private def within_015_015 : CountSegment 1 10 13120 1936 1938 993 993 := CountSegment.join atomic_1937 atomic_1938
private theorem within_015_015_agreement : RowAgreement 1936 1938 := RowAgreement.join atomic_1937_agreement atomic_1938_agreement
private def within_015_016 : CountSegment 1 10 13120 1938 1940 993 994 := CountSegment.join atomic_1939 atomic_1940
private theorem within_015_016_agreement : RowAgreement 1938 1940 := RowAgreement.join atomic_1939_agreement atomic_1940_agreement
private def within_015_017 : CountSegment 1 10 13120 1936 1940 993 994 := CountSegment.join within_015_015 within_015_016
private theorem within_015_017_agreement : RowAgreement 1936 1940 := RowAgreement.join within_015_015_agreement within_015_016_agreement
private def within_015_018 : CountSegment 1 10 13120 1940 1942 994 997 := CountSegment.join atomic_1941 atomic_1942
private theorem within_015_018_agreement : RowAgreement 1940 1942 := RowAgreement.join atomic_1941_agreement atomic_1942_agreement
private def within_015_019 : CountSegment 1 10 13120 1942 1944 997 999 := CountSegment.join atomic_1943 atomic_1944
private theorem within_015_019_agreement : RowAgreement 1942 1944 := RowAgreement.join atomic_1943_agreement atomic_1944_agreement
private def within_015_020 : CountSegment 1 10 13120 1940 1944 994 999 := CountSegment.join within_015_018 within_015_019
private theorem within_015_020_agreement : RowAgreement 1940 1944 := RowAgreement.join within_015_018_agreement within_015_019_agreement
private def within_015_021 : CountSegment 1 10 13120 1936 1944 993 999 := CountSegment.join within_015_017 within_015_020
private theorem within_015_021_agreement : RowAgreement 1936 1944 := RowAgreement.join within_015_017_agreement within_015_020_agreement
private def within_015_022 : CountSegment 1 10 13120 1944 1946 999 1001 := CountSegment.join atomic_1945 atomic_1946
private theorem within_015_022_agreement : RowAgreement 1944 1946 := RowAgreement.join atomic_1945_agreement atomic_1946_agreement
private def within_015_023 : CountSegment 1 10 13120 1946 1948 1001 1002 := CountSegment.join atomic_1947 atomic_1948
private theorem within_015_023_agreement : RowAgreement 1946 1948 := RowAgreement.join atomic_1947_agreement atomic_1948_agreement
private def within_015_024 : CountSegment 1 10 13120 1944 1948 999 1002 := CountSegment.join within_015_022 within_015_023
private theorem within_015_024_agreement : RowAgreement 1944 1948 := RowAgreement.join within_015_022_agreement within_015_023_agreement
private def within_015_025 : CountSegment 1 10 13120 1948 1950 1002 1004 := CountSegment.join atomic_1949 atomic_1950
private theorem within_015_025_agreement : RowAgreement 1948 1950 := RowAgreement.join atomic_1949_agreement atomic_1950_agreement
private def within_015_026 : CountSegment 1 10 13120 1950 1952 1004 1004 := CountSegment.join atomic_1951 atomic_1952
private theorem within_015_026_agreement : RowAgreement 1950 1952 := RowAgreement.join atomic_1951_agreement atomic_1952_agreement
private def within_015_027 : CountSegment 1 10 13120 1948 1952 1002 1004 := CountSegment.join within_015_025 within_015_026
private theorem within_015_027_agreement : RowAgreement 1948 1952 := RowAgreement.join within_015_025_agreement within_015_026_agreement
private def within_015_028 : CountSegment 1 10 13120 1944 1952 999 1004 := CountSegment.join within_015_024 within_015_027
private theorem within_015_028_agreement : RowAgreement 1944 1952 := RowAgreement.join within_015_024_agreement within_015_027_agreement
private def within_015_029 : CountSegment 1 10 13120 1936 1952 993 1004 := CountSegment.join within_015_021 within_015_028
private theorem within_015_029_agreement : RowAgreement 1936 1952 := RowAgreement.join within_015_021_agreement within_015_028_agreement
private def within_015_030 : CountSegment 1 10 13120 1920 1952 986 1004 := CountSegment.join within_015_014 within_015_029
private theorem within_015_030_agreement : RowAgreement 1920 1952 := RowAgreement.join within_015_014_agreement within_015_029_agreement
private def within_015_031 : CountSegment 1 10 13120 1952 1954 1004 1005 := CountSegment.join atomic_1953 atomic_1954
private theorem within_015_031_agreement : RowAgreement 1952 1954 := RowAgreement.join atomic_1953_agreement atomic_1954_agreement
private def within_015_032 : CountSegment 1 10 13120 1954 1956 1005 1006 := CountSegment.join atomic_1955 atomic_1956
private theorem within_015_032_agreement : RowAgreement 1954 1956 := RowAgreement.join atomic_1955_agreement atomic_1956_agreement
private def within_015_033 : CountSegment 1 10 13120 1952 1956 1004 1006 := CountSegment.join within_015_031 within_015_032
private theorem within_015_033_agreement : RowAgreement 1952 1956 := RowAgreement.join within_015_031_agreement within_015_032_agreement
private def within_015_034 : CountSegment 1 10 13120 1956 1958 1006 1007 := CountSegment.join atomic_1957 atomic_1958
private theorem within_015_034_agreement : RowAgreement 1956 1958 := RowAgreement.join atomic_1957_agreement atomic_1958_agreement
private def within_015_035 : CountSegment 1 10 13120 1958 1960 1007 1009 := CountSegment.join atomic_1959 atomic_1960
private theorem within_015_035_agreement : RowAgreement 1958 1960 := RowAgreement.join atomic_1959_agreement atomic_1960_agreement
private def within_015_036 : CountSegment 1 10 13120 1956 1960 1006 1009 := CountSegment.join within_015_034 within_015_035
private theorem within_015_036_agreement : RowAgreement 1956 1960 := RowAgreement.join within_015_034_agreement within_015_035_agreement
private def within_015_037 : CountSegment 1 10 13120 1952 1960 1004 1009 := CountSegment.join within_015_033 within_015_036
private theorem within_015_037_agreement : RowAgreement 1952 1960 := RowAgreement.join within_015_033_agreement within_015_036_agreement
private def within_015_038 : CountSegment 1 10 13120 1960 1962 1009 1010 := CountSegment.join atomic_1961 atomic_1962
private theorem within_015_038_agreement : RowAgreement 1960 1962 := RowAgreement.join atomic_1961_agreement atomic_1962_agreement
private def within_015_039 : CountSegment 1 10 13120 1962 1964 1010 1011 := CountSegment.join atomic_1963 atomic_1964
private theorem within_015_039_agreement : RowAgreement 1962 1964 := RowAgreement.join atomic_1963_agreement atomic_1964_agreement
private def within_015_040 : CountSegment 1 10 13120 1960 1964 1009 1011 := CountSegment.join within_015_038 within_015_039
private theorem within_015_040_agreement : RowAgreement 1960 1964 := RowAgreement.join within_015_038_agreement within_015_039_agreement
private def within_015_041 : CountSegment 1 10 13120 1964 1966 1011 1011 := CountSegment.join atomic_1965 atomic_1966
private theorem within_015_041_agreement : RowAgreement 1964 1966 := RowAgreement.join atomic_1965_agreement atomic_1966_agreement
private def within_015_042 : CountSegment 1 10 13120 1966 1968 1011 1011 := CountSegment.join atomic_1967 atomic_1968
private theorem within_015_042_agreement : RowAgreement 1966 1968 := RowAgreement.join atomic_1967_agreement atomic_1968_agreement
private def within_015_043 : CountSegment 1 10 13120 1964 1968 1011 1011 := CountSegment.join within_015_041 within_015_042
private theorem within_015_043_agreement : RowAgreement 1964 1968 := RowAgreement.join within_015_041_agreement within_015_042_agreement
private def within_015_044 : CountSegment 1 10 13120 1960 1968 1009 1011 := CountSegment.join within_015_040 within_015_043
private theorem within_015_044_agreement : RowAgreement 1960 1968 := RowAgreement.join within_015_040_agreement within_015_043_agreement
private def within_015_045 : CountSegment 1 10 13120 1952 1968 1004 1011 := CountSegment.join within_015_037 within_015_044
private theorem within_015_045_agreement : RowAgreement 1952 1968 := RowAgreement.join within_015_037_agreement within_015_044_agreement
private def within_015_046 : CountSegment 1 10 13120 1968 1970 1011 1012 := CountSegment.join atomic_1969 atomic_1970
private theorem within_015_046_agreement : RowAgreement 1968 1970 := RowAgreement.join atomic_1969_agreement atomic_1970_agreement
private def within_015_047 : CountSegment 1 10 13120 1970 1972 1012 1012 := CountSegment.join atomic_1971 atomic_1972
private theorem within_015_047_agreement : RowAgreement 1970 1972 := RowAgreement.join atomic_1971_agreement atomic_1972_agreement
private def within_015_048 : CountSegment 1 10 13120 1968 1972 1011 1012 := CountSegment.join within_015_046 within_015_047
private theorem within_015_048_agreement : RowAgreement 1968 1972 := RowAgreement.join within_015_046_agreement within_015_047_agreement
private def within_015_049 : CountSegment 1 10 13120 1972 1974 1012 1013 := CountSegment.join atomic_1973 atomic_1974
private theorem within_015_049_agreement : RowAgreement 1972 1974 := RowAgreement.join atomic_1973_agreement atomic_1974_agreement
private def within_015_050 : CountSegment 1 10 13120 1974 1976 1013 1013 := CountSegment.join atomic_1975 atomic_1976
private theorem within_015_050_agreement : RowAgreement 1974 1976 := RowAgreement.join atomic_1975_agreement atomic_1976_agreement
private def within_015_051 : CountSegment 1 10 13120 1972 1976 1012 1013 := CountSegment.join within_015_049 within_015_050
private theorem within_015_051_agreement : RowAgreement 1972 1976 := RowAgreement.join within_015_049_agreement within_015_050_agreement
private def within_015_052 : CountSegment 1 10 13120 1968 1976 1011 1013 := CountSegment.join within_015_048 within_015_051
private theorem within_015_052_agreement : RowAgreement 1968 1976 := RowAgreement.join within_015_048_agreement within_015_051_agreement
private def within_015_053 : CountSegment 1 10 13120 1976 1978 1013 1014 := CountSegment.join atomic_1977 atomic_1978
private theorem within_015_053_agreement : RowAgreement 1976 1978 := RowAgreement.join atomic_1977_agreement atomic_1978_agreement
private def within_015_054 : CountSegment 1 10 13120 1978 1980 1014 1014 := CountSegment.join atomic_1979 atomic_1980
private theorem within_015_054_agreement : RowAgreement 1978 1980 := RowAgreement.join atomic_1979_agreement atomic_1980_agreement
private def within_015_055 : CountSegment 1 10 13120 1976 1980 1013 1014 := CountSegment.join within_015_053 within_015_054
private theorem within_015_055_agreement : RowAgreement 1976 1980 := RowAgreement.join within_015_053_agreement within_015_054_agreement
private def within_015_056 : CountSegment 1 10 13120 1980 1982 1014 1014 := CountSegment.join atomic_1981 atomic_1982
private theorem within_015_056_agreement : RowAgreement 1980 1982 := RowAgreement.join atomic_1981_agreement atomic_1982_agreement
private def within_015_057 : CountSegment 1 10 13120 1982 1984 1014 1016 := CountSegment.join atomic_1983 atomic_1984
private theorem within_015_057_agreement : RowAgreement 1982 1984 := RowAgreement.join atomic_1983_agreement atomic_1984_agreement
private def within_015_058 : CountSegment 1 10 13120 1980 1984 1014 1016 := CountSegment.join within_015_056 within_015_057
private theorem within_015_058_agreement : RowAgreement 1980 1984 := RowAgreement.join within_015_056_agreement within_015_057_agreement
private def within_015_059 : CountSegment 1 10 13120 1976 1984 1013 1016 := CountSegment.join within_015_055 within_015_058
private theorem within_015_059_agreement : RowAgreement 1976 1984 := RowAgreement.join within_015_055_agreement within_015_058_agreement
private def within_015_060 : CountSegment 1 10 13120 1968 1984 1011 1016 := CountSegment.join within_015_052 within_015_059
private theorem within_015_060_agreement : RowAgreement 1968 1984 := RowAgreement.join within_015_052_agreement within_015_059_agreement
private def within_015_061 : CountSegment 1 10 13120 1952 1984 1004 1016 := CountSegment.join within_015_045 within_015_060
private theorem within_015_061_agreement : RowAgreement 1952 1984 := RowAgreement.join within_015_045_agreement within_015_060_agreement
private def within_015_062 : CountSegment 1 10 13120 1920 1984 986 1016 := CountSegment.join within_015_030 within_015_061
private theorem within_015_062_agreement : RowAgreement 1920 1984 := RowAgreement.join within_015_030_agreement within_015_061_agreement
private def within_015_063 : CountSegment 1 10 13120 1984 1986 1016 1016 := CountSegment.join atomic_1985 atomic_1986
private theorem within_015_063_agreement : RowAgreement 1984 1986 := RowAgreement.join atomic_1985_agreement atomic_1986_agreement
private def within_015_064 : CountSegment 1 10 13120 1986 1988 1016 1017 := CountSegment.join atomic_1987 atomic_1988
private theorem within_015_064_agreement : RowAgreement 1986 1988 := RowAgreement.join atomic_1987_agreement atomic_1988_agreement
private def within_015_065 : CountSegment 1 10 13120 1984 1988 1016 1017 := CountSegment.join within_015_063 within_015_064
private theorem within_015_065_agreement : RowAgreement 1984 1988 := RowAgreement.join within_015_063_agreement within_015_064_agreement
private def within_015_066 : CountSegment 1 10 13120 1988 1990 1017 1017 := CountSegment.join atomic_1989 atomic_1990
private theorem within_015_066_agreement : RowAgreement 1988 1990 := RowAgreement.join atomic_1989_agreement atomic_1990_agreement
private def within_015_067 : CountSegment 1 10 13120 1990 1992 1017 1018 := CountSegment.join atomic_1991 atomic_1992
private theorem within_015_067_agreement : RowAgreement 1990 1992 := RowAgreement.join atomic_1991_agreement atomic_1992_agreement
private def within_015_068 : CountSegment 1 10 13120 1988 1992 1017 1018 := CountSegment.join within_015_066 within_015_067
private theorem within_015_068_agreement : RowAgreement 1988 1992 := RowAgreement.join within_015_066_agreement within_015_067_agreement
private def within_015_069 : CountSegment 1 10 13120 1984 1992 1016 1018 := CountSegment.join within_015_065 within_015_068
private theorem within_015_069_agreement : RowAgreement 1984 1992 := RowAgreement.join within_015_065_agreement within_015_068_agreement
private def within_015_070 : CountSegment 1 10 13120 1992 1994 1018 1019 := CountSegment.join atomic_1993 atomic_1994
private theorem within_015_070_agreement : RowAgreement 1992 1994 := RowAgreement.join atomic_1993_agreement atomic_1994_agreement
private def within_015_071 : CountSegment 1 10 13120 1994 1996 1019 1020 := CountSegment.join atomic_1995 atomic_1996
private theorem within_015_071_agreement : RowAgreement 1994 1996 := RowAgreement.join atomic_1995_agreement atomic_1996_agreement
private def within_015_072 : CountSegment 1 10 13120 1992 1996 1018 1020 := CountSegment.join within_015_070 within_015_071
private theorem within_015_072_agreement : RowAgreement 1992 1996 := RowAgreement.join within_015_070_agreement within_015_071_agreement
private def within_015_073 : CountSegment 1 10 13120 1996 1998 1020 1023 := CountSegment.join atomic_1997 atomic_1998
private theorem within_015_073_agreement : RowAgreement 1996 1998 := RowAgreement.join atomic_1997_agreement atomic_1998_agreement
private def within_015_074 : CountSegment 1 10 13120 1998 2000 1023 1024 := CountSegment.join atomic_1999 atomic_2000
private theorem within_015_074_agreement : RowAgreement 1998 2000 := RowAgreement.join atomic_1999_agreement atomic_2000_agreement
private def within_015_075 : CountSegment 1 10 13120 1996 2000 1020 1024 := CountSegment.join within_015_073 within_015_074
private theorem within_015_075_agreement : RowAgreement 1996 2000 := RowAgreement.join within_015_073_agreement within_015_074_agreement
private def within_015_076 : CountSegment 1 10 13120 1992 2000 1018 1024 := CountSegment.join within_015_072 within_015_075
private theorem within_015_076_agreement : RowAgreement 1992 2000 := RowAgreement.join within_015_072_agreement within_015_075_agreement
private def within_015_077 : CountSegment 1 10 13120 1984 2000 1016 1024 := CountSegment.join within_015_069 within_015_076
private theorem within_015_077_agreement : RowAgreement 1984 2000 := RowAgreement.join within_015_069_agreement within_015_076_agreement
private def within_015_078 : CountSegment 1 10 13120 2000 2002 1024 1025 := CountSegment.join atomic_2001 atomic_2002
private theorem within_015_078_agreement : RowAgreement 2000 2002 := RowAgreement.join atomic_2001_agreement atomic_2002_agreement
private def within_015_079 : CountSegment 1 10 13120 2002 2004 1025 1026 := CountSegment.join atomic_2003 atomic_2004
private theorem within_015_079_agreement : RowAgreement 2002 2004 := RowAgreement.join atomic_2003_agreement atomic_2004_agreement
private def within_015_080 : CountSegment 1 10 13120 2000 2004 1024 1026 := CountSegment.join within_015_078 within_015_079
private theorem within_015_080_agreement : RowAgreement 2000 2004 := RowAgreement.join within_015_078_agreement within_015_079_agreement
private def within_015_081 : CountSegment 1 10 13120 2004 2006 1026 1026 := CountSegment.join atomic_2005 atomic_2006
private theorem within_015_081_agreement : RowAgreement 2004 2006 := RowAgreement.join atomic_2005_agreement atomic_2006_agreement
private def within_015_082 : CountSegment 1 10 13120 2006 2008 1026 1026 := CountSegment.join atomic_2007 atomic_2008
private theorem within_015_082_agreement : RowAgreement 2006 2008 := RowAgreement.join atomic_2007_agreement atomic_2008_agreement
private def within_015_083 : CountSegment 1 10 13120 2004 2008 1026 1026 := CountSegment.join within_015_081 within_015_082
private theorem within_015_083_agreement : RowAgreement 2004 2008 := RowAgreement.join within_015_081_agreement within_015_082_agreement
private def within_015_084 : CountSegment 1 10 13120 2000 2008 1024 1026 := CountSegment.join within_015_080 within_015_083
private theorem within_015_084_agreement : RowAgreement 2000 2008 := RowAgreement.join within_015_080_agreement within_015_083_agreement
private def within_015_085 : CountSegment 1 10 13120 2008 2010 1026 1027 := CountSegment.join atomic_2009 atomic_2010
private theorem within_015_085_agreement : RowAgreement 2008 2010 := RowAgreement.join atomic_2009_agreement atomic_2010_agreement
private def within_015_086 : CountSegment 1 10 13120 2010 2012 1027 1028 := CountSegment.join atomic_2011 atomic_2012
private theorem within_015_086_agreement : RowAgreement 2010 2012 := RowAgreement.join atomic_2011_agreement atomic_2012_agreement
private def within_015_087 : CountSegment 1 10 13120 2008 2012 1026 1028 := CountSegment.join within_015_085 within_015_086
private theorem within_015_087_agreement : RowAgreement 2008 2012 := RowAgreement.join within_015_085_agreement within_015_086_agreement
private def within_015_088 : CountSegment 1 10 13120 2012 2014 1028 1031 := CountSegment.join atomic_2013 atomic_2014
private theorem within_015_088_agreement : RowAgreement 2012 2014 := RowAgreement.join atomic_2013_agreement atomic_2014_agreement
private def within_015_089 : CountSegment 1 10 13120 2014 2016 1031 1032 := CountSegment.join atomic_2015 atomic_2016
private theorem within_015_089_agreement : RowAgreement 2014 2016 := RowAgreement.join atomic_2015_agreement atomic_2016_agreement
private def within_015_090 : CountSegment 1 10 13120 2012 2016 1028 1032 := CountSegment.join within_015_088 within_015_089
private theorem within_015_090_agreement : RowAgreement 2012 2016 := RowAgreement.join within_015_088_agreement within_015_089_agreement
private def within_015_091 : CountSegment 1 10 13120 2008 2016 1026 1032 := CountSegment.join within_015_087 within_015_090
private theorem within_015_091_agreement : RowAgreement 2008 2016 := RowAgreement.join within_015_087_agreement within_015_090_agreement
private def within_015_092 : CountSegment 1 10 13120 2000 2016 1024 1032 := CountSegment.join within_015_084 within_015_091
private theorem within_015_092_agreement : RowAgreement 2000 2016 := RowAgreement.join within_015_084_agreement within_015_091_agreement
private def within_015_093 : CountSegment 1 10 13120 1984 2016 1016 1032 := CountSegment.join within_015_077 within_015_092
private theorem within_015_093_agreement : RowAgreement 1984 2016 := RowAgreement.join within_015_077_agreement within_015_092_agreement
private def within_015_094 : CountSegment 1 10 13120 2016 2018 1032 1033 := CountSegment.join atomic_2017 atomic_2018
private theorem within_015_094_agreement : RowAgreement 2016 2018 := RowAgreement.join atomic_2017_agreement atomic_2018_agreement
private def within_015_095 : CountSegment 1 10 13120 2018 2020 1033 1033 := CountSegment.join atomic_2019 atomic_2020
private theorem within_015_095_agreement : RowAgreement 2018 2020 := RowAgreement.join atomic_2019_agreement atomic_2020_agreement
private def within_015_096 : CountSegment 1 10 13120 2016 2020 1032 1033 := CountSegment.join within_015_094 within_015_095
private theorem within_015_096_agreement : RowAgreement 2016 2020 := RowAgreement.join within_015_094_agreement within_015_095_agreement
private def within_015_097 : CountSegment 1 10 13120 2020 2022 1033 1033 := CountSegment.join atomic_2021 atomic_2022
private theorem within_015_097_agreement : RowAgreement 2020 2022 := RowAgreement.join atomic_2021_agreement atomic_2022_agreement
private def within_015_098 : CountSegment 1 10 13120 2022 2024 1033 1035 := CountSegment.join atomic_2023 atomic_2024
private theorem within_015_098_agreement : RowAgreement 2022 2024 := RowAgreement.join atomic_2023_agreement atomic_2024_agreement
private def within_015_099 : CountSegment 1 10 13120 2020 2024 1033 1035 := CountSegment.join within_015_097 within_015_098
private theorem within_015_099_agreement : RowAgreement 2020 2024 := RowAgreement.join within_015_097_agreement within_015_098_agreement
private def within_015_100 : CountSegment 1 10 13120 2016 2024 1032 1035 := CountSegment.join within_015_096 within_015_099
private theorem within_015_100_agreement : RowAgreement 2016 2024 := RowAgreement.join within_015_096_agreement within_015_099_agreement
private def within_015_101 : CountSegment 1 10 13120 2024 2026 1035 1035 := CountSegment.join atomic_2025 atomic_2026
private theorem within_015_101_agreement : RowAgreement 2024 2026 := RowAgreement.join atomic_2025_agreement atomic_2026_agreement
private def within_015_102 : CountSegment 1 10 13120 2026 2028 1035 1037 := CountSegment.join atomic_2027 atomic_2028
private theorem within_015_102_agreement : RowAgreement 2026 2028 := RowAgreement.join atomic_2027_agreement atomic_2028_agreement
private def within_015_103 : CountSegment 1 10 13120 2024 2028 1035 1037 := CountSegment.join within_015_101 within_015_102
private theorem within_015_103_agreement : RowAgreement 2024 2028 := RowAgreement.join within_015_101_agreement within_015_102_agreement
private def within_015_104 : CountSegment 1 10 13120 2028 2030 1037 1037 := CountSegment.join atomic_2029 atomic_2030
private theorem within_015_104_agreement : RowAgreement 2028 2030 := RowAgreement.join atomic_2029_agreement atomic_2030_agreement
private def within_015_105 : CountSegment 1 10 13120 2030 2032 1037 1038 := CountSegment.join atomic_2031 atomic_2032
private theorem within_015_105_agreement : RowAgreement 2030 2032 := RowAgreement.join atomic_2031_agreement atomic_2032_agreement
private def within_015_106 : CountSegment 1 10 13120 2028 2032 1037 1038 := CountSegment.join within_015_104 within_015_105
private theorem within_015_106_agreement : RowAgreement 2028 2032 := RowAgreement.join within_015_104_agreement within_015_105_agreement
private def within_015_107 : CountSegment 1 10 13120 2024 2032 1035 1038 := CountSegment.join within_015_103 within_015_106
private theorem within_015_107_agreement : RowAgreement 2024 2032 := RowAgreement.join within_015_103_agreement within_015_106_agreement
private def within_015_108 : CountSegment 1 10 13120 2016 2032 1032 1038 := CountSegment.join within_015_100 within_015_107
private theorem within_015_108_agreement : RowAgreement 2016 2032 := RowAgreement.join within_015_100_agreement within_015_107_agreement
private def within_015_109 : CountSegment 1 10 13120 2032 2034 1038 1040 := CountSegment.join atomic_2033 atomic_2034
private theorem within_015_109_agreement : RowAgreement 2032 2034 := RowAgreement.join atomic_2033_agreement atomic_2034_agreement
private def within_015_110 : CountSegment 1 10 13120 2034 2036 1040 1041 := CountSegment.join atomic_2035 atomic_2036
private theorem within_015_110_agreement : RowAgreement 2034 2036 := RowAgreement.join atomic_2035_agreement atomic_2036_agreement
private def within_015_111 : CountSegment 1 10 13120 2032 2036 1038 1041 := CountSegment.join within_015_109 within_015_110
private theorem within_015_111_agreement : RowAgreement 2032 2036 := RowAgreement.join within_015_109_agreement within_015_110_agreement
private def within_015_112 : CountSegment 1 10 13120 2036 2038 1041 1042 := CountSegment.join atomic_2037 atomic_2038
private theorem within_015_112_agreement : RowAgreement 2036 2038 := RowAgreement.join atomic_2037_agreement atomic_2038_agreement
private def within_015_113 : CountSegment 1 10 13120 2038 2040 1042 1043 := CountSegment.join atomic_2039 atomic_2040
private theorem within_015_113_agreement : RowAgreement 2038 2040 := RowAgreement.join atomic_2039_agreement atomic_2040_agreement
private def within_015_114 : CountSegment 1 10 13120 2036 2040 1041 1043 := CountSegment.join within_015_112 within_015_113
private theorem within_015_114_agreement : RowAgreement 2036 2040 := RowAgreement.join within_015_112_agreement within_015_113_agreement
private def within_015_115 : CountSegment 1 10 13120 2032 2040 1038 1043 := CountSegment.join within_015_111 within_015_114
private theorem within_015_115_agreement : RowAgreement 2032 2040 := RowAgreement.join within_015_111_agreement within_015_114_agreement
private def within_015_116 : CountSegment 1 10 13120 2040 2042 1043 1044 := CountSegment.join atomic_2041 atomic_2042
private theorem within_015_116_agreement : RowAgreement 2040 2042 := RowAgreement.join atomic_2041_agreement atomic_2042_agreement
private def within_015_117 : CountSegment 1 10 13120 2042 2044 1044 1045 := CountSegment.join atomic_2043 atomic_2044
private theorem within_015_117_agreement : RowAgreement 2042 2044 := RowAgreement.join atomic_2043_agreement atomic_2044_agreement
private def within_015_118 : CountSegment 1 10 13120 2040 2044 1043 1045 := CountSegment.join within_015_116 within_015_117
private theorem within_015_118_agreement : RowAgreement 2040 2044 := RowAgreement.join within_015_116_agreement within_015_117_agreement
private def within_015_119 : CountSegment 1 10 13120 2044 2046 1045 1046 := CountSegment.join atomic_2045 atomic_2046
private theorem within_015_119_agreement : RowAgreement 2044 2046 := RowAgreement.join atomic_2045_agreement atomic_2046_agreement
private def within_015_120 : CountSegment 1 10 13120 2046 2048 1046 1046 := CountSegment.join atomic_2047 atomic_2048
private theorem within_015_120_agreement : RowAgreement 2046 2048 := RowAgreement.join atomic_2047_agreement atomic_2048_agreement
private def within_015_121 : CountSegment 1 10 13120 2044 2048 1045 1046 := CountSegment.join within_015_119 within_015_120
private theorem within_015_121_agreement : RowAgreement 2044 2048 := RowAgreement.join within_015_119_agreement within_015_120_agreement
private def within_015_122 : CountSegment 1 10 13120 2040 2048 1043 1046 := CountSegment.join within_015_118 within_015_121
private theorem within_015_122_agreement : RowAgreement 2040 2048 := RowAgreement.join within_015_118_agreement within_015_121_agreement
private def within_015_123 : CountSegment 1 10 13120 2032 2048 1038 1046 := CountSegment.join within_015_115 within_015_122
private theorem within_015_123_agreement : RowAgreement 2032 2048 := RowAgreement.join within_015_115_agreement within_015_122_agreement
private def within_015_124 : CountSegment 1 10 13120 2016 2048 1032 1046 := CountSegment.join within_015_108 within_015_123
private theorem within_015_124_agreement : RowAgreement 2016 2048 := RowAgreement.join within_015_108_agreement within_015_123_agreement
private def within_015_125 : CountSegment 1 10 13120 1984 2048 1016 1046 := CountSegment.join within_015_093 within_015_124
private theorem within_015_125_agreement : RowAgreement 1984 2048 := RowAgreement.join within_015_093_agreement within_015_124_agreement
private def within_015_126 : CountSegment 1 10 13120 1920 2048 986 1046 := CountSegment.join within_015_062 within_015_125
private theorem within_015_126_agreement : RowAgreement 1920 2048 := RowAgreement.join within_015_062_agreement within_015_125_agreement
def segment_015 : CountSegment 1 10 13120 1920 2048 986 1046 := within_015_126
theorem agreement_015 : RowAgreement 1920 2048 := within_015_126_agreement
end ProjectEulerSolutions.P58Numerical
