import ProjectEulerSolutions.Lib.SpiralPrimeSegments
import ProjectEulerSolutions.Lib.SpiralPrimeCertificates.All
import ProjectEulerSolutions.Lib.MillerRabin
import ProjectEulerSolutions.Lib.SpiralMillerAgreement
/-! Finite spiral row and counter certificates. Python execution is checked separately. -/
namespace ProjectEulerSolutions.P58Numerical
open SpiralPrimes
set_option Elab.async false
def row_2945 : RowCertificate 2945 where
  first := false
  first_correct := by
    have composite : ¬ Nat.Prime (corner 2945 1) := by change ¬ Nat.Prime 34697991; exact PrattCertificates.composite_of_product (factor := 3) (cofactor := 11565997) rfl (by decide) (by decide)
    exact decide_eq_false composite
  second := false
  second_correct := by
    have composite : ¬ Nat.Prime (corner 2945 2) := by change ¬ Nat.Prime 34692101; exact PrattCertificates.composite_of_product (factor := 1381) (cofactor := 25121) rfl (by decide) (by decide)
    exact decide_eq_false composite
  third := false
  third_correct := by
    have composite : ¬ Nat.Prime (corner 2945 3) := by change ¬ Nat.Prime 34686211; exact PrattCertificates.composite_of_product (factor := 7) (cofactor := 4955173) rfl (by decide) (by decide)
    exact decide_eq_false composite
theorem row_count_2945 : row 2945 = 0 := (row_2945.checked).trans rfl
theorem model_count_2945 : modelRow 2945 = 0 := by decide
theorem modelrow_agrees_2945 : modelRow 2945 = row 2945 := model_count_2945.trans row_count_2945.symm
def row_2946 : RowCertificate 2946 where
  first := false
  first_correct := by
    have composite : ¬ Nat.Prime (corner 2946 1) := by change ¬ Nat.Prime 34721557; exact PrattCertificates.composite_of_product (factor := 13) (cofactor := 2670889) rfl (by decide) (by decide)
    exact decide_eq_false composite
  second := false
  second_correct := by
    have composite : ¬ Nat.Prime (corner 2946 2) := by change ¬ Nat.Prime 34715665; exact PrattCertificates.composite_of_product (factor := 5) (cofactor := 6943133) rfl (by decide) (by decide)
    exact decide_eq_false composite
  third := false
  third_correct := by
    have composite : ¬ Nat.Prime (corner 2946 3) := by change ¬ Nat.Prime 34709773; exact PrattCertificates.composite_of_product (factor := 7) (cofactor := 4958539) rfl (by decide) (by decide)
    exact decide_eq_false composite
theorem row_count_2946 : row 2946 = 0 := (row_2946.checked).trans rfl
theorem model_count_2946 : modelRow 2946 = 0 := by decide
theorem modelrow_agrees_2946 : modelRow 2946 = row 2946 := model_count_2946.trans row_count_2946.symm
def row_2947 : RowCertificate 2947 where
  first := true
  first_correct := by
    have prime : Nat.Prime (corner 2947 1) := by change Nat.Prime 34745131; exact ProjectEulerSolutions.P58PrimeNodes.p_34745131.prime
    exact decide_eq_true prime
  second := false
  second_correct := by
    have composite : ¬ Nat.Prime (corner 2947 2) := by change ¬ Nat.Prime 34739237; exact PrattCertificates.composite_of_product (factor := 13) (cofactor := 2672249) rfl (by decide) (by decide)
    exact decide_eq_false composite
  third := false
  third_correct := by
    have composite : ¬ Nat.Prime (corner 2947 3) := by change ¬ Nat.Prime 34733343; exact PrattCertificates.composite_of_product (factor := 3) (cofactor := 11577781) rfl (by decide) (by decide)
    exact decide_eq_false composite
theorem row_count_2947 : row 2947 = 1 := (row_2947.checked).trans rfl
theorem model_count_2947 : modelRow 2947 = 1 := by decide
theorem modelrow_agrees_2947 : modelRow 2947 = row 2947 := model_count_2947.trans row_count_2947.symm
def row_2948 : RowCertificate 2948 where
  first := false
  first_correct := by
    have composite : ¬ Nat.Prime (corner 2948 1) := by change ¬ Nat.Prime 34768713; exact PrattCertificates.composite_of_product (factor := 3) (cofactor := 11589571) rfl (by decide) (by decide)
    exact decide_eq_false composite
  second := false
  second_correct := by
    have composite : ¬ Nat.Prime (corner 2948 2) := by change ¬ Nat.Prime 34762817; exact PrattCertificates.composite_of_product (factor := 197) (cofactor := 176461) rfl (by decide) (by decide)
    exact decide_eq_false composite
  third := false
  third_correct := by
    have composite : ¬ Nat.Prime (corner 2948 3) := by change ¬ Nat.Prime 34756921; exact PrattCertificates.composite_of_product (factor := 31) (cofactor := 1121191) rfl (by decide) (by decide)
    exact decide_eq_false composite
theorem row_count_2948 : row 2948 = 0 := (row_2948.checked).trans rfl
theorem model_count_2948 : modelRow 2948 = 0 := by decide
theorem modelrow_agrees_2948 : modelRow 2948 = row 2948 := model_count_2948.trans row_count_2948.symm
def row_2949 : RowCertificate 2949 where
  first := false
  first_correct := by
    have composite : ¬ Nat.Prime (corner 2949 1) := by change ¬ Nat.Prime 34792303; exact PrattCertificates.composite_of_product (factor := 7) (cofactor := 4970329) rfl (by decide) (by decide)
    exact decide_eq_false composite
  second := false
  second_correct := by
    have composite : ¬ Nat.Prime (corner 2949 2) := by change ¬ Nat.Prime 34786405; exact PrattCertificates.composite_of_product (factor := 5) (cofactor := 6957281) rfl (by decide) (by decide)
    exact decide_eq_false composite
  third := false
  third_correct := by
    have composite : ¬ Nat.Prime (corner 2949 3) := by change ¬ Nat.Prime 34780507; exact PrattCertificates.composite_of_product (factor := 19) (cofactor := 1830553) rfl (by decide) (by decide)
    exact decide_eq_false composite
theorem row_count_2949 : row 2949 = 0 := (row_2949.checked).trans rfl
theorem model_count_2949 : modelRow 2949 = 0 := by decide
theorem modelrow_agrees_2949 : modelRow 2949 = row 2949 := model_count_2949.trans row_count_2949.symm
def row_2950 : RowCertificate 2950 where
  first := false
  first_correct := by
    have composite : ¬ Nat.Prime (corner 2950 1) := by change ¬ Nat.Prime 34815901; exact PrattCertificates.composite_of_product (factor := 2113) (cofactor := 16477) rfl (by decide) (by decide)
    exact decide_eq_false composite
  second := false
  second_correct := by
    have composite : ¬ Nat.Prime (corner 2950 2) := by change ¬ Nat.Prime 34810001; exact PrattCertificates.composite_of_product (factor := 509) (cofactor := 68389) rfl (by decide) (by decide)
    exact decide_eq_false composite
  third := false
  third_correct := by
    have composite : ¬ Nat.Prime (corner 2950 3) := by change ¬ Nat.Prime 34804101; exact PrattCertificates.composite_of_product (factor := 3) (cofactor := 11601367) rfl (by decide) (by decide)
    exact decide_eq_false composite
theorem row_count_2950 : row 2950 = 0 := (row_2950.checked).trans rfl
theorem model_count_2950 : modelRow 2950 = 0 := by decide
theorem modelrow_agrees_2950 : modelRow 2950 = row 2950 := model_count_2950.trans row_count_2950.symm
def row_2951 : RowCertificate 2951 where
  first := false
  first_correct := by
    have composite : ¬ Nat.Prime (corner 2951 1) := by change ¬ Nat.Prime 34839507; exact PrattCertificates.composite_of_product (factor := 3) (cofactor := 11613169) rfl (by decide) (by decide)
    exact decide_eq_false composite
  second := false
  second_correct := by
    have composite : ¬ Nat.Prime (corner 2951 2) := by change ¬ Nat.Prime 34833605; exact PrattCertificates.composite_of_product (factor := 5) (cofactor := 6966721) rfl (by decide) (by decide)
    exact decide_eq_false composite
  third := false
  third_correct := by
    have composite : ¬ Nat.Prime (corner 2951 3) := by change ¬ Nat.Prime 34827703; exact PrattCertificates.composite_of_product (factor := 19) (cofactor := 1833037) rfl (by decide) (by decide)
    exact decide_eq_false composite
theorem row_count_2951 : row 2951 = 0 := (row_2951.checked).trans rfl
theorem model_count_2951 : modelRow 2951 = 0 := by decide
theorem modelrow_agrees_2951 : modelRow 2951 = row 2951 := model_count_2951.trans row_count_2951.symm
def row_2952 : RowCertificate 2952 where
  first := false
  first_correct := by
    have composite : ¬ Nat.Prime (corner 2952 1) := by change ¬ Nat.Prime 34863121; exact PrattCertificates.composite_of_product (factor := 73) (cofactor := 477577) rfl (by decide) (by decide)
    exact decide_eq_false composite
  second := false
  second_correct := by
    have composite : ¬ Nat.Prime (corner 2952 2) := by change ¬ Nat.Prime 34857217; exact PrattCertificates.composite_of_product (factor := 29) (cofactor := 1201973) rfl (by decide) (by decide)
    exact decide_eq_false composite
  third := false
  third_correct := by
    have composite : ¬ Nat.Prime (corner 2952 3) := by change ¬ Nat.Prime 34851313; exact PrattCertificates.composite_of_product (factor := 7) (cofactor := 4978759) rfl (by decide) (by decide)
    exact decide_eq_false composite
theorem row_count_2952 : row 2952 = 0 := (row_2952.checked).trans rfl
theorem model_count_2952 : modelRow 2952 = 0 := by decide
theorem modelrow_agrees_2952 : modelRow 2952 = row 2952 := model_count_2952.trans row_count_2952.symm
def row_2953 : RowCertificate 2953 where
  first := true
  first_correct := by
    have prime : Nat.Prime (corner 2953 1) := by change Nat.Prime 34886743; exact ProjectEulerSolutions.P58PrimeNodes.p_34886743.prime
    exact decide_eq_true prime
  second := false
  second_correct := by
    have composite : ¬ Nat.Prime (corner 2953 2) := by change ¬ Nat.Prime 34880837; exact PrattCertificates.composite_of_product (factor := 53) (cofactor := 658129) rfl (by decide) (by decide)
    exact decide_eq_false composite
  third := false
  third_correct := by
    have composite : ¬ Nat.Prime (corner 2953 3) := by change ¬ Nat.Prime 34874931; exact PrattCertificates.composite_of_product (factor := 3) (cofactor := 11624977) rfl (by decide) (by decide)
    exact decide_eq_false composite
theorem row_count_2953 : row 2953 = 1 := (row_2953.checked).trans rfl
theorem model_count_2953 : modelRow 2953 = 1 := by decide
theorem modelrow_agrees_2953 : modelRow 2953 = row 2953 := model_count_2953.trans row_count_2953.symm
def row_2954 : RowCertificate 2954 where
  first := false
  first_correct := by
    have composite : ¬ Nat.Prime (corner 2954 1) := by change ¬ Nat.Prime 34910373; exact PrattCertificates.composite_of_product (factor := 3) (cofactor := 11636791) rfl (by decide) (by decide)
    exact decide_eq_false composite
  second := false
  second_correct := by
    have composite : ¬ Nat.Prime (corner 2954 2) := by change ¬ Nat.Prime 34904465; exact PrattCertificates.composite_of_product (factor := 5) (cofactor := 6980893) rfl (by decide) (by decide)
    exact decide_eq_false composite
  third := true
  third_correct := by
    have prime : Nat.Prime (corner 2954 3) := by change Nat.Prime 34898557; exact ProjectEulerSolutions.P58PrimeNodes.p_34898557.prime
    exact decide_eq_true prime
theorem row_count_2954 : row 2954 = 1 := (row_2954.checked).trans rfl
theorem model_count_2954 : modelRow 2954 = 1 := by decide
theorem modelrow_agrees_2954 : modelRow 2954 = row 2954 := model_count_2954.trans row_count_2954.symm
def row_2955 : RowCertificate 2955 where
  first := false
  first_correct := by
    have composite : ¬ Nat.Prime (corner 2955 1) := by change ¬ Nat.Prime 34934011; exact PrattCertificates.composite_of_product (factor := 7) (cofactor := 4990573) rfl (by decide) (by decide)
    exact decide_eq_false composite
  second := false
  second_correct := by
    have composite : ¬ Nat.Prime (corner 2955 2) := by change ¬ Nat.Prime 34928101; exact PrattCertificates.composite_of_product (factor := 13) (cofactor := 2686777) rfl (by decide) (by decide)
    exact decide_eq_false composite
  third := false
  third_correct := by
    have composite : ¬ Nat.Prime (corner 2955 3) := by change ¬ Nat.Prime 34922191; exact PrattCertificates.composite_of_product (factor := 37) (cofactor := 943843) rfl (by decide) (by decide)
    exact decide_eq_false composite
theorem row_count_2955 : row 2955 = 0 := (row_2955.checked).trans rfl
theorem model_count_2955 : modelRow 2955 = 0 := by decide
theorem modelrow_agrees_2955 : modelRow 2955 = row 2955 := model_count_2955.trans row_count_2955.symm
def row_2956 : RowCertificate 2956 where
  first := false
  first_correct := by
    have composite : ¬ Nat.Prime (corner 2956 1) := by change ¬ Nat.Prime 34957657; exact PrattCertificates.composite_of_product (factor := 7) (cofactor := 4993951) rfl (by decide) (by decide)
    exact decide_eq_false composite
  second := false
  second_correct := by
    have composite : ¬ Nat.Prime (corner 2956 2) := by change ¬ Nat.Prime 34951745; exact PrattCertificates.composite_of_product (factor := 5) (cofactor := 6990349) rfl (by decide) (by decide)
    exact decide_eq_false composite
  third := false
  third_correct := by
    have composite : ¬ Nat.Prime (corner 2956 3) := by change ¬ Nat.Prime 34945833; exact PrattCertificates.composite_of_product (factor := 3) (cofactor := 11648611) rfl (by decide) (by decide)
    exact decide_eq_false composite
theorem row_count_2956 : row 2956 = 0 := (row_2956.checked).trans rfl
theorem model_count_2956 : modelRow 2956 = 0 := by decide
theorem modelrow_agrees_2956 : modelRow 2956 = row 2956 := model_count_2956.trans row_count_2956.symm
def row_2957 : RowCertificate 2957 where
  first := false
  first_correct := by
    have composite : ¬ Nat.Prime (corner 2957 1) := by change ¬ Nat.Prime 34981311; exact PrattCertificates.composite_of_product (factor := 3) (cofactor := 11660437) rfl (by decide) (by decide)
    exact decide_eq_false composite
  second := false
  second_correct := by
    have composite : ¬ Nat.Prime (corner 2957 2) := by change ¬ Nat.Prime 34975397; exact PrattCertificates.composite_of_product (factor := 37) (cofactor := 945281) rfl (by decide) (by decide)
    exact decide_eq_false composite
  third := true
  third_correct := by
    have prime : Nat.Prime (corner 2957 3) := by change Nat.Prime 34969483; exact ProjectEulerSolutions.P58PrimeNodes.p_34969483.prime
    exact decide_eq_true prime
theorem row_count_2957 : row 2957 = 1 := (row_2957.checked).trans rfl
theorem model_count_2957 : modelRow 2957 = 1 := by decide
theorem modelrow_agrees_2957 : modelRow 2957 = row 2957 := model_count_2957.trans row_count_2957.symm
def row_2958 : RowCertificate 2958 where
  first := false
  first_correct := by
    have composite : ¬ Nat.Prime (corner 2958 1) := by change ¬ Nat.Prime 35004973; exact PrattCertificates.composite_of_product (factor := 19) (cofactor := 1842367) rfl (by decide) (by decide)
    exact decide_eq_false composite
  second := false
  second_correct := by
    have composite : ¬ Nat.Prime (corner 2958 2) := by change ¬ Nat.Prime 34999057; exact PrattCertificates.composite_of_product (factor := 149) (cofactor := 234893) rfl (by decide) (by decide)
    exact decide_eq_false composite
  third := false
  third_correct := by
    have composite : ¬ Nat.Prime (corner 2958 3) := by change ¬ Nat.Prime 34993141; exact PrattCertificates.composite_of_product (factor := 31) (cofactor := 1128811) rfl (by decide) (by decide)
    exact decide_eq_false composite
theorem row_count_2958 : row 2958 = 0 := (row_2958.checked).trans rfl
theorem model_count_2958 : modelRow 2958 = 0 := by decide
theorem modelrow_agrees_2958 : modelRow 2958 = row 2958 := model_count_2958.trans row_count_2958.symm
def row_2959 : RowCertificate 2959 where
  first := false
  first_correct := by
    have composite : ¬ Nat.Prime (corner 2959 1) := by change ¬ Nat.Prime 35028643; exact PrattCertificates.composite_of_product (factor := 13) (cofactor := 2694511) rfl (by decide) (by decide)
    exact decide_eq_false composite
  second := false
  second_correct := by
    have composite : ¬ Nat.Prime (corner 2959 2) := by change ¬ Nat.Prime 35022725; exact PrattCertificates.composite_of_product (factor := 5) (cofactor := 7004545) rfl (by decide) (by decide)
    exact decide_eq_false composite
  third := false
  third_correct := by
    have composite : ¬ Nat.Prime (corner 2959 3) := by change ¬ Nat.Prime 35016807; exact PrattCertificates.composite_of_product (factor := 3) (cofactor := 11672269) rfl (by decide) (by decide)
    exact decide_eq_false composite
theorem row_count_2959 : row 2959 = 0 := (row_2959.checked).trans rfl
theorem model_count_2959 : modelRow 2959 = 0 := by decide
theorem modelrow_agrees_2959 : modelRow 2959 = row 2959 := model_count_2959.trans row_count_2959.symm
def row_2960 : RowCertificate 2960 where
  first := false
  first_correct := by
    have composite : ¬ Nat.Prime (corner 2960 1) := by change ¬ Nat.Prime 35052321; exact PrattCertificates.composite_of_product (factor := 3) (cofactor := 11684107) rfl (by decide) (by decide)
    exact decide_eq_false composite
  second := false
  second_correct := by
    have composite : ¬ Nat.Prime (corner 2960 2) := by change ¬ Nat.Prime 35046401; exact PrattCertificates.composite_of_product (factor := 13) (cofactor := 2695877) rfl (by decide) (by decide)
    exact decide_eq_false composite
  third := false
  third_correct := by
    have composite : ¬ Nat.Prime (corner 2960 3) := by change ¬ Nat.Prime 35040481; exact PrattCertificates.composite_of_product (factor := 7) (cofactor := 5005783) rfl (by decide) (by decide)
    exact decide_eq_false composite
theorem row_count_2960 : row 2960 = 0 := (row_2960.checked).trans rfl
theorem model_count_2960 : modelRow 2960 = 0 := by decide
theorem modelrow_agrees_2960 : modelRow 2960 = row 2960 := model_count_2960.trans row_count_2960.symm
def row_2961 : RowCertificate 2961 where
  first := false
  first_correct := by
    have composite : ¬ Nat.Prime (corner 2961 1) := by change ¬ Nat.Prime 35076007; exact PrattCertificates.composite_of_product (factor := 211) (cofactor := 166237) rfl (by decide) (by decide)
    exact decide_eq_false composite
  second := false
  second_correct := by
    have composite : ¬ Nat.Prime (corner 2961 2) := by change ¬ Nat.Prime 35070085; exact PrattCertificates.composite_of_product (factor := 5) (cofactor := 7014017) rfl (by decide) (by decide)
    exact decide_eq_false composite
  third := false
  third_correct := by
    have composite : ¬ Nat.Prime (corner 2961 3) := by change ¬ Nat.Prime 35064163; exact PrattCertificates.composite_of_product (factor := 73) (cofactor := 480331) rfl (by decide) (by decide)
    exact decide_eq_false composite
theorem row_count_2961 : row 2961 = 0 := (row_2961.checked).trans rfl
theorem model_count_2961 : modelRow 2961 = 0 := by decide
theorem modelrow_agrees_2961 : modelRow 2961 = row 2961 := model_count_2961.trans row_count_2961.symm
def row_2962 : RowCertificate 2962 where
  first := false
  first_correct := by
    have composite : ¬ Nat.Prime (corner 2962 1) := by change ¬ Nat.Prime 35099701; exact PrattCertificates.composite_of_product (factor := 7) (cofactor := 5014243) rfl (by decide) (by decide)
    exact decide_eq_false composite
  second := false
  second_correct := by
    have composite : ¬ Nat.Prime (corner 2962 2) := by change ¬ Nat.Prime 35093777; exact PrattCertificates.composite_of_product (factor := 197) (cofactor := 178141) rfl (by decide) (by decide)
    exact decide_eq_false composite
  third := false
  third_correct := by
    have composite : ¬ Nat.Prime (corner 2962 3) := by change ¬ Nat.Prime 35087853; exact PrattCertificates.composite_of_product (factor := 3) (cofactor := 11695951) rfl (by decide) (by decide)
    exact decide_eq_false composite
theorem row_count_2962 : row 2962 = 0 := (row_2962.checked).trans rfl
theorem model_count_2962 : modelRow 2962 = 0 := by decide
theorem modelrow_agrees_2962 : modelRow 2962 = row 2962 := model_count_2962.trans row_count_2962.symm
def row_2963 : RowCertificate 2963 where
  first := false
  first_correct := by
    have composite : ¬ Nat.Prime (corner 2963 1) := by change ¬ Nat.Prime 35123403; exact PrattCertificates.composite_of_product (factor := 3) (cofactor := 11707801) rfl (by decide) (by decide)
    exact decide_eq_false composite
  second := false
  second_correct := by
    have composite : ¬ Nat.Prime (corner 2963 2) := by change ¬ Nat.Prime 35117477; exact PrattCertificates.composite_of_product (factor := 37) (cofactor := 949121) rfl (by decide) (by decide)
    exact decide_eq_false composite
  third := false
  third_correct := by
    have composite : ¬ Nat.Prime (corner 2963 3) := by change ¬ Nat.Prime 35111551; exact PrattCertificates.composite_of_product (factor := 67) (cofactor := 524053) rfl (by decide) (by decide)
    exact decide_eq_false composite
theorem row_count_2963 : row 2963 = 0 := (row_2963.checked).trans rfl
theorem model_count_2963 : modelRow 2963 = 0 := by decide
theorem modelrow_agrees_2963 : modelRow 2963 = row 2963 := model_count_2963.trans row_count_2963.symm
def row_2964 : RowCertificate 2964 where
  first := false
  first_correct := by
    have composite : ¬ Nat.Prime (corner 2964 1) := by change ¬ Nat.Prime 35147113; exact PrattCertificates.composite_of_product (factor := 829) (cofactor := 42397) rfl (by decide) (by decide)
    exact decide_eq_false composite
  second := false
  second_correct := by
    have composite : ¬ Nat.Prime (corner 2964 2) := by change ¬ Nat.Prime 35141185; exact PrattCertificates.composite_of_product (factor := 5) (cofactor := 7028237) rfl (by decide) (by decide)
    exact decide_eq_false composite
  third := false
  third_correct := by
    have composite : ¬ Nat.Prime (corner 2964 3) := by change ¬ Nat.Prime 35135257; exact PrattCertificates.composite_of_product (factor := 43) (cofactor := 817099) rfl (by decide) (by decide)
    exact decide_eq_false composite
theorem row_count_2964 : row 2964 = 0 := (row_2964.checked).trans rfl
theorem model_count_2964 : modelRow 2964 = 0 := by decide
theorem modelrow_agrees_2964 : modelRow 2964 = row 2964 := model_count_2964.trans row_count_2964.symm
def row_2965 : RowCertificate 2965 where
  first := false
  first_correct := by
    have composite : ¬ Nat.Prime (corner 2965 1) := by change ¬ Nat.Prime 35170831; exact PrattCertificates.composite_of_product (factor := 37) (cofactor := 950563) rfl (by decide) (by decide)
    exact decide_eq_false composite
  second := true
  second_correct := by
    have prime : Nat.Prime (corner 2965 2) := by change Nat.Prime 35164901; exact ProjectEulerSolutions.P58PrimeNodes.p_35164901.prime
    exact decide_eq_true prime
  third := false
  third_correct := by
    have composite : ¬ Nat.Prime (corner 2965 3) := by change ¬ Nat.Prime 35158971; exact PrattCertificates.composite_of_product (factor := 3) (cofactor := 11719657) rfl (by decide) (by decide)
    exact decide_eq_false composite
theorem row_count_2965 : row 2965 = 1 := (row_2965.checked).trans rfl
theorem model_count_2965 : modelRow 2965 = 1 := by decide
theorem modelrow_agrees_2965 : modelRow 2965 = row 2965 := model_count_2965.trans row_count_2965.symm
def row_2966 : RowCertificate 2966 where
  first := false
  first_correct := by
    have composite : ¬ Nat.Prime (corner 2966 1) := by change ¬ Nat.Prime 35194557; exact PrattCertificates.composite_of_product (factor := 3) (cofactor := 11731519) rfl (by decide) (by decide)
    exact decide_eq_false composite
  second := false
  second_correct := by
    have composite : ¬ Nat.Prime (corner 2966 2) := by change ¬ Nat.Prime 35188625; exact PrattCertificates.composite_of_product (factor := 5) (cofactor := 7037725) rfl (by decide) (by decide)
    exact decide_eq_false composite
  third := false
  third_correct := by
    have composite : ¬ Nat.Prime (corner 2966 3) := by change ¬ Nat.Prime 35182693; exact PrattCertificates.composite_of_product (factor := 7) (cofactor := 5026099) rfl (by decide) (by decide)
    exact decide_eq_false composite
theorem row_count_2966 : row 2966 = 0 := (row_2966.checked).trans rfl
theorem model_count_2966 : modelRow 2966 = 0 := by decide
theorem modelrow_agrees_2966 : modelRow 2966 = row 2966 := model_count_2966.trans row_count_2966.symm
def row_2967 : RowCertificate 2967 where
  first := false
  first_correct := by
    have composite : ¬ Nat.Prime (corner 2967 1) := by change ¬ Nat.Prime 35218291; exact PrattCertificates.composite_of_product (factor := 139) (cofactor := 253369) rfl (by decide) (by decide)
    exact decide_eq_false composite
  second := false
  second_correct := by
    have composite : ¬ Nat.Prime (corner 2967 2) := by change ¬ Nat.Prime 35212357; exact PrattCertificates.composite_of_product (factor := 3209) (cofactor := 10973) rfl (by decide) (by decide)
    exact decide_eq_false composite
  third := false
  third_correct := by
    have composite : ¬ Nat.Prime (corner 2967 3) := by change ¬ Nat.Prime 35206423; exact PrattCertificates.composite_of_product (factor := 7) (cofactor := 5029489) rfl (by decide) (by decide)
    exact decide_eq_false composite
theorem row_count_2967 : row 2967 = 0 := (row_2967.checked).trans rfl
theorem model_count_2967 : modelRow 2967 = 0 := by decide
theorem modelrow_agrees_2967 : modelRow 2967 = row 2967 := model_count_2967.trans row_count_2967.symm
def row_2968 : RowCertificate 2968 where
  first := true
  first_correct := by
    have prime : Nat.Prime (corner 2968 1) := by change Nat.Prime 35242033; exact ProjectEulerSolutions.P58PrimeNodes.p_35242033.prime
    exact decide_eq_true prime
  second := false
  second_correct := by
    have composite : ¬ Nat.Prime (corner 2968 2) := by change ¬ Nat.Prime 35236097; exact PrattCertificates.composite_of_product (factor := 13) (cofactor := 2710469) rfl (by decide) (by decide)
    exact decide_eq_false composite
  third := false
  third_correct := by
    have composite : ¬ Nat.Prime (corner 2968 3) := by change ¬ Nat.Prime 35230161; exact PrattCertificates.composite_of_product (factor := 3) (cofactor := 11743387) rfl (by decide) (by decide)
    exact decide_eq_false composite
theorem row_count_2968 : row 2968 = 1 := (row_2968.checked).trans rfl
theorem model_count_2968 : modelRow 2968 = 1 := by decide
theorem modelrow_agrees_2968 : modelRow 2968 = row 2968 := model_count_2968.trans row_count_2968.symm
def row_2969 : RowCertificate 2969 where
  first := false
  first_correct := by
    have composite : ¬ Nat.Prime (corner 2969 1) := by change ¬ Nat.Prime 35265783; exact PrattCertificates.composite_of_product (factor := 3) (cofactor := 11755261) rfl (by decide) (by decide)
    exact decide_eq_false composite
  second := false
  second_correct := by
    have composite : ¬ Nat.Prime (corner 2969 2) := by change ¬ Nat.Prime 35259845; exact PrattCertificates.composite_of_product (factor := 5) (cofactor := 7051969) rfl (by decide) (by decide)
    exact decide_eq_false composite
  third := false
  third_correct := by
    have composite : ¬ Nat.Prime (corner 2969 3) := by change ¬ Nat.Prime 35253907; exact PrattCertificates.composite_of_product (factor := 13) (cofactor := 2711839) rfl (by decide) (by decide)
    exact decide_eq_false composite
theorem row_count_2969 : row 2969 = 0 := (row_2969.checked).trans rfl
theorem model_count_2969 : modelRow 2969 = 0 := by decide
theorem modelrow_agrees_2969 : modelRow 2969 = row 2969 := model_count_2969.trans row_count_2969.symm
def row_2970 : RowCertificate 2970 where
  first := false
  first_correct := by
    have composite : ¬ Nat.Prime (corner 2970 1) := by change ¬ Nat.Prime 35289541; exact PrattCertificates.composite_of_product (factor := 7) (cofactor := 5041363) rfl (by decide) (by decide)
    exact decide_eq_false composite
  second := false
  second_correct := by
    have composite : ¬ Nat.Prime (corner 2970 2) := by change ¬ Nat.Prime 35283601; exact PrattCertificates.composite_of_product (factor := 73) (cofactor := 483337) rfl (by decide) (by decide)
    exact decide_eq_false composite
  third := false
  third_correct := by
    have composite : ¬ Nat.Prime (corner 2970 3) := by change ¬ Nat.Prime 35277661; exact PrattCertificates.composite_of_product (factor := 19) (cofactor := 1856719) rfl (by decide) (by decide)
    exact decide_eq_false composite
theorem row_count_2970 : row 2970 = 0 := (row_2970.checked).trans rfl
theorem model_count_2970 : modelRow 2970 = 0 := by decide
theorem modelrow_agrees_2970 : modelRow 2970 = row 2970 := model_count_2970.trans row_count_2970.symm
def row_2971 : RowCertificate 2971 where
  first := true
  first_correct := by
    have prime : Nat.Prime (corner 2971 1) := by change Nat.Prime 35313307; exact ProjectEulerSolutions.P58PrimeNodes.p_35313307.prime
    exact decide_eq_true prime
  second := false
  second_correct := by
    have composite : ¬ Nat.Prime (corner 2971 2) := by change ¬ Nat.Prime 35307365; exact PrattCertificates.composite_of_product (factor := 5) (cofactor := 7061473) rfl (by decide) (by decide)
    exact decide_eq_false composite
  third := false
  third_correct := by
    have composite : ¬ Nat.Prime (corner 2971 3) := by change ¬ Nat.Prime 35301423; exact PrattCertificates.composite_of_product (factor := 3) (cofactor := 11767141) rfl (by decide) (by decide)
    exact decide_eq_false composite
theorem row_count_2971 : row 2971 = 1 := (row_2971.checked).trans rfl
theorem model_count_2971 : modelRow 2971 = 1 := by decide
theorem modelrow_agrees_2971 : modelRow 2971 = row 2971 := model_count_2971.trans row_count_2971.symm
def row_2972 : RowCertificate 2972 where
  first := false
  first_correct := by
    have composite : ¬ Nat.Prime (corner 2972 1) := by change ¬ Nat.Prime 35337081; exact PrattCertificates.composite_of_product (factor := 3) (cofactor := 11779027) rfl (by decide) (by decide)
    exact decide_eq_false composite
  second := true
  second_correct := by
    have prime : Nat.Prime (corner 2972 2) := by change Nat.Prime 35331137; exact ProjectEulerSolutions.P58PrimeNodes.p_35331137.prime
    exact decide_eq_true prime
  third := false
  third_correct := by
    have composite : ¬ Nat.Prime (corner 2972 3) := by change ¬ Nat.Prime 35325193; exact PrattCertificates.composite_of_product (factor := 5743) (cofactor := 6151) rfl (by decide) (by decide)
    exact decide_eq_false composite
theorem row_count_2972 : row 2972 = 1 := (row_2972.checked).trans rfl
theorem model_count_2972 : modelRow 2972 = 1 := by decide
theorem modelrow_agrees_2972 : modelRow 2972 = row 2972 := model_count_2972.trans row_count_2972.symm
def row_2973 : RowCertificate 2973 where
  first := false
  first_correct := by
    have composite : ¬ Nat.Prime (corner 2973 1) := by change ¬ Nat.Prime 35360863; exact PrattCertificates.composite_of_product (factor := 31) (cofactor := 1140673) rfl (by decide) (by decide)
    exact decide_eq_false composite
  second := false
  second_correct := by
    have composite : ¬ Nat.Prime (corner 2973 2) := by change ¬ Nat.Prime 35354917; exact PrattCertificates.composite_of_product (factor := 13) (cofactor := 2719609) rfl (by decide) (by decide)
    exact decide_eq_false composite
  third := false
  third_correct := by
    have composite : ¬ Nat.Prime (corner 2973 3) := by change ¬ Nat.Prime 35348971; exact PrattCertificates.composite_of_product (factor := 7) (cofactor := 5049853) rfl (by decide) (by decide)
    exact decide_eq_false composite
theorem row_count_2973 : row 2973 = 0 := (row_2973.checked).trans rfl
theorem model_count_2973 : modelRow 2973 = 0 := by decide
theorem modelrow_agrees_2973 : modelRow 2973 = row 2973 := model_count_2973.trans row_count_2973.symm
def row_2974 : RowCertificate 2974 where
  first := false
  first_correct := by
    have composite : ¬ Nat.Prime (corner 2974 1) := by change ¬ Nat.Prime 35384653; exact PrattCertificates.composite_of_product (factor := 79) (cofactor := 447907) rfl (by decide) (by decide)
    exact decide_eq_false composite
  second := false
  second_correct := by
    have composite : ¬ Nat.Prime (corner 2974 2) := by change ¬ Nat.Prime 35378705; exact PrattCertificates.composite_of_product (factor := 5) (cofactor := 7075741) rfl (by decide) (by decide)
    exact decide_eq_false composite
  third := false
  third_correct := by
    have composite : ¬ Nat.Prime (corner 2974 3) := by change ¬ Nat.Prime 35372757; exact PrattCertificates.composite_of_product (factor := 3) (cofactor := 11790919) rfl (by decide) (by decide)
    exact decide_eq_false composite
theorem row_count_2974 : row 2974 = 0 := (row_2974.checked).trans rfl
theorem model_count_2974 : modelRow 2974 = 0 := by decide
theorem modelrow_agrees_2974 : modelRow 2974 = row 2974 := model_count_2974.trans row_count_2974.symm
def row_2975 : RowCertificate 2975 where
  first := false
  first_correct := by
    have composite : ¬ Nat.Prime (corner 2975 1) := by change ¬ Nat.Prime 35408451; exact PrattCertificates.composite_of_product (factor := 3) (cofactor := 11802817) rfl (by decide) (by decide)
    exact decide_eq_false composite
  second := false
  second_correct := by
    have composite : ¬ Nat.Prime (corner 2975 2) := by change ¬ Nat.Prime 35402501; exact PrattCertificates.composite_of_product (factor := 389) (cofactor := 91009) rfl (by decide) (by decide)
    exact decide_eq_false composite
  third := false
  third_correct := by
    have composite : ¬ Nat.Prime (corner 2975 3) := by change ¬ Nat.Prime 35396551; exact PrattCertificates.composite_of_product (factor := 109) (cofactor := 324739) rfl (by decide) (by decide)
    exact decide_eq_false composite
theorem row_count_2975 : row 2975 = 0 := (row_2975.checked).trans rfl
theorem model_count_2975 : modelRow 2975 = 0 := by decide
theorem modelrow_agrees_2975 : modelRow 2975 = row 2975 := model_count_2975.trans row_count_2975.symm
def row_2976 : RowCertificate 2976 where
  first := false
  first_correct := by
    have composite : ¬ Nat.Prime (corner 2976 1) := by change ¬ Nat.Prime 35432257; exact PrattCertificates.composite_of_product (factor := 7) (cofactor := 5061751) rfl (by decide) (by decide)
    exact decide_eq_false composite
  second := false
  second_correct := by
    have composite : ¬ Nat.Prime (corner 2976 2) := by change ¬ Nat.Prime 35426305; exact PrattCertificates.composite_of_product (factor := 5) (cofactor := 7085261) rfl (by decide) (by decide)
    exact decide_eq_false composite
  third := true
  third_correct := by
    have prime : Nat.Prime (corner 2976 3) := by change Nat.Prime 35420353; exact ProjectEulerSolutions.P58PrimeNodes.p_35420353.prime
    exact decide_eq_true prime
theorem row_count_2976 : row 2976 = 1 := (row_2976.checked).trans rfl
theorem model_count_2976 : modelRow 2976 = 1 := by decide
theorem modelrow_agrees_2976 : modelRow 2976 = row 2976 := model_count_2976.trans row_count_2976.symm
def row_2977 : RowCertificate 2977 where
  first := false
  first_correct := by
    have composite : ¬ Nat.Prime (corner 2977 1) := by change ¬ Nat.Prime 35456071; exact PrattCertificates.composite_of_product (factor := 7) (cofactor := 5065153) rfl (by decide) (by decide)
    exact decide_eq_false composite
  second := false
  second_correct := by
    have composite : ¬ Nat.Prime (corner 2977 2) := by change ¬ Nat.Prime 35450117; exact PrattCertificates.composite_of_product (factor := 17) (cofactor := 2085301) rfl (by decide) (by decide)
    exact decide_eq_false composite
  third := false
  third_correct := by
    have composite : ¬ Nat.Prime (corner 2977 3) := by change ¬ Nat.Prime 35444163; exact PrattCertificates.composite_of_product (factor := 3) (cofactor := 11814721) rfl (by decide) (by decide)
    exact decide_eq_false composite
theorem row_count_2977 : row 2977 = 0 := (row_2977.checked).trans rfl
theorem model_count_2977 : modelRow 2977 = 0 := by decide
theorem modelrow_agrees_2977 : modelRow 2977 = row 2977 := model_count_2977.trans row_count_2977.symm
def row_2978 : RowCertificate 2978 where
  first := false
  first_correct := by
    have composite : ¬ Nat.Prime (corner 2978 1) := by change ¬ Nat.Prime 35479893; exact PrattCertificates.composite_of_product (factor := 3) (cofactor := 11826631) rfl (by decide) (by decide)
    exact decide_eq_false composite
  second := false
  second_correct := by
    have composite : ¬ Nat.Prime (corner 2978 2) := by change ¬ Nat.Prime 35473937; exact PrattCertificates.composite_of_product (factor := 1201) (cofactor := 29537) rfl (by decide) (by decide)
    exact decide_eq_false composite
  third := false
  third_correct := by
    have composite : ¬ Nat.Prime (corner 2978 3) := by change ¬ Nat.Prime 35467981; exact PrattCertificates.composite_of_product (factor := 367) (cofactor := 96643) rfl (by decide) (by decide)
    exact decide_eq_false composite
theorem row_count_2978 : row 2978 = 0 := (row_2978.checked).trans rfl
theorem model_count_2978 : modelRow 2978 = 0 := by decide
theorem modelrow_agrees_2978 : modelRow 2978 = row 2978 := model_count_2978.trans row_count_2978.symm
def row_2979 : RowCertificate 2979 where
  first := false
  first_correct := by
    have composite : ¬ Nat.Prime (corner 2979 1) := by change ¬ Nat.Prime 35503723; exact PrattCertificates.composite_of_product (factor := 19) (cofactor := 1868617) rfl (by decide) (by decide)
    exact decide_eq_false composite
  second := false
  second_correct := by
    have composite : ¬ Nat.Prime (corner 2979 2) := by change ¬ Nat.Prime 35497765; exact PrattCertificates.composite_of_product (factor := 5) (cofactor := 7099553) rfl (by decide) (by decide)
    exact decide_eq_false composite
  third := false
  third_correct := by
    have composite : ¬ Nat.Prime (corner 2979 3) := by change ¬ Nat.Prime 35491807; exact PrattCertificates.composite_of_product (factor := 13) (cofactor := 2730139) rfl (by decide) (by decide)
    exact decide_eq_false composite
theorem row_count_2979 : row 2979 = 0 := (row_2979.checked).trans rfl
theorem model_count_2979 : modelRow 2979 = 0 := by decide
theorem modelrow_agrees_2979 : modelRow 2979 = row 2979 := model_count_2979.trans row_count_2979.symm
def row_2980 : RowCertificate 2980 where
  first := false
  first_correct := by
    have composite : ¬ Nat.Prime (corner 2980 1) := by change ¬ Nat.Prime 35527561; exact PrattCertificates.composite_of_product (factor := 1777) (cofactor := 19993) rfl (by decide) (by decide)
    exact decide_eq_false composite
  second := true
  second_correct := by
    have prime : Nat.Prime (corner 2980 2) := by change Nat.Prime 35521601; exact ProjectEulerSolutions.P58PrimeNodes.p_35521601.prime
    exact decide_eq_true prime
  third := false
  third_correct := by
    have composite : ¬ Nat.Prime (corner 2980 3) := by change ¬ Nat.Prime 35515641; exact PrattCertificates.composite_of_product (factor := 3) (cofactor := 11838547) rfl (by decide) (by decide)
    exact decide_eq_false composite
theorem row_count_2980 : row 2980 = 1 := (row_2980.checked).trans rfl
theorem model_count_2980 : modelRow 2980 = 1 := by decide
theorem modelrow_agrees_2980 : modelRow 2980 = row 2980 := model_count_2980.trans row_count_2980.symm
def row_2981 : RowCertificate 2981 where
  first := false
  first_correct := by
    have composite : ¬ Nat.Prime (corner 2981 1) := by change ¬ Nat.Prime 35551407; exact PrattCertificates.composite_of_product (factor := 3) (cofactor := 11850469) rfl (by decide) (by decide)
    exact decide_eq_false composite
  second := false
  second_correct := by
    have composite : ¬ Nat.Prime (corner 2981 2) := by change ¬ Nat.Prime 35545445; exact PrattCertificates.composite_of_product (factor := 5) (cofactor := 7109089) rfl (by decide) (by decide)
    exact decide_eq_false composite
  third := false
  third_correct := by
    have composite : ¬ Nat.Prime (corner 2981 3) := by change ¬ Nat.Prime 35539483; exact PrattCertificates.composite_of_product (factor := 7) (cofactor := 5077069) rfl (by decide) (by decide)
    exact decide_eq_false composite
theorem row_count_2981 : row 2981 = 0 := (row_2981.checked).trans rfl
theorem model_count_2981 : modelRow 2981 = 0 := by decide
theorem modelrow_agrees_2981 : modelRow 2981 = row 2981 := model_count_2981.trans row_count_2981.symm
def row_2982 : RowCertificate 2982 where
  first := false
  first_correct := by
    have composite : ¬ Nat.Prime (corner 2982 1) := by change ¬ Nat.Prime 35575261; exact PrattCertificates.composite_of_product (factor := 61) (cofactor := 583201) rfl (by decide) (by decide)
    exact decide_eq_false composite
  second := true
  second_correct := by
    have prime : Nat.Prime (corner 2982 2) := by change Nat.Prime 35569297; exact ProjectEulerSolutions.P58PrimeNodes.p_35569297.prime
    exact decide_eq_true prime
  third := false
  third_correct := by
    have composite : ¬ Nat.Prime (corner 2982 3) := by change ¬ Nat.Prime 35563333; exact PrattCertificates.composite_of_product (factor := 13) (cofactor := 2735641) rfl (by decide) (by decide)
    exact decide_eq_false composite
theorem row_count_2982 : row 2982 = 1 := (row_2982.checked).trans rfl
theorem model_count_2982 : modelRow 2982 = 1 := by decide
theorem modelrow_agrees_2982 : modelRow 2982 = row 2982 := model_count_2982.trans row_count_2982.symm
def row_2983 : RowCertificate 2983 where
  first := false
  first_correct := by
    have composite : ¬ Nat.Prime (corner 2983 1) := by change ¬ Nat.Prime 35599123; exact PrattCertificates.composite_of_product (factor := 7) (cofactor := 5085589) rfl (by decide) (by decide)
    exact decide_eq_false composite
  second := false
  second_correct := by
    have composite : ¬ Nat.Prime (corner 2983 2) := by change ¬ Nat.Prime 35593157; exact PrattCertificates.composite_of_product (factor := 53) (cofactor := 671569) rfl (by decide) (by decide)
    exact decide_eq_false composite
  third := false
  third_correct := by
    have composite : ¬ Nat.Prime (corner 2983 3) := by change ¬ Nat.Prime 35587191; exact PrattCertificates.composite_of_product (factor := 3) (cofactor := 11862397) rfl (by decide) (by decide)
    exact decide_eq_false composite
theorem row_count_2983 : row 2983 = 0 := (row_2983.checked).trans rfl
theorem model_count_2983 : modelRow 2983 = 0 := by decide
theorem modelrow_agrees_2983 : modelRow 2983 = row 2983 := model_count_2983.trans row_count_2983.symm
def row_2984 : RowCertificate 2984 where
  first := false
  first_correct := by
    have composite : ¬ Nat.Prime (corner 2984 1) := by change ¬ Nat.Prime 35622993; exact PrattCertificates.composite_of_product (factor := 3) (cofactor := 11874331) rfl (by decide) (by decide)
    exact decide_eq_false composite
  second := false
  second_correct := by
    have composite : ¬ Nat.Prime (corner 2984 2) := by change ¬ Nat.Prime 35617025; exact PrattCertificates.composite_of_product (factor := 5) (cofactor := 7123405) rfl (by decide) (by decide)
    exact decide_eq_false composite
  third := false
  third_correct := by
    have composite : ¬ Nat.Prime (corner 2984 3) := by change ¬ Nat.Prime 35611057; exact PrattCertificates.composite_of_product (factor := 37) (cofactor := 962461) rfl (by decide) (by decide)
    exact decide_eq_false composite
theorem row_count_2984 : row 2984 = 0 := (row_2984.checked).trans rfl
theorem model_count_2984 : modelRow 2984 = 0 := by decide
theorem modelrow_agrees_2984 : modelRow 2984 = row 2984 := model_count_2984.trans row_count_2984.symm
def row_2985 : RowCertificate 2985 where
  first := false
  first_correct := by
    have composite : ¬ Nat.Prime (corner 2985 1) := by change ¬ Nat.Prime 35646871; exact PrattCertificates.composite_of_product (factor := 13) (cofactor := 2742067) rfl (by decide) (by decide)
    exact decide_eq_false composite
  second := true
  second_correct := by
    have prime : Nat.Prime (corner 2985 2) := by change Nat.Prime 35640901; exact ProjectEulerSolutions.P58PrimeNodes.p_35640901.prime
    exact decide_eq_true prime
  third := true
  third_correct := by
    have prime : Nat.Prime (corner 2985 3) := by change Nat.Prime 35634931; exact ProjectEulerSolutions.P58PrimeNodes.p_35634931.prime
    exact decide_eq_true prime
theorem row_count_2985 : row 2985 = 2 := (row_2985.checked).trans rfl
theorem model_count_2985 : modelRow 2985 = 2 := by decide
theorem modelrow_agrees_2985 : modelRow 2985 = row 2985 := model_count_2985.trans row_count_2985.symm
def row_2986 : RowCertificate 2986 where
  first := false
  first_correct := by
    have composite : ¬ Nat.Prime (corner 2986 1) := by change ¬ Nat.Prime 35670757; exact PrattCertificates.composite_of_product (factor := 163) (cofactor := 218839) rfl (by decide) (by decide)
    exact decide_eq_false composite
  second := false
  second_correct := by
    have composite : ¬ Nat.Prime (corner 2986 2) := by change ¬ Nat.Prime 35664785; exact PrattCertificates.composite_of_product (factor := 5) (cofactor := 7132957) rfl (by decide) (by decide)
    exact decide_eq_false composite
  third := false
  third_correct := by
    have composite : ¬ Nat.Prime (corner 2986 3) := by change ¬ Nat.Prime 35658813; exact PrattCertificates.composite_of_product (factor := 3) (cofactor := 11886271) rfl (by decide) (by decide)
    exact decide_eq_false composite
theorem row_count_2986 : row 2986 = 0 := (row_2986.checked).trans rfl
theorem model_count_2986 : modelRow 2986 = 0 := by decide
theorem modelrow_agrees_2986 : modelRow 2986 = row 2986 := model_count_2986.trans row_count_2986.symm
def row_2987 : RowCertificate 2987 where
  first := false
  first_correct := by
    have composite : ¬ Nat.Prime (corner 2987 1) := by change ¬ Nat.Prime 35694651; exact PrattCertificates.composite_of_product (factor := 3) (cofactor := 11898217) rfl (by decide) (by decide)
    exact decide_eq_false composite
  second := false
  second_correct := by
    have composite : ¬ Nat.Prime (corner 2987 2) := by change ¬ Nat.Prime 35688677; exact PrattCertificates.composite_of_product (factor := 113) (cofactor := 315829) rfl (by decide) (by decide)
    exact decide_eq_false composite
  third := false
  third_correct := by
    have composite : ¬ Nat.Prime (corner 2987 3) := by change ¬ Nat.Prime 35682703; exact PrattCertificates.composite_of_product (factor := 7) (cofactor := 5097529) rfl (by decide) (by decide)
    exact decide_eq_false composite
theorem row_count_2987 : row 2987 = 0 := (row_2987.checked).trans rfl
theorem model_count_2987 : modelRow 2987 = 0 := by decide
theorem modelrow_agrees_2987 : modelRow 2987 = row 2987 := model_count_2987.trans row_count_2987.symm
def row_2988 : RowCertificate 2988 where
  first := false
  first_correct := by
    have composite : ¬ Nat.Prime (corner 2988 1) := by change ¬ Nat.Prime 35718553; exact PrattCertificates.composite_of_product (factor := 13) (cofactor := 2747581) rfl (by decide) (by decide)
    exact decide_eq_false composite
  second := false
  second_correct := by
    have composite : ¬ Nat.Prime (corner 2988 2) := by change ¬ Nat.Prime 35712577; exact PrattCertificates.composite_of_product (factor := 617) (cofactor := 57881) rfl (by decide) (by decide)
    exact decide_eq_false composite
  third := false
  third_correct := by
    have composite : ¬ Nat.Prime (corner 2988 3) := by change ¬ Nat.Prime 35706601; exact PrattCertificates.composite_of_product (factor := 7) (cofactor := 5100943) rfl (by decide) (by decide)
    exact decide_eq_false composite
theorem row_count_2988 : row 2988 = 0 := (row_2988.checked).trans rfl
theorem model_count_2988 : modelRow 2988 = 0 := by decide
theorem modelrow_agrees_2988 : modelRow 2988 = row 2988 := model_count_2988.trans row_count_2988.symm
def row_2989 : RowCertificate 2989 where
  first := false
  first_correct := by
    have composite : ¬ Nat.Prime (corner 2989 1) := by change ¬ Nat.Prime 35742463; exact PrattCertificates.composite_of_product (factor := 97) (cofactor := 368479) rfl (by decide) (by decide)
    exact decide_eq_false composite
  second := false
  second_correct := by
    have composite : ¬ Nat.Prime (corner 2989 2) := by change ¬ Nat.Prime 35736485; exact PrattCertificates.composite_of_product (factor := 5) (cofactor := 7147297) rfl (by decide) (by decide)
    exact decide_eq_false composite
  third := false
  third_correct := by
    have composite : ¬ Nat.Prime (corner 2989 3) := by change ¬ Nat.Prime 35730507; exact PrattCertificates.composite_of_product (factor := 3) (cofactor := 11910169) rfl (by decide) (by decide)
    exact decide_eq_false composite
theorem row_count_2989 : row 2989 = 0 := (row_2989.checked).trans rfl
theorem model_count_2989 : modelRow 2989 = 0 := by decide
theorem modelrow_agrees_2989 : modelRow 2989 = row 2989 := model_count_2989.trans row_count_2989.symm
def row_2990 : RowCertificate 2990 where
  first := false
  first_correct := by
    have composite : ¬ Nat.Prime (corner 2990 1) := by change ¬ Nat.Prime 35766381; exact PrattCertificates.composite_of_product (factor := 3) (cofactor := 11922127) rfl (by decide) (by decide)
    exact decide_eq_false composite
  second := false
  second_correct := by
    have composite : ¬ Nat.Prime (corner 2990 2) := by change ¬ Nat.Prime 35760401; exact PrattCertificates.composite_of_product (factor := 17) (cofactor := 2103553) rfl (by decide) (by decide)
    exact decide_eq_false composite
  third := true
  third_correct := by
    have prime : Nat.Prime (corner 2990 3) := by change Nat.Prime 35754421; exact ProjectEulerSolutions.P58PrimeNodes.p_35754421.prime
    exact decide_eq_true prime
theorem row_count_2990 : row 2990 = 1 := (row_2990.checked).trans rfl
theorem model_count_2990 : modelRow 2990 = 1 := by decide
theorem modelrow_agrees_2990 : modelRow 2990 = row 2990 := model_count_2990.trans row_count_2990.symm
def row_2991 : RowCertificate 2991 where
  first := false
  first_correct := by
    have composite : ¬ Nat.Prime (corner 2991 1) := by change ¬ Nat.Prime 35790307; exact PrattCertificates.composite_of_product (factor := 7) (cofactor := 5112901) rfl (by decide) (by decide)
    exact decide_eq_false composite
  second := false
  second_correct := by
    have composite : ¬ Nat.Prime (corner 2991 2) := by change ¬ Nat.Prime 35784325; exact PrattCertificates.composite_of_product (factor := 5) (cofactor := 7156865) rfl (by decide) (by decide)
    exact decide_eq_false composite
  third := false
  third_correct := by
    have composite : ¬ Nat.Prime (corner 2991 3) := by change ¬ Nat.Prime 35778343; exact PrattCertificates.composite_of_product (factor := 223) (cofactor := 160441) rfl (by decide) (by decide)
    exact decide_eq_false composite
theorem row_count_2991 : row 2991 = 0 := (row_2991.checked).trans rfl
theorem model_count_2991 : modelRow 2991 = 0 := by decide
theorem modelrow_agrees_2991 : modelRow 2991 = row 2991 := model_count_2991.trans row_count_2991.symm
def row_2992 : RowCertificate 2992 where
  first := true
  first_correct := by
    have prime : Nat.Prime (corner 2992 1) := by change Nat.Prime 35814241; exact ProjectEulerSolutions.P58PrimeNodes.p_35814241.prime
    exact decide_eq_true prime
  second := true
  second_correct := by
    have prime : Nat.Prime (corner 2992 2) := by change Nat.Prime 35808257; exact ProjectEulerSolutions.P58PrimeNodes.p_35808257.prime
    exact decide_eq_true prime
  third := false
  third_correct := by
    have composite : ¬ Nat.Prime (corner 2992 3) := by change ¬ Nat.Prime 35802273; exact PrattCertificates.composite_of_product (factor := 3) (cofactor := 11934091) rfl (by decide) (by decide)
    exact decide_eq_false composite
theorem row_count_2992 : row 2992 = 2 := (row_2992.checked).trans rfl
theorem model_count_2992 : modelRow 2992 = 2 := by decide
theorem modelrow_agrees_2992 : modelRow 2992 = row 2992 := model_count_2992.trans row_count_2992.symm
def row_2993 : RowCertificate 2993 where
  first := false
  first_correct := by
    have composite : ¬ Nat.Prime (corner 2993 1) := by change ¬ Nat.Prime 35838183; exact PrattCertificates.composite_of_product (factor := 3) (cofactor := 11946061) rfl (by decide) (by decide)
    exact decide_eq_false composite
  second := false
  second_correct := by
    have composite : ¬ Nat.Prime (corner 2993 2) := by change ¬ Nat.Prime 35832197; exact PrattCertificates.composite_of_product (factor := 29) (cofactor := 1235593) rfl (by decide) (by decide)
    exact decide_eq_false composite
  third := false
  third_correct := by
    have composite : ¬ Nat.Prime (corner 2993 3) := by change ¬ Nat.Prime 35826211; exact PrattCertificates.composite_of_product (factor := 601) (cofactor := 59611) rfl (by decide) (by decide)
    exact decide_eq_false composite
theorem row_count_2993 : row 2993 = 0 := (row_2993.checked).trans rfl
theorem model_count_2993 : modelRow 2993 = 0 := by decide
theorem modelrow_agrees_2993 : modelRow 2993 = row 2993 := model_count_2993.trans row_count_2993.symm
def row_2994 : RowCertificate 2994 where
  first := false
  first_correct := by
    have composite : ¬ Nat.Prime (corner 2994 1) := by change ¬ Nat.Prime 35862133; exact PrattCertificates.composite_of_product (factor := 31) (cofactor := 1156843) rfl (by decide) (by decide)
    exact decide_eq_false composite
  second := false
  second_correct := by
    have composite : ¬ Nat.Prime (corner 2994 2) := by change ¬ Nat.Prime 35856145; exact PrattCertificates.composite_of_product (factor := 5) (cofactor := 7171229) rfl (by decide) (by decide)
    exact decide_eq_false composite
  third := false
  third_correct := by
    have composite : ¬ Nat.Prime (corner 2994 3) := by change ¬ Nat.Prime 35850157; exact PrattCertificates.composite_of_product (factor := 7) (cofactor := 5121451) rfl (by decide) (by decide)
    exact decide_eq_false composite
theorem row_count_2994 : row 2994 = 0 := (row_2994.checked).trans rfl
theorem model_count_2994 : modelRow 2994 = 0 := by decide
theorem modelrow_agrees_2994 : modelRow 2994 = row 2994 := model_count_2994.trans row_count_2994.symm
def row_2995 : RowCertificate 2995 where
  first := false
  first_correct := by
    have composite : ¬ Nat.Prime (corner 2995 1) := by change ¬ Nat.Prime 35886091; exact PrattCertificates.composite_of_product (factor := 271) (cofactor := 132421) rfl (by decide) (by decide)
    exact decide_eq_false composite
  second := true
  second_correct := by
    have prime : Nat.Prime (corner 2995 2) := by change Nat.Prime 35880101; exact ProjectEulerSolutions.P58PrimeNodes.p_35880101.prime
    exact decide_eq_true prime
  third := false
  third_correct := by
    have composite : ¬ Nat.Prime (corner 2995 3) := by change ¬ Nat.Prime 35874111; exact PrattCertificates.composite_of_product (factor := 3) (cofactor := 11958037) rfl (by decide) (by decide)
    exact decide_eq_false composite
theorem row_count_2995 : row 2995 = 1 := (row_2995.checked).trans rfl
theorem model_count_2995 : modelRow 2995 = 1 := by decide
theorem modelrow_agrees_2995 : modelRow 2995 = row 2995 := model_count_2995.trans row_count_2995.symm
def row_2996 : RowCertificate 2996 where
  first := false
  first_correct := by
    have composite : ¬ Nat.Prime (corner 2996 1) := by change ¬ Nat.Prime 35910057; exact PrattCertificates.composite_of_product (factor := 3) (cofactor := 11970019) rfl (by decide) (by decide)
    exact decide_eq_false composite
  second := false
  second_correct := by
    have composite : ¬ Nat.Prime (corner 2996 2) := by change ¬ Nat.Prime 35904065; exact PrattCertificates.composite_of_product (factor := 5) (cofactor := 7180813) rfl (by decide) (by decide)
    exact decide_eq_false composite
  third := false
  third_correct := by
    have composite : ¬ Nat.Prime (corner 2996 3) := by change ¬ Nat.Prime 35898073; exact PrattCertificates.composite_of_product (factor := 61) (cofactor := 588493) rfl (by decide) (by decide)
    exact decide_eq_false composite
theorem row_count_2996 : row 2996 = 0 := (row_2996.checked).trans rfl
theorem model_count_2996 : modelRow 2996 = 0 := by decide
theorem modelrow_agrees_2996 : modelRow 2996 = row 2996 := model_count_2996.trans row_count_2996.symm
def row_2997 : RowCertificate 2997 where
  first := false
  first_correct := by
    have composite : ¬ Nat.Prime (corner 2997 1) := by change ¬ Nat.Prime 35934031; exact PrattCertificates.composite_of_product (factor := 7) (cofactor := 5133433) rfl (by decide) (by decide)
    exact decide_eq_false composite
  second := false
  second_correct := by
    have composite : ¬ Nat.Prime (corner 2997 2) := by change ¬ Nat.Prime 35928037; exact PrattCertificates.composite_of_product (factor := 157) (cofactor := 228841) rfl (by decide) (by decide)
    exact decide_eq_false composite
  third := true
  third_correct := by
    have prime : Nat.Prime (corner 2997 3) := by change Nat.Prime 35922043; exact ProjectEulerSolutions.P58PrimeNodes.p_35922043.prime
    exact decide_eq_true prime
theorem row_count_2997 : row 2997 = 1 := (row_2997.checked).trans rfl
theorem model_count_2997 : modelRow 2997 = 1 := by decide
theorem modelrow_agrees_2997 : modelRow 2997 = row 2997 := model_count_2997.trans row_count_2997.symm
def row_2998 : RowCertificate 2998 where
  first := false
  first_correct := by
    have composite : ¬ Nat.Prime (corner 2998 1) := by change ¬ Nat.Prime 35958013; exact PrattCertificates.composite_of_product (factor := 7) (cofactor := 5136859) rfl (by decide) (by decide)
    exact decide_eq_false composite
  second := true
  second_correct := by
    have prime : Nat.Prime (corner 2998 2) := by change Nat.Prime 35952017; exact ProjectEulerSolutions.P58PrimeNodes.p_35952017.prime
    exact decide_eq_true prime
  third := false
  third_correct := by
    have composite : ¬ Nat.Prime (corner 2998 3) := by change ¬ Nat.Prime 35946021; exact PrattCertificates.composite_of_product (factor := 3) (cofactor := 11982007) rfl (by decide) (by decide)
    exact decide_eq_false composite
theorem row_count_2998 : row 2998 = 1 := (row_2998.checked).trans rfl
theorem model_count_2998 : modelRow 2998 = 1 := by decide
theorem modelrow_agrees_2998 : modelRow 2998 = row 2998 := model_count_2998.trans row_count_2998.symm
def row_2999 : RowCertificate 2999 where
  first := false
  first_correct := by
    have composite : ¬ Nat.Prime (corner 2999 1) := by change ¬ Nat.Prime 35982003; exact PrattCertificates.composite_of_product (factor := 3) (cofactor := 11994001) rfl (by decide) (by decide)
    exact decide_eq_false composite
  second := false
  second_correct := by
    have composite : ¬ Nat.Prime (corner 2999 2) := by change ¬ Nat.Prime 35976005; exact PrattCertificates.composite_of_product (factor := 5) (cofactor := 7195201) rfl (by decide) (by decide)
    exact decide_eq_false composite
  third := false
  third_correct := by
    have composite : ¬ Nat.Prime (corner 2999 3) := by change ¬ Nat.Prime 35970007; exact PrattCertificates.composite_of_product (factor := 463) (cofactor := 77689) rfl (by decide) (by decide)
    exact decide_eq_false composite
theorem row_count_2999 : row 2999 = 0 := (row_2999.checked).trans rfl
theorem model_count_2999 : modelRow 2999 = 0 := by decide
theorem modelrow_agrees_2999 : modelRow 2999 = row 2999 := model_count_2999.trans row_count_2999.symm
def row_3000 : RowCertificate 3000 where
  first := false
  first_correct := by
    have composite : ¬ Nat.Prime (corner 3000 1) := by change ¬ Nat.Prime 36006001; exact PrattCertificates.composite_of_product (factor := 67) (cofactor := 537403) rfl (by decide) (by decide)
    exact decide_eq_false composite
  second := false
  second_correct := by
    have composite : ¬ Nat.Prime (corner 3000 2) := by change ¬ Nat.Prime 36000001; exact PrattCertificates.composite_of_product (factor := 37) (cofactor := 972973) rfl (by decide) (by decide)
    exact decide_eq_false composite
  third := false
  third_correct := by
    have composite : ¬ Nat.Prime (corner 3000 3) := by change ¬ Nat.Prime 35994001; exact PrattCertificates.composite_of_product (factor := 4639) (cofactor := 7759) rfl (by decide) (by decide)
    exact decide_eq_false composite
theorem row_count_3000 : row 3000 = 0 := (row_3000.checked).trans rfl
theorem model_count_3000 : modelRow 3000 = 0 := by decide
theorem modelrow_agrees_3000 : modelRow 3000 = row 3000 := model_count_3000.trans row_count_3000.symm
def row_3001 : RowCertificate 3001 where
  first := false
  first_correct := by
    have composite : ¬ Nat.Prime (corner 3001 1) := by change ¬ Nat.Prime 36030007; exact PrattCertificates.composite_of_product (factor := 13) (cofactor := 2771539) rfl (by decide) (by decide)
    exact decide_eq_false composite
  second := false
  second_correct := by
    have composite : ¬ Nat.Prime (corner 3001 2) := by change ¬ Nat.Prime 36024005; exact PrattCertificates.composite_of_product (factor := 5) (cofactor := 7204801) rfl (by decide) (by decide)
    exact decide_eq_false composite
  third := false
  third_correct := by
    have composite : ¬ Nat.Prime (corner 3001 3) := by change ¬ Nat.Prime 36018003; exact PrattCertificates.composite_of_product (factor := 3) (cofactor := 12006001) rfl (by decide) (by decide)
    exact decide_eq_false composite
theorem row_count_3001 : row 3001 = 0 := (row_3001.checked).trans rfl
theorem model_count_3001 : modelRow 3001 = 0 := by decide
theorem modelrow_agrees_3001 : modelRow 3001 = row 3001 := model_count_3001.trans row_count_3001.symm
def row_3002 : RowCertificate 3002 where
  first := false
  first_correct := by
    have composite : ¬ Nat.Prime (corner 3002 1) := by change ¬ Nat.Prime 36054021; exact PrattCertificates.composite_of_product (factor := 3) (cofactor := 12018007) rfl (by decide) (by decide)
    exact decide_eq_false composite
  second := false
  second_correct := by
    have composite : ¬ Nat.Prime (corner 3002 2) := by change ¬ Nat.Prime 36048017; exact PrattCertificates.composite_of_product (factor := 113) (cofactor := 319009) rfl (by decide) (by decide)
    exact decide_eq_false composite
  third := false
  third_correct := by
    have composite : ¬ Nat.Prime (corner 3002 3) := by change ¬ Nat.Prime 36042013; exact PrattCertificates.composite_of_product (factor := 7) (cofactor := 5148859) rfl (by decide) (by decide)
    exact decide_eq_false composite
theorem row_count_3002 : row 3002 = 0 := (row_3002.checked).trans rfl
theorem model_count_3002 : modelRow 3002 = 0 := by decide
theorem modelrow_agrees_3002 : modelRow 3002 = row 3002 := model_count_3002.trans row_count_3002.symm
def row_3003 : RowCertificate 3003 where
  first := true
  first_correct := by
    have prime : Nat.Prime (corner 3003 1) := by change Nat.Prime 36078043; exact ProjectEulerSolutions.P58PrimeNodes.p_36078043.prime
    exact decide_eq_true prime
  second := true
  second_correct := by
    have prime : Nat.Prime (corner 3003 2) := by change Nat.Prime 36072037; exact ProjectEulerSolutions.P58PrimeNodes.p_36072037.prime
    exact decide_eq_true prime
  third := true
  third_correct := by
    have prime : Nat.Prime (corner 3003 3) := by change Nat.Prime 36066031; exact ProjectEulerSolutions.P58PrimeNodes.p_36066031.prime
    exact decide_eq_true prime
theorem row_count_3003 : row 3003 = 3 := (row_3003.checked).trans rfl
theorem model_count_3003 : modelRow 3003 = 3 := by decide
theorem modelrow_agrees_3003 : modelRow 3003 = row 3003 := model_count_3003.trans row_count_3003.symm
def row_3004 : RowCertificate 3004 where
  first := false
  first_correct := by
    have composite : ¬ Nat.Prime (corner 3004 1) := by change ¬ Nat.Prime 36102073; exact PrattCertificates.composite_of_product (factor := 7) (cofactor := 5157439) rfl (by decide) (by decide)
    exact decide_eq_false composite
  second := false
  second_correct := by
    have composite : ¬ Nat.Prime (corner 3004 2) := by change ¬ Nat.Prime 36096065; exact PrattCertificates.composite_of_product (factor := 5) (cofactor := 7219213) rfl (by decide) (by decide)
    exact decide_eq_false composite
  third := false
  third_correct := by
    have composite : ¬ Nat.Prime (corner 3004 3) := by change ¬ Nat.Prime 36090057; exact PrattCertificates.composite_of_product (factor := 3) (cofactor := 12030019) rfl (by decide) (by decide)
    exact decide_eq_false composite
theorem row_count_3004 : row 3004 = 0 := (row_3004.checked).trans rfl
theorem model_count_3004 : modelRow 3004 = 0 := by decide
theorem modelrow_agrees_3004 : modelRow 3004 = row 3004 := model_count_3004.trans row_count_3004.symm
def row_3005 : RowCertificate 3005 where
  first := false
  first_correct := by
    have composite : ¬ Nat.Prime (corner 3005 1) := by change ¬ Nat.Prime 36126111; exact PrattCertificates.composite_of_product (factor := 3) (cofactor := 12042037) rfl (by decide) (by decide)
    exact decide_eq_false composite
  second := true
  second_correct := by
    have prime : Nat.Prime (corner 3005 2) := by change Nat.Prime 36120101; exact ProjectEulerSolutions.P58PrimeNodes.p_36120101.prime
    exact decide_eq_true prime
  third := false
  third_correct := by
    have composite : ¬ Nat.Prime (corner 3005 3) := by change ¬ Nat.Prime 36114091; exact PrattCertificates.composite_of_product (factor := 13) (cofactor := 2778007) rfl (by decide) (by decide)
    exact decide_eq_false composite
theorem row_count_3005 : row 3005 = 1 := (row_3005.checked).trans rfl
theorem model_count_3005 : modelRow 3005 = 1 := by decide
theorem modelrow_agrees_3005 : modelRow 3005 = row 3005 := model_count_3005.trans row_count_3005.symm
def row_3006 : RowCertificate 3006 where
  first := false
  first_correct := by
    have composite : ¬ Nat.Prime (corner 3006 1) := by change ¬ Nat.Prime 36150157; exact PrattCertificates.composite_of_product (factor := 379) (cofactor := 95383) rfl (by decide) (by decide)
    exact decide_eq_false composite
  second := false
  second_correct := by
    have composite : ¬ Nat.Prime (corner 3006 2) := by change ¬ Nat.Prime 36144145; exact PrattCertificates.composite_of_product (factor := 5) (cofactor := 7228829) rfl (by decide) (by decide)
    exact decide_eq_false composite
  third := false
  third_correct := by
    have composite : ¬ Nat.Prime (corner 3006 3) := by change ¬ Nat.Prime 36138133; exact PrattCertificates.composite_of_product (factor := 19) (cofactor := 1902007) rfl (by decide) (by decide)
    exact decide_eq_false composite
theorem row_count_3006 : row 3006 = 0 := (row_3006.checked).trans rfl
theorem model_count_3006 : modelRow 3006 = 0 := by decide
theorem modelrow_agrees_3006 : modelRow 3006 = row 3006 := model_count_3006.trans row_count_3006.symm
def row_3007 : RowCertificate 3007 where
  first := true
  first_correct := by
    have prime : Nat.Prime (corner 3007 1) := by change Nat.Prime 36174211; exact ProjectEulerSolutions.P58PrimeNodes.p_36174211.prime
    exact decide_eq_true prime
  second := false
  second_correct := by
    have composite : ¬ Nat.Prime (corner 3007 2) := by change ¬ Nat.Prime 36168197; exact PrattCertificates.composite_of_product (factor := 13) (cofactor := 2782169) rfl (by decide) (by decide)
    exact decide_eq_false composite
  third := false
  third_correct := by
    have composite : ¬ Nat.Prime (corner 3007 3) := by change ¬ Nat.Prime 36162183; exact PrattCertificates.composite_of_product (factor := 3) (cofactor := 12054061) rfl (by decide) (by decide)
    exact decide_eq_false composite
theorem row_count_3007 : row 3007 = 1 := (row_3007.checked).trans rfl
theorem model_count_3007 : modelRow 3007 = 1 := by decide
theorem modelrow_agrees_3007 : modelRow 3007 = row 3007 := model_count_3007.trans row_count_3007.symm
def row_3008 : RowCertificate 3008 where
  first := false
  first_correct := by
    have composite : ¬ Nat.Prime (corner 3008 1) := by change ¬ Nat.Prime 36198273; exact PrattCertificates.composite_of_product (factor := 3) (cofactor := 12066091) rfl (by decide) (by decide)
    exact decide_eq_false composite
  second := true
  second_correct := by
    have prime : Nat.Prime (corner 3008 2) := by change Nat.Prime 36192257; exact ProjectEulerSolutions.P58PrimeNodes.p_36192257.prime
    exact decide_eq_true prime
  third := false
  third_correct := by
    have composite : ¬ Nat.Prime (corner 3008 3) := by change ¬ Nat.Prime 36186241; exact PrattCertificates.composite_of_product (factor := 7) (cofactor := 5169463) rfl (by decide) (by decide)
    exact decide_eq_false composite
theorem row_count_3008 : row 3008 = 1 := (row_3008.checked).trans rfl
theorem model_count_3008 : modelRow 3008 = 1 := by decide
theorem modelrow_agrees_3008 : modelRow 3008 = row 3008 := model_count_3008.trans row_count_3008.symm
def row_3009 : RowCertificate 3009 where
  first := true
  first_correct := by
    have prime : Nat.Prime (corner 3009 1) := by change Nat.Prime 36222343; exact ProjectEulerSolutions.P58PrimeNodes.p_36222343.prime
    exact decide_eq_true prime
  second := false
  second_correct := by
    have composite : ¬ Nat.Prime (corner 3009 2) := by change ¬ Nat.Prime 36216325; exact PrattCertificates.composite_of_product (factor := 5) (cofactor := 7243265) rfl (by decide) (by decide)
    exact decide_eq_false composite
  third := false
  third_correct := by
    have composite : ¬ Nat.Prime (corner 3009 3) := by change ¬ Nat.Prime 36210307; exact PrattCertificates.composite_of_product (factor := 7) (cofactor := 5172901) rfl (by decide) (by decide)
    exact decide_eq_false composite
theorem row_count_3009 : row 3009 = 1 := (row_3009.checked).trans rfl
theorem model_count_3009 : modelRow 3009 = 1 := by decide
theorem modelrow_agrees_3009 : modelRow 3009 = row 3009 := model_count_3009.trans row_count_3009.symm
def row_3010 : RowCertificate 3010 where
  first := false
  first_correct := by
    have composite : ¬ Nat.Prime (corner 3010 1) := by change ¬ Nat.Prime 36246421; exact PrattCertificates.composite_of_product (factor := 37) (cofactor := 979633) rfl (by decide) (by decide)
    exact decide_eq_false composite
  second := false
  second_correct := by
    have composite : ¬ Nat.Prime (corner 3010 2) := by change ¬ Nat.Prime 36240401; exact PrattCertificates.composite_of_product (factor := 29) (cofactor := 1249669) rfl (by decide) (by decide)
    exact decide_eq_false composite
  third := false
  third_correct := by
    have composite : ¬ Nat.Prime (corner 3010 3) := by change ¬ Nat.Prime 36234381; exact PrattCertificates.composite_of_product (factor := 3) (cofactor := 12078127) rfl (by decide) (by decide)
    exact decide_eq_false composite
theorem row_count_3010 : row 3010 = 0 := (row_3010.checked).trans rfl
theorem model_count_3010 : modelRow 3010 = 0 := by decide
theorem modelrow_agrees_3010 : modelRow 3010 = row 3010 := model_count_3010.trans row_count_3010.symm
def row_3011 : RowCertificate 3011 where
  first := false
  first_correct := by
    have composite : ¬ Nat.Prime (corner 3011 1) := by change ¬ Nat.Prime 36270507; exact PrattCertificates.composite_of_product (factor := 3) (cofactor := 12090169) rfl (by decide) (by decide)
    exact decide_eq_false composite
  second := false
  second_correct := by
    have composite : ¬ Nat.Prime (corner 3011 2) := by change ¬ Nat.Prime 36264485; exact PrattCertificates.composite_of_product (factor := 5) (cofactor := 7252897) rfl (by decide) (by decide)
    exact decide_eq_false composite
  third := false
  third_correct := by
    have composite : ¬ Nat.Prime (corner 3011 3) := by change ¬ Nat.Prime 36258463; exact PrattCertificates.composite_of_product (factor := 181) (cofactor := 200323) rfl (by decide) (by decide)
    exact decide_eq_false composite
theorem row_count_3011 : row 3011 = 0 := (row_3011.checked).trans rfl
theorem model_count_3011 : modelRow 3011 = 0 := by decide
theorem modelrow_agrees_3011 : modelRow 3011 = row 3011 := model_count_3011.trans row_count_3011.symm
def row_3012 : RowCertificate 3012 where
  first := false
  first_correct := by
    have composite : ¬ Nat.Prime (corner 3012 1) := by change ¬ Nat.Prime 36294601; exact PrattCertificates.composite_of_product (factor := 7) (cofactor := 5184943) rfl (by decide) (by decide)
    exact decide_eq_false composite
  second := false
  second_correct := by
    have composite : ¬ Nat.Prime (corner 3012 2) := by change ¬ Nat.Prime 36288577; exact PrattCertificates.composite_of_product (factor := 13) (cofactor := 2791429) rfl (by decide) (by decide)
    exact decide_eq_false composite
  third := true
  third_correct := by
    have prime : Nat.Prime (corner 3012 3) := by change Nat.Prime 36282553; exact ProjectEulerSolutions.P58PrimeNodes.p_36282553.prime
    exact decide_eq_true prime
theorem row_count_3012 : row 3012 = 1 := (row_3012.checked).trans rfl
theorem model_count_3012 : modelRow 3012 = 1 := by decide
theorem modelrow_agrees_3012 : modelRow 3012 = row 3012 := model_count_3012.trans row_count_3012.symm
def row_3013 : RowCertificate 3013 where
  first := false
  first_correct := by
    have composite : ¬ Nat.Prime (corner 3013 1) := by change ¬ Nat.Prime 36318703; exact PrattCertificates.composite_of_product (factor := 43) (cofactor := 844621) rfl (by decide) (by decide)
    exact decide_eq_false composite
  second := false
  second_correct := by
    have composite : ¬ Nat.Prime (corner 3013 2) := by change ¬ Nat.Prime 36312677; exact PrattCertificates.composite_of_product (factor := 653) (cofactor := 55609) rfl (by decide) (by decide)
    exact decide_eq_false composite
  third := false
  third_correct := by
    have composite : ¬ Nat.Prime (corner 3013 3) := by change ¬ Nat.Prime 36306651; exact PrattCertificates.composite_of_product (factor := 3) (cofactor := 12102217) rfl (by decide) (by decide)
    exact decide_eq_false composite
theorem row_count_3013 : row 3013 = 0 := (row_3013.checked).trans rfl
theorem model_count_3013 : modelRow 3013 = 0 := by decide
theorem modelrow_agrees_3013 : modelRow 3013 = row 3013 := model_count_3013.trans row_count_3013.symm
def row_3014 : RowCertificate 3014 where
  first := false
  first_correct := by
    have composite : ¬ Nat.Prime (corner 3014 1) := by change ¬ Nat.Prime 36342813; exact PrattCertificates.composite_of_product (factor := 3) (cofactor := 12114271) rfl (by decide) (by decide)
    exact decide_eq_false composite
  second := false
  second_correct := by
    have composite : ¬ Nat.Prime (corner 3014 2) := by change ¬ Nat.Prime 36336785; exact PrattCertificates.composite_of_product (factor := 5) (cofactor := 7267357) rfl (by decide) (by decide)
    exact decide_eq_false composite
  third := false
  third_correct := by
    have composite : ¬ Nat.Prime (corner 3014 3) := by change ¬ Nat.Prime 36330757; exact PrattCertificates.composite_of_product (factor := 79) (cofactor := 459883) rfl (by decide) (by decide)
    exact decide_eq_false composite
theorem row_count_3014 : row 3014 = 0 := (row_3014.checked).trans rfl
theorem model_count_3014 : modelRow 3014 = 0 := by decide
theorem modelrow_agrees_3014 : modelRow 3014 = row 3014 := model_count_3014.trans row_count_3014.symm
def row_3015 : RowCertificate 3015 where
  first := false
  first_correct := by
    have composite : ¬ Nat.Prime (corner 3015 1) := by change ¬ Nat.Prime 36366931; exact PrattCertificates.composite_of_product (factor := 19) (cofactor := 1914049) rfl (by decide) (by decide)
    exact decide_eq_false composite
  second := true
  second_correct := by
    have prime : Nat.Prime (corner 3015 2) := by change Nat.Prime 36360901; exact ProjectEulerSolutions.P58PrimeNodes.p_36360901.prime
    exact decide_eq_true prime
  third := false
  third_correct := by
    have composite : ¬ Nat.Prime (corner 3015 3) := by change ¬ Nat.Prime 36354871; exact PrattCertificates.composite_of_product (factor := 7) (cofactor := 5193553) rfl (by decide) (by decide)
    exact decide_eq_false composite
theorem row_count_3015 : row 3015 = 1 := (row_3015.checked).trans rfl
theorem model_count_3015 : modelRow 3015 = 1 := by decide
theorem modelrow_agrees_3015 : modelRow 3015 = row 3015 := model_count_3015.trans row_count_3015.symm
def row_3016 : RowCertificate 3016 where
  first := false
  first_correct := by
    have composite : ¬ Nat.Prime (corner 3016 1) := by change ¬ Nat.Prime 36391057; exact PrattCertificates.composite_of_product (factor := 1129) (cofactor := 32233) rfl (by decide) (by decide)
    exact decide_eq_false composite
  second := false
  second_correct := by
    have composite : ¬ Nat.Prime (corner 3016 2) := by change ¬ Nat.Prime 36385025; exact PrattCertificates.composite_of_product (factor := 5) (cofactor := 7277005) rfl (by decide) (by decide)
    exact decide_eq_false composite
  third := false
  third_correct := by
    have composite : ¬ Nat.Prime (corner 3016 3) := by change ¬ Nat.Prime 36378993; exact PrattCertificates.composite_of_product (factor := 3) (cofactor := 12126331) rfl (by decide) (by decide)
    exact decide_eq_false composite
theorem row_count_3016 : row 3016 = 0 := (row_3016.checked).trans rfl
theorem model_count_3016 : modelRow 3016 = 0 := by decide
theorem modelrow_agrees_3016 : modelRow 3016 = row 3016 := model_count_3016.trans row_count_3016.symm
def row_3017 : RowCertificate 3017 where
  first := false
  first_correct := by
    have composite : ¬ Nat.Prime (corner 3017 1) := by change ¬ Nat.Prime 36415191; exact PrattCertificates.composite_of_product (factor := 3) (cofactor := 12138397) rfl (by decide) (by decide)
    exact decide_eq_false composite
  second := false
  second_correct := by
    have composite : ¬ Nat.Prime (corner 3017 2) := by change ¬ Nat.Prime 36409157; exact PrattCertificates.composite_of_product (factor := 277) (cofactor := 131441) rfl (by decide) (by decide)
    exact decide_eq_false composite
  third := true
  third_correct := by
    have prime : Nat.Prime (corner 3017 3) := by change Nat.Prime 36403123; exact ProjectEulerSolutions.P58PrimeNodes.p_36403123.prime
    exact decide_eq_true prime
theorem row_count_3017 : row 3017 = 1 := (row_3017.checked).trans rfl
theorem model_count_3017 : modelRow 3017 = 1 := by decide
theorem modelrow_agrees_3017 : modelRow 3017 = row 3017 := model_count_3017.trans row_count_3017.symm
def row_3018 : RowCertificate 3018 where
  first := false
  first_correct := by
    have composite : ¬ Nat.Prime (corner 3018 1) := by change ¬ Nat.Prime 36439333; exact PrattCertificates.composite_of_product (factor := 7) (cofactor := 5205619) rfl (by decide) (by decide)
    exact decide_eq_false composite
  second := false
  second_correct := by
    have composite : ¬ Nat.Prime (corner 3018 2) := by change ¬ Nat.Prime 36433297; exact PrattCertificates.composite_of_product (factor := 41) (cofactor := 888617) rfl (by decide) (by decide)
    exact decide_eq_false composite
  third := false
  third_correct := by
    have composite : ¬ Nat.Prime (corner 3018 3) := by change ¬ Nat.Prime 36427261; exact PrattCertificates.composite_of_product (factor := 13) (cofactor := 2802097) rfl (by decide) (by decide)
    exact decide_eq_false composite
theorem row_count_3018 : row 3018 = 0 := (row_3018.checked).trans rfl
theorem model_count_3018 : modelRow 3018 = 0 := by decide
theorem modelrow_agrees_3018 : modelRow 3018 = row 3018 := model_count_3018.trans row_count_3018.symm
def row_3019 : RowCertificate 3019 where
  first := false
  first_correct := by
    have composite : ¬ Nat.Prime (corner 3019 1) := by change ¬ Nat.Prime 36463483; exact PrattCertificates.composite_of_product (factor := 7) (cofactor := 5209069) rfl (by decide) (by decide)
    exact decide_eq_false composite
  second := false
  second_correct := by
    have composite : ¬ Nat.Prime (corner 3019 2) := by change ¬ Nat.Prime 36457445; exact PrattCertificates.composite_of_product (factor := 5) (cofactor := 7291489) rfl (by decide) (by decide)
    exact decide_eq_false composite
  third := false
  third_correct := by
    have composite : ¬ Nat.Prime (corner 3019 3) := by change ¬ Nat.Prime 36451407; exact PrattCertificates.composite_of_product (factor := 3) (cofactor := 12150469) rfl (by decide) (by decide)
    exact decide_eq_false composite
theorem row_count_3019 : row 3019 = 0 := (row_3019.checked).trans rfl
theorem model_count_3019 : modelRow 3019 = 0 := by decide
theorem modelrow_agrees_3019 : modelRow 3019 = row 3019 := model_count_3019.trans row_count_3019.symm
def row_3020 : RowCertificate 3020 where
  first := false
  first_correct := by
    have composite : ¬ Nat.Prime (corner 3020 1) := by change ¬ Nat.Prime 36487641; exact PrattCertificates.composite_of_product (factor := 3) (cofactor := 12162547) rfl (by decide) (by decide)
    exact decide_eq_false composite
  second := false
  second_correct := by
    have composite : ¬ Nat.Prime (corner 3020 2) := by change ¬ Nat.Prime 36481601; exact PrattCertificates.composite_of_product (factor := 13) (cofactor := 2806277) rfl (by decide) (by decide)
    exact decide_eq_false composite
  third := false
  third_correct := by
    have composite : ¬ Nat.Prime (corner 3020 3) := by change ¬ Nat.Prime 36475561; exact PrattCertificates.composite_of_product (factor := 31) (cofactor := 1176631) rfl (by decide) (by decide)
    exact decide_eq_false composite
theorem row_count_3020 : row 3020 = 0 := (row_3020.checked).trans rfl
theorem model_count_3020 : modelRow 3020 = 0 := by decide
theorem modelrow_agrees_3020 : modelRow 3020 = row 3020 := model_count_3020.trans row_count_3020.symm
def row_3021 : RowCertificate 3021 where
  first := true
  first_correct := by
    have prime : Nat.Prime (corner 3021 1) := by change Nat.Prime 36511807; exact ProjectEulerSolutions.P58PrimeNodes.p_36511807.prime
    exact decide_eq_true prime
  second := false
  second_correct := by
    have composite : ¬ Nat.Prime (corner 3021 2) := by change ¬ Nat.Prime 36505765; exact PrattCertificates.composite_of_product (factor := 5) (cofactor := 7301153) rfl (by decide) (by decide)
    exact decide_eq_false composite
  third := false
  third_correct := by
    have composite : ¬ Nat.Prime (corner 3021 3) := by change ¬ Nat.Prime 36499723; exact PrattCertificates.composite_of_product (factor := 13) (cofactor := 2807671) rfl (by decide) (by decide)
    exact decide_eq_false composite
theorem row_count_3021 : row 3021 = 1 := (row_3021.checked).trans rfl
theorem model_count_3021 : modelRow 3021 = 1 := by decide
theorem modelrow_agrees_3021 : modelRow 3021 = row 3021 := model_count_3021.trans row_count_3021.symm
def row_3022 : RowCertificate 3022 where
  first := false
  first_correct := by
    have composite : ¬ Nat.Prime (corner 3022 1) := by change ¬ Nat.Prime 36535981; exact PrattCertificates.composite_of_product (factor := 2251) (cofactor := 16231) rfl (by decide) (by decide)
    exact decide_eq_false composite
  second := false
  second_correct := by
    have composite : ¬ Nat.Prime (corner 3022 2) := by change ¬ Nat.Prime 36529937; exact PrattCertificates.composite_of_product (factor := 29) (cofactor := 1259653) rfl (by decide) (by decide)
    exact decide_eq_false composite
  third := false
  third_correct := by
    have composite : ¬ Nat.Prime (corner 3022 3) := by change ¬ Nat.Prime 36523893; exact PrattCertificates.composite_of_product (factor := 3) (cofactor := 12174631) rfl (by decide) (by decide)
    exact decide_eq_false composite
theorem row_count_3022 : row 3022 = 0 := (row_3022.checked).trans rfl
theorem model_count_3022 : modelRow 3022 = 0 := by decide
theorem modelrow_agrees_3022 : modelRow 3022 = row 3022 := model_count_3022.trans row_count_3022.symm
def row_3023 : RowCertificate 3023 where
  first := false
  first_correct := by
    have composite : ¬ Nat.Prime (corner 3023 1) := by change ¬ Nat.Prime 36560163; exact PrattCertificates.composite_of_product (factor := 3) (cofactor := 12186721) rfl (by decide) (by decide)
    exact decide_eq_false composite
  second := true
  second_correct := by
    have prime : Nat.Prime (corner 3023 2) := by change Nat.Prime 36554117; exact ProjectEulerSolutions.P58PrimeNodes.p_36554117.prime
    exact decide_eq_true prime
  third := false
  third_correct := by
    have composite : ¬ Nat.Prime (corner 3023 3) := by change ¬ Nat.Prime 36548071; exact PrattCertificates.composite_of_product (factor := 7) (cofactor := 5221153) rfl (by decide) (by decide)
    exact decide_eq_false composite
theorem row_count_3023 : row 3023 = 1 := (row_3023.checked).trans rfl
theorem model_count_3023 : modelRow 3023 = 1 := by decide
theorem modelrow_agrees_3023 : modelRow 3023 = row 3023 := model_count_3023.trans row_count_3023.symm
def row_3024 : RowCertificate 3024 where
  first := false
  first_correct := by
    have composite : ¬ Nat.Prime (corner 3024 1) := by change ¬ Nat.Prime 36584353; exact PrattCertificates.composite_of_product (factor := 13) (cofactor := 2814181) rfl (by decide) (by decide)
    exact decide_eq_false composite
  second := false
  second_correct := by
    have composite : ¬ Nat.Prime (corner 3024 2) := by change ¬ Nat.Prime 36578305; exact PrattCertificates.composite_of_product (factor := 5) (cofactor := 7315661) rfl (by decide) (by decide)
    exact decide_eq_false composite
  third := true
  third_correct := by
    have prime : Nat.Prime (corner 3024 3) := by change Nat.Prime 36572257; exact ProjectEulerSolutions.P58PrimeNodes.p_36572257.prime
    exact decide_eq_true prime
theorem row_count_3024 : row 3024 = 1 := (row_3024.checked).trans rfl
theorem model_count_3024 : modelRow 3024 = 1 := by decide
theorem modelrow_agrees_3024 : modelRow 3024 = row 3024 := model_count_3024.trans row_count_3024.symm
def row_3025 : RowCertificate 3025 where
  first := false
  first_correct := by
    have composite : ¬ Nat.Prime (corner 3025 1) := by change ¬ Nat.Prime 36608551; exact PrattCertificates.composite_of_product (factor := 7) (cofactor := 5229793) rfl (by decide) (by decide)
    exact decide_eq_false composite
  second := false
  second_correct := by
    have composite : ¬ Nat.Prime (corner 3025 2) := by change ¬ Nat.Prime 36602501; exact PrattCertificates.composite_of_product (factor := 13) (cofactor := 2815577) rfl (by decide) (by decide)
    exact decide_eq_false composite
  third := false
  third_correct := by
    have composite : ¬ Nat.Prime (corner 3025 3) := by change ¬ Nat.Prime 36596451; exact PrattCertificates.composite_of_product (factor := 3) (cofactor := 12198817) rfl (by decide) (by decide)
    exact decide_eq_false composite
theorem row_count_3025 : row 3025 = 0 := (row_3025.checked).trans rfl
theorem model_count_3025 : modelRow 3025 = 0 := by decide
theorem modelrow_agrees_3025 : modelRow 3025 = row 3025 := model_count_3025.trans row_count_3025.symm
def row_3026 : RowCertificate 3026 where
  first := false
  first_correct := by
    have composite : ¬ Nat.Prime (corner 3026 1) := by change ¬ Nat.Prime 36632757; exact PrattCertificates.composite_of_product (factor := 3) (cofactor := 12210919) rfl (by decide) (by decide)
    exact decide_eq_false composite
  second := false
  second_correct := by
    have composite : ¬ Nat.Prime (corner 3026 2) := by change ¬ Nat.Prime 36626705; exact PrattCertificates.composite_of_product (factor := 5) (cofactor := 7325341) rfl (by decide) (by decide)
    exact decide_eq_false composite
  third := true
  third_correct := by
    have prime : Nat.Prime (corner 3026 3) := by change Nat.Prime 36620653; exact ProjectEulerSolutions.P58PrimeNodes.p_36620653.prime
    exact decide_eq_true prime
theorem row_count_3026 : row 3026 = 1 := (row_3026.checked).trans rfl
theorem model_count_3026 : modelRow 3026 = 1 := by decide
theorem modelrow_agrees_3026 : modelRow 3026 = row 3026 := model_count_3026.trans row_count_3026.symm
def row_3027 : RowCertificate 3027 where
  first := false
  first_correct := by
    have composite : ¬ Nat.Prime (corner 3027 1) := by change ¬ Nat.Prime 36656971; exact PrattCertificates.composite_of_product (factor := 13) (cofactor := 2819767) rfl (by decide) (by decide)
    exact decide_eq_false composite
  second := false
  second_correct := by
    have composite : ¬ Nat.Prime (corner 3027 2) := by change ¬ Nat.Prime 36650917; exact PrattCertificates.composite_of_product (factor := 1201) (cofactor := 30517) rfl (by decide) (by decide)
    exact decide_eq_false composite
  third := false
  third_correct := by
    have composite : ¬ Nat.Prime (corner 3027 3) := by change ¬ Nat.Prime 36644863; exact PrattCertificates.composite_of_product (factor := 19) (cofactor := 1928677) rfl (by decide) (by decide)
    exact decide_eq_false composite
theorem row_count_3027 : row 3027 = 0 := (row_3027.checked).trans rfl
theorem model_count_3027 : modelRow 3027 = 0 := by decide
theorem modelrow_agrees_3027 : modelRow 3027 = row 3027 := model_count_3027.trans row_count_3027.symm
def row_3028 : RowCertificate 3028 where
  first := false
  first_correct := by
    have composite : ¬ Nat.Prime (corner 3028 1) := by change ¬ Nat.Prime 36681193; exact PrattCertificates.composite_of_product (factor := 43) (cofactor := 853051) rfl (by decide) (by decide)
    exact decide_eq_false composite
  second := false
  second_correct := by
    have composite : ¬ Nat.Prime (corner 3028 2) := by change ¬ Nat.Prime 36675137; exact PrattCertificates.composite_of_product (factor := 17) (cofactor := 2157361) rfl (by decide) (by decide)
    exact decide_eq_false composite
  third := false
  third_correct := by
    have composite : ¬ Nat.Prime (corner 3028 3) := by change ¬ Nat.Prime 36669081; exact PrattCertificates.composite_of_product (factor := 3) (cofactor := 12223027) rfl (by decide) (by decide)
    exact decide_eq_false composite
theorem row_count_3028 : row 3028 = 0 := (row_3028.checked).trans rfl
theorem model_count_3028 : modelRow 3028 = 0 := by decide
theorem modelrow_agrees_3028 : modelRow 3028 = row 3028 := model_count_3028.trans row_count_3028.symm
def row_3029 : RowCertificate 3029 where
  first := false
  first_correct := by
    have composite : ¬ Nat.Prime (corner 3029 1) := by change ¬ Nat.Prime 36705423; exact PrattCertificates.composite_of_product (factor := 3) (cofactor := 12235141) rfl (by decide) (by decide)
    exact decide_eq_false composite
  second := false
  second_correct := by
    have composite : ¬ Nat.Prime (corner 3029 2) := by change ¬ Nat.Prime 36699365; exact PrattCertificates.composite_of_product (factor := 5) (cofactor := 7339873) rfl (by decide) (by decide)
    exact decide_eq_false composite
  third := false
  third_correct := by
    have composite : ¬ Nat.Prime (corner 3029 3) := by change ¬ Nat.Prime 36693307; exact PrattCertificates.composite_of_product (factor := 7) (cofactor := 5241901) rfl (by decide) (by decide)
    exact decide_eq_false composite
theorem row_count_3029 : row 3029 = 0 := (row_3029.checked).trans rfl
theorem model_count_3029 : modelRow 3029 = 0 := by decide
theorem modelrow_agrees_3029 : modelRow 3029 = row 3029 := model_count_3029.trans row_count_3029.symm
def row_3030 : RowCertificate 3030 where
  first := false
  first_correct := by
    have composite : ¬ Nat.Prime (corner 3030 1) := by change ¬ Nat.Prime 36729661; exact PrattCertificates.composite_of_product (factor := 223) (cofactor := 164707) rfl (by decide) (by decide)
    exact decide_eq_false composite
  second := true
  second_correct := by
    have prime : Nat.Prime (corner 3030 2) := by change Nat.Prime 36723601; exact ProjectEulerSolutions.P58PrimeNodes.p_36723601.prime
    exact decide_eq_true prime
  third := false
  third_correct := by
    have composite : ¬ Nat.Prime (corner 3030 3) := by change ¬ Nat.Prime 36717541; exact PrattCertificates.composite_of_product (factor := 7) (cofactor := 5245363) rfl (by decide) (by decide)
    exact decide_eq_false composite
theorem row_count_3030 : row 3030 = 1 := (row_3030.checked).trans rfl
theorem model_count_3030 : modelRow 3030 = 1 := by decide
theorem modelrow_agrees_3030 : modelRow 3030 = row 3030 := model_count_3030.trans row_count_3030.symm
def row_3031 : RowCertificate 3031 where
  first := false
  first_correct := by
    have composite : ¬ Nat.Prime (corner 3031 1) := by change ¬ Nat.Prime 36753907; exact PrattCertificates.composite_of_product (factor := 199) (cofactor := 184693) rfl (by decide) (by decide)
    exact decide_eq_false composite
  second := false
  second_correct := by
    have composite : ¬ Nat.Prime (corner 3031 2) := by change ¬ Nat.Prime 36747845; exact PrattCertificates.composite_of_product (factor := 5) (cofactor := 7349569) rfl (by decide) (by decide)
    exact decide_eq_false composite
  third := false
  third_correct := by
    have composite : ¬ Nat.Prime (corner 3031 3) := by change ¬ Nat.Prime 36741783; exact PrattCertificates.composite_of_product (factor := 3) (cofactor := 12247261) rfl (by decide) (by decide)
    exact decide_eq_false composite
theorem row_count_3031 : row 3031 = 0 := (row_3031.checked).trans rfl
theorem model_count_3031 : modelRow 3031 = 0 := by decide
theorem modelrow_agrees_3031 : modelRow 3031 = row 3031 := model_count_3031.trans row_count_3031.symm
def row_3032 : RowCertificate 3032 where
  first := false
  first_correct := by
    have composite : ¬ Nat.Prime (corner 3032 1) := by change ¬ Nat.Prime 36778161; exact PrattCertificates.composite_of_product (factor := 3) (cofactor := 12259387) rfl (by decide) (by decide)
    exact decide_eq_false composite
  second := false
  second_correct := by
    have composite : ¬ Nat.Prime (corner 3032 2) := by change ¬ Nat.Prime 36772097; exact PrattCertificates.composite_of_product (factor := 193) (cofactor := 190529) rfl (by decide) (by decide)
    exact decide_eq_false composite
  third := true
  third_correct := by
    have prime : Nat.Prime (corner 3032 3) := by change Nat.Prime 36766033; exact ProjectEulerSolutions.P58PrimeNodes.p_36766033.prime
    exact decide_eq_true prime
theorem row_count_3032 : row 3032 = 1 := (row_3032.checked).trans rfl
theorem model_count_3032 : modelRow 3032 = 1 := by decide
theorem modelrow_agrees_3032 : modelRow 3032 = row 3032 := model_count_3032.trans row_count_3032.symm
def row_3033 : RowCertificate 3033 where
  first := false
  first_correct := by
    have composite : ¬ Nat.Prime (corner 3033 1) := by change ¬ Nat.Prime 36802423; exact PrattCertificates.composite_of_product (factor := 7) (cofactor := 5257489) rfl (by decide) (by decide)
    exact decide_eq_false composite
  second := false
  second_correct := by
    have composite : ¬ Nat.Prime (corner 3033 2) := by change ¬ Nat.Prime 36796357; exact PrattCertificates.composite_of_product (factor := 13) (cofactor := 2830489) rfl (by decide) (by decide)
    exact decide_eq_false composite
  third := true
  third_correct := by
    have prime : Nat.Prime (corner 3033 3) := by change Nat.Prime 36790291; exact ProjectEulerSolutions.P58PrimeNodes.p_36790291.prime
    exact decide_eq_true prime
theorem row_count_3033 : row 3033 = 1 := (row_3033.checked).trans rfl
theorem model_count_3033 : modelRow 3033 = 1 := by decide
theorem modelrow_agrees_3033 : modelRow 3033 = row 3033 := model_count_3033.trans row_count_3033.symm
def row_3034 : RowCertificate 3034 where
  first := false
  first_correct := by
    have composite : ¬ Nat.Prime (corner 3034 1) := by change ¬ Nat.Prime 36826693; exact PrattCertificates.composite_of_product (factor := 19) (cofactor := 1938247) rfl (by decide) (by decide)
    exact decide_eq_false composite
  second := false
  second_correct := by
    have composite : ¬ Nat.Prime (corner 3034 2) := by change ¬ Nat.Prime 36820625; exact PrattCertificates.composite_of_product (factor := 5) (cofactor := 7364125) rfl (by decide) (by decide)
    exact decide_eq_false composite
  third := false
  third_correct := by
    have composite : ¬ Nat.Prime (corner 3034 3) := by change ¬ Nat.Prime 36814557; exact PrattCertificates.composite_of_product (factor := 3) (cofactor := 12271519) rfl (by decide) (by decide)
    exact decide_eq_false composite
theorem row_count_3034 : row 3034 = 0 := (row_3034.checked).trans rfl
theorem model_count_3034 : modelRow 3034 = 0 := by decide
theorem modelrow_agrees_3034 : modelRow 3034 = row 3034 := model_count_3034.trans row_count_3034.symm
def row_3035 : RowCertificate 3035 where
  first := false
  first_correct := by
    have composite : ¬ Nat.Prime (corner 3035 1) := by change ¬ Nat.Prime 36850971; exact PrattCertificates.composite_of_product (factor := 3) (cofactor := 12283657) rfl (by decide) (by decide)
    exact decide_eq_false composite
  second := false
  second_correct := by
    have composite : ¬ Nat.Prime (corner 3035 2) := by change ¬ Nat.Prime 36844901; exact PrattCertificates.composite_of_product (factor := 101) (cofactor := 364801) rfl (by decide) (by decide)
    exact decide_eq_false composite
  third := false
  third_correct := by
    have composite : ¬ Nat.Prime (corner 3035 3) := by change ¬ Nat.Prime 36838831; exact PrattCertificates.composite_of_product (factor := 43) (cofactor := 856717) rfl (by decide) (by decide)
    exact decide_eq_false composite
theorem row_count_3035 : row 3035 = 0 := (row_3035.checked).trans rfl
theorem model_count_3035 : modelRow 3035 = 0 := by decide
theorem modelrow_agrees_3035 : modelRow 3035 = row 3035 := model_count_3035.trans row_count_3035.symm
def row_3036 : RowCertificate 3036 where
  first := false
  first_correct := by
    have composite : ¬ Nat.Prime (corner 3036 1) := by change ¬ Nat.Prime 36875257; exact PrattCertificates.composite_of_product (factor := 19) (cofactor := 1940803) rfl (by decide) (by decide)
    exact decide_eq_false composite
  second := false
  second_correct := by
    have composite : ¬ Nat.Prime (corner 3036 2) := by change ¬ Nat.Prime 36869185; exact PrattCertificates.composite_of_product (factor := 5) (cofactor := 7373837) rfl (by decide) (by decide)
    exact decide_eq_false composite
  third := false
  third_correct := by
    have composite : ¬ Nat.Prime (corner 3036 3) := by change ¬ Nat.Prime 36863113; exact PrattCertificates.composite_of_product (factor := 7) (cofactor := 5266159) rfl (by decide) (by decide)
    exact decide_eq_false composite
theorem row_count_3036 : row 3036 = 0 := (row_3036.checked).trans rfl
theorem model_count_3036 : modelRow 3036 = 0 := by decide
theorem modelrow_agrees_3036 : modelRow 3036 = row 3036 := model_count_3036.trans row_count_3036.symm
def row_3037 : RowCertificate 3037 where
  first := false
  first_correct := by
    have composite : ¬ Nat.Prime (corner 3037 1) := by change ¬ Nat.Prime 36899551; exact PrattCertificates.composite_of_product (factor := 13) (cofactor := 2838427) rfl (by decide) (by decide)
    exact decide_eq_false composite
  second := false
  second_correct := by
    have composite : ¬ Nat.Prime (corner 3037 2) := by change ¬ Nat.Prime 36893477; exact PrattCertificates.composite_of_product (factor := 37) (cofactor := 997121) rfl (by decide) (by decide)
    exact decide_eq_false composite
  third := false
  third_correct := by
    have composite : ¬ Nat.Prime (corner 3037 3) := by change ¬ Nat.Prime 36887403; exact PrattCertificates.composite_of_product (factor := 3) (cofactor := 12295801) rfl (by decide) (by decide)
    exact decide_eq_false composite
theorem row_count_3037 : row 3037 = 0 := (row_3037.checked).trans rfl
theorem model_count_3037 : modelRow 3037 = 0 := by decide
theorem modelrow_agrees_3037 : modelRow 3037 = row 3037 := model_count_3037.trans row_count_3037.symm
def row_3038 : RowCertificate 3038 where
  first := false
  first_correct := by
    have composite : ¬ Nat.Prime (corner 3038 1) := by change ¬ Nat.Prime 36923853; exact PrattCertificates.composite_of_product (factor := 3) (cofactor := 12307951) rfl (by decide) (by decide)
    exact decide_eq_false composite
  second := false
  second_correct := by
    have composite : ¬ Nat.Prime (corner 3038 2) := by change ¬ Nat.Prime 36917777; exact PrattCertificates.composite_of_product (factor := 13) (cofactor := 2839829) rfl (by decide) (by decide)
    exact decide_eq_false composite
  third := false
  third_correct := by
    have composite : ¬ Nat.Prime (corner 3038 3) := by change ¬ Nat.Prime 36911701; exact PrattCertificates.composite_of_product (factor := 97) (cofactor := 380533) rfl (by decide) (by decide)
    exact decide_eq_false composite
theorem row_count_3038 : row 3038 = 0 := (row_3038.checked).trans rfl
theorem model_count_3038 : modelRow 3038 = 0 := by decide
theorem modelrow_agrees_3038 : modelRow 3038 = row 3038 := model_count_3038.trans row_count_3038.symm
def row_3039 : RowCertificate 3039 where
  first := false
  first_correct := by
    have composite : ¬ Nat.Prime (corner 3039 1) := by change ¬ Nat.Prime 36948163; exact PrattCertificates.composite_of_product (factor := 7) (cofactor := 5278309) rfl (by decide) (by decide)
    exact decide_eq_false composite
  second := false
  second_correct := by
    have composite : ¬ Nat.Prime (corner 3039 2) := by change ¬ Nat.Prime 36942085; exact PrattCertificates.composite_of_product (factor := 5) (cofactor := 7388417) rfl (by decide) (by decide)
    exact decide_eq_false composite
  third := true
  third_correct := by
    have prime : Nat.Prime (corner 3039 3) := by change Nat.Prime 36936007; exact ProjectEulerSolutions.P58PrimeNodes.p_36936007.prime
    exact decide_eq_true prime
theorem row_count_3039 : row 3039 = 1 := (row_3039.checked).trans rfl
theorem model_count_3039 : modelRow 3039 = 1 := by decide
theorem modelrow_agrees_3039 : modelRow 3039 = row 3039 := model_count_3039.trans row_count_3039.symm
def row_3040 : RowCertificate 3040 where
  first := false
  first_correct := by
    have composite : ¬ Nat.Prime (corner 3040 1) := by change ¬ Nat.Prime 36972481; exact PrattCertificates.composite_of_product (factor := 7) (cofactor := 5281783) rfl (by decide) (by decide)
    exact decide_eq_false composite
  second := false
  second_correct := by
    have composite : ¬ Nat.Prime (corner 3040 2) := by change ¬ Nat.Prime 36966401; exact PrattCertificates.composite_of_product (factor := 3181) (cofactor := 11621) rfl (by decide) (by decide)
    exact decide_eq_false composite
  third := false
  third_correct := by
    have composite : ¬ Nat.Prime (corner 3040 3) := by change ¬ Nat.Prime 36960321; exact PrattCertificates.composite_of_product (factor := 3) (cofactor := 12320107) rfl (by decide) (by decide)
    exact decide_eq_false composite
theorem row_count_3040 : row 3040 = 0 := (row_3040.checked).trans rfl
theorem model_count_3040 : modelRow 3040 = 0 := by decide
theorem modelrow_agrees_3040 : modelRow 3040 = row 3040 := model_count_3040.trans row_count_3040.symm
def row_3041 : RowCertificate 3041 where
  first := false
  first_correct := by
    have composite : ¬ Nat.Prime (corner 3041 1) := by change ¬ Nat.Prime 36996807; exact PrattCertificates.composite_of_product (factor := 3) (cofactor := 12332269) rfl (by decide) (by decide)
    exact decide_eq_false composite
  second := false
  second_correct := by
    have composite : ¬ Nat.Prime (corner 3041 2) := by change ¬ Nat.Prime 36990725; exact PrattCertificates.composite_of_product (factor := 5) (cofactor := 7398145) rfl (by decide) (by decide)
    exact decide_eq_false composite
  third := false
  third_correct := by
    have composite : ¬ Nat.Prime (corner 3041 3) := by change ¬ Nat.Prime 36984643; exact PrattCertificates.composite_of_product (factor := 31) (cofactor := 1193053) rfl (by decide) (by decide)
    exact decide_eq_false composite
theorem row_count_3041 : row 3041 = 0 := (row_3041.checked).trans rfl
theorem model_count_3041 : modelRow 3041 = 0 := by decide
theorem modelrow_agrees_3041 : modelRow 3041 = row 3041 := model_count_3041.trans row_count_3041.symm
def row_3042 : RowCertificate 3042 where
  first := false
  first_correct := by
    have composite : ¬ Nat.Prime (corner 3042 1) := by change ¬ Nat.Prime 37021141; exact PrattCertificates.composite_of_product (factor := 6007) (cofactor := 6163) rfl (by decide) (by decide)
    exact decide_eq_false composite
  second := false
  second_correct := by
    have composite : ¬ Nat.Prime (corner 3042 2) := by change ¬ Nat.Prime 37015057; exact PrattCertificates.composite_of_product (factor := 5449) (cofactor := 6793) rfl (by decide) (by decide)
    exact decide_eq_false composite
  third := false
  third_correct := by
    have composite : ¬ Nat.Prime (corner 3042 3) := by change ¬ Nat.Prime 37008973; exact PrattCertificates.composite_of_product (factor := 757) (cofactor := 48889) rfl (by decide) (by decide)
    exact decide_eq_false composite
theorem row_count_3042 : row 3042 = 0 := (row_3042.checked).trans rfl
theorem model_count_3042 : modelRow 3042 = 0 := by decide
theorem modelrow_agrees_3042 : modelRow 3042 = row 3042 := model_count_3042.trans row_count_3042.symm
def row_3043 : RowCertificate 3043 where
  first := false
  first_correct := by
    have composite : ¬ Nat.Prime (corner 3043 1) := by change ¬ Nat.Prime 37045483; exact PrattCertificates.composite_of_product (factor := 61) (cofactor := 607303) rfl (by decide) (by decide)
    exact decide_eq_false composite
  second := false
  second_correct := by
    have composite : ¬ Nat.Prime (corner 3043 2) := by change ¬ Nat.Prime 37039397; exact PrattCertificates.composite_of_product (factor := 73) (cofactor := 507389) rfl (by decide) (by decide)
    exact decide_eq_false composite
  third := false
  third_correct := by
    have composite : ¬ Nat.Prime (corner 3043 3) := by change ¬ Nat.Prime 37033311; exact PrattCertificates.composite_of_product (factor := 3) (cofactor := 12344437) rfl (by decide) (by decide)
    exact decide_eq_false composite
theorem row_count_3043 : row 3043 = 0 := (row_3043.checked).trans rfl
theorem model_count_3043 : modelRow 3043 = 0 := by decide
theorem modelrow_agrees_3043 : modelRow 3043 = row 3043 := model_count_3043.trans row_count_3043.symm
def row_3044 : RowCertificate 3044 where
  first := false
  first_correct := by
    have composite : ¬ Nat.Prime (corner 3044 1) := by change ¬ Nat.Prime 37069833; exact PrattCertificates.composite_of_product (factor := 3) (cofactor := 12356611) rfl (by decide) (by decide)
    exact decide_eq_false composite
  second := false
  second_correct := by
    have composite : ¬ Nat.Prime (corner 3044 2) := by change ¬ Nat.Prime 37063745; exact PrattCertificates.composite_of_product (factor := 5) (cofactor := 7412749) rfl (by decide) (by decide)
    exact decide_eq_false composite
  third := false
  third_correct := by
    have composite : ¬ Nat.Prime (corner 3044 3) := by change ¬ Nat.Prime 37057657; exact PrattCertificates.composite_of_product (factor := 7) (cofactor := 5293951) rfl (by decide) (by decide)
    exact decide_eq_false composite
theorem row_count_3044 : row 3044 = 0 := (row_3044.checked).trans rfl
theorem model_count_3044 : modelRow 3044 = 0 := by decide
theorem modelrow_agrees_3044 : modelRow 3044 = row 3044 := model_count_3044.trans row_count_3044.symm
def row_3045 : RowCertificate 3045 where
  first := true
  first_correct := by
    have prime : Nat.Prime (corner 3045 1) := by change Nat.Prime 37094191; exact ProjectEulerSolutions.P58PrimeNodes.p_37094191.prime
    exact decide_eq_true prime
  second := false
  second_correct := by
    have composite : ¬ Nat.Prime (corner 3045 2) := by change ¬ Nat.Prime 37088101; exact PrattCertificates.composite_of_product (factor := 17) (cofactor := 2181653) rfl (by decide) (by decide)
    exact decide_eq_false composite
  third := false
  third_correct := by
    have composite : ¬ Nat.Prime (corner 3045 3) := by change ¬ Nat.Prime 37082011; exact PrattCertificates.composite_of_product (factor := 163) (cofactor := 227497) rfl (by decide) (by decide)
    exact decide_eq_false composite
theorem row_count_3045 : row 3045 = 1 := (row_3045.checked).trans rfl
theorem model_count_3045 : modelRow 3045 = 1 := by decide
theorem modelrow_agrees_3045 : modelRow 3045 = row 3045 := model_count_3045.trans row_count_3045.symm
def row_3046 : RowCertificate 3046 where
  first := false
  first_correct := by
    have composite : ¬ Nat.Prime (corner 3046 1) := by change ¬ Nat.Prime 37118557; exact PrattCertificates.composite_of_product (factor := 7) (cofactor := 5302651) rfl (by decide) (by decide)
    exact decide_eq_false composite
  second := false
  second_correct := by
    have composite : ¬ Nat.Prime (corner 3046 2) := by change ¬ Nat.Prime 37112465; exact PrattCertificates.composite_of_product (factor := 5) (cofactor := 7422493) rfl (by decide) (by decide)
    exact decide_eq_false composite
  third := false
  third_correct := by
    have composite : ¬ Nat.Prime (corner 3046 3) := by change ¬ Nat.Prime 37106373; exact PrattCertificates.composite_of_product (factor := 3) (cofactor := 12368791) rfl (by decide) (by decide)
    exact decide_eq_false composite
theorem row_count_3046 : row 3046 = 0 := (row_3046.checked).trans rfl
theorem model_count_3046 : modelRow 3046 = 0 := by decide
theorem modelrow_agrees_3046 : modelRow 3046 = row 3046 := model_count_3046.trans row_count_3046.symm
def row_3047 : RowCertificate 3047 where
  first := false
  first_correct := by
    have composite : ¬ Nat.Prime (corner 3047 1) := by change ¬ Nat.Prime 37142931; exact PrattCertificates.composite_of_product (factor := 3) (cofactor := 12380977) rfl (by decide) (by decide)
    exact decide_eq_false composite
  second := false
  second_correct := by
    have composite : ¬ Nat.Prime (corner 3047 2) := by change ¬ Nat.Prime 37136837; exact PrattCertificates.composite_of_product (factor := 4729) (cofactor := 7853) rfl (by decide) (by decide)
    exact decide_eq_false composite
  third := false
  third_correct := by
    have composite : ¬ Nat.Prime (corner 3047 3) := by change ¬ Nat.Prime 37130743; exact PrattCertificates.composite_of_product (factor := 13) (cofactor := 2856211) rfl (by decide) (by decide)
    exact decide_eq_false composite
theorem row_count_3047 : row 3047 = 0 := (row_3047.checked).trans rfl
theorem model_count_3047 : modelRow 3047 = 0 := by decide
theorem modelrow_agrees_3047 : modelRow 3047 = row 3047 := model_count_3047.trans row_count_3047.symm
def row_3048 : RowCertificate 3048 where
  first := true
  first_correct := by
    have prime : Nat.Prime (corner 3048 1) := by change Nat.Prime 37167313; exact ProjectEulerSolutions.P58PrimeNodes.p_37167313.prime
    exact decide_eq_true prime
  second := true
  second_correct := by
    have prime : Nat.Prime (corner 3048 2) := by change Nat.Prime 37161217; exact ProjectEulerSolutions.P58PrimeNodes.p_37161217.prime
    exact decide_eq_true prime
  third := true
  third_correct := by
    have prime : Nat.Prime (corner 3048 3) := by change Nat.Prime 37155121; exact ProjectEulerSolutions.P58PrimeNodes.p_37155121.prime
    exact decide_eq_true prime
theorem row_count_3048 : row 3048 = 3 := (row_3048.checked).trans rfl
theorem model_count_3048 : modelRow 3048 = 3 := by decide
theorem modelrow_agrees_3048 : modelRow 3048 = row 3048 := model_count_3048.trans row_count_3048.symm
def row_3049 : RowCertificate 3049 where
  first := false
  first_correct := by
    have composite : ¬ Nat.Prime (corner 3049 1) := by change ¬ Nat.Prime 37191703; exact PrattCertificates.composite_of_product (factor := 487) (cofactor := 76369) rfl (by decide) (by decide)
    exact decide_eq_false composite
  second := false
  second_correct := by
    have composite : ¬ Nat.Prime (corner 3049 2) := by change ¬ Nat.Prime 37185605; exact PrattCertificates.composite_of_product (factor := 5) (cofactor := 7437121) rfl (by decide) (by decide)
    exact decide_eq_false composite
  third := false
  third_correct := by
    have composite : ¬ Nat.Prime (corner 3049 3) := by change ¬ Nat.Prime 37179507; exact PrattCertificates.composite_of_product (factor := 3) (cofactor := 12393169) rfl (by decide) (by decide)
    exact decide_eq_false composite
theorem row_count_3049 : row 3049 = 0 := (row_3049.checked).trans rfl
theorem model_count_3049 : modelRow 3049 = 0 := by decide
theorem modelrow_agrees_3049 : modelRow 3049 = row 3049 := model_count_3049.trans row_count_3049.symm
def row_3050 : RowCertificate 3050 where
  first := false
  first_correct := by
    have composite : ¬ Nat.Prime (corner 3050 1) := by change ¬ Nat.Prime 37216101; exact PrattCertificates.composite_of_product (factor := 3) (cofactor := 12405367) rfl (by decide) (by decide)
    exact decide_eq_false composite
  second := false
  second_correct := by
    have composite : ¬ Nat.Prime (corner 3050 2) := by change ¬ Nat.Prime 37210001; exact PrattCertificates.composite_of_product (factor := 41) (cofactor := 907561) rfl (by decide) (by decide)
    exact decide_eq_false composite
  third := false
  third_correct := by
    have composite : ¬ Nat.Prime (corner 3050 3) := by change ¬ Nat.Prime 37203901; exact PrattCertificates.composite_of_product (factor := 7) (cofactor := 5314843) rfl (by decide) (by decide)
    exact decide_eq_false composite
theorem row_count_3050 : row 3050 = 0 := (row_3050.checked).trans rfl
theorem model_count_3050 : modelRow 3050 = 0 := by decide
theorem modelrow_agrees_3050 : modelRow 3050 = row 3050 := model_count_3050.trans row_count_3050.symm
def row_3051 : RowCertificate 3051 where
  first := true
  first_correct := by
    have prime : Nat.Prime (corner 3051 1) := by change Nat.Prime 37240507; exact ProjectEulerSolutions.P58PrimeNodes.p_37240507.prime
    exact decide_eq_true prime
  second := false
  second_correct := by
    have composite : ¬ Nat.Prime (corner 3051 2) := by change ¬ Nat.Prime 37234405; exact PrattCertificates.composite_of_product (factor := 5) (cofactor := 7446881) rfl (by decide) (by decide)
    exact decide_eq_false composite
  third := false
  third_correct := by
    have composite : ¬ Nat.Prime (corner 3051 3) := by change ¬ Nat.Prime 37228303; exact PrattCertificates.composite_of_product (factor := 7) (cofactor := 5318329) rfl (by decide) (by decide)
    exact decide_eq_false composite
theorem row_count_3051 : row 3051 = 1 := (row_3051.checked).trans rfl
theorem model_count_3051 : modelRow 3051 = 1 := by decide
theorem modelrow_agrees_3051 : modelRow 3051 = row 3051 := model_count_3051.trans row_count_3051.symm
def row_3052 : RowCertificate 3052 where
  first := false
  first_correct := by
    have composite : ¬ Nat.Prime (corner 3052 1) := by change ¬ Nat.Prime 37264921; exact PrattCertificates.composite_of_product (factor := 211) (cofactor := 176611) rfl (by decide) (by decide)
    exact decide_eq_false composite
  second := false
  second_correct := by
    have composite : ¬ Nat.Prime (corner 3052 2) := by change ¬ Nat.Prime 37258817; exact PrattCertificates.composite_of_product (factor := 2593) (cofactor := 14369) rfl (by decide) (by decide)
    exact decide_eq_false composite
  third := false
  third_correct := by
    have composite : ¬ Nat.Prime (corner 3052 3) := by change ¬ Nat.Prime 37252713; exact PrattCertificates.composite_of_product (factor := 3) (cofactor := 12417571) rfl (by decide) (by decide)
    exact decide_eq_false composite
theorem row_count_3052 : row 3052 = 0 := (row_3052.checked).trans rfl
theorem model_count_3052 : modelRow 3052 = 0 := by decide
theorem modelrow_agrees_3052 : modelRow 3052 = row 3052 := model_count_3052.trans row_count_3052.symm
def row_3053 : RowCertificate 3053 where
  first := false
  first_correct := by
    have composite : ¬ Nat.Prime (corner 3053 1) := by change ¬ Nat.Prime 37289343; exact PrattCertificates.composite_of_product (factor := 3) (cofactor := 12429781) rfl (by decide) (by decide)
    exact decide_eq_false composite
  second := false
  second_correct := by
    have composite : ¬ Nat.Prime (corner 3053 2) := by change ¬ Nat.Prime 37283237; exact PrattCertificates.composite_of_product (factor := 1777) (cofactor := 20981) rfl (by decide) (by decide)
    exact decide_eq_false composite
  third := false
  third_correct := by
    have composite : ¬ Nat.Prime (corner 3053 3) := by change ¬ Nat.Prime 37277131; exact PrattCertificates.composite_of_product (factor := 181) (cofactor := 205951) rfl (by decide) (by decide)
    exact decide_eq_false composite
theorem row_count_3053 : row 3053 = 0 := (row_3053.checked).trans rfl
theorem model_count_3053 : modelRow 3053 = 0 := by decide
theorem modelrow_agrees_3053 : modelRow 3053 = row 3053 := model_count_3053.trans row_count_3053.symm
def row_3054 : RowCertificate 3054 where
  first := false
  first_correct := by
    have composite : ¬ Nat.Prime (corner 3054 1) := by change ¬ Nat.Prime 37313773; exact PrattCertificates.composite_of_product (factor := 7) (cofactor := 5330539) rfl (by decide) (by decide)
    exact decide_eq_false composite
  second := false
  second_correct := by
    have composite : ¬ Nat.Prime (corner 3054 2) := by change ¬ Nat.Prime 37307665; exact PrattCertificates.composite_of_product (factor := 5) (cofactor := 7461533) rfl (by decide) (by decide)
    exact decide_eq_false composite
  third := true
  third_correct := by
    have prime : Nat.Prime (corner 3054 3) := by change Nat.Prime 37301557; exact ProjectEulerSolutions.P58PrimeNodes.p_37301557.prime
    exact decide_eq_true prime
theorem row_count_3054 : row 3054 = 1 := (row_3054.checked).trans rfl
theorem model_count_3054 : modelRow 3054 = 1 := by decide
theorem modelrow_agrees_3054 : modelRow 3054 = row 3054 := model_count_3054.trans row_count_3054.symm
def row_3055 : RowCertificate 3055 where
  first := false
  first_correct := by
    have composite : ¬ Nat.Prime (corner 3055 1) := by change ¬ Nat.Prime 37338211; exact PrattCertificates.composite_of_product (factor := 19) (cofactor := 1965169) rfl (by decide) (by decide)
    exact decide_eq_false composite
  second := true
  second_correct := by
    have prime : Nat.Prime (corner 3055 2) := by change Nat.Prime 37332101; exact ProjectEulerSolutions.P58PrimeNodes.p_37332101.prime
    exact decide_eq_true prime
  third := false
  third_correct := by
    have composite : ¬ Nat.Prime (corner 3055 3) := by change ¬ Nat.Prime 37325991; exact PrattCertificates.composite_of_product (factor := 3) (cofactor := 12441997) rfl (by decide) (by decide)
    exact decide_eq_false composite
theorem row_count_3055 : row 3055 = 1 := (row_3055.checked).trans rfl
theorem model_count_3055 : modelRow 3055 = 1 := by decide
theorem modelrow_agrees_3055 : modelRow 3055 = row 3055 := model_count_3055.trans row_count_3055.symm
def row_3056 : RowCertificate 3056 where
  first := false
  first_correct := by
    have composite : ¬ Nat.Prime (corner 3056 1) := by change ¬ Nat.Prime 37362657; exact PrattCertificates.composite_of_product (factor := 3) (cofactor := 12454219) rfl (by decide) (by decide)
    exact decide_eq_false composite
  second := false
  second_correct := by
    have composite : ¬ Nat.Prime (corner 3056 2) := by change ¬ Nat.Prime 37356545; exact PrattCertificates.composite_of_product (factor := 5) (cofactor := 7471309) rfl (by decide) (by decide)
    exact decide_eq_false composite
  third := true
  third_correct := by
    have prime : Nat.Prime (corner 3056 3) := by change Nat.Prime 37350433; exact ProjectEulerSolutions.P58PrimeNodes.p_37350433.prime
    exact decide_eq_true prime
theorem row_count_3056 : row 3056 = 1 := (row_3056.checked).trans rfl
theorem model_count_3056 : modelRow 3056 = 1 := by decide
theorem modelrow_agrees_3056 : modelRow 3056 = row 3056 := model_count_3056.trans row_count_3056.symm
def row_3057 : RowCertificate 3057 where
  first := true
  first_correct := by
    have prime : Nat.Prime (corner 3057 1) := by change Nat.Prime 37387111; exact ProjectEulerSolutions.P58PrimeNodes.p_37387111.prime
    exact decide_eq_true prime
  second := false
  second_correct := by
    have composite : ¬ Nat.Prime (corner 3057 2) := by change ¬ Nat.Prime 37380997; exact PrattCertificates.composite_of_product (factor := 977) (cofactor := 38261) rfl (by decide) (by decide)
    exact decide_eq_false composite
  third := false
  third_correct := by
    have composite : ¬ Nat.Prime (corner 3057 3) := by change ¬ Nat.Prime 37374883; exact PrattCertificates.composite_of_product (factor := 7) (cofactor := 5339269) rfl (by decide) (by decide)
    exact decide_eq_false composite
theorem row_count_3057 : row 3057 = 1 := (row_3057.checked).trans rfl
theorem model_count_3057 : modelRow 3057 = 1 := by decide
theorem modelrow_agrees_3057 : modelRow 3057 = row 3057 := model_count_3057.trans row_count_3057.symm
def row_3058 : RowCertificate 3058 where
  first := false
  first_correct := by
    have composite : ¬ Nat.Prime (corner 3058 1) := by change ¬ Nat.Prime 37411573; exact PrattCertificates.composite_of_product (factor := 4099) (cofactor := 9127) rfl (by decide) (by decide)
    exact decide_eq_false composite
  second := false
  second_correct := by
    have composite : ¬ Nat.Prime (corner 3058 2) := by change ¬ Nat.Prime 37405457; exact PrattCertificates.composite_of_product (factor := 17) (cofactor := 2200321) rfl (by decide) (by decide)
    exact decide_eq_false composite
  third := false
  third_correct := by
    have composite : ¬ Nat.Prime (corner 3058 3) := by change ¬ Nat.Prime 37399341; exact PrattCertificates.composite_of_product (factor := 3) (cofactor := 12466447) rfl (by decide) (by decide)
    exact decide_eq_false composite
theorem row_count_3058 : row 3058 = 0 := (row_3058.checked).trans rfl
theorem model_count_3058 : modelRow 3058 = 0 := by decide
theorem modelrow_agrees_3058 : modelRow 3058 = row 3058 := model_count_3058.trans row_count_3058.symm
def row_3059 : RowCertificate 3059 where
  first := false
  first_correct := by
    have composite : ¬ Nat.Prime (corner 3059 1) := by change ¬ Nat.Prime 37436043; exact PrattCertificates.composite_of_product (factor := 3) (cofactor := 12478681) rfl (by decide) (by decide)
    exact decide_eq_false composite
  second := false
  second_correct := by
    have composite : ¬ Nat.Prime (corner 3059 2) := by change ¬ Nat.Prime 37429925; exact PrattCertificates.composite_of_product (factor := 5) (cofactor := 7485985) rfl (by decide) (by decide)
    exact decide_eq_false composite
  third := true
  third_correct := by
    have prime : Nat.Prime (corner 3059 3) := by change Nat.Prime 37423807; exact ProjectEulerSolutions.P58PrimeNodes.p_37423807.prime
    exact decide_eq_true prime
theorem row_count_3059 : row 3059 = 1 := (row_3059.checked).trans rfl
theorem model_count_3059 : modelRow 3059 = 1 := by decide
theorem modelrow_agrees_3059 : modelRow 3059 = row 3059 := model_count_3059.trans row_count_3059.symm
def row_3060 : RowCertificate 3060 where
  first := false
  first_correct := by
    have composite : ¬ Nat.Prime (corner 3060 1) := by change ¬ Nat.Prime 37460521; exact PrattCertificates.composite_of_product (factor := 7) (cofactor := 5351503) rfl (by decide) (by decide)
    exact decide_eq_false composite
  second := true
  second_correct := by
    have prime : Nat.Prime (corner 3060 2) := by change Nat.Prime 37454401; exact ProjectEulerSolutions.P58PrimeNodes.p_37454401.prime
    exact decide_eq_true prime
  third := false
  third_correct := by
    have composite : ¬ Nat.Prime (corner 3060 3) := by change ¬ Nat.Prime 37448281; exact PrattCertificates.composite_of_product (factor := 13) (cofactor := 2880637) rfl (by decide) (by decide)
    exact decide_eq_false composite
theorem row_count_3060 : row 3060 = 1 := (row_3060.checked).trans rfl
theorem model_count_3060 : modelRow 3060 = 1 := by decide
theorem modelrow_agrees_3060 : modelRow 3060 = row 3060 := model_count_3060.trans row_count_3060.symm
def row_3061 : RowCertificate 3061 where
  first := false
  first_correct := by
    have composite : ¬ Nat.Prime (corner 3061 1) := by change ¬ Nat.Prime 37485007; exact PrattCertificates.composite_of_product (factor := 7) (cofactor := 5355001) rfl (by decide) (by decide)
    exact decide_eq_false composite
  second := false
  second_correct := by
    have composite : ¬ Nat.Prime (corner 3061 2) := by change ¬ Nat.Prime 37478885; exact PrattCertificates.composite_of_product (factor := 5) (cofactor := 7495777) rfl (by decide) (by decide)
    exact decide_eq_false composite
  third := false
  third_correct := by
    have composite : ¬ Nat.Prime (corner 3061 3) := by change ¬ Nat.Prime 37472763; exact PrattCertificates.composite_of_product (factor := 3) (cofactor := 12490921) rfl (by decide) (by decide)
    exact decide_eq_false composite
theorem row_count_3061 : row 3061 = 0 := (row_3061.checked).trans rfl
theorem model_count_3061 : modelRow 3061 = 0 := by decide
theorem modelrow_agrees_3061 : modelRow 3061 = row 3061 := model_count_3061.trans row_count_3061.symm
def row_3062 : RowCertificate 3062 where
  first := false
  first_correct := by
    have composite : ¬ Nat.Prime (corner 3062 1) := by change ¬ Nat.Prime 37509501; exact PrattCertificates.composite_of_product (factor := 3) (cofactor := 12503167) rfl (by decide) (by decide)
    exact decide_eq_false composite
  second := false
  second_correct := by
    have composite : ¬ Nat.Prime (corner 3062 2) := by change ¬ Nat.Prime 37503377; exact PrattCertificates.composite_of_product (factor := 17) (cofactor := 2206081) rfl (by decide) (by decide)
    exact decide_eq_false composite
  third := false
  third_correct := by
    have composite : ¬ Nat.Prime (corner 3062 3) := by change ¬ Nat.Prime 37497253; exact PrattCertificates.composite_of_product (factor := 73) (cofactor := 513661) rfl (by decide) (by decide)
    exact decide_eq_false composite
theorem row_count_3062 : row 3062 = 0 := (row_3062.checked).trans rfl
theorem model_count_3062 : modelRow 3062 = 0 := by decide
theorem modelrow_agrees_3062 : modelRow 3062 = row 3062 := model_count_3062.trans row_count_3062.symm
def row_3063 : RowCertificate 3063 where
  first := false
  first_correct := by
    have composite : ¬ Nat.Prime (corner 3063 1) := by change ¬ Nat.Prime 37534003; exact PrattCertificates.composite_of_product (factor := 13) (cofactor := 2887231) rfl (by decide) (by decide)
    exact decide_eq_false composite
  second := true
  second_correct := by
    have prime : Nat.Prime (corner 3063 2) := by change Nat.Prime 37527877; exact ProjectEulerSolutions.P58PrimeNodes.p_37527877.prime
    exact decide_eq_true prime
  third := false
  third_correct := by
    have composite : ¬ Nat.Prime (corner 3063 3) := by change ¬ Nat.Prime 37521751; exact PrattCertificates.composite_of_product (factor := 19) (cofactor := 1974829) rfl (by decide) (by decide)
    exact decide_eq_false composite
theorem row_count_3063 : row 3063 = 1 := (row_3063.checked).trans rfl
theorem model_count_3063 : modelRow 3063 = 1 := by decide
theorem modelrow_agrees_3063 : modelRow 3063 = row 3063 := model_count_3063.trans row_count_3063.symm
def row_3064 : RowCertificate 3064 where
  first := true
  first_correct := by
    have prime : Nat.Prime (corner 3064 1) := by change Nat.Prime 37558513; exact ProjectEulerSolutions.P58PrimeNodes.p_37558513.prime
    exact decide_eq_true prime
  second := false
  second_correct := by
    have composite : ¬ Nat.Prime (corner 3064 2) := by change ¬ Nat.Prime 37552385; exact PrattCertificates.composite_of_product (factor := 5) (cofactor := 7510477) rfl (by decide) (by decide)
    exact decide_eq_false composite
  third := false
  third_correct := by
    have composite : ¬ Nat.Prime (corner 3064 3) := by change ¬ Nat.Prime 37546257; exact PrattCertificates.composite_of_product (factor := 3) (cofactor := 12515419) rfl (by decide) (by decide)
    exact decide_eq_false composite
theorem row_count_3064 : row 3064 = 1 := (row_3064.checked).trans rfl
theorem model_count_3064 : modelRow 3064 = 1 := by decide
theorem modelrow_agrees_3064 : modelRow 3064 = row 3064 := model_count_3064.trans row_count_3064.symm
def row_3065 : RowCertificate 3065 where
  first := false
  first_correct := by
    have composite : ¬ Nat.Prime (corner 3065 1) := by change ¬ Nat.Prime 37583031; exact PrattCertificates.composite_of_product (factor := 3) (cofactor := 12527677) rfl (by decide) (by decide)
    exact decide_eq_false composite
  second := true
  second_correct := by
    have prime : Nat.Prime (corner 3065 2) := by change Nat.Prime 37576901; exact ProjectEulerSolutions.P58PrimeNodes.p_37576901.prime
    exact decide_eq_true prime
  third := false
  third_correct := by
    have composite : ¬ Nat.Prime (corner 3065 3) := by change ¬ Nat.Prime 37570771; exact PrattCertificates.composite_of_product (factor := 7) (cofactor := 5367253) rfl (by decide) (by decide)
    exact decide_eq_false composite
theorem row_count_3065 : row 3065 = 1 := (row_3065.checked).trans rfl
theorem model_count_3065 : modelRow 3065 = 1 := by decide
theorem modelrow_agrees_3065 : modelRow 3065 = row 3065 := model_count_3065.trans row_count_3065.symm
def row_3066 : RowCertificate 3066 where
  first := false
  first_correct := by
    have composite : ¬ Nat.Prime (corner 3066 1) := by change ¬ Nat.Prime 37607557; exact PrattCertificates.composite_of_product (factor := 13) (cofactor := 2892889) rfl (by decide) (by decide)
    exact decide_eq_false composite
  second := false
  second_correct := by
    have composite : ¬ Nat.Prime (corner 3066 2) := by change ¬ Nat.Prime 37601425; exact PrattCertificates.composite_of_product (factor := 5) (cofactor := 7520285) rfl (by decide) (by decide)
    exact decide_eq_false composite
  third := false
  third_correct := by
    have composite : ¬ Nat.Prime (corner 3066 3) := by change ¬ Nat.Prime 37595293; exact PrattCertificates.composite_of_product (factor := 37) (cofactor := 1016089) rfl (by decide) (by decide)
    exact decide_eq_false composite
theorem row_count_3066 : row 3066 = 0 := (row_3066.checked).trans rfl
theorem model_count_3066 : modelRow 3066 = 0 := by decide
theorem modelrow_agrees_3066 : modelRow 3066 = row 3066 := model_count_3066.trans row_count_3066.symm
def row_3067 : RowCertificate 3067 where
  first := false
  first_correct := by
    have composite : ¬ Nat.Prime (corner 3067 1) := by change ¬ Nat.Prime 37632091; exact PrattCertificates.composite_of_product (factor := 7) (cofactor := 5376013) rfl (by decide) (by decide)
    exact decide_eq_false composite
  second := true
  second_correct := by
    have prime : Nat.Prime (corner 3067 2) := by change Nat.Prime 37625957; exact ProjectEulerSolutions.P58PrimeNodes.p_37625957.prime
    exact decide_eq_true prime
  third := false
  third_correct := by
    have composite : ¬ Nat.Prime (corner 3067 3) := by change ¬ Nat.Prime 37619823; exact PrattCertificates.composite_of_product (factor := 3) (cofactor := 12539941) rfl (by decide) (by decide)
    exact decide_eq_false composite
theorem row_count_3067 : row 3067 = 1 := (row_3067.checked).trans rfl
theorem model_count_3067 : modelRow 3067 = 1 := by decide
theorem modelrow_agrees_3067 : modelRow 3067 = row 3067 := model_count_3067.trans row_count_3067.symm
def row_3068 : RowCertificate 3068 where
  first := false
  first_correct := by
    have composite : ¬ Nat.Prime (corner 3068 1) := by change ¬ Nat.Prime 37656633; exact PrattCertificates.composite_of_product (factor := 3) (cofactor := 12552211) rfl (by decide) (by decide)
    exact decide_eq_false composite
  second := false
  second_correct := by
    have composite : ¬ Nat.Prime (corner 3068 2) := by change ¬ Nat.Prime 37650497; exact PrattCertificates.composite_of_product (factor := 29) (cofactor := 1298293) rfl (by decide) (by decide)
    exact decide_eq_false composite
  third := false
  third_correct := by
    have composite : ¬ Nat.Prime (corner 3068 3) := by change ¬ Nat.Prime 37644361; exact PrattCertificates.composite_of_product (factor := 157) (cofactor := 239773) rfl (by decide) (by decide)
    exact decide_eq_false composite
theorem row_count_3068 : row 3068 = 0 := (row_3068.checked).trans rfl
theorem model_count_3068 : modelRow 3068 = 0 := by decide
theorem modelrow_agrees_3068 : modelRow 3068 = row 3068 := model_count_3068.trans row_count_3068.symm
def row_3069 : RowCertificate 3069 where
  first := false
  first_correct := by
    have composite : ¬ Nat.Prime (corner 3069 1) := by change ¬ Nat.Prime 37681183; exact PrattCertificates.composite_of_product (factor := 79) (cofactor := 476977) rfl (by decide) (by decide)
    exact decide_eq_false composite
  second := false
  second_correct := by
    have composite : ¬ Nat.Prime (corner 3069 2) := by change ¬ Nat.Prime 37675045; exact PrattCertificates.composite_of_product (factor := 5) (cofactor := 7535009) rfl (by decide) (by decide)
    exact decide_eq_false composite
  third := true
  third_correct := by
    have prime : Nat.Prime (corner 3069 3) := by change Nat.Prime 37668907; exact ProjectEulerSolutions.P58PrimeNodes.p_37668907.prime
    exact decide_eq_true prime
theorem row_count_3069 : row 3069 = 1 := (row_3069.checked).trans rfl
theorem model_count_3069 : modelRow 3069 = 1 := by decide
theorem modelrow_agrees_3069 : modelRow 3069 = row 3069 := model_count_3069.trans row_count_3069.symm
def row_3070 : RowCertificate 3070 where
  first := false
  first_correct := by
    have composite : ¬ Nat.Prime (corner 3070 1) := by change ¬ Nat.Prime 37705741; exact PrattCertificates.composite_of_product (factor := 73) (cofactor := 516517) rfl (by decide) (by decide)
    exact decide_eq_false composite
  second := true
  second_correct := by
    have prime : Nat.Prime (corner 3070 2) := by change Nat.Prime 37699601; exact ProjectEulerSolutions.P58PrimeNodes.p_37699601.prime
    exact decide_eq_true prime
  third := false
  third_correct := by
    have composite : ¬ Nat.Prime (corner 3070 3) := by change ¬ Nat.Prime 37693461; exact PrattCertificates.composite_of_product (factor := 3) (cofactor := 12564487) rfl (by decide) (by decide)
    exact decide_eq_false composite
theorem row_count_3070 : row 3070 = 1 := (row_3070.checked).trans rfl
theorem model_count_3070 : modelRow 3070 = 1 := by decide
theorem modelrow_agrees_3070 : modelRow 3070 = row 3070 := model_count_3070.trans row_count_3070.symm
def row_3071 : RowCertificate 3071 where
  first := false
  first_correct := by
    have composite : ¬ Nat.Prime (corner 3071 1) := by change ¬ Nat.Prime 37730307; exact PrattCertificates.composite_of_product (factor := 3) (cofactor := 12576769) rfl (by decide) (by decide)
    exact decide_eq_false composite
  second := false
  second_correct := by
    have composite : ¬ Nat.Prime (corner 3071 2) := by change ¬ Nat.Prime 37724165; exact PrattCertificates.composite_of_product (factor := 5) (cofactor := 7544833) rfl (by decide) (by decide)
    exact decide_eq_false composite
  third := false
  third_correct := by
    have composite : ¬ Nat.Prime (corner 3071 3) := by change ¬ Nat.Prime 37718023; exact PrattCertificates.composite_of_product (factor := 7) (cofactor := 5388289) rfl (by decide) (by decide)
    exact decide_eq_false composite
theorem row_count_3071 : row 3071 = 0 := (row_3071.checked).trans rfl
theorem model_count_3071 : modelRow 3071 = 0 := by decide
theorem modelrow_agrees_3071 : modelRow 3071 = row 3071 := model_count_3071.trans row_count_3071.symm
def row_3072 : RowCertificate 3072 where
  first := false
  first_correct := by
    have composite : ¬ Nat.Prime (corner 3072 1) := by change ¬ Nat.Prime 37754881; exact PrattCertificates.composite_of_product (factor := 19) (cofactor := 1987099) rfl (by decide) (by decide)
    exact decide_eq_false composite
  second := false
  second_correct := by
    have composite : ¬ Nat.Prime (corner 3072 2) := by change ¬ Nat.Prime 37748737; exact PrattCertificates.composite_of_product (factor := 13) (cofactor := 2903749) rfl (by decide) (by decide)
    exact decide_eq_false composite
  third := false
  third_correct := by
    have composite : ¬ Nat.Prime (corner 3072 3) := by change ¬ Nat.Prime 37742593; exact PrattCertificates.composite_of_product (factor := 7) (cofactor := 5391799) rfl (by decide) (by decide)
    exact decide_eq_false composite
theorem row_count_3072 : row 3072 = 0 := (row_3072.checked).trans rfl
theorem model_count_3072 : modelRow 3072 = 0 := by decide
theorem modelrow_agrees_3072 : modelRow 3072 = row 3072 := model_count_3072.trans row_count_3072.symm
private def atomic_2945 : CountSegment 1 10 13120 2944 2945 1439 1439 := CountSegment.single (first := 2944) (initial := 1439) row_count_2945 (by decide)
private theorem atomic_2945_agreement : RowAgreement 2944 2945 := RowAgreement.single (first := 2944) modelrow_agrees_2945
private def atomic_2946 : CountSegment 1 10 13120 2945 2946 1439 1439 := CountSegment.single (first := 2945) (initial := 1439) row_count_2946 (by decide)
private theorem atomic_2946_agreement : RowAgreement 2945 2946 := RowAgreement.single (first := 2945) modelrow_agrees_2946
private def atomic_2947 : CountSegment 1 10 13120 2946 2947 1439 1440 := CountSegment.single (first := 2946) (initial := 1439) row_count_2947 (by decide)
private theorem atomic_2947_agreement : RowAgreement 2946 2947 := RowAgreement.single (first := 2946) modelrow_agrees_2947
private def atomic_2948 : CountSegment 1 10 13120 2947 2948 1440 1440 := CountSegment.single (first := 2947) (initial := 1440) row_count_2948 (by decide)
private theorem atomic_2948_agreement : RowAgreement 2947 2948 := RowAgreement.single (first := 2947) modelrow_agrees_2948
private def atomic_2949 : CountSegment 1 10 13120 2948 2949 1440 1440 := CountSegment.single (first := 2948) (initial := 1440) row_count_2949 (by decide)
private theorem atomic_2949_agreement : RowAgreement 2948 2949 := RowAgreement.single (first := 2948) modelrow_agrees_2949
private def atomic_2950 : CountSegment 1 10 13120 2949 2950 1440 1440 := CountSegment.single (first := 2949) (initial := 1440) row_count_2950 (by decide)
private theorem atomic_2950_agreement : RowAgreement 2949 2950 := RowAgreement.single (first := 2949) modelrow_agrees_2950
private def atomic_2951 : CountSegment 1 10 13120 2950 2951 1440 1440 := CountSegment.single (first := 2950) (initial := 1440) row_count_2951 (by decide)
private theorem atomic_2951_agreement : RowAgreement 2950 2951 := RowAgreement.single (first := 2950) modelrow_agrees_2951
private def atomic_2952 : CountSegment 1 10 13120 2951 2952 1440 1440 := CountSegment.single (first := 2951) (initial := 1440) row_count_2952 (by decide)
private theorem atomic_2952_agreement : RowAgreement 2951 2952 := RowAgreement.single (first := 2951) modelrow_agrees_2952
private def atomic_2953 : CountSegment 1 10 13120 2952 2953 1440 1441 := CountSegment.single (first := 2952) (initial := 1440) row_count_2953 (by decide)
private theorem atomic_2953_agreement : RowAgreement 2952 2953 := RowAgreement.single (first := 2952) modelrow_agrees_2953
private def atomic_2954 : CountSegment 1 10 13120 2953 2954 1441 1442 := CountSegment.single (first := 2953) (initial := 1441) row_count_2954 (by decide)
private theorem atomic_2954_agreement : RowAgreement 2953 2954 := RowAgreement.single (first := 2953) modelrow_agrees_2954
private def atomic_2955 : CountSegment 1 10 13120 2954 2955 1442 1442 := CountSegment.single (first := 2954) (initial := 1442) row_count_2955 (by decide)
private theorem atomic_2955_agreement : RowAgreement 2954 2955 := RowAgreement.single (first := 2954) modelrow_agrees_2955
private def atomic_2956 : CountSegment 1 10 13120 2955 2956 1442 1442 := CountSegment.single (first := 2955) (initial := 1442) row_count_2956 (by decide)
private theorem atomic_2956_agreement : RowAgreement 2955 2956 := RowAgreement.single (first := 2955) modelrow_agrees_2956
private def atomic_2957 : CountSegment 1 10 13120 2956 2957 1442 1443 := CountSegment.single (first := 2956) (initial := 1442) row_count_2957 (by decide)
private theorem atomic_2957_agreement : RowAgreement 2956 2957 := RowAgreement.single (first := 2956) modelrow_agrees_2957
private def atomic_2958 : CountSegment 1 10 13120 2957 2958 1443 1443 := CountSegment.single (first := 2957) (initial := 1443) row_count_2958 (by decide)
private theorem atomic_2958_agreement : RowAgreement 2957 2958 := RowAgreement.single (first := 2957) modelrow_agrees_2958
private def atomic_2959 : CountSegment 1 10 13120 2958 2959 1443 1443 := CountSegment.single (first := 2958) (initial := 1443) row_count_2959 (by decide)
private theorem atomic_2959_agreement : RowAgreement 2958 2959 := RowAgreement.single (first := 2958) modelrow_agrees_2959
private def atomic_2960 : CountSegment 1 10 13120 2959 2960 1443 1443 := CountSegment.single (first := 2959) (initial := 1443) row_count_2960 (by decide)
private theorem atomic_2960_agreement : RowAgreement 2959 2960 := RowAgreement.single (first := 2959) modelrow_agrees_2960
private def atomic_2961 : CountSegment 1 10 13120 2960 2961 1443 1443 := CountSegment.single (first := 2960) (initial := 1443) row_count_2961 (by decide)
private theorem atomic_2961_agreement : RowAgreement 2960 2961 := RowAgreement.single (first := 2960) modelrow_agrees_2961
private def atomic_2962 : CountSegment 1 10 13120 2961 2962 1443 1443 := CountSegment.single (first := 2961) (initial := 1443) row_count_2962 (by decide)
private theorem atomic_2962_agreement : RowAgreement 2961 2962 := RowAgreement.single (first := 2961) modelrow_agrees_2962
private def atomic_2963 : CountSegment 1 10 13120 2962 2963 1443 1443 := CountSegment.single (first := 2962) (initial := 1443) row_count_2963 (by decide)
private theorem atomic_2963_agreement : RowAgreement 2962 2963 := RowAgreement.single (first := 2962) modelrow_agrees_2963
private def atomic_2964 : CountSegment 1 10 13120 2963 2964 1443 1443 := CountSegment.single (first := 2963) (initial := 1443) row_count_2964 (by decide)
private theorem atomic_2964_agreement : RowAgreement 2963 2964 := RowAgreement.single (first := 2963) modelrow_agrees_2964
private def atomic_2965 : CountSegment 1 10 13120 2964 2965 1443 1444 := CountSegment.single (first := 2964) (initial := 1443) row_count_2965 (by decide)
private theorem atomic_2965_agreement : RowAgreement 2964 2965 := RowAgreement.single (first := 2964) modelrow_agrees_2965
private def atomic_2966 : CountSegment 1 10 13120 2965 2966 1444 1444 := CountSegment.single (first := 2965) (initial := 1444) row_count_2966 (by decide)
private theorem atomic_2966_agreement : RowAgreement 2965 2966 := RowAgreement.single (first := 2965) modelrow_agrees_2966
private def atomic_2967 : CountSegment 1 10 13120 2966 2967 1444 1444 := CountSegment.single (first := 2966) (initial := 1444) row_count_2967 (by decide)
private theorem atomic_2967_agreement : RowAgreement 2966 2967 := RowAgreement.single (first := 2966) modelrow_agrees_2967
private def atomic_2968 : CountSegment 1 10 13120 2967 2968 1444 1445 := CountSegment.single (first := 2967) (initial := 1444) row_count_2968 (by decide)
private theorem atomic_2968_agreement : RowAgreement 2967 2968 := RowAgreement.single (first := 2967) modelrow_agrees_2968
private def atomic_2969 : CountSegment 1 10 13120 2968 2969 1445 1445 := CountSegment.single (first := 2968) (initial := 1445) row_count_2969 (by decide)
private theorem atomic_2969_agreement : RowAgreement 2968 2969 := RowAgreement.single (first := 2968) modelrow_agrees_2969
private def atomic_2970 : CountSegment 1 10 13120 2969 2970 1445 1445 := CountSegment.single (first := 2969) (initial := 1445) row_count_2970 (by decide)
private theorem atomic_2970_agreement : RowAgreement 2969 2970 := RowAgreement.single (first := 2969) modelrow_agrees_2970
private def atomic_2971 : CountSegment 1 10 13120 2970 2971 1445 1446 := CountSegment.single (first := 2970) (initial := 1445) row_count_2971 (by decide)
private theorem atomic_2971_agreement : RowAgreement 2970 2971 := RowAgreement.single (first := 2970) modelrow_agrees_2971
private def atomic_2972 : CountSegment 1 10 13120 2971 2972 1446 1447 := CountSegment.single (first := 2971) (initial := 1446) row_count_2972 (by decide)
private theorem atomic_2972_agreement : RowAgreement 2971 2972 := RowAgreement.single (first := 2971) modelrow_agrees_2972
private def atomic_2973 : CountSegment 1 10 13120 2972 2973 1447 1447 := CountSegment.single (first := 2972) (initial := 1447) row_count_2973 (by decide)
private theorem atomic_2973_agreement : RowAgreement 2972 2973 := RowAgreement.single (first := 2972) modelrow_agrees_2973
private def atomic_2974 : CountSegment 1 10 13120 2973 2974 1447 1447 := CountSegment.single (first := 2973) (initial := 1447) row_count_2974 (by decide)
private theorem atomic_2974_agreement : RowAgreement 2973 2974 := RowAgreement.single (first := 2973) modelrow_agrees_2974
private def atomic_2975 : CountSegment 1 10 13120 2974 2975 1447 1447 := CountSegment.single (first := 2974) (initial := 1447) row_count_2975 (by decide)
private theorem atomic_2975_agreement : RowAgreement 2974 2975 := RowAgreement.single (first := 2974) modelrow_agrees_2975
private def atomic_2976 : CountSegment 1 10 13120 2975 2976 1447 1448 := CountSegment.single (first := 2975) (initial := 1447) row_count_2976 (by decide)
private theorem atomic_2976_agreement : RowAgreement 2975 2976 := RowAgreement.single (first := 2975) modelrow_agrees_2976
private def atomic_2977 : CountSegment 1 10 13120 2976 2977 1448 1448 := CountSegment.single (first := 2976) (initial := 1448) row_count_2977 (by decide)
private theorem atomic_2977_agreement : RowAgreement 2976 2977 := RowAgreement.single (first := 2976) modelrow_agrees_2977
private def atomic_2978 : CountSegment 1 10 13120 2977 2978 1448 1448 := CountSegment.single (first := 2977) (initial := 1448) row_count_2978 (by decide)
private theorem atomic_2978_agreement : RowAgreement 2977 2978 := RowAgreement.single (first := 2977) modelrow_agrees_2978
private def atomic_2979 : CountSegment 1 10 13120 2978 2979 1448 1448 := CountSegment.single (first := 2978) (initial := 1448) row_count_2979 (by decide)
private theorem atomic_2979_agreement : RowAgreement 2978 2979 := RowAgreement.single (first := 2978) modelrow_agrees_2979
private def atomic_2980 : CountSegment 1 10 13120 2979 2980 1448 1449 := CountSegment.single (first := 2979) (initial := 1448) row_count_2980 (by decide)
private theorem atomic_2980_agreement : RowAgreement 2979 2980 := RowAgreement.single (first := 2979) modelrow_agrees_2980
private def atomic_2981 : CountSegment 1 10 13120 2980 2981 1449 1449 := CountSegment.single (first := 2980) (initial := 1449) row_count_2981 (by decide)
private theorem atomic_2981_agreement : RowAgreement 2980 2981 := RowAgreement.single (first := 2980) modelrow_agrees_2981
private def atomic_2982 : CountSegment 1 10 13120 2981 2982 1449 1450 := CountSegment.single (first := 2981) (initial := 1449) row_count_2982 (by decide)
private theorem atomic_2982_agreement : RowAgreement 2981 2982 := RowAgreement.single (first := 2981) modelrow_agrees_2982
private def atomic_2983 : CountSegment 1 10 13120 2982 2983 1450 1450 := CountSegment.single (first := 2982) (initial := 1450) row_count_2983 (by decide)
private theorem atomic_2983_agreement : RowAgreement 2982 2983 := RowAgreement.single (first := 2982) modelrow_agrees_2983
private def atomic_2984 : CountSegment 1 10 13120 2983 2984 1450 1450 := CountSegment.single (first := 2983) (initial := 1450) row_count_2984 (by decide)
private theorem atomic_2984_agreement : RowAgreement 2983 2984 := RowAgreement.single (first := 2983) modelrow_agrees_2984
private def atomic_2985 : CountSegment 1 10 13120 2984 2985 1450 1452 := CountSegment.single (first := 2984) (initial := 1450) row_count_2985 (by decide)
private theorem atomic_2985_agreement : RowAgreement 2984 2985 := RowAgreement.single (first := 2984) modelrow_agrees_2985
private def atomic_2986 : CountSegment 1 10 13120 2985 2986 1452 1452 := CountSegment.single (first := 2985) (initial := 1452) row_count_2986 (by decide)
private theorem atomic_2986_agreement : RowAgreement 2985 2986 := RowAgreement.single (first := 2985) modelrow_agrees_2986
private def atomic_2987 : CountSegment 1 10 13120 2986 2987 1452 1452 := CountSegment.single (first := 2986) (initial := 1452) row_count_2987 (by decide)
private theorem atomic_2987_agreement : RowAgreement 2986 2987 := RowAgreement.single (first := 2986) modelrow_agrees_2987
private def atomic_2988 : CountSegment 1 10 13120 2987 2988 1452 1452 := CountSegment.single (first := 2987) (initial := 1452) row_count_2988 (by decide)
private theorem atomic_2988_agreement : RowAgreement 2987 2988 := RowAgreement.single (first := 2987) modelrow_agrees_2988
private def atomic_2989 : CountSegment 1 10 13120 2988 2989 1452 1452 := CountSegment.single (first := 2988) (initial := 1452) row_count_2989 (by decide)
private theorem atomic_2989_agreement : RowAgreement 2988 2989 := RowAgreement.single (first := 2988) modelrow_agrees_2989
private def atomic_2990 : CountSegment 1 10 13120 2989 2990 1452 1453 := CountSegment.single (first := 2989) (initial := 1452) row_count_2990 (by decide)
private theorem atomic_2990_agreement : RowAgreement 2989 2990 := RowAgreement.single (first := 2989) modelrow_agrees_2990
private def atomic_2991 : CountSegment 1 10 13120 2990 2991 1453 1453 := CountSegment.single (first := 2990) (initial := 1453) row_count_2991 (by decide)
private theorem atomic_2991_agreement : RowAgreement 2990 2991 := RowAgreement.single (first := 2990) modelrow_agrees_2991
private def atomic_2992 : CountSegment 1 10 13120 2991 2992 1453 1455 := CountSegment.single (first := 2991) (initial := 1453) row_count_2992 (by decide)
private theorem atomic_2992_agreement : RowAgreement 2991 2992 := RowAgreement.single (first := 2991) modelrow_agrees_2992
private def atomic_2993 : CountSegment 1 10 13120 2992 2993 1455 1455 := CountSegment.single (first := 2992) (initial := 1455) row_count_2993 (by decide)
private theorem atomic_2993_agreement : RowAgreement 2992 2993 := RowAgreement.single (first := 2992) modelrow_agrees_2993
private def atomic_2994 : CountSegment 1 10 13120 2993 2994 1455 1455 := CountSegment.single (first := 2993) (initial := 1455) row_count_2994 (by decide)
private theorem atomic_2994_agreement : RowAgreement 2993 2994 := RowAgreement.single (first := 2993) modelrow_agrees_2994
private def atomic_2995 : CountSegment 1 10 13120 2994 2995 1455 1456 := CountSegment.single (first := 2994) (initial := 1455) row_count_2995 (by decide)
private theorem atomic_2995_agreement : RowAgreement 2994 2995 := RowAgreement.single (first := 2994) modelrow_agrees_2995
private def atomic_2996 : CountSegment 1 10 13120 2995 2996 1456 1456 := CountSegment.single (first := 2995) (initial := 1456) row_count_2996 (by decide)
private theorem atomic_2996_agreement : RowAgreement 2995 2996 := RowAgreement.single (first := 2995) modelrow_agrees_2996
private def atomic_2997 : CountSegment 1 10 13120 2996 2997 1456 1457 := CountSegment.single (first := 2996) (initial := 1456) row_count_2997 (by decide)
private theorem atomic_2997_agreement : RowAgreement 2996 2997 := RowAgreement.single (first := 2996) modelrow_agrees_2997
private def atomic_2998 : CountSegment 1 10 13120 2997 2998 1457 1458 := CountSegment.single (first := 2997) (initial := 1457) row_count_2998 (by decide)
private theorem atomic_2998_agreement : RowAgreement 2997 2998 := RowAgreement.single (first := 2997) modelrow_agrees_2998
private def atomic_2999 : CountSegment 1 10 13120 2998 2999 1458 1458 := CountSegment.single (first := 2998) (initial := 1458) row_count_2999 (by decide)
private theorem atomic_2999_agreement : RowAgreement 2998 2999 := RowAgreement.single (first := 2998) modelrow_agrees_2999
private def atomic_3000 : CountSegment 1 10 13120 2999 3000 1458 1458 := CountSegment.single (first := 2999) (initial := 1458) row_count_3000 (by decide)
private theorem atomic_3000_agreement : RowAgreement 2999 3000 := RowAgreement.single (first := 2999) modelrow_agrees_3000
private def atomic_3001 : CountSegment 1 10 13120 3000 3001 1458 1458 := CountSegment.single (first := 3000) (initial := 1458) row_count_3001 (by decide)
private theorem atomic_3001_agreement : RowAgreement 3000 3001 := RowAgreement.single (first := 3000) modelrow_agrees_3001
private def atomic_3002 : CountSegment 1 10 13120 3001 3002 1458 1458 := CountSegment.single (first := 3001) (initial := 1458) row_count_3002 (by decide)
private theorem atomic_3002_agreement : RowAgreement 3001 3002 := RowAgreement.single (first := 3001) modelrow_agrees_3002
private def atomic_3003 : CountSegment 1 10 13120 3002 3003 1458 1461 := CountSegment.single (first := 3002) (initial := 1458) row_count_3003 (by decide)
private theorem atomic_3003_agreement : RowAgreement 3002 3003 := RowAgreement.single (first := 3002) modelrow_agrees_3003
private def atomic_3004 : CountSegment 1 10 13120 3003 3004 1461 1461 := CountSegment.single (first := 3003) (initial := 1461) row_count_3004 (by decide)
private theorem atomic_3004_agreement : RowAgreement 3003 3004 := RowAgreement.single (first := 3003) modelrow_agrees_3004
private def atomic_3005 : CountSegment 1 10 13120 3004 3005 1461 1462 := CountSegment.single (first := 3004) (initial := 1461) row_count_3005 (by decide)
private theorem atomic_3005_agreement : RowAgreement 3004 3005 := RowAgreement.single (first := 3004) modelrow_agrees_3005
private def atomic_3006 : CountSegment 1 10 13120 3005 3006 1462 1462 := CountSegment.single (first := 3005) (initial := 1462) row_count_3006 (by decide)
private theorem atomic_3006_agreement : RowAgreement 3005 3006 := RowAgreement.single (first := 3005) modelrow_agrees_3006
private def atomic_3007 : CountSegment 1 10 13120 3006 3007 1462 1463 := CountSegment.single (first := 3006) (initial := 1462) row_count_3007 (by decide)
private theorem atomic_3007_agreement : RowAgreement 3006 3007 := RowAgreement.single (first := 3006) modelrow_agrees_3007
private def atomic_3008 : CountSegment 1 10 13120 3007 3008 1463 1464 := CountSegment.single (first := 3007) (initial := 1463) row_count_3008 (by decide)
private theorem atomic_3008_agreement : RowAgreement 3007 3008 := RowAgreement.single (first := 3007) modelrow_agrees_3008
private def atomic_3009 : CountSegment 1 10 13120 3008 3009 1464 1465 := CountSegment.single (first := 3008) (initial := 1464) row_count_3009 (by decide)
private theorem atomic_3009_agreement : RowAgreement 3008 3009 := RowAgreement.single (first := 3008) modelrow_agrees_3009
private def atomic_3010 : CountSegment 1 10 13120 3009 3010 1465 1465 := CountSegment.single (first := 3009) (initial := 1465) row_count_3010 (by decide)
private theorem atomic_3010_agreement : RowAgreement 3009 3010 := RowAgreement.single (first := 3009) modelrow_agrees_3010
private def atomic_3011 : CountSegment 1 10 13120 3010 3011 1465 1465 := CountSegment.single (first := 3010) (initial := 1465) row_count_3011 (by decide)
private theorem atomic_3011_agreement : RowAgreement 3010 3011 := RowAgreement.single (first := 3010) modelrow_agrees_3011
private def atomic_3012 : CountSegment 1 10 13120 3011 3012 1465 1466 := CountSegment.single (first := 3011) (initial := 1465) row_count_3012 (by decide)
private theorem atomic_3012_agreement : RowAgreement 3011 3012 := RowAgreement.single (first := 3011) modelrow_agrees_3012
private def atomic_3013 : CountSegment 1 10 13120 3012 3013 1466 1466 := CountSegment.single (first := 3012) (initial := 1466) row_count_3013 (by decide)
private theorem atomic_3013_agreement : RowAgreement 3012 3013 := RowAgreement.single (first := 3012) modelrow_agrees_3013
private def atomic_3014 : CountSegment 1 10 13120 3013 3014 1466 1466 := CountSegment.single (first := 3013) (initial := 1466) row_count_3014 (by decide)
private theorem atomic_3014_agreement : RowAgreement 3013 3014 := RowAgreement.single (first := 3013) modelrow_agrees_3014
private def atomic_3015 : CountSegment 1 10 13120 3014 3015 1466 1467 := CountSegment.single (first := 3014) (initial := 1466) row_count_3015 (by decide)
private theorem atomic_3015_agreement : RowAgreement 3014 3015 := RowAgreement.single (first := 3014) modelrow_agrees_3015
private def atomic_3016 : CountSegment 1 10 13120 3015 3016 1467 1467 := CountSegment.single (first := 3015) (initial := 1467) row_count_3016 (by decide)
private theorem atomic_3016_agreement : RowAgreement 3015 3016 := RowAgreement.single (first := 3015) modelrow_agrees_3016
private def atomic_3017 : CountSegment 1 10 13120 3016 3017 1467 1468 := CountSegment.single (first := 3016) (initial := 1467) row_count_3017 (by decide)
private theorem atomic_3017_agreement : RowAgreement 3016 3017 := RowAgreement.single (first := 3016) modelrow_agrees_3017
private def atomic_3018 : CountSegment 1 10 13120 3017 3018 1468 1468 := CountSegment.single (first := 3017) (initial := 1468) row_count_3018 (by decide)
private theorem atomic_3018_agreement : RowAgreement 3017 3018 := RowAgreement.single (first := 3017) modelrow_agrees_3018
private def atomic_3019 : CountSegment 1 10 13120 3018 3019 1468 1468 := CountSegment.single (first := 3018) (initial := 1468) row_count_3019 (by decide)
private theorem atomic_3019_agreement : RowAgreement 3018 3019 := RowAgreement.single (first := 3018) modelrow_agrees_3019
private def atomic_3020 : CountSegment 1 10 13120 3019 3020 1468 1468 := CountSegment.single (first := 3019) (initial := 1468) row_count_3020 (by decide)
private theorem atomic_3020_agreement : RowAgreement 3019 3020 := RowAgreement.single (first := 3019) modelrow_agrees_3020
private def atomic_3021 : CountSegment 1 10 13120 3020 3021 1468 1469 := CountSegment.single (first := 3020) (initial := 1468) row_count_3021 (by decide)
private theorem atomic_3021_agreement : RowAgreement 3020 3021 := RowAgreement.single (first := 3020) modelrow_agrees_3021
private def atomic_3022 : CountSegment 1 10 13120 3021 3022 1469 1469 := CountSegment.single (first := 3021) (initial := 1469) row_count_3022 (by decide)
private theorem atomic_3022_agreement : RowAgreement 3021 3022 := RowAgreement.single (first := 3021) modelrow_agrees_3022
private def atomic_3023 : CountSegment 1 10 13120 3022 3023 1469 1470 := CountSegment.single (first := 3022) (initial := 1469) row_count_3023 (by decide)
private theorem atomic_3023_agreement : RowAgreement 3022 3023 := RowAgreement.single (first := 3022) modelrow_agrees_3023
private def atomic_3024 : CountSegment 1 10 13120 3023 3024 1470 1471 := CountSegment.single (first := 3023) (initial := 1470) row_count_3024 (by decide)
private theorem atomic_3024_agreement : RowAgreement 3023 3024 := RowAgreement.single (first := 3023) modelrow_agrees_3024
private def atomic_3025 : CountSegment 1 10 13120 3024 3025 1471 1471 := CountSegment.single (first := 3024) (initial := 1471) row_count_3025 (by decide)
private theorem atomic_3025_agreement : RowAgreement 3024 3025 := RowAgreement.single (first := 3024) modelrow_agrees_3025
private def atomic_3026 : CountSegment 1 10 13120 3025 3026 1471 1472 := CountSegment.single (first := 3025) (initial := 1471) row_count_3026 (by decide)
private theorem atomic_3026_agreement : RowAgreement 3025 3026 := RowAgreement.single (first := 3025) modelrow_agrees_3026
private def atomic_3027 : CountSegment 1 10 13120 3026 3027 1472 1472 := CountSegment.single (first := 3026) (initial := 1472) row_count_3027 (by decide)
private theorem atomic_3027_agreement : RowAgreement 3026 3027 := RowAgreement.single (first := 3026) modelrow_agrees_3027
private def atomic_3028 : CountSegment 1 10 13120 3027 3028 1472 1472 := CountSegment.single (first := 3027) (initial := 1472) row_count_3028 (by decide)
private theorem atomic_3028_agreement : RowAgreement 3027 3028 := RowAgreement.single (first := 3027) modelrow_agrees_3028
private def atomic_3029 : CountSegment 1 10 13120 3028 3029 1472 1472 := CountSegment.single (first := 3028) (initial := 1472) row_count_3029 (by decide)
private theorem atomic_3029_agreement : RowAgreement 3028 3029 := RowAgreement.single (first := 3028) modelrow_agrees_3029
private def atomic_3030 : CountSegment 1 10 13120 3029 3030 1472 1473 := CountSegment.single (first := 3029) (initial := 1472) row_count_3030 (by decide)
private theorem atomic_3030_agreement : RowAgreement 3029 3030 := RowAgreement.single (first := 3029) modelrow_agrees_3030
private def atomic_3031 : CountSegment 1 10 13120 3030 3031 1473 1473 := CountSegment.single (first := 3030) (initial := 1473) row_count_3031 (by decide)
private theorem atomic_3031_agreement : RowAgreement 3030 3031 := RowAgreement.single (first := 3030) modelrow_agrees_3031
private def atomic_3032 : CountSegment 1 10 13120 3031 3032 1473 1474 := CountSegment.single (first := 3031) (initial := 1473) row_count_3032 (by decide)
private theorem atomic_3032_agreement : RowAgreement 3031 3032 := RowAgreement.single (first := 3031) modelrow_agrees_3032
private def atomic_3033 : CountSegment 1 10 13120 3032 3033 1474 1475 := CountSegment.single (first := 3032) (initial := 1474) row_count_3033 (by decide)
private theorem atomic_3033_agreement : RowAgreement 3032 3033 := RowAgreement.single (first := 3032) modelrow_agrees_3033
private def atomic_3034 : CountSegment 1 10 13120 3033 3034 1475 1475 := CountSegment.single (first := 3033) (initial := 1475) row_count_3034 (by decide)
private theorem atomic_3034_agreement : RowAgreement 3033 3034 := RowAgreement.single (first := 3033) modelrow_agrees_3034
private def atomic_3035 : CountSegment 1 10 13120 3034 3035 1475 1475 := CountSegment.single (first := 3034) (initial := 1475) row_count_3035 (by decide)
private theorem atomic_3035_agreement : RowAgreement 3034 3035 := RowAgreement.single (first := 3034) modelrow_agrees_3035
private def atomic_3036 : CountSegment 1 10 13120 3035 3036 1475 1475 := CountSegment.single (first := 3035) (initial := 1475) row_count_3036 (by decide)
private theorem atomic_3036_agreement : RowAgreement 3035 3036 := RowAgreement.single (first := 3035) modelrow_agrees_3036
private def atomic_3037 : CountSegment 1 10 13120 3036 3037 1475 1475 := CountSegment.single (first := 3036) (initial := 1475) row_count_3037 (by decide)
private theorem atomic_3037_agreement : RowAgreement 3036 3037 := RowAgreement.single (first := 3036) modelrow_agrees_3037
private def atomic_3038 : CountSegment 1 10 13120 3037 3038 1475 1475 := CountSegment.single (first := 3037) (initial := 1475) row_count_3038 (by decide)
private theorem atomic_3038_agreement : RowAgreement 3037 3038 := RowAgreement.single (first := 3037) modelrow_agrees_3038
private def atomic_3039 : CountSegment 1 10 13120 3038 3039 1475 1476 := CountSegment.single (first := 3038) (initial := 1475) row_count_3039 (by decide)
private theorem atomic_3039_agreement : RowAgreement 3038 3039 := RowAgreement.single (first := 3038) modelrow_agrees_3039
private def atomic_3040 : CountSegment 1 10 13120 3039 3040 1476 1476 := CountSegment.single (first := 3039) (initial := 1476) row_count_3040 (by decide)
private theorem atomic_3040_agreement : RowAgreement 3039 3040 := RowAgreement.single (first := 3039) modelrow_agrees_3040
private def atomic_3041 : CountSegment 1 10 13120 3040 3041 1476 1476 := CountSegment.single (first := 3040) (initial := 1476) row_count_3041 (by decide)
private theorem atomic_3041_agreement : RowAgreement 3040 3041 := RowAgreement.single (first := 3040) modelrow_agrees_3041
private def atomic_3042 : CountSegment 1 10 13120 3041 3042 1476 1476 := CountSegment.single (first := 3041) (initial := 1476) row_count_3042 (by decide)
private theorem atomic_3042_agreement : RowAgreement 3041 3042 := RowAgreement.single (first := 3041) modelrow_agrees_3042
private def atomic_3043 : CountSegment 1 10 13120 3042 3043 1476 1476 := CountSegment.single (first := 3042) (initial := 1476) row_count_3043 (by decide)
private theorem atomic_3043_agreement : RowAgreement 3042 3043 := RowAgreement.single (first := 3042) modelrow_agrees_3043
private def atomic_3044 : CountSegment 1 10 13120 3043 3044 1476 1476 := CountSegment.single (first := 3043) (initial := 1476) row_count_3044 (by decide)
private theorem atomic_3044_agreement : RowAgreement 3043 3044 := RowAgreement.single (first := 3043) modelrow_agrees_3044
private def atomic_3045 : CountSegment 1 10 13120 3044 3045 1476 1477 := CountSegment.single (first := 3044) (initial := 1476) row_count_3045 (by decide)
private theorem atomic_3045_agreement : RowAgreement 3044 3045 := RowAgreement.single (first := 3044) modelrow_agrees_3045
private def atomic_3046 : CountSegment 1 10 13120 3045 3046 1477 1477 := CountSegment.single (first := 3045) (initial := 1477) row_count_3046 (by decide)
private theorem atomic_3046_agreement : RowAgreement 3045 3046 := RowAgreement.single (first := 3045) modelrow_agrees_3046
private def atomic_3047 : CountSegment 1 10 13120 3046 3047 1477 1477 := CountSegment.single (first := 3046) (initial := 1477) row_count_3047 (by decide)
private theorem atomic_3047_agreement : RowAgreement 3046 3047 := RowAgreement.single (first := 3046) modelrow_agrees_3047
private def atomic_3048 : CountSegment 1 10 13120 3047 3048 1477 1480 := CountSegment.single (first := 3047) (initial := 1477) row_count_3048 (by decide)
private theorem atomic_3048_agreement : RowAgreement 3047 3048 := RowAgreement.single (first := 3047) modelrow_agrees_3048
private def atomic_3049 : CountSegment 1 10 13120 3048 3049 1480 1480 := CountSegment.single (first := 3048) (initial := 1480) row_count_3049 (by decide)
private theorem atomic_3049_agreement : RowAgreement 3048 3049 := RowAgreement.single (first := 3048) modelrow_agrees_3049
private def atomic_3050 : CountSegment 1 10 13120 3049 3050 1480 1480 := CountSegment.single (first := 3049) (initial := 1480) row_count_3050 (by decide)
private theorem atomic_3050_agreement : RowAgreement 3049 3050 := RowAgreement.single (first := 3049) modelrow_agrees_3050
private def atomic_3051 : CountSegment 1 10 13120 3050 3051 1480 1481 := CountSegment.single (first := 3050) (initial := 1480) row_count_3051 (by decide)
private theorem atomic_3051_agreement : RowAgreement 3050 3051 := RowAgreement.single (first := 3050) modelrow_agrees_3051
private def atomic_3052 : CountSegment 1 10 13120 3051 3052 1481 1481 := CountSegment.single (first := 3051) (initial := 1481) row_count_3052 (by decide)
private theorem atomic_3052_agreement : RowAgreement 3051 3052 := RowAgreement.single (first := 3051) modelrow_agrees_3052
private def atomic_3053 : CountSegment 1 10 13120 3052 3053 1481 1481 := CountSegment.single (first := 3052) (initial := 1481) row_count_3053 (by decide)
private theorem atomic_3053_agreement : RowAgreement 3052 3053 := RowAgreement.single (first := 3052) modelrow_agrees_3053
private def atomic_3054 : CountSegment 1 10 13120 3053 3054 1481 1482 := CountSegment.single (first := 3053) (initial := 1481) row_count_3054 (by decide)
private theorem atomic_3054_agreement : RowAgreement 3053 3054 := RowAgreement.single (first := 3053) modelrow_agrees_3054
private def atomic_3055 : CountSegment 1 10 13120 3054 3055 1482 1483 := CountSegment.single (first := 3054) (initial := 1482) row_count_3055 (by decide)
private theorem atomic_3055_agreement : RowAgreement 3054 3055 := RowAgreement.single (first := 3054) modelrow_agrees_3055
private def atomic_3056 : CountSegment 1 10 13120 3055 3056 1483 1484 := CountSegment.single (first := 3055) (initial := 1483) row_count_3056 (by decide)
private theorem atomic_3056_agreement : RowAgreement 3055 3056 := RowAgreement.single (first := 3055) modelrow_agrees_3056
private def atomic_3057 : CountSegment 1 10 13120 3056 3057 1484 1485 := CountSegment.single (first := 3056) (initial := 1484) row_count_3057 (by decide)
private theorem atomic_3057_agreement : RowAgreement 3056 3057 := RowAgreement.single (first := 3056) modelrow_agrees_3057
private def atomic_3058 : CountSegment 1 10 13120 3057 3058 1485 1485 := CountSegment.single (first := 3057) (initial := 1485) row_count_3058 (by decide)
private theorem atomic_3058_agreement : RowAgreement 3057 3058 := RowAgreement.single (first := 3057) modelrow_agrees_3058
private def atomic_3059 : CountSegment 1 10 13120 3058 3059 1485 1486 := CountSegment.single (first := 3058) (initial := 1485) row_count_3059 (by decide)
private theorem atomic_3059_agreement : RowAgreement 3058 3059 := RowAgreement.single (first := 3058) modelrow_agrees_3059
private def atomic_3060 : CountSegment 1 10 13120 3059 3060 1486 1487 := CountSegment.single (first := 3059) (initial := 1486) row_count_3060 (by decide)
private theorem atomic_3060_agreement : RowAgreement 3059 3060 := RowAgreement.single (first := 3059) modelrow_agrees_3060
private def atomic_3061 : CountSegment 1 10 13120 3060 3061 1487 1487 := CountSegment.single (first := 3060) (initial := 1487) row_count_3061 (by decide)
private theorem atomic_3061_agreement : RowAgreement 3060 3061 := RowAgreement.single (first := 3060) modelrow_agrees_3061
private def atomic_3062 : CountSegment 1 10 13120 3061 3062 1487 1487 := CountSegment.single (first := 3061) (initial := 1487) row_count_3062 (by decide)
private theorem atomic_3062_agreement : RowAgreement 3061 3062 := RowAgreement.single (first := 3061) modelrow_agrees_3062
private def atomic_3063 : CountSegment 1 10 13120 3062 3063 1487 1488 := CountSegment.single (first := 3062) (initial := 1487) row_count_3063 (by decide)
private theorem atomic_3063_agreement : RowAgreement 3062 3063 := RowAgreement.single (first := 3062) modelrow_agrees_3063
private def atomic_3064 : CountSegment 1 10 13120 3063 3064 1488 1489 := CountSegment.single (first := 3063) (initial := 1488) row_count_3064 (by decide)
private theorem atomic_3064_agreement : RowAgreement 3063 3064 := RowAgreement.single (first := 3063) modelrow_agrees_3064
private def atomic_3065 : CountSegment 1 10 13120 3064 3065 1489 1490 := CountSegment.single (first := 3064) (initial := 1489) row_count_3065 (by decide)
private theorem atomic_3065_agreement : RowAgreement 3064 3065 := RowAgreement.single (first := 3064) modelrow_agrees_3065
private def atomic_3066 : CountSegment 1 10 13120 3065 3066 1490 1490 := CountSegment.single (first := 3065) (initial := 1490) row_count_3066 (by decide)
private theorem atomic_3066_agreement : RowAgreement 3065 3066 := RowAgreement.single (first := 3065) modelrow_agrees_3066
private def atomic_3067 : CountSegment 1 10 13120 3066 3067 1490 1491 := CountSegment.single (first := 3066) (initial := 1490) row_count_3067 (by decide)
private theorem atomic_3067_agreement : RowAgreement 3066 3067 := RowAgreement.single (first := 3066) modelrow_agrees_3067
private def atomic_3068 : CountSegment 1 10 13120 3067 3068 1491 1491 := CountSegment.single (first := 3067) (initial := 1491) row_count_3068 (by decide)
private theorem atomic_3068_agreement : RowAgreement 3067 3068 := RowAgreement.single (first := 3067) modelrow_agrees_3068
private def atomic_3069 : CountSegment 1 10 13120 3068 3069 1491 1492 := CountSegment.single (first := 3068) (initial := 1491) row_count_3069 (by decide)
private theorem atomic_3069_agreement : RowAgreement 3068 3069 := RowAgreement.single (first := 3068) modelrow_agrees_3069
private def atomic_3070 : CountSegment 1 10 13120 3069 3070 1492 1493 := CountSegment.single (first := 3069) (initial := 1492) row_count_3070 (by decide)
private theorem atomic_3070_agreement : RowAgreement 3069 3070 := RowAgreement.single (first := 3069) modelrow_agrees_3070
private def atomic_3071 : CountSegment 1 10 13120 3070 3071 1493 1493 := CountSegment.single (first := 3070) (initial := 1493) row_count_3071 (by decide)
private theorem atomic_3071_agreement : RowAgreement 3070 3071 := RowAgreement.single (first := 3070) modelrow_agrees_3071
private def atomic_3072 : CountSegment 1 10 13120 3071 3072 1493 1493 := CountSegment.single (first := 3071) (initial := 1493) row_count_3072 (by decide)
private theorem atomic_3072_agreement : RowAgreement 3071 3072 := RowAgreement.single (first := 3071) modelrow_agrees_3072
private def within_023_000 : CountSegment 1 10 13120 2944 2946 1439 1439 := CountSegment.join atomic_2945 atomic_2946
private theorem within_023_000_agreement : RowAgreement 2944 2946 := RowAgreement.join atomic_2945_agreement atomic_2946_agreement
private def within_023_001 : CountSegment 1 10 13120 2946 2948 1439 1440 := CountSegment.join atomic_2947 atomic_2948
private theorem within_023_001_agreement : RowAgreement 2946 2948 := RowAgreement.join atomic_2947_agreement atomic_2948_agreement
private def within_023_002 : CountSegment 1 10 13120 2944 2948 1439 1440 := CountSegment.join within_023_000 within_023_001
private theorem within_023_002_agreement : RowAgreement 2944 2948 := RowAgreement.join within_023_000_agreement within_023_001_agreement
private def within_023_003 : CountSegment 1 10 13120 2948 2950 1440 1440 := CountSegment.join atomic_2949 atomic_2950
private theorem within_023_003_agreement : RowAgreement 2948 2950 := RowAgreement.join atomic_2949_agreement atomic_2950_agreement
private def within_023_004 : CountSegment 1 10 13120 2950 2952 1440 1440 := CountSegment.join atomic_2951 atomic_2952
private theorem within_023_004_agreement : RowAgreement 2950 2952 := RowAgreement.join atomic_2951_agreement atomic_2952_agreement
private def within_023_005 : CountSegment 1 10 13120 2948 2952 1440 1440 := CountSegment.join within_023_003 within_023_004
private theorem within_023_005_agreement : RowAgreement 2948 2952 := RowAgreement.join within_023_003_agreement within_023_004_agreement
private def within_023_006 : CountSegment 1 10 13120 2944 2952 1439 1440 := CountSegment.join within_023_002 within_023_005
private theorem within_023_006_agreement : RowAgreement 2944 2952 := RowAgreement.join within_023_002_agreement within_023_005_agreement
private def within_023_007 : CountSegment 1 10 13120 2952 2954 1440 1442 := CountSegment.join atomic_2953 atomic_2954
private theorem within_023_007_agreement : RowAgreement 2952 2954 := RowAgreement.join atomic_2953_agreement atomic_2954_agreement
private def within_023_008 : CountSegment 1 10 13120 2954 2956 1442 1442 := CountSegment.join atomic_2955 atomic_2956
private theorem within_023_008_agreement : RowAgreement 2954 2956 := RowAgreement.join atomic_2955_agreement atomic_2956_agreement
private def within_023_009 : CountSegment 1 10 13120 2952 2956 1440 1442 := CountSegment.join within_023_007 within_023_008
private theorem within_023_009_agreement : RowAgreement 2952 2956 := RowAgreement.join within_023_007_agreement within_023_008_agreement
private def within_023_010 : CountSegment 1 10 13120 2956 2958 1442 1443 := CountSegment.join atomic_2957 atomic_2958
private theorem within_023_010_agreement : RowAgreement 2956 2958 := RowAgreement.join atomic_2957_agreement atomic_2958_agreement
private def within_023_011 : CountSegment 1 10 13120 2958 2960 1443 1443 := CountSegment.join atomic_2959 atomic_2960
private theorem within_023_011_agreement : RowAgreement 2958 2960 := RowAgreement.join atomic_2959_agreement atomic_2960_agreement
private def within_023_012 : CountSegment 1 10 13120 2956 2960 1442 1443 := CountSegment.join within_023_010 within_023_011
private theorem within_023_012_agreement : RowAgreement 2956 2960 := RowAgreement.join within_023_010_agreement within_023_011_agreement
private def within_023_013 : CountSegment 1 10 13120 2952 2960 1440 1443 := CountSegment.join within_023_009 within_023_012
private theorem within_023_013_agreement : RowAgreement 2952 2960 := RowAgreement.join within_023_009_agreement within_023_012_agreement
private def within_023_014 : CountSegment 1 10 13120 2944 2960 1439 1443 := CountSegment.join within_023_006 within_023_013
private theorem within_023_014_agreement : RowAgreement 2944 2960 := RowAgreement.join within_023_006_agreement within_023_013_agreement
private def within_023_015 : CountSegment 1 10 13120 2960 2962 1443 1443 := CountSegment.join atomic_2961 atomic_2962
private theorem within_023_015_agreement : RowAgreement 2960 2962 := RowAgreement.join atomic_2961_agreement atomic_2962_agreement
private def within_023_016 : CountSegment 1 10 13120 2962 2964 1443 1443 := CountSegment.join atomic_2963 atomic_2964
private theorem within_023_016_agreement : RowAgreement 2962 2964 := RowAgreement.join atomic_2963_agreement atomic_2964_agreement
private def within_023_017 : CountSegment 1 10 13120 2960 2964 1443 1443 := CountSegment.join within_023_015 within_023_016
private theorem within_023_017_agreement : RowAgreement 2960 2964 := RowAgreement.join within_023_015_agreement within_023_016_agreement
private def within_023_018 : CountSegment 1 10 13120 2964 2966 1443 1444 := CountSegment.join atomic_2965 atomic_2966
private theorem within_023_018_agreement : RowAgreement 2964 2966 := RowAgreement.join atomic_2965_agreement atomic_2966_agreement
private def within_023_019 : CountSegment 1 10 13120 2966 2968 1444 1445 := CountSegment.join atomic_2967 atomic_2968
private theorem within_023_019_agreement : RowAgreement 2966 2968 := RowAgreement.join atomic_2967_agreement atomic_2968_agreement
private def within_023_020 : CountSegment 1 10 13120 2964 2968 1443 1445 := CountSegment.join within_023_018 within_023_019
private theorem within_023_020_agreement : RowAgreement 2964 2968 := RowAgreement.join within_023_018_agreement within_023_019_agreement
private def within_023_021 : CountSegment 1 10 13120 2960 2968 1443 1445 := CountSegment.join within_023_017 within_023_020
private theorem within_023_021_agreement : RowAgreement 2960 2968 := RowAgreement.join within_023_017_agreement within_023_020_agreement
private def within_023_022 : CountSegment 1 10 13120 2968 2970 1445 1445 := CountSegment.join atomic_2969 atomic_2970
private theorem within_023_022_agreement : RowAgreement 2968 2970 := RowAgreement.join atomic_2969_agreement atomic_2970_agreement
private def within_023_023 : CountSegment 1 10 13120 2970 2972 1445 1447 := CountSegment.join atomic_2971 atomic_2972
private theorem within_023_023_agreement : RowAgreement 2970 2972 := RowAgreement.join atomic_2971_agreement atomic_2972_agreement
private def within_023_024 : CountSegment 1 10 13120 2968 2972 1445 1447 := CountSegment.join within_023_022 within_023_023
private theorem within_023_024_agreement : RowAgreement 2968 2972 := RowAgreement.join within_023_022_agreement within_023_023_agreement
private def within_023_025 : CountSegment 1 10 13120 2972 2974 1447 1447 := CountSegment.join atomic_2973 atomic_2974
private theorem within_023_025_agreement : RowAgreement 2972 2974 := RowAgreement.join atomic_2973_agreement atomic_2974_agreement
private def within_023_026 : CountSegment 1 10 13120 2974 2976 1447 1448 := CountSegment.join atomic_2975 atomic_2976
private theorem within_023_026_agreement : RowAgreement 2974 2976 := RowAgreement.join atomic_2975_agreement atomic_2976_agreement
private def within_023_027 : CountSegment 1 10 13120 2972 2976 1447 1448 := CountSegment.join within_023_025 within_023_026
private theorem within_023_027_agreement : RowAgreement 2972 2976 := RowAgreement.join within_023_025_agreement within_023_026_agreement
private def within_023_028 : CountSegment 1 10 13120 2968 2976 1445 1448 := CountSegment.join within_023_024 within_023_027
private theorem within_023_028_agreement : RowAgreement 2968 2976 := RowAgreement.join within_023_024_agreement within_023_027_agreement
private def within_023_029 : CountSegment 1 10 13120 2960 2976 1443 1448 := CountSegment.join within_023_021 within_023_028
private theorem within_023_029_agreement : RowAgreement 2960 2976 := RowAgreement.join within_023_021_agreement within_023_028_agreement
private def within_023_030 : CountSegment 1 10 13120 2944 2976 1439 1448 := CountSegment.join within_023_014 within_023_029
private theorem within_023_030_agreement : RowAgreement 2944 2976 := RowAgreement.join within_023_014_agreement within_023_029_agreement
private def within_023_031 : CountSegment 1 10 13120 2976 2978 1448 1448 := CountSegment.join atomic_2977 atomic_2978
private theorem within_023_031_agreement : RowAgreement 2976 2978 := RowAgreement.join atomic_2977_agreement atomic_2978_agreement
private def within_023_032 : CountSegment 1 10 13120 2978 2980 1448 1449 := CountSegment.join atomic_2979 atomic_2980
private theorem within_023_032_agreement : RowAgreement 2978 2980 := RowAgreement.join atomic_2979_agreement atomic_2980_agreement
private def within_023_033 : CountSegment 1 10 13120 2976 2980 1448 1449 := CountSegment.join within_023_031 within_023_032
private theorem within_023_033_agreement : RowAgreement 2976 2980 := RowAgreement.join within_023_031_agreement within_023_032_agreement
private def within_023_034 : CountSegment 1 10 13120 2980 2982 1449 1450 := CountSegment.join atomic_2981 atomic_2982
private theorem within_023_034_agreement : RowAgreement 2980 2982 := RowAgreement.join atomic_2981_agreement atomic_2982_agreement
private def within_023_035 : CountSegment 1 10 13120 2982 2984 1450 1450 := CountSegment.join atomic_2983 atomic_2984
private theorem within_023_035_agreement : RowAgreement 2982 2984 := RowAgreement.join atomic_2983_agreement atomic_2984_agreement
private def within_023_036 : CountSegment 1 10 13120 2980 2984 1449 1450 := CountSegment.join within_023_034 within_023_035
private theorem within_023_036_agreement : RowAgreement 2980 2984 := RowAgreement.join within_023_034_agreement within_023_035_agreement
private def within_023_037 : CountSegment 1 10 13120 2976 2984 1448 1450 := CountSegment.join within_023_033 within_023_036
private theorem within_023_037_agreement : RowAgreement 2976 2984 := RowAgreement.join within_023_033_agreement within_023_036_agreement
private def within_023_038 : CountSegment 1 10 13120 2984 2986 1450 1452 := CountSegment.join atomic_2985 atomic_2986
private theorem within_023_038_agreement : RowAgreement 2984 2986 := RowAgreement.join atomic_2985_agreement atomic_2986_agreement
private def within_023_039 : CountSegment 1 10 13120 2986 2988 1452 1452 := CountSegment.join atomic_2987 atomic_2988
private theorem within_023_039_agreement : RowAgreement 2986 2988 := RowAgreement.join atomic_2987_agreement atomic_2988_agreement
private def within_023_040 : CountSegment 1 10 13120 2984 2988 1450 1452 := CountSegment.join within_023_038 within_023_039
private theorem within_023_040_agreement : RowAgreement 2984 2988 := RowAgreement.join within_023_038_agreement within_023_039_agreement
private def within_023_041 : CountSegment 1 10 13120 2988 2990 1452 1453 := CountSegment.join atomic_2989 atomic_2990
private theorem within_023_041_agreement : RowAgreement 2988 2990 := RowAgreement.join atomic_2989_agreement atomic_2990_agreement
private def within_023_042 : CountSegment 1 10 13120 2990 2992 1453 1455 := CountSegment.join atomic_2991 atomic_2992
private theorem within_023_042_agreement : RowAgreement 2990 2992 := RowAgreement.join atomic_2991_agreement atomic_2992_agreement
private def within_023_043 : CountSegment 1 10 13120 2988 2992 1452 1455 := CountSegment.join within_023_041 within_023_042
private theorem within_023_043_agreement : RowAgreement 2988 2992 := RowAgreement.join within_023_041_agreement within_023_042_agreement
private def within_023_044 : CountSegment 1 10 13120 2984 2992 1450 1455 := CountSegment.join within_023_040 within_023_043
private theorem within_023_044_agreement : RowAgreement 2984 2992 := RowAgreement.join within_023_040_agreement within_023_043_agreement
private def within_023_045 : CountSegment 1 10 13120 2976 2992 1448 1455 := CountSegment.join within_023_037 within_023_044
private theorem within_023_045_agreement : RowAgreement 2976 2992 := RowAgreement.join within_023_037_agreement within_023_044_agreement
private def within_023_046 : CountSegment 1 10 13120 2992 2994 1455 1455 := CountSegment.join atomic_2993 atomic_2994
private theorem within_023_046_agreement : RowAgreement 2992 2994 := RowAgreement.join atomic_2993_agreement atomic_2994_agreement
private def within_023_047 : CountSegment 1 10 13120 2994 2996 1455 1456 := CountSegment.join atomic_2995 atomic_2996
private theorem within_023_047_agreement : RowAgreement 2994 2996 := RowAgreement.join atomic_2995_agreement atomic_2996_agreement
private def within_023_048 : CountSegment 1 10 13120 2992 2996 1455 1456 := CountSegment.join within_023_046 within_023_047
private theorem within_023_048_agreement : RowAgreement 2992 2996 := RowAgreement.join within_023_046_agreement within_023_047_agreement
private def within_023_049 : CountSegment 1 10 13120 2996 2998 1456 1458 := CountSegment.join atomic_2997 atomic_2998
private theorem within_023_049_agreement : RowAgreement 2996 2998 := RowAgreement.join atomic_2997_agreement atomic_2998_agreement
private def within_023_050 : CountSegment 1 10 13120 2998 3000 1458 1458 := CountSegment.join atomic_2999 atomic_3000
private theorem within_023_050_agreement : RowAgreement 2998 3000 := RowAgreement.join atomic_2999_agreement atomic_3000_agreement
private def within_023_051 : CountSegment 1 10 13120 2996 3000 1456 1458 := CountSegment.join within_023_049 within_023_050
private theorem within_023_051_agreement : RowAgreement 2996 3000 := RowAgreement.join within_023_049_agreement within_023_050_agreement
private def within_023_052 : CountSegment 1 10 13120 2992 3000 1455 1458 := CountSegment.join within_023_048 within_023_051
private theorem within_023_052_agreement : RowAgreement 2992 3000 := RowAgreement.join within_023_048_agreement within_023_051_agreement
private def within_023_053 : CountSegment 1 10 13120 3000 3002 1458 1458 := CountSegment.join atomic_3001 atomic_3002
private theorem within_023_053_agreement : RowAgreement 3000 3002 := RowAgreement.join atomic_3001_agreement atomic_3002_agreement
private def within_023_054 : CountSegment 1 10 13120 3002 3004 1458 1461 := CountSegment.join atomic_3003 atomic_3004
private theorem within_023_054_agreement : RowAgreement 3002 3004 := RowAgreement.join atomic_3003_agreement atomic_3004_agreement
private def within_023_055 : CountSegment 1 10 13120 3000 3004 1458 1461 := CountSegment.join within_023_053 within_023_054
private theorem within_023_055_agreement : RowAgreement 3000 3004 := RowAgreement.join within_023_053_agreement within_023_054_agreement
private def within_023_056 : CountSegment 1 10 13120 3004 3006 1461 1462 := CountSegment.join atomic_3005 atomic_3006
private theorem within_023_056_agreement : RowAgreement 3004 3006 := RowAgreement.join atomic_3005_agreement atomic_3006_agreement
private def within_023_057 : CountSegment 1 10 13120 3006 3008 1462 1464 := CountSegment.join atomic_3007 atomic_3008
private theorem within_023_057_agreement : RowAgreement 3006 3008 := RowAgreement.join atomic_3007_agreement atomic_3008_agreement
private def within_023_058 : CountSegment 1 10 13120 3004 3008 1461 1464 := CountSegment.join within_023_056 within_023_057
private theorem within_023_058_agreement : RowAgreement 3004 3008 := RowAgreement.join within_023_056_agreement within_023_057_agreement
private def within_023_059 : CountSegment 1 10 13120 3000 3008 1458 1464 := CountSegment.join within_023_055 within_023_058
private theorem within_023_059_agreement : RowAgreement 3000 3008 := RowAgreement.join within_023_055_agreement within_023_058_agreement
private def within_023_060 : CountSegment 1 10 13120 2992 3008 1455 1464 := CountSegment.join within_023_052 within_023_059
private theorem within_023_060_agreement : RowAgreement 2992 3008 := RowAgreement.join within_023_052_agreement within_023_059_agreement
private def within_023_061 : CountSegment 1 10 13120 2976 3008 1448 1464 := CountSegment.join within_023_045 within_023_060
private theorem within_023_061_agreement : RowAgreement 2976 3008 := RowAgreement.join within_023_045_agreement within_023_060_agreement
private def within_023_062 : CountSegment 1 10 13120 2944 3008 1439 1464 := CountSegment.join within_023_030 within_023_061
private theorem within_023_062_agreement : RowAgreement 2944 3008 := RowAgreement.join within_023_030_agreement within_023_061_agreement
private def within_023_063 : CountSegment 1 10 13120 3008 3010 1464 1465 := CountSegment.join atomic_3009 atomic_3010
private theorem within_023_063_agreement : RowAgreement 3008 3010 := RowAgreement.join atomic_3009_agreement atomic_3010_agreement
private def within_023_064 : CountSegment 1 10 13120 3010 3012 1465 1466 := CountSegment.join atomic_3011 atomic_3012
private theorem within_023_064_agreement : RowAgreement 3010 3012 := RowAgreement.join atomic_3011_agreement atomic_3012_agreement
private def within_023_065 : CountSegment 1 10 13120 3008 3012 1464 1466 := CountSegment.join within_023_063 within_023_064
private theorem within_023_065_agreement : RowAgreement 3008 3012 := RowAgreement.join within_023_063_agreement within_023_064_agreement
private def within_023_066 : CountSegment 1 10 13120 3012 3014 1466 1466 := CountSegment.join atomic_3013 atomic_3014
private theorem within_023_066_agreement : RowAgreement 3012 3014 := RowAgreement.join atomic_3013_agreement atomic_3014_agreement
private def within_023_067 : CountSegment 1 10 13120 3014 3016 1466 1467 := CountSegment.join atomic_3015 atomic_3016
private theorem within_023_067_agreement : RowAgreement 3014 3016 := RowAgreement.join atomic_3015_agreement atomic_3016_agreement
private def within_023_068 : CountSegment 1 10 13120 3012 3016 1466 1467 := CountSegment.join within_023_066 within_023_067
private theorem within_023_068_agreement : RowAgreement 3012 3016 := RowAgreement.join within_023_066_agreement within_023_067_agreement
private def within_023_069 : CountSegment 1 10 13120 3008 3016 1464 1467 := CountSegment.join within_023_065 within_023_068
private theorem within_023_069_agreement : RowAgreement 3008 3016 := RowAgreement.join within_023_065_agreement within_023_068_agreement
private def within_023_070 : CountSegment 1 10 13120 3016 3018 1467 1468 := CountSegment.join atomic_3017 atomic_3018
private theorem within_023_070_agreement : RowAgreement 3016 3018 := RowAgreement.join atomic_3017_agreement atomic_3018_agreement
private def within_023_071 : CountSegment 1 10 13120 3018 3020 1468 1468 := CountSegment.join atomic_3019 atomic_3020
private theorem within_023_071_agreement : RowAgreement 3018 3020 := RowAgreement.join atomic_3019_agreement atomic_3020_agreement
private def within_023_072 : CountSegment 1 10 13120 3016 3020 1467 1468 := CountSegment.join within_023_070 within_023_071
private theorem within_023_072_agreement : RowAgreement 3016 3020 := RowAgreement.join within_023_070_agreement within_023_071_agreement
private def within_023_073 : CountSegment 1 10 13120 3020 3022 1468 1469 := CountSegment.join atomic_3021 atomic_3022
private theorem within_023_073_agreement : RowAgreement 3020 3022 := RowAgreement.join atomic_3021_agreement atomic_3022_agreement
private def within_023_074 : CountSegment 1 10 13120 3022 3024 1469 1471 := CountSegment.join atomic_3023 atomic_3024
private theorem within_023_074_agreement : RowAgreement 3022 3024 := RowAgreement.join atomic_3023_agreement atomic_3024_agreement
private def within_023_075 : CountSegment 1 10 13120 3020 3024 1468 1471 := CountSegment.join within_023_073 within_023_074
private theorem within_023_075_agreement : RowAgreement 3020 3024 := RowAgreement.join within_023_073_agreement within_023_074_agreement
private def within_023_076 : CountSegment 1 10 13120 3016 3024 1467 1471 := CountSegment.join within_023_072 within_023_075
private theorem within_023_076_agreement : RowAgreement 3016 3024 := RowAgreement.join within_023_072_agreement within_023_075_agreement
private def within_023_077 : CountSegment 1 10 13120 3008 3024 1464 1471 := CountSegment.join within_023_069 within_023_076
private theorem within_023_077_agreement : RowAgreement 3008 3024 := RowAgreement.join within_023_069_agreement within_023_076_agreement
private def within_023_078 : CountSegment 1 10 13120 3024 3026 1471 1472 := CountSegment.join atomic_3025 atomic_3026
private theorem within_023_078_agreement : RowAgreement 3024 3026 := RowAgreement.join atomic_3025_agreement atomic_3026_agreement
private def within_023_079 : CountSegment 1 10 13120 3026 3028 1472 1472 := CountSegment.join atomic_3027 atomic_3028
private theorem within_023_079_agreement : RowAgreement 3026 3028 := RowAgreement.join atomic_3027_agreement atomic_3028_agreement
private def within_023_080 : CountSegment 1 10 13120 3024 3028 1471 1472 := CountSegment.join within_023_078 within_023_079
private theorem within_023_080_agreement : RowAgreement 3024 3028 := RowAgreement.join within_023_078_agreement within_023_079_agreement
private def within_023_081 : CountSegment 1 10 13120 3028 3030 1472 1473 := CountSegment.join atomic_3029 atomic_3030
private theorem within_023_081_agreement : RowAgreement 3028 3030 := RowAgreement.join atomic_3029_agreement atomic_3030_agreement
private def within_023_082 : CountSegment 1 10 13120 3030 3032 1473 1474 := CountSegment.join atomic_3031 atomic_3032
private theorem within_023_082_agreement : RowAgreement 3030 3032 := RowAgreement.join atomic_3031_agreement atomic_3032_agreement
private def within_023_083 : CountSegment 1 10 13120 3028 3032 1472 1474 := CountSegment.join within_023_081 within_023_082
private theorem within_023_083_agreement : RowAgreement 3028 3032 := RowAgreement.join within_023_081_agreement within_023_082_agreement
private def within_023_084 : CountSegment 1 10 13120 3024 3032 1471 1474 := CountSegment.join within_023_080 within_023_083
private theorem within_023_084_agreement : RowAgreement 3024 3032 := RowAgreement.join within_023_080_agreement within_023_083_agreement
private def within_023_085 : CountSegment 1 10 13120 3032 3034 1474 1475 := CountSegment.join atomic_3033 atomic_3034
private theorem within_023_085_agreement : RowAgreement 3032 3034 := RowAgreement.join atomic_3033_agreement atomic_3034_agreement
private def within_023_086 : CountSegment 1 10 13120 3034 3036 1475 1475 := CountSegment.join atomic_3035 atomic_3036
private theorem within_023_086_agreement : RowAgreement 3034 3036 := RowAgreement.join atomic_3035_agreement atomic_3036_agreement
private def within_023_087 : CountSegment 1 10 13120 3032 3036 1474 1475 := CountSegment.join within_023_085 within_023_086
private theorem within_023_087_agreement : RowAgreement 3032 3036 := RowAgreement.join within_023_085_agreement within_023_086_agreement
private def within_023_088 : CountSegment 1 10 13120 3036 3038 1475 1475 := CountSegment.join atomic_3037 atomic_3038
private theorem within_023_088_agreement : RowAgreement 3036 3038 := RowAgreement.join atomic_3037_agreement atomic_3038_agreement
private def within_023_089 : CountSegment 1 10 13120 3038 3040 1475 1476 := CountSegment.join atomic_3039 atomic_3040
private theorem within_023_089_agreement : RowAgreement 3038 3040 := RowAgreement.join atomic_3039_agreement atomic_3040_agreement
private def within_023_090 : CountSegment 1 10 13120 3036 3040 1475 1476 := CountSegment.join within_023_088 within_023_089
private theorem within_023_090_agreement : RowAgreement 3036 3040 := RowAgreement.join within_023_088_agreement within_023_089_agreement
private def within_023_091 : CountSegment 1 10 13120 3032 3040 1474 1476 := CountSegment.join within_023_087 within_023_090
private theorem within_023_091_agreement : RowAgreement 3032 3040 := RowAgreement.join within_023_087_agreement within_023_090_agreement
private def within_023_092 : CountSegment 1 10 13120 3024 3040 1471 1476 := CountSegment.join within_023_084 within_023_091
private theorem within_023_092_agreement : RowAgreement 3024 3040 := RowAgreement.join within_023_084_agreement within_023_091_agreement
private def within_023_093 : CountSegment 1 10 13120 3008 3040 1464 1476 := CountSegment.join within_023_077 within_023_092
private theorem within_023_093_agreement : RowAgreement 3008 3040 := RowAgreement.join within_023_077_agreement within_023_092_agreement
private def within_023_094 : CountSegment 1 10 13120 3040 3042 1476 1476 := CountSegment.join atomic_3041 atomic_3042
private theorem within_023_094_agreement : RowAgreement 3040 3042 := RowAgreement.join atomic_3041_agreement atomic_3042_agreement
private def within_023_095 : CountSegment 1 10 13120 3042 3044 1476 1476 := CountSegment.join atomic_3043 atomic_3044
private theorem within_023_095_agreement : RowAgreement 3042 3044 := RowAgreement.join atomic_3043_agreement atomic_3044_agreement
private def within_023_096 : CountSegment 1 10 13120 3040 3044 1476 1476 := CountSegment.join within_023_094 within_023_095
private theorem within_023_096_agreement : RowAgreement 3040 3044 := RowAgreement.join within_023_094_agreement within_023_095_agreement
private def within_023_097 : CountSegment 1 10 13120 3044 3046 1476 1477 := CountSegment.join atomic_3045 atomic_3046
private theorem within_023_097_agreement : RowAgreement 3044 3046 := RowAgreement.join atomic_3045_agreement atomic_3046_agreement
private def within_023_098 : CountSegment 1 10 13120 3046 3048 1477 1480 := CountSegment.join atomic_3047 atomic_3048
private theorem within_023_098_agreement : RowAgreement 3046 3048 := RowAgreement.join atomic_3047_agreement atomic_3048_agreement
private def within_023_099 : CountSegment 1 10 13120 3044 3048 1476 1480 := CountSegment.join within_023_097 within_023_098
private theorem within_023_099_agreement : RowAgreement 3044 3048 := RowAgreement.join within_023_097_agreement within_023_098_agreement
private def within_023_100 : CountSegment 1 10 13120 3040 3048 1476 1480 := CountSegment.join within_023_096 within_023_099
private theorem within_023_100_agreement : RowAgreement 3040 3048 := RowAgreement.join within_023_096_agreement within_023_099_agreement
private def within_023_101 : CountSegment 1 10 13120 3048 3050 1480 1480 := CountSegment.join atomic_3049 atomic_3050
private theorem within_023_101_agreement : RowAgreement 3048 3050 := RowAgreement.join atomic_3049_agreement atomic_3050_agreement
private def within_023_102 : CountSegment 1 10 13120 3050 3052 1480 1481 := CountSegment.join atomic_3051 atomic_3052
private theorem within_023_102_agreement : RowAgreement 3050 3052 := RowAgreement.join atomic_3051_agreement atomic_3052_agreement
private def within_023_103 : CountSegment 1 10 13120 3048 3052 1480 1481 := CountSegment.join within_023_101 within_023_102
private theorem within_023_103_agreement : RowAgreement 3048 3052 := RowAgreement.join within_023_101_agreement within_023_102_agreement
private def within_023_104 : CountSegment 1 10 13120 3052 3054 1481 1482 := CountSegment.join atomic_3053 atomic_3054
private theorem within_023_104_agreement : RowAgreement 3052 3054 := RowAgreement.join atomic_3053_agreement atomic_3054_agreement
private def within_023_105 : CountSegment 1 10 13120 3054 3056 1482 1484 := CountSegment.join atomic_3055 atomic_3056
private theorem within_023_105_agreement : RowAgreement 3054 3056 := RowAgreement.join atomic_3055_agreement atomic_3056_agreement
private def within_023_106 : CountSegment 1 10 13120 3052 3056 1481 1484 := CountSegment.join within_023_104 within_023_105
private theorem within_023_106_agreement : RowAgreement 3052 3056 := RowAgreement.join within_023_104_agreement within_023_105_agreement
private def within_023_107 : CountSegment 1 10 13120 3048 3056 1480 1484 := CountSegment.join within_023_103 within_023_106
private theorem within_023_107_agreement : RowAgreement 3048 3056 := RowAgreement.join within_023_103_agreement within_023_106_agreement
private def within_023_108 : CountSegment 1 10 13120 3040 3056 1476 1484 := CountSegment.join within_023_100 within_023_107
private theorem within_023_108_agreement : RowAgreement 3040 3056 := RowAgreement.join within_023_100_agreement within_023_107_agreement
private def within_023_109 : CountSegment 1 10 13120 3056 3058 1484 1485 := CountSegment.join atomic_3057 atomic_3058
private theorem within_023_109_agreement : RowAgreement 3056 3058 := RowAgreement.join atomic_3057_agreement atomic_3058_agreement
private def within_023_110 : CountSegment 1 10 13120 3058 3060 1485 1487 := CountSegment.join atomic_3059 atomic_3060
private theorem within_023_110_agreement : RowAgreement 3058 3060 := RowAgreement.join atomic_3059_agreement atomic_3060_agreement
private def within_023_111 : CountSegment 1 10 13120 3056 3060 1484 1487 := CountSegment.join within_023_109 within_023_110
private theorem within_023_111_agreement : RowAgreement 3056 3060 := RowAgreement.join within_023_109_agreement within_023_110_agreement
private def within_023_112 : CountSegment 1 10 13120 3060 3062 1487 1487 := CountSegment.join atomic_3061 atomic_3062
private theorem within_023_112_agreement : RowAgreement 3060 3062 := RowAgreement.join atomic_3061_agreement atomic_3062_agreement
private def within_023_113 : CountSegment 1 10 13120 3062 3064 1487 1489 := CountSegment.join atomic_3063 atomic_3064
private theorem within_023_113_agreement : RowAgreement 3062 3064 := RowAgreement.join atomic_3063_agreement atomic_3064_agreement
private def within_023_114 : CountSegment 1 10 13120 3060 3064 1487 1489 := CountSegment.join within_023_112 within_023_113
private theorem within_023_114_agreement : RowAgreement 3060 3064 := RowAgreement.join within_023_112_agreement within_023_113_agreement
private def within_023_115 : CountSegment 1 10 13120 3056 3064 1484 1489 := CountSegment.join within_023_111 within_023_114
private theorem within_023_115_agreement : RowAgreement 3056 3064 := RowAgreement.join within_023_111_agreement within_023_114_agreement
private def within_023_116 : CountSegment 1 10 13120 3064 3066 1489 1490 := CountSegment.join atomic_3065 atomic_3066
private theorem within_023_116_agreement : RowAgreement 3064 3066 := RowAgreement.join atomic_3065_agreement atomic_3066_agreement
private def within_023_117 : CountSegment 1 10 13120 3066 3068 1490 1491 := CountSegment.join atomic_3067 atomic_3068
private theorem within_023_117_agreement : RowAgreement 3066 3068 := RowAgreement.join atomic_3067_agreement atomic_3068_agreement
private def within_023_118 : CountSegment 1 10 13120 3064 3068 1489 1491 := CountSegment.join within_023_116 within_023_117
private theorem within_023_118_agreement : RowAgreement 3064 3068 := RowAgreement.join within_023_116_agreement within_023_117_agreement
private def within_023_119 : CountSegment 1 10 13120 3068 3070 1491 1493 := CountSegment.join atomic_3069 atomic_3070
private theorem within_023_119_agreement : RowAgreement 3068 3070 := RowAgreement.join atomic_3069_agreement atomic_3070_agreement
private def within_023_120 : CountSegment 1 10 13120 3070 3072 1493 1493 := CountSegment.join atomic_3071 atomic_3072
private theorem within_023_120_agreement : RowAgreement 3070 3072 := RowAgreement.join atomic_3071_agreement atomic_3072_agreement
private def within_023_121 : CountSegment 1 10 13120 3068 3072 1491 1493 := CountSegment.join within_023_119 within_023_120
private theorem within_023_121_agreement : RowAgreement 3068 3072 := RowAgreement.join within_023_119_agreement within_023_120_agreement
private def within_023_122 : CountSegment 1 10 13120 3064 3072 1489 1493 := CountSegment.join within_023_118 within_023_121
private theorem within_023_122_agreement : RowAgreement 3064 3072 := RowAgreement.join within_023_118_agreement within_023_121_agreement
private def within_023_123 : CountSegment 1 10 13120 3056 3072 1484 1493 := CountSegment.join within_023_115 within_023_122
private theorem within_023_123_agreement : RowAgreement 3056 3072 := RowAgreement.join within_023_115_agreement within_023_122_agreement
private def within_023_124 : CountSegment 1 10 13120 3040 3072 1476 1493 := CountSegment.join within_023_108 within_023_123
private theorem within_023_124_agreement : RowAgreement 3040 3072 := RowAgreement.join within_023_108_agreement within_023_123_agreement
private def within_023_125 : CountSegment 1 10 13120 3008 3072 1464 1493 := CountSegment.join within_023_093 within_023_124
private theorem within_023_125_agreement : RowAgreement 3008 3072 := RowAgreement.join within_023_093_agreement within_023_124_agreement
private def within_023_126 : CountSegment 1 10 13120 2944 3072 1439 1493 := CountSegment.join within_023_062 within_023_125
private theorem within_023_126_agreement : RowAgreement 2944 3072 := RowAgreement.join within_023_062_agreement within_023_125_agreement
def segment_023 : CountSegment 1 10 13120 2944 3072 1439 1493 := within_023_126
theorem agreement_023 : RowAgreement 2944 3072 := within_023_126_agreement
end ProjectEulerSolutions.P58Numerical
