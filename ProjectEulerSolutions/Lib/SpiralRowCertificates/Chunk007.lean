import ProjectEulerSolutions.Lib.SpiralPrimeSegments
import ProjectEulerSolutions.Lib.SpiralPrimeCertificates.All
import ProjectEulerSolutions.Lib.MillerRabin
import ProjectEulerSolutions.Lib.SpiralMillerAgreement
/-! Finite spiral row and counter certificates. Python execution is checked separately. -/
namespace ProjectEulerSolutions.P58Numerical
open SpiralPrimes
set_option Elab.async false
def row_897 : RowCertificate 897 where
  first := false
  first_correct := by
    have composite : ¬ Nat.Prime (corner 897 1) := by change ¬ Nat.Prime 3220231; exact PrattCertificates.composite_of_product (factor := 7) (cofactor := 460033) rfl (by decide) (by decide)
    exact decide_eq_false composite
  second := true
  second_correct := by
    have prime : Nat.Prime (corner 897 2) := by change Nat.Prime 3218437; exact ProjectEulerSolutions.P58PrimeNodes.p_3218437.prime
    exact decide_eq_true prime
  third := false
  third_correct := by
    have composite : ¬ Nat.Prime (corner 897 3) := by change ¬ Nat.Prime 3216643; exact PrattCertificates.composite_of_product (factor := 19) (cofactor := 169297) rfl (by decide) (by decide)
    exact decide_eq_false composite
theorem row_count_897 : row 897 = 1 := (row_897.checked).trans rfl
theorem model_count_897 : modelRow 897 = 1 := by decide
theorem modelrow_agrees_897 : modelRow 897 = row 897 := model_count_897.trans row_count_897.symm
def row_898 : RowCertificate 898 where
  first := false
  first_correct := by
    have composite : ¬ Nat.Prime (corner 898 1) := by change ¬ Nat.Prime 3227413; exact PrattCertificates.composite_of_product (factor := 7) (cofactor := 461059) rfl (by decide) (by decide)
    exact decide_eq_false composite
  second := false
  second_correct := by
    have composite : ¬ Nat.Prime (corner 898 2) := by change ¬ Nat.Prime 3225617; exact PrattCertificates.composite_of_product (factor := 461) (cofactor := 6997) rfl (by decide) (by decide)
    exact decide_eq_false composite
  third := false
  third_correct := by
    have composite : ¬ Nat.Prime (corner 898 3) := by change ¬ Nat.Prime 3223821; exact PrattCertificates.composite_of_product (factor := 3) (cofactor := 1074607) rfl (by decide) (by decide)
    exact decide_eq_false composite
theorem row_count_898 : row 898 = 0 := (row_898.checked).trans rfl
theorem model_count_898 : modelRow 898 = 0 := by decide
theorem modelrow_agrees_898 : modelRow 898 = row 898 := model_count_898.trans row_count_898.symm
def row_899 : RowCertificate 899 where
  first := false
  first_correct := by
    have composite : ¬ Nat.Prime (corner 899 1) := by change ¬ Nat.Prime 3234603; exact PrattCertificates.composite_of_product (factor := 3) (cofactor := 1078201) rfl (by decide) (by decide)
    exact decide_eq_false composite
  second := false
  second_correct := by
    have composite : ¬ Nat.Prime (corner 899 2) := by change ¬ Nat.Prime 3232805; exact PrattCertificates.composite_of_product (factor := 5) (cofactor := 646561) rfl (by decide) (by decide)
    exact decide_eq_false composite
  third := false
  third_correct := by
    have composite : ¬ Nat.Prime (corner 899 3) := by change ¬ Nat.Prime 3231007; exact PrattCertificates.composite_of_product (factor := 13) (cofactor := 248539) rfl (by decide) (by decide)
    exact decide_eq_false composite
theorem row_count_899 : row 899 = 0 := (row_899.checked).trans rfl
theorem model_count_899 : modelRow 899 = 0 := by decide
theorem modelrow_agrees_899 : modelRow 899 = row 899 := model_count_899.trans row_count_899.symm
def row_900 : RowCertificate 900 where
  first := true
  first_correct := by
    have prime : Nat.Prime (corner 900 1) := by change Nat.Prime 3241801; exact ProjectEulerSolutions.P58PrimeNodes.p_3241801.prime
    exact decide_eq_true prime
  second := false
  second_correct := by
    have composite : ¬ Nat.Prime (corner 900 2) := by change ¬ Nat.Prime 3240001; exact PrattCertificates.composite_of_product (factor := 1741) (cofactor := 1861) rfl (by decide) (by decide)
    exact decide_eq_false composite
  third := false
  third_correct := by
    have composite : ¬ Nat.Prime (corner 900 3) := by change ¬ Nat.Prime 3238201; exact PrattCertificates.composite_of_product (factor := 43) (cofactor := 75307) rfl (by decide) (by decide)
    exact decide_eq_false composite
theorem row_count_900 : row 900 = 1 := (row_900.checked).trans rfl
theorem model_count_900 : modelRow 900 = 1 := by decide
theorem modelrow_agrees_900 : modelRow 900 = row 900 := model_count_900.trans row_count_900.symm
def row_901 : RowCertificate 901 where
  first := false
  first_correct := by
    have composite : ¬ Nat.Prime (corner 901 1) := by change ¬ Nat.Prime 3249007; exact PrattCertificates.composite_of_product (factor := 37) (cofactor := 87811) rfl (by decide) (by decide)
    exact decide_eq_false composite
  second := false
  second_correct := by
    have composite : ¬ Nat.Prime (corner 901 2) := by change ¬ Nat.Prime 3247205; exact PrattCertificates.composite_of_product (factor := 5) (cofactor := 649441) rfl (by decide) (by decide)
    exact decide_eq_false composite
  third := false
  third_correct := by
    have composite : ¬ Nat.Prime (corner 901 3) := by change ¬ Nat.Prime 3245403; exact PrattCertificates.composite_of_product (factor := 3) (cofactor := 1081801) rfl (by decide) (by decide)
    exact decide_eq_false composite
theorem row_count_901 : row 901 = 0 := (row_901.checked).trans rfl
theorem model_count_901 : modelRow 901 = 0 := by decide
theorem modelrow_agrees_901 : modelRow 901 = row 901 := model_count_901.trans row_count_901.symm
def row_902 : RowCertificate 902 where
  first := false
  first_correct := by
    have composite : ¬ Nat.Prime (corner 902 1) := by change ¬ Nat.Prime 3256221; exact PrattCertificates.composite_of_product (factor := 3) (cofactor := 1085407) rfl (by decide) (by decide)
    exact decide_eq_false composite
  second := false
  second_correct := by
    have composite : ¬ Nat.Prime (corner 902 2) := by change ¬ Nat.Prime 3254417; exact PrattCertificates.composite_of_product (factor := 613) (cofactor := 5309) rfl (by decide) (by decide)
    exact decide_eq_false composite
  third := false
  third_correct := by
    have composite : ¬ Nat.Prime (corner 902 3) := by change ¬ Nat.Prime 3252613; exact PrattCertificates.composite_of_product (factor := 7) (cofactor := 464659) rfl (by decide) (by decide)
    exact decide_eq_false composite
theorem row_count_902 : row 902 = 0 := (row_902.checked).trans rfl
theorem model_count_902 : modelRow 902 = 0 := by decide
theorem modelrow_agrees_902 : modelRow 902 = row 902 := model_count_902.trans row_count_902.symm
def row_903 : RowCertificate 903 where
  first := true
  first_correct := by
    have prime : Nat.Prime (corner 903 1) := by change Nat.Prime 3263443; exact ProjectEulerSolutions.P58PrimeNodes.p_3263443.prime
    exact decide_eq_true prime
  second := false
  second_correct := by
    have composite : ¬ Nat.Prime (corner 903 2) := by change ¬ Nat.Prime 3261637; exact PrattCertificates.composite_of_product (factor := 17) (cofactor := 191861) rfl (by decide) (by decide)
    exact decide_eq_false composite
  third := false
  third_correct := by
    have composite : ¬ Nat.Prime (corner 903 3) := by change ¬ Nat.Prime 3259831; exact PrattCertificates.composite_of_product (factor := 1621) (cofactor := 2011) rfl (by decide) (by decide)
    exact decide_eq_false composite
theorem row_count_903 : row 903 = 1 := (row_903.checked).trans rfl
theorem model_count_903 : modelRow 903 = 1 := by decide
theorem modelrow_agrees_903 : modelRow 903 = row 903 := model_count_903.trans row_count_903.symm
def row_904 : RowCertificate 904 where
  first := false
  first_correct := by
    have composite : ¬ Nat.Prime (corner 904 1) := by change ¬ Nat.Prime 3270673; exact PrattCertificates.composite_of_product (factor := 7) (cofactor := 467239) rfl (by decide) (by decide)
    exact decide_eq_false composite
  second := false
  second_correct := by
    have composite : ¬ Nat.Prime (corner 904 2) := by change ¬ Nat.Prime 3268865; exact PrattCertificates.composite_of_product (factor := 5) (cofactor := 653773) rfl (by decide) (by decide)
    exact decide_eq_false composite
  third := false
  third_correct := by
    have composite : ¬ Nat.Prime (corner 904 3) := by change ¬ Nat.Prime 3267057; exact PrattCertificates.composite_of_product (factor := 3) (cofactor := 1089019) rfl (by decide) (by decide)
    exact decide_eq_false composite
theorem row_count_904 : row 904 = 0 := (row_904.checked).trans rfl
theorem model_count_904 : modelRow 904 = 0 := by decide
theorem modelrow_agrees_904 : modelRow 904 = row 904 := model_count_904.trans row_count_904.symm
def row_905 : RowCertificate 905 where
  first := false
  first_correct := by
    have composite : ¬ Nat.Prime (corner 905 1) := by change ¬ Nat.Prime 3277911; exact PrattCertificates.composite_of_product (factor := 3) (cofactor := 1092637) rfl (by decide) (by decide)
    exact decide_eq_false composite
  second := false
  second_correct := by
    have composite : ¬ Nat.Prime (corner 905 2) := by change ¬ Nat.Prime 3276101; exact PrattCertificates.composite_of_product (factor := 29) (cofactor := 112969) rfl (by decide) (by decide)
    exact decide_eq_false composite
  third := true
  third_correct := by
    have prime : Nat.Prime (corner 905 3) := by change Nat.Prime 3274291; exact ProjectEulerSolutions.P58PrimeNodes.p_3274291.prime
    exact decide_eq_true prime
theorem row_count_905 : row 905 = 1 := (row_905.checked).trans rfl
theorem model_count_905 : modelRow 905 = 1 := by decide
theorem modelrow_agrees_905 : modelRow 905 = row 905 := model_count_905.trans row_count_905.symm
def row_906 : RowCertificate 906 where
  first := false
  first_correct := by
    have composite : ¬ Nat.Prime (corner 906 1) := by change ¬ Nat.Prime 3285157; exact PrattCertificates.composite_of_product (factor := 19) (cofactor := 172903) rfl (by decide) (by decide)
    exact decide_eq_false composite
  second := false
  second_correct := by
    have composite : ¬ Nat.Prime (corner 906 2) := by change ¬ Nat.Prime 3283345; exact PrattCertificates.composite_of_product (factor := 5) (cofactor := 656669) rfl (by decide) (by decide)
    exact decide_eq_false composite
  third := true
  third_correct := by
    have prime : Nat.Prime (corner 906 3) := by change Nat.Prime 3281533; exact ProjectEulerSolutions.P58PrimeNodes.p_3281533.prime
    exact decide_eq_true prime
theorem row_count_906 : row 906 = 1 := (row_906.checked).trans rfl
theorem model_count_906 : modelRow 906 = 1 := by decide
theorem modelrow_agrees_906 : modelRow 906 = row 906 := model_count_906.trans row_count_906.symm
def row_907 : RowCertificate 907 where
  first := true
  first_correct := by
    have prime : Nat.Prime (corner 907 1) := by change Nat.Prime 3292411; exact ProjectEulerSolutions.P58PrimeNodes.p_3292411.prime
    exact decide_eq_true prime
  second := false
  second_correct := by
    have composite : ¬ Nat.Prime (corner 907 2) := by change ¬ Nat.Prime 3290597; exact PrattCertificates.composite_of_product (factor := 89) (cofactor := 36973) rfl (by decide) (by decide)
    exact decide_eq_false composite
  third := false
  third_correct := by
    have composite : ¬ Nat.Prime (corner 907 3) := by change ¬ Nat.Prime 3288783; exact PrattCertificates.composite_of_product (factor := 3) (cofactor := 1096261) rfl (by decide) (by decide)
    exact decide_eq_false composite
theorem row_count_907 : row 907 = 1 := (row_907.checked).trans rfl
theorem model_count_907 : modelRow 907 = 1 := by decide
theorem modelrow_agrees_907 : modelRow 907 = row 907 := model_count_907.trans row_count_907.symm
def row_908 : RowCertificate 908 where
  first := false
  first_correct := by
    have composite : ¬ Nat.Prime (corner 908 1) := by change ¬ Nat.Prime 3299673; exact PrattCertificates.composite_of_product (factor := 3) (cofactor := 1099891) rfl (by decide) (by decide)
    exact decide_eq_false composite
  second := true
  second_correct := by
    have prime : Nat.Prime (corner 908 2) := by change Nat.Prime 3297857; exact ProjectEulerSolutions.P58PrimeNodes.p_3297857.prime
    exact decide_eq_true prime
  third := false
  third_correct := by
    have composite : ¬ Nat.Prime (corner 908 3) := by change ¬ Nat.Prime 3296041; exact PrattCertificates.composite_of_product (factor := 7) (cofactor := 470863) rfl (by decide) (by decide)
    exact decide_eq_false composite
theorem row_count_908 : row 908 = 1 := (row_908.checked).trans rfl
theorem model_count_908 : modelRow 908 = 1 := by decide
theorem modelrow_agrees_908 : modelRow 908 = row 908 := model_count_908.trans row_count_908.symm
def row_909 : RowCertificate 909 where
  first := true
  first_correct := by
    have prime : Nat.Prime (corner 909 1) := by change Nat.Prime 3306943; exact ProjectEulerSolutions.P58PrimeNodes.p_3306943.prime
    exact decide_eq_true prime
  second := false
  second_correct := by
    have composite : ¬ Nat.Prime (corner 909 2) := by change ¬ Nat.Prime 3305125; exact PrattCertificates.composite_of_product (factor := 5) (cofactor := 661025) rfl (by decide) (by decide)
    exact decide_eq_false composite
  third := false
  third_correct := by
    have composite : ¬ Nat.Prime (corner 909 3) := by change ¬ Nat.Prime 3303307; exact PrattCertificates.composite_of_product (factor := 7) (cofactor := 471901) rfl (by decide) (by decide)
    exact decide_eq_false composite
theorem row_count_909 : row 909 = 1 := (row_909.checked).trans rfl
theorem model_count_909 : modelRow 909 = 1 := by decide
theorem modelrow_agrees_909 : modelRow 909 = row 909 := model_count_909.trans row_count_909.symm
def row_910 : RowCertificate 910 where
  first := false
  first_correct := by
    have composite : ¬ Nat.Prime (corner 910 1) := by change ¬ Nat.Prime 3314221; exact PrattCertificates.composite_of_product (factor := 823) (cofactor := 4027) rfl (by decide) (by decide)
    exact decide_eq_false composite
  second := false
  second_correct := by
    have composite : ¬ Nat.Prime (corner 910 2) := by change ¬ Nat.Prime 3312401; exact PrattCertificates.composite_of_product (factor := 109) (cofactor := 30389) rfl (by decide) (by decide)
    exact decide_eq_false composite
  third := false
  third_correct := by
    have composite : ¬ Nat.Prime (corner 910 3) := by change ¬ Nat.Prime 3310581; exact PrattCertificates.composite_of_product (factor := 3) (cofactor := 1103527) rfl (by decide) (by decide)
    exact decide_eq_false composite
theorem row_count_910 : row 910 = 0 := (row_910.checked).trans rfl
theorem model_count_910 : modelRow 910 = 0 := by decide
theorem modelrow_agrees_910 : modelRow 910 = row 910 := model_count_910.trans row_count_910.symm
def row_911 : RowCertificate 911 where
  first := false
  first_correct := by
    have composite : ¬ Nat.Prime (corner 911 1) := by change ¬ Nat.Prime 3321507; exact PrattCertificates.composite_of_product (factor := 3) (cofactor := 1107169) rfl (by decide) (by decide)
    exact decide_eq_false composite
  second := false
  second_correct := by
    have composite : ¬ Nat.Prime (corner 911 2) := by change ¬ Nat.Prime 3319685; exact PrattCertificates.composite_of_product (factor := 5) (cofactor := 663937) rfl (by decide) (by decide)
    exact decide_eq_false composite
  third := false
  third_correct := by
    have composite : ¬ Nat.Prime (corner 911 3) := by change ¬ Nat.Prime 3317863; exact PrattCertificates.composite_of_product (factor := 193) (cofactor := 17191) rfl (by decide) (by decide)
    exact decide_eq_false composite
theorem row_count_911 : row 911 = 0 := (row_911.checked).trans rfl
theorem model_count_911 : modelRow 911 = 0 := by decide
theorem modelrow_agrees_911 : modelRow 911 = row 911 := model_count_911.trans row_count_911.symm
def row_912 : RowCertificate 912 where
  first := false
  first_correct := by
    have composite : ¬ Nat.Prime (corner 912 1) := by change ¬ Nat.Prime 3328801; exact PrattCertificates.composite_of_product (factor := 7) (cofactor := 475543) rfl (by decide) (by decide)
    exact decide_eq_false composite
  second := true
  second_correct := by
    have prime : Nat.Prime (corner 912 2) := by change Nat.Prime 3326977; exact ProjectEulerSolutions.P58PrimeNodes.p_3326977.prime
    exact decide_eq_true prime
  third := false
  third_correct := by
    have composite : ¬ Nat.Prime (corner 912 3) := by change ¬ Nat.Prime 3325153; exact PrattCertificates.composite_of_product (factor := 13) (cofactor := 255781) rfl (by decide) (by decide)
    exact decide_eq_false composite
theorem row_count_912 : row 912 = 1 := (row_912.checked).trans rfl
theorem model_count_912 : modelRow 912 = 1 := by decide
theorem modelrow_agrees_912 : modelRow 912 = row 912 := model_count_912.trans row_count_912.symm
def row_913 : RowCertificate 913 where
  first := true
  first_correct := by
    have prime : Nat.Prime (corner 913 1) := by change Nat.Prime 3336103; exact ProjectEulerSolutions.P58PrimeNodes.p_3336103.prime
    exact decide_eq_true prime
  second := false
  second_correct := by
    have composite : ¬ Nat.Prime (corner 913 2) := by change ¬ Nat.Prime 3334277; exact PrattCertificates.composite_of_product (factor := 1229) (cofactor := 2713) rfl (by decide) (by decide)
    exact decide_eq_false composite
  third := false
  third_correct := by
    have composite : ¬ Nat.Prime (corner 913 3) := by change ¬ Nat.Prime 3332451; exact PrattCertificates.composite_of_product (factor := 3) (cofactor := 1110817) rfl (by decide) (by decide)
    exact decide_eq_false composite
theorem row_count_913 : row 913 = 1 := (row_913.checked).trans rfl
theorem model_count_913 : modelRow 913 = 1 := by decide
theorem modelrow_agrees_913 : modelRow 913 = row 913 := model_count_913.trans row_count_913.symm
def row_914 : RowCertificate 914 where
  first := false
  first_correct := by
    have composite : ¬ Nat.Prime (corner 914 1) := by change ¬ Nat.Prime 3343413; exact PrattCertificates.composite_of_product (factor := 3) (cofactor := 1114471) rfl (by decide) (by decide)
    exact decide_eq_false composite
  second := false
  second_correct := by
    have composite : ¬ Nat.Prime (corner 914 2) := by change ¬ Nat.Prime 3341585; exact PrattCertificates.composite_of_product (factor := 5) (cofactor := 668317) rfl (by decide) (by decide)
    exact decide_eq_false composite
  third := false
  third_correct := by
    have composite : ¬ Nat.Prime (corner 914 3) := by change ¬ Nat.Prime 3339757; exact PrattCertificates.composite_of_product (factor := 601) (cofactor := 5557) rfl (by decide) (by decide)
    exact decide_eq_false composite
theorem row_count_914 : row 914 = 0 := (row_914.checked).trans rfl
theorem model_count_914 : modelRow 914 = 0 := by decide
theorem modelrow_agrees_914 : modelRow 914 = row 914 := model_count_914.trans row_count_914.symm
def row_915 : RowCertificate 915 where
  first := false
  first_correct := by
    have composite : ¬ Nat.Prime (corner 915 1) := by change ¬ Nat.Prime 3350731; exact PrattCertificates.composite_of_product (factor := 463) (cofactor := 7237) rfl (by decide) (by decide)
    exact decide_eq_false composite
  second := false
  second_correct := by
    have composite : ¬ Nat.Prime (corner 915 2) := by change ¬ Nat.Prime 3348901; exact PrattCertificates.composite_of_product (factor := 389) (cofactor := 8609) rfl (by decide) (by decide)
    exact decide_eq_false composite
  third := false
  third_correct := by
    have composite : ¬ Nat.Prime (corner 915 3) := by change ¬ Nat.Prime 3347071; exact PrattCertificates.composite_of_product (factor := 7) (cofactor := 478153) rfl (by decide) (by decide)
    exact decide_eq_false composite
theorem row_count_915 : row 915 = 0 := (row_915.checked).trans rfl
theorem model_count_915 : modelRow 915 = 0 := by decide
theorem modelrow_agrees_915 : modelRow 915 = row 915 := model_count_915.trans row_count_915.symm
def row_916 : RowCertificate 916 where
  first := true
  first_correct := by
    have prime : Nat.Prime (corner 916 1) := by change Nat.Prime 3358057; exact ProjectEulerSolutions.P58PrimeNodes.p_3358057.prime
    exact decide_eq_true prime
  second := false
  second_correct := by
    have composite : ¬ Nat.Prime (corner 916 2) := by change ¬ Nat.Prime 3356225; exact PrattCertificates.composite_of_product (factor := 5) (cofactor := 671245) rfl (by decide) (by decide)
    exact decide_eq_false composite
  third := false
  third_correct := by
    have composite : ¬ Nat.Prime (corner 916 3) := by change ¬ Nat.Prime 3354393; exact PrattCertificates.composite_of_product (factor := 3) (cofactor := 1118131) rfl (by decide) (by decide)
    exact decide_eq_false composite
theorem row_count_916 : row 916 = 1 := (row_916.checked).trans rfl
theorem model_count_916 : modelRow 916 = 1 := by decide
theorem modelrow_agrees_916 : modelRow 916 = row 916 := model_count_916.trans row_count_916.symm
def row_917 : RowCertificate 917 where
  first := false
  first_correct := by
    have composite : ¬ Nat.Prime (corner 917 1) := by change ¬ Nat.Prime 3365391; exact PrattCertificates.composite_of_product (factor := 3) (cofactor := 1121797) rfl (by decide) (by decide)
    exact decide_eq_false composite
  second := false
  second_correct := by
    have composite : ¬ Nat.Prime (corner 917 2) := by change ¬ Nat.Prime 3363557; exact PrattCertificates.composite_of_product (factor := 1249) (cofactor := 2693) rfl (by decide) (by decide)
    exact decide_eq_false composite
  third := false
  third_correct := by
    have composite : ¬ Nat.Prime (corner 917 3) := by change ¬ Nat.Prime 3361723; exact PrattCertificates.composite_of_product (factor := 73) (cofactor := 46051) rfl (by decide) (by decide)
    exact decide_eq_false composite
theorem row_count_917 : row 917 = 0 := (row_917.checked).trans rfl
theorem model_count_917 : modelRow 917 = 0 := by decide
theorem modelrow_agrees_917 : modelRow 917 = row 917 := model_count_917.trans row_count_917.symm
def row_918 : RowCertificate 918 where
  first := false
  first_correct := by
    have composite : ¬ Nat.Prime (corner 918 1) := by change ¬ Nat.Prime 3372733; exact PrattCertificates.composite_of_product (factor := 7) (cofactor := 481819) rfl (by decide) (by decide)
    exact decide_eq_false composite
  second := false
  second_correct := by
    have composite : ¬ Nat.Prime (corner 918 2) := by change ¬ Nat.Prime 3370897; exact PrattCertificates.composite_of_product (factor := 41) (cofactor := 82217) rfl (by decide) (by decide)
    exact decide_eq_false composite
  third := false
  third_correct := by
    have composite : ¬ Nat.Prime (corner 918 3) := by change ¬ Nat.Prime 3369061; exact PrattCertificates.composite_of_product (factor := 19) (cofactor := 177319) rfl (by decide) (by decide)
    exact decide_eq_false composite
theorem row_count_918 : row 918 = 0 := (row_918.checked).trans rfl
theorem model_count_918 : modelRow 918 = 0 := by decide
theorem modelrow_agrees_918 : modelRow 918 = row 918 := model_count_918.trans row_count_918.symm
def row_919 : RowCertificate 919 where
  first := false
  first_correct := by
    have composite : ¬ Nat.Prime (corner 919 1) := by change ¬ Nat.Prime 3380083; exact PrattCertificates.composite_of_product (factor := 7) (cofactor := 482869) rfl (by decide) (by decide)
    exact decide_eq_false composite
  second := false
  second_correct := by
    have composite : ¬ Nat.Prime (corner 919 2) := by change ¬ Nat.Prime 3378245; exact PrattCertificates.composite_of_product (factor := 5) (cofactor := 675649) rfl (by decide) (by decide)
    exact decide_eq_false composite
  third := false
  third_correct := by
    have composite : ¬ Nat.Prime (corner 919 3) := by change ¬ Nat.Prime 3376407; exact PrattCertificates.composite_of_product (factor := 3) (cofactor := 1125469) rfl (by decide) (by decide)
    exact decide_eq_false composite
theorem row_count_919 : row 919 = 0 := (row_919.checked).trans rfl
theorem model_count_919 : modelRow 919 = 0 := by decide
theorem modelrow_agrees_919 : modelRow 919 = row 919 := model_count_919.trans row_count_919.symm
def row_920 : RowCertificate 920 where
  first := false
  first_correct := by
    have composite : ¬ Nat.Prime (corner 920 1) := by change ¬ Nat.Prime 3387441; exact PrattCertificates.composite_of_product (factor := 3) (cofactor := 1129147) rfl (by decide) (by decide)
    exact decide_eq_false composite
  second := false
  second_correct := by
    have composite : ¬ Nat.Prime (corner 920 2) := by change ¬ Nat.Prime 3385601; exact PrattCertificates.composite_of_product (factor := 17) (cofactor := 199153) rfl (by decide) (by decide)
    exact decide_eq_false composite
  third := false
  third_correct := by
    have composite : ¬ Nat.Prime (corner 920 3) := by change ¬ Nat.Prime 3383761; exact PrattCertificates.composite_of_product (factor := 37) (cofactor := 91453) rfl (by decide) (by decide)
    exact decide_eq_false composite
theorem row_count_920 : row 920 = 0 := (row_920.checked).trans rfl
theorem model_count_920 : modelRow 920 = 0 := by decide
theorem modelrow_agrees_920 : modelRow 920 = row 920 := model_count_920.trans row_count_920.symm
def row_921 : RowCertificate 921 where
  first := false
  first_correct := by
    have composite : ¬ Nat.Prime (corner 921 1) := by change ¬ Nat.Prime 3394807; exact PrattCertificates.composite_of_product (factor := 13) (cofactor := 261139) rfl (by decide) (by decide)
    exact decide_eq_false composite
  second := false
  second_correct := by
    have composite : ¬ Nat.Prime (corner 921 2) := by change ¬ Nat.Prime 3392965; exact PrattCertificates.composite_of_product (factor := 5) (cofactor := 678593) rfl (by decide) (by decide)
    exact decide_eq_false composite
  third := true
  third_correct := by
    have prime : Nat.Prime (corner 921 3) := by change Nat.Prime 3391123; exact ProjectEulerSolutions.P58PrimeNodes.p_3391123.prime
    exact decide_eq_true prime
theorem row_count_921 : row 921 = 1 := (row_921.checked).trans rfl
theorem model_count_921 : modelRow 921 = 1 := by decide
theorem modelrow_agrees_921 : modelRow 921 = row 921 := model_count_921.trans row_count_921.symm
def row_922 : RowCertificate 922 where
  first := false
  first_correct := by
    have composite : ¬ Nat.Prime (corner 922 1) := by change ¬ Nat.Prime 3402181; exact PrattCertificates.composite_of_product (factor := 151) (cofactor := 22531) rfl (by decide) (by decide)
    exact decide_eq_false composite
  second := false
  second_correct := by
    have composite : ¬ Nat.Prime (corner 922 2) := by change ¬ Nat.Prime 3400337; exact PrattCertificates.composite_of_product (factor := 29) (cofactor := 117253) rfl (by decide) (by decide)
    exact decide_eq_false composite
  third := false
  third_correct := by
    have composite : ¬ Nat.Prime (corner 922 3) := by change ¬ Nat.Prime 3398493; exact PrattCertificates.composite_of_product (factor := 3) (cofactor := 1132831) rfl (by decide) (by decide)
    exact decide_eq_false composite
theorem row_count_922 : row 922 = 0 := (row_922.checked).trans rfl
theorem model_count_922 : modelRow 922 = 0 := by decide
theorem modelrow_agrees_922 : modelRow 922 = row 922 := model_count_922.trans row_count_922.symm
def row_923 : RowCertificate 923 where
  first := false
  first_correct := by
    have composite : ¬ Nat.Prime (corner 923 1) := by change ¬ Nat.Prime 3409563; exact PrattCertificates.composite_of_product (factor := 3) (cofactor := 1136521) rfl (by decide) (by decide)
    exact decide_eq_false composite
  second := false
  second_correct := by
    have composite : ¬ Nat.Prime (corner 923 2) := by change ¬ Nat.Prime 3407717; exact PrattCertificates.composite_of_product (factor := 733) (cofactor := 4649) rfl (by decide) (by decide)
    exact decide_eq_false composite
  third := false
  third_correct := by
    have composite : ¬ Nat.Prime (corner 923 3) := by change ¬ Nat.Prime 3405871; exact PrattCertificates.composite_of_product (factor := 7) (cofactor := 486553) rfl (by decide) (by decide)
    exact decide_eq_false composite
theorem row_count_923 : row 923 = 0 := (row_923.checked).trans rfl
theorem model_count_923 : modelRow 923 = 0 := by decide
theorem modelrow_agrees_923 : modelRow 923 = row 923 := model_count_923.trans row_count_923.symm
def row_924 : RowCertificate 924 where
  first := true
  first_correct := by
    have prime : Nat.Prime (corner 924 1) := by change Nat.Prime 3416953; exact ProjectEulerSolutions.P58PrimeNodes.p_3416953.prime
    exact decide_eq_true prime
  second := false
  second_correct := by
    have composite : ¬ Nat.Prime (corner 924 2) := by change ¬ Nat.Prime 3415105; exact PrattCertificates.composite_of_product (factor := 5) (cofactor := 683021) rfl (by decide) (by decide)
    exact decide_eq_false composite
  third := true
  third_correct := by
    have prime : Nat.Prime (corner 924 3) := by change Nat.Prime 3413257; exact ProjectEulerSolutions.P58PrimeNodes.p_3413257.prime
    exact decide_eq_true prime
theorem row_count_924 : row 924 = 2 := (row_924.checked).trans rfl
theorem model_count_924 : modelRow 924 = 2 := by decide
theorem modelrow_agrees_924 : modelRow 924 = row 924 := model_count_924.trans row_count_924.symm
def row_925 : RowCertificate 925 where
  first := false
  first_correct := by
    have composite : ¬ Nat.Prime (corner 925 1) := by change ¬ Nat.Prime 3424351; exact PrattCertificates.composite_of_product (factor := 7) (cofactor := 489193) rfl (by decide) (by decide)
    exact decide_eq_false composite
  second := true
  second_correct := by
    have prime : Nat.Prime (corner 925 2) := by change Nat.Prime 3422501; exact ProjectEulerSolutions.P58PrimeNodes.p_3422501.prime
    exact decide_eq_true prime
  third := false
  third_correct := by
    have composite : ¬ Nat.Prime (corner 925 3) := by change ¬ Nat.Prime 3420651; exact PrattCertificates.composite_of_product (factor := 3) (cofactor := 1140217) rfl (by decide) (by decide)
    exact decide_eq_false composite
theorem row_count_925 : row 925 = 1 := (row_925.checked).trans rfl
theorem model_count_925 : modelRow 925 = 1 := by decide
theorem modelrow_agrees_925 : modelRow 925 = row 925 := model_count_925.trans row_count_925.symm
def row_926 : RowCertificate 926 where
  first := false
  first_correct := by
    have composite : ¬ Nat.Prime (corner 926 1) := by change ¬ Nat.Prime 3431757; exact PrattCertificates.composite_of_product (factor := 3) (cofactor := 1143919) rfl (by decide) (by decide)
    exact decide_eq_false composite
  second := false
  second_correct := by
    have composite : ¬ Nat.Prime (corner 926 2) := by change ¬ Nat.Prime 3429905; exact PrattCertificates.composite_of_product (factor := 5) (cofactor := 685981) rfl (by decide) (by decide)
    exact decide_eq_false composite
  third := false
  third_correct := by
    have composite : ¬ Nat.Prime (corner 926 3) := by change ¬ Nat.Prime 3428053; exact PrattCertificates.composite_of_product (factor := 163) (cofactor := 21031) rfl (by decide) (by decide)
    exact decide_eq_false composite
theorem row_count_926 : row 926 = 0 := (row_926.checked).trans rfl
theorem model_count_926 : modelRow 926 = 0 := by decide
theorem modelrow_agrees_926 : modelRow 926 = row 926 := model_count_926.trans row_count_926.symm
def row_927 : RowCertificate 927 where
  first := false
  first_correct := by
    have composite : ¬ Nat.Prime (corner 927 1) := by change ¬ Nat.Prime 3439171; exact PrattCertificates.composite_of_product (factor := 19) (cofactor := 181009) rfl (by decide) (by decide)
    exact decide_eq_false composite
  second := false
  second_correct := by
    have composite : ¬ Nat.Prime (corner 927 2) := by change ¬ Nat.Prime 3437317; exact PrattCertificates.composite_of_product (factor := 13) (cofactor := 264409) rfl (by decide) (by decide)
    exact decide_eq_false composite
  third := true
  third_correct := by
    have prime : Nat.Prime (corner 927 3) := by change Nat.Prime 3435463; exact ProjectEulerSolutions.P58PrimeNodes.p_3435463.prime
    exact decide_eq_true prime
theorem row_count_927 : row 927 = 1 := (row_927.checked).trans rfl
theorem model_count_927 : modelRow 927 = 1 := by decide
theorem modelrow_agrees_927 : modelRow 927 = row 927 := model_count_927.trans row_count_927.symm
def row_928 : RowCertificate 928 where
  first := false
  first_correct := by
    have composite : ¬ Nat.Prime (corner 928 1) := by change ¬ Nat.Prime 3446593; exact PrattCertificates.composite_of_product (factor := 499) (cofactor := 6907) rfl (by decide) (by decide)
    exact decide_eq_false composite
  second := false
  second_correct := by
    have composite : ¬ Nat.Prime (corner 928 2) := by change ¬ Nat.Prime 3444737; exact PrattCertificates.composite_of_product (factor := 37) (cofactor := 93101) rfl (by decide) (by decide)
    exact decide_eq_false composite
  third := false
  third_correct := by
    have composite : ¬ Nat.Prime (corner 928 3) := by change ¬ Nat.Prime 3442881; exact PrattCertificates.composite_of_product (factor := 3) (cofactor := 1147627) rfl (by decide) (by decide)
    exact decide_eq_false composite
theorem row_count_928 : row 928 = 0 := (row_928.checked).trans rfl
theorem model_count_928 : modelRow 928 = 0 := by decide
theorem modelrow_agrees_928 : modelRow 928 = row 928 := model_count_928.trans row_count_928.symm
def row_929 : RowCertificate 929 where
  first := false
  first_correct := by
    have composite : ¬ Nat.Prime (corner 929 1) := by change ¬ Nat.Prime 3454023; exact PrattCertificates.composite_of_product (factor := 3) (cofactor := 1151341) rfl (by decide) (by decide)
    exact decide_eq_false composite
  second := false
  second_correct := by
    have composite : ¬ Nat.Prime (corner 929 2) := by change ¬ Nat.Prime 3452165; exact PrattCertificates.composite_of_product (factor := 5) (cofactor := 690433) rfl (by decide) (by decide)
    exact decide_eq_false composite
  third := false
  third_correct := by
    have composite : ¬ Nat.Prime (corner 929 3) := by change ¬ Nat.Prime 3450307; exact PrattCertificates.composite_of_product (factor := 7) (cofactor := 492901) rfl (by decide) (by decide)
    exact decide_eq_false composite
theorem row_count_929 : row 929 = 0 := (row_929.checked).trans rfl
theorem model_count_929 : modelRow 929 = 0 := by decide
theorem modelrow_agrees_929 : modelRow 929 = row 929 := model_count_929.trans row_count_929.symm
def row_930 : RowCertificate 930 where
  first := false
  first_correct := by
    have composite : ¬ Nat.Prime (corner 930 1) := by change ¬ Nat.Prime 3461461; exact PrattCertificates.composite_of_product (factor := 37) (cofactor := 93553) rfl (by decide) (by decide)
    exact decide_eq_false composite
  second := true
  second_correct := by
    have prime : Nat.Prime (corner 930 2) := by change Nat.Prime 3459601; exact ProjectEulerSolutions.P58PrimeNodes.p_3459601.prime
    exact decide_eq_true prime
  third := false
  third_correct := by
    have composite : ¬ Nat.Prime (corner 930 3) := by change ¬ Nat.Prime 3457741; exact PrattCertificates.composite_of_product (factor := 7) (cofactor := 493963) rfl (by decide) (by decide)
    exact decide_eq_false composite
theorem row_count_930 : row 930 = 1 := (row_930.checked).trans rfl
theorem model_count_930 : modelRow 930 = 1 := by decide
theorem modelrow_agrees_930 : modelRow 930 = row 930 := model_count_930.trans row_count_930.symm
def row_931 : RowCertificate 931 where
  first := false
  first_correct := by
    have composite : ¬ Nat.Prime (corner 931 1) := by change ¬ Nat.Prime 3468907; exact PrattCertificates.composite_of_product (factor := 13) (cofactor := 266839) rfl (by decide) (by decide)
    exact decide_eq_false composite
  second := false
  second_correct := by
    have composite : ¬ Nat.Prime (corner 931 2) := by change ¬ Nat.Prime 3467045; exact PrattCertificates.composite_of_product (factor := 5) (cofactor := 693409) rfl (by decide) (by decide)
    exact decide_eq_false composite
  third := false
  third_correct := by
    have composite : ¬ Nat.Prime (corner 931 3) := by change ¬ Nat.Prime 3465183; exact PrattCertificates.composite_of_product (factor := 3) (cofactor := 1155061) rfl (by decide) (by decide)
    exact decide_eq_false composite
theorem row_count_931 : row 931 = 0 := (row_931.checked).trans rfl
theorem model_count_931 : modelRow 931 = 0 := by decide
theorem modelrow_agrees_931 : modelRow 931 = row 931 := model_count_931.trans row_count_931.symm
def row_932 : RowCertificate 932 where
  first := false
  first_correct := by
    have composite : ¬ Nat.Prime (corner 932 1) := by change ¬ Nat.Prime 3476361; exact PrattCertificates.composite_of_product (factor := 3) (cofactor := 1158787) rfl (by decide) (by decide)
    exact decide_eq_false composite
  second := false
  second_correct := by
    have composite : ¬ Nat.Prime (corner 932 2) := by change ¬ Nat.Prime 3474497; exact PrattCertificates.composite_of_product (factor := 13) (cofactor := 267269) rfl (by decide) (by decide)
    exact decide_eq_false composite
  third := true
  third_correct := by
    have prime : Nat.Prime (corner 932 3) := by change Nat.Prime 3472633; exact ProjectEulerSolutions.P58PrimeNodes.p_3472633.prime
    exact decide_eq_true prime
theorem row_count_932 : row 932 = 1 := (row_932.checked).trans rfl
theorem model_count_932 : modelRow 932 = 1 := by decide
theorem modelrow_agrees_932 : modelRow 932 = row 932 := model_count_932.trans row_count_932.symm
def row_933 : RowCertificate 933 where
  first := false
  first_correct := by
    have composite : ¬ Nat.Prime (corner 933 1) := by change ¬ Nat.Prime 3483823; exact PrattCertificates.composite_of_product (factor := 7) (cofactor := 497689) rfl (by decide) (by decide)
    exact decide_eq_false composite
  second := false
  second_correct := by
    have composite : ¬ Nat.Prime (corner 933 2) := by change ¬ Nat.Prime 3481957; exact PrattCertificates.composite_of_product (factor := 17) (cofactor := 204821) rfl (by decide) (by decide)
    exact decide_eq_false composite
  third := false
  third_correct := by
    have composite : ¬ Nat.Prime (corner 933 3) := by change ¬ Nat.Prime 3480091; exact PrattCertificates.composite_of_product (factor := 31) (cofactor := 112261) rfl (by decide) (by decide)
    exact decide_eq_false composite
theorem row_count_933 : row 933 = 0 := (row_933.checked).trans rfl
theorem model_count_933 : modelRow 933 = 0 := by decide
theorem modelrow_agrees_933 : modelRow 933 = row 933 := model_count_933.trans row_count_933.symm
def row_934 : RowCertificate 934 where
  first := false
  first_correct := by
    have composite : ¬ Nat.Prime (corner 934 1) := by change ¬ Nat.Prime 3491293; exact PrattCertificates.composite_of_product (factor := 13) (cofactor := 268561) rfl (by decide) (by decide)
    exact decide_eq_false composite
  second := false
  second_correct := by
    have composite : ¬ Nat.Prime (corner 934 2) := by change ¬ Nat.Prime 3489425; exact PrattCertificates.composite_of_product (factor := 5) (cofactor := 697885) rfl (by decide) (by decide)
    exact decide_eq_false composite
  third := false
  third_correct := by
    have composite : ¬ Nat.Prime (corner 934 3) := by change ¬ Nat.Prime 3487557; exact PrattCertificates.composite_of_product (factor := 3) (cofactor := 1162519) rfl (by decide) (by decide)
    exact decide_eq_false composite
theorem row_count_934 : row 934 = 0 := (row_934.checked).trans rfl
theorem model_count_934 : modelRow 934 = 0 := by decide
theorem modelrow_agrees_934 : modelRow 934 = row 934 := model_count_934.trans row_count_934.symm
def row_935 : RowCertificate 935 where
  first := false
  first_correct := by
    have composite : ¬ Nat.Prime (corner 935 1) := by change ¬ Nat.Prime 3498771; exact PrattCertificates.composite_of_product (factor := 3) (cofactor := 1166257) rfl (by decide) (by decide)
    exact decide_eq_false composite
  second := true
  second_correct := by
    have prime : Nat.Prime (corner 935 2) := by change Nat.Prime 3496901; exact ProjectEulerSolutions.P58PrimeNodes.p_3496901.prime
    exact decide_eq_true prime
  third := false
  third_correct := by
    have composite : ¬ Nat.Prime (corner 935 3) := by change ¬ Nat.Prime 3495031; exact PrattCertificates.composite_of_product (factor := 19) (cofactor := 183949) rfl (by decide) (by decide)
    exact decide_eq_false composite
theorem row_count_935 : row 935 = 1 := (row_935.checked).trans rfl
theorem model_count_935 : modelRow 935 = 1 := by decide
theorem modelrow_agrees_935 : modelRow 935 = row 935 := model_count_935.trans row_count_935.symm
def row_936 : RowCertificate 936 where
  first := false
  first_correct := by
    have composite : ¬ Nat.Prime (corner 936 1) := by change ¬ Nat.Prime 3506257; exact PrattCertificates.composite_of_product (factor := 79) (cofactor := 44383) rfl (by decide) (by decide)
    exact decide_eq_false composite
  second := false
  second_correct := by
    have composite : ¬ Nat.Prime (corner 936 2) := by change ¬ Nat.Prime 3504385; exact PrattCertificates.composite_of_product (factor := 5) (cofactor := 700877) rfl (by decide) (by decide)
    exact decide_eq_false composite
  third := false
  third_correct := by
    have composite : ¬ Nat.Prime (corner 936 3) := by change ¬ Nat.Prime 3502513; exact PrattCertificates.composite_of_product (factor := 7) (cofactor := 500359) rfl (by decide) (by decide)
    exact decide_eq_false composite
theorem row_count_936 : row 936 = 0 := (row_936.checked).trans rfl
theorem model_count_936 : modelRow 936 = 0 := by decide
theorem modelrow_agrees_936 : modelRow 936 = row 936 := model_count_936.trans row_count_936.symm
def row_937 : RowCertificate 937 where
  first := false
  first_correct := by
    have composite : ¬ Nat.Prime (corner 937 1) := by change ¬ Nat.Prime 3513751; exact PrattCertificates.composite_of_product (factor := 1801) (cofactor := 1951) rfl (by decide) (by decide)
    exact decide_eq_false composite
  second := false
  second_correct := by
    have composite : ¬ Nat.Prime (corner 937 2) := by change ¬ Nat.Prime 3511877; exact PrattCertificates.composite_of_product (factor := 17) (cofactor := 206581) rfl (by decide) (by decide)
    exact decide_eq_false composite
  third := false
  third_correct := by
    have composite : ¬ Nat.Prime (corner 937 3) := by change ¬ Nat.Prime 3510003; exact PrattCertificates.composite_of_product (factor := 3) (cofactor := 1170001) rfl (by decide) (by decide)
    exact decide_eq_false composite
theorem row_count_937 : row 937 = 0 := (row_937.checked).trans rfl
theorem model_count_937 : modelRow 937 = 0 := by decide
theorem modelrow_agrees_937 : modelRow 937 = row 937 := model_count_937.trans row_count_937.symm
def row_938 : RowCertificate 938 where
  first := false
  first_correct := by
    have composite : ¬ Nat.Prime (corner 938 1) := by change ¬ Nat.Prime 3521253; exact PrattCertificates.composite_of_product (factor := 3) (cofactor := 1173751) rfl (by decide) (by decide)
    exact decide_eq_false composite
  second := true
  second_correct := by
    have prime : Nat.Prime (corner 938 2) := by change Nat.Prime 3519377; exact ProjectEulerSolutions.P58PrimeNodes.p_3519377.prime
    exact decide_eq_true prime
  third := false
  third_correct := by
    have composite : ¬ Nat.Prime (corner 938 3) := by change ¬ Nat.Prime 3517501; exact PrattCertificates.composite_of_product (factor := 13) (cofactor := 270577) rfl (by decide) (by decide)
    exact decide_eq_false composite
theorem row_count_938 : row 938 = 1 := (row_938.checked).trans rfl
theorem model_count_938 : modelRow 938 = 1 := by decide
theorem modelrow_agrees_938 : modelRow 938 = row 938 := model_count_938.trans row_count_938.symm
def row_939 : RowCertificate 939 where
  first := false
  first_correct := by
    have composite : ¬ Nat.Prime (corner 939 1) := by change ¬ Nat.Prime 3528763; exact PrattCertificates.composite_of_product (factor := 7) (cofactor := 504109) rfl (by decide) (by decide)
    exact decide_eq_false composite
  second := false
  second_correct := by
    have composite : ¬ Nat.Prime (corner 939 2) := by change ¬ Nat.Prime 3526885; exact PrattCertificates.composite_of_product (factor := 5) (cofactor := 705377) rfl (by decide) (by decide)
    exact decide_eq_false composite
  third := false
  third_correct := by
    have composite : ¬ Nat.Prime (corner 939 3) := by change ¬ Nat.Prime 3525007; exact PrattCertificates.composite_of_product (factor := 61) (cofactor := 57787) rfl (by decide) (by decide)
    exact decide_eq_false composite
theorem row_count_939 : row 939 = 0 := (row_939.checked).trans rfl
theorem model_count_939 : modelRow 939 = 0 := by decide
theorem modelrow_agrees_939 : modelRow 939 = row 939 := model_count_939.trans row_count_939.symm
def row_940 : RowCertificate 940 where
  first := false
  first_correct := by
    have composite : ¬ Nat.Prime (corner 940 1) := by change ¬ Nat.Prime 3536281; exact PrattCertificates.composite_of_product (factor := 7) (cofactor := 505183) rfl (by decide) (by decide)
    exact decide_eq_false composite
  second := false
  second_correct := by
    have composite : ¬ Nat.Prime (corner 940 2) := by change ¬ Nat.Prime 3534401; exact PrattCertificates.composite_of_product (factor := 13) (cofactor := 271877) rfl (by decide) (by decide)
    exact decide_eq_false composite
  third := false
  third_correct := by
    have composite : ¬ Nat.Prime (corner 940 3) := by change ¬ Nat.Prime 3532521; exact PrattCertificates.composite_of_product (factor := 3) (cofactor := 1177507) rfl (by decide) (by decide)
    exact decide_eq_false composite
theorem row_count_940 : row 940 = 0 := (row_940.checked).trans rfl
theorem model_count_940 : modelRow 940 = 0 := by decide
theorem modelrow_agrees_940 : modelRow 940 = row 940 := model_count_940.trans row_count_940.symm
def row_941 : RowCertificate 941 where
  first := false
  first_correct := by
    have composite : ¬ Nat.Prime (corner 941 1) := by change ¬ Nat.Prime 3543807; exact PrattCertificates.composite_of_product (factor := 3) (cofactor := 1181269) rfl (by decide) (by decide)
    exact decide_eq_false composite
  second := false
  second_correct := by
    have composite : ¬ Nat.Prime (corner 941 2) := by change ¬ Nat.Prime 3541925; exact PrattCertificates.composite_of_product (factor := 5) (cofactor := 708385) rfl (by decide) (by decide)
    exact decide_eq_false composite
  third := false
  third_correct := by
    have composite : ¬ Nat.Prime (corner 941 3) := by change ¬ Nat.Prime 3540043; exact PrattCertificates.composite_of_product (factor := 13) (cofactor := 272311) rfl (by decide) (by decide)
    exact decide_eq_false composite
theorem row_count_941 : row 941 = 0 := (row_941.checked).trans rfl
theorem model_count_941 : modelRow 941 = 0 := by decide
theorem modelrow_agrees_941 : modelRow 941 = row 941 := model_count_941.trans row_count_941.symm
def row_942 : RowCertificate 942 where
  first := false
  first_correct := by
    have composite : ¬ Nat.Prime (corner 942 1) := by change ¬ Nat.Prime 3551341; exact PrattCertificates.composite_of_product (factor := 211) (cofactor := 16831) rfl (by decide) (by decide)
    exact decide_eq_false composite
  second := true
  second_correct := by
    have prime : Nat.Prime (corner 942 2) := by change Nat.Prime 3549457; exact ProjectEulerSolutions.P58PrimeNodes.p_3549457.prime
    exact decide_eq_true prime
  third := false
  third_correct := by
    have composite : ¬ Nat.Prime (corner 942 3) := by change ¬ Nat.Prime 3547573; exact PrattCertificates.composite_of_product (factor := 199) (cofactor := 17827) rfl (by decide) (by decide)
    exact decide_eq_false composite
theorem row_count_942 : row 942 = 1 := (row_942.checked).trans rfl
theorem model_count_942 : modelRow 942 = 1 := by decide
theorem modelrow_agrees_942 : modelRow 942 = row 942 := model_count_942.trans row_count_942.symm
def row_943 : RowCertificate 943 where
  first := true
  first_correct := by
    have prime : Nat.Prime (corner 943 1) := by change Nat.Prime 3558883; exact ProjectEulerSolutions.P58PrimeNodes.p_3558883.prime
    exact decide_eq_true prime
  second := false
  second_correct := by
    have composite : ¬ Nat.Prime (corner 943 2) := by change ¬ Nat.Prime 3556997; exact PrattCertificates.composite_of_product (factor := 109) (cofactor := 32633) rfl (by decide) (by decide)
    exact decide_eq_false composite
  third := false
  third_correct := by
    have composite : ¬ Nat.Prime (corner 943 3) := by change ¬ Nat.Prime 3555111; exact PrattCertificates.composite_of_product (factor := 3) (cofactor := 1185037) rfl (by decide) (by decide)
    exact decide_eq_false composite
theorem row_count_943 : row 943 = 1 := (row_943.checked).trans rfl
theorem model_count_943 : modelRow 943 = 1 := by decide
theorem modelrow_agrees_943 : modelRow 943 = row 943 := model_count_943.trans row_count_943.symm
def row_944 : RowCertificate 944 where
  first := false
  first_correct := by
    have composite : ¬ Nat.Prime (corner 944 1) := by change ¬ Nat.Prime 3566433; exact PrattCertificates.composite_of_product (factor := 3) (cofactor := 1188811) rfl (by decide) (by decide)
    exact decide_eq_false composite
  second := false
  second_correct := by
    have composite : ¬ Nat.Prime (corner 944 2) := by change ¬ Nat.Prime 3564545; exact PrattCertificates.composite_of_product (factor := 5) (cofactor := 712909) rfl (by decide) (by decide)
    exact decide_eq_false composite
  third := false
  third_correct := by
    have composite : ¬ Nat.Prime (corner 944 3) := by change ¬ Nat.Prime 3562657; exact PrattCertificates.composite_of_product (factor := 7) (cofactor := 508951) rfl (by decide) (by decide)
    exact decide_eq_false composite
theorem row_count_944 : row 944 = 0 := (row_944.checked).trans rfl
theorem model_count_944 : modelRow 944 = 0 := by decide
theorem modelrow_agrees_944 : modelRow 944 = row 944 := model_count_944.trans row_count_944.symm
def row_945 : RowCertificate 945 where
  first := false
  first_correct := by
    have composite : ¬ Nat.Prime (corner 945 1) := by change ¬ Nat.Prime 3573991; exact PrattCertificates.composite_of_product (factor := 919) (cofactor := 3889) rfl (by decide) (by decide)
    exact decide_eq_false composite
  second := false
  second_correct := by
    have composite : ¬ Nat.Prime (corner 945 2) := by change ¬ Nat.Prime 3572101; exact PrattCertificates.composite_of_product (factor := 13) (cofactor := 274777) rfl (by decide) (by decide)
    exact decide_eq_false composite
  third := false
  third_correct := by
    have composite : ¬ Nat.Prime (corner 945 3) := by change ¬ Nat.Prime 3570211; exact PrattCertificates.composite_of_product (factor := 73) (cofactor := 48907) rfl (by decide) (by decide)
    exact decide_eq_false composite
theorem row_count_945 : row 945 = 0 := (row_945.checked).trans rfl
theorem model_count_945 : modelRow 945 = 0 := by decide
theorem modelrow_agrees_945 : modelRow 945 = row 945 := model_count_945.trans row_count_945.symm
def row_946 : RowCertificate 946 where
  first := false
  first_correct := by
    have composite : ¬ Nat.Prime (corner 946 1) := by change ¬ Nat.Prime 3581557; exact PrattCertificates.composite_of_product (factor := 7) (cofactor := 511651) rfl (by decide) (by decide)
    exact decide_eq_false composite
  second := false
  second_correct := by
    have composite : ¬ Nat.Prime (corner 946 2) := by change ¬ Nat.Prime 3579665; exact PrattCertificates.composite_of_product (factor := 5) (cofactor := 715933) rfl (by decide) (by decide)
    exact decide_eq_false composite
  third := false
  third_correct := by
    have composite : ¬ Nat.Prime (corner 946 3) := by change ¬ Nat.Prime 3577773; exact PrattCertificates.composite_of_product (factor := 3) (cofactor := 1192591) rfl (by decide) (by decide)
    exact decide_eq_false composite
theorem row_count_946 : row 946 = 0 := (row_946.checked).trans rfl
theorem model_count_946 : modelRow 946 = 0 := by decide
theorem modelrow_agrees_946 : modelRow 946 = row 946 := model_count_946.trans row_count_946.symm
def row_947 : RowCertificate 947 where
  first := false
  first_correct := by
    have composite : ¬ Nat.Prime (corner 947 1) := by change ¬ Nat.Prime 3589131; exact PrattCertificates.composite_of_product (factor := 3) (cofactor := 1196377) rfl (by decide) (by decide)
    exact decide_eq_false composite
  second := true
  second_correct := by
    have prime : Nat.Prime (corner 947 2) := by change Nat.Prime 3587237; exact ProjectEulerSolutions.P58PrimeNodes.p_3587237.prime
    exact decide_eq_true prime
  third := false
  third_correct := by
    have composite : ¬ Nat.Prime (corner 947 3) := by change ¬ Nat.Prime 3585343; exact PrattCertificates.composite_of_product (factor := 337) (cofactor := 10639) rfl (by decide) (by decide)
    exact decide_eq_false composite
theorem row_count_947 : row 947 = 1 := (row_947.checked).trans rfl
theorem model_count_947 : modelRow 947 = 1 := by decide
theorem modelrow_agrees_947 : modelRow 947 = row 947 := model_count_947.trans row_count_947.symm
def row_948 : RowCertificate 948 where
  first := false
  first_correct := by
    have composite : ¬ Nat.Prime (corner 948 1) := by change ¬ Nat.Prime 3596713; exact PrattCertificates.composite_of_product (factor := 31) (cofactor := 116023) rfl (by decide) (by decide)
    exact decide_eq_false composite
  second := false
  second_correct := by
    have composite : ¬ Nat.Prime (corner 948 2) := by change ¬ Nat.Prime 3594817; exact PrattCertificates.composite_of_product (factor := 293) (cofactor := 12269) rfl (by decide) (by decide)
    exact decide_eq_false composite
  third := true
  third_correct := by
    have prime : Nat.Prime (corner 948 3) := by change Nat.Prime 3592921; exact ProjectEulerSolutions.P58PrimeNodes.p_3592921.prime
    exact decide_eq_true prime
theorem row_count_948 : row 948 = 1 := (row_948.checked).trans rfl
theorem model_count_948 : modelRow 948 = 1 := by decide
theorem modelrow_agrees_948 : modelRow 948 = row 948 := model_count_948.trans row_count_948.symm
def row_949 : RowCertificate 949 where
  first := false
  first_correct := by
    have composite : ¬ Nat.Prime (corner 949 1) := by change ¬ Nat.Prime 3604303; exact PrattCertificates.composite_of_product (factor := 43) (cofactor := 83821) rfl (by decide) (by decide)
    exact decide_eq_false composite
  second := false
  second_correct := by
    have composite : ¬ Nat.Prime (corner 949 2) := by change ¬ Nat.Prime 3602405; exact PrattCertificates.composite_of_product (factor := 5) (cofactor := 720481) rfl (by decide) (by decide)
    exact decide_eq_false composite
  third := false
  third_correct := by
    have composite : ¬ Nat.Prime (corner 949 3) := by change ¬ Nat.Prime 3600507; exact PrattCertificates.composite_of_product (factor := 3) (cofactor := 1200169) rfl (by decide) (by decide)
    exact decide_eq_false composite
theorem row_count_949 : row 949 = 0 := (row_949.checked).trans rfl
theorem model_count_949 : modelRow 949 = 0 := by decide
theorem modelrow_agrees_949 : modelRow 949 = row 949 := model_count_949.trans row_count_949.symm
def row_950 : RowCertificate 950 where
  first := false
  first_correct := by
    have composite : ¬ Nat.Prime (corner 950 1) := by change ¬ Nat.Prime 3611901; exact PrattCertificates.composite_of_product (factor := 3) (cofactor := 1203967) rfl (by decide) (by decide)
    exact decide_eq_false composite
  second := false
  second_correct := by
    have composite : ¬ Nat.Prime (corner 950 2) := by change ¬ Nat.Prime 3610001; exact PrattCertificates.composite_of_product (factor := 17) (cofactor := 212353) rfl (by decide) (by decide)
    exact decide_eq_false composite
  third := false
  third_correct := by
    have composite : ¬ Nat.Prime (corner 950 3) := by change ¬ Nat.Prime 3608101; exact PrattCertificates.composite_of_product (factor := 7) (cofactor := 515443) rfl (by decide) (by decide)
    exact decide_eq_false composite
theorem row_count_950 : row 950 = 0 := (row_950.checked).trans rfl
theorem model_count_950 : modelRow 950 = 0 := by decide
theorem modelrow_agrees_950 : modelRow 950 = row 950 := model_count_950.trans row_count_950.symm
def row_951 : RowCertificate 951 where
  first := true
  first_correct := by
    have prime : Nat.Prime (corner 951 1) := by change Nat.Prime 3619507; exact ProjectEulerSolutions.P58PrimeNodes.p_3619507.prime
    exact decide_eq_true prime
  second := false
  second_correct := by
    have composite : ¬ Nat.Prime (corner 951 2) := by change ¬ Nat.Prime 3617605; exact PrattCertificates.composite_of_product (factor := 5) (cofactor := 723521) rfl (by decide) (by decide)
    exact decide_eq_false composite
  third := false
  third_correct := by
    have composite : ¬ Nat.Prime (corner 951 3) := by change ¬ Nat.Prime 3615703; exact PrattCertificates.composite_of_product (factor := 7) (cofactor := 516529) rfl (by decide) (by decide)
    exact decide_eq_false composite
theorem row_count_951 : row 951 = 1 := (row_951.checked).trans rfl
theorem model_count_951 : modelRow 951 = 1 := by decide
theorem modelrow_agrees_951 : modelRow 951 = row 951 := model_count_951.trans row_count_951.symm
def row_952 : RowCertificate 952 where
  first := false
  first_correct := by
    have composite : ¬ Nat.Prime (corner 952 1) := by change ¬ Nat.Prime 3627121; exact PrattCertificates.composite_of_product (factor := 61) (cofactor := 59461) rfl (by decide) (by decide)
    exact decide_eq_false composite
  second := false
  second_correct := by
    have composite : ¬ Nat.Prime (corner 952 2) := by change ¬ Nat.Prime 3625217; exact PrattCertificates.composite_of_product (factor := 829) (cofactor := 4373) rfl (by decide) (by decide)
    exact decide_eq_false composite
  third := false
  third_correct := by
    have composite : ¬ Nat.Prime (corner 952 3) := by change ¬ Nat.Prime 3623313; exact PrattCertificates.composite_of_product (factor := 3) (cofactor := 1207771) rfl (by decide) (by decide)
    exact decide_eq_false composite
theorem row_count_952 : row 952 = 0 := (row_952.checked).trans rfl
theorem model_count_952 : modelRow 952 = 0 := by decide
theorem modelrow_agrees_952 : modelRow 952 = row 952 := model_count_952.trans row_count_952.symm
def row_953 : RowCertificate 953 where
  first := false
  first_correct := by
    have composite : ¬ Nat.Prime (corner 953 1) := by change ¬ Nat.Prime 3634743; exact PrattCertificates.composite_of_product (factor := 3) (cofactor := 1211581) rfl (by decide) (by decide)
    exact decide_eq_false composite
  second := false
  second_correct := by
    have composite : ¬ Nat.Prime (corner 953 2) := by change ¬ Nat.Prime 3632837; exact PrattCertificates.composite_of_product (factor := 13) (cofactor := 279449) rfl (by decide) (by decide)
    exact decide_eq_false composite
  third := false
  third_correct := by
    have composite : ¬ Nat.Prime (corner 953 3) := by change ¬ Nat.Prime 3630931; exact PrattCertificates.composite_of_product (factor := 67) (cofactor := 54193) rfl (by decide) (by decide)
    exact decide_eq_false composite
theorem row_count_953 : row 953 = 0 := (row_953.checked).trans rfl
theorem model_count_953 : modelRow 953 = 0 := by decide
theorem modelrow_agrees_953 : modelRow 953 = row 953 := model_count_953.trans row_count_953.symm
def row_954 : RowCertificate 954 where
  first := false
  first_correct := by
    have composite : ¬ Nat.Prime (corner 954 1) := by change ¬ Nat.Prime 3642373; exact PrattCertificates.composite_of_product (factor := 7) (cofactor := 520339) rfl (by decide) (by decide)
    exact decide_eq_false composite
  second := false
  second_correct := by
    have composite : ¬ Nat.Prime (corner 954 2) := by change ¬ Nat.Prime 3640465; exact PrattCertificates.composite_of_product (factor := 5) (cofactor := 728093) rfl (by decide) (by decide)
    exact decide_eq_false composite
  third := false
  third_correct := by
    have composite : ¬ Nat.Prime (corner 954 3) := by change ¬ Nat.Prime 3638557; exact PrattCertificates.composite_of_product (factor := 13) (cofactor := 279889) rfl (by decide) (by decide)
    exact decide_eq_false composite
theorem row_count_954 : row 954 = 0 := (row_954.checked).trans rfl
theorem model_count_954 : modelRow 954 = 0 := by decide
theorem modelrow_agrees_954 : modelRow 954 = row 954 := model_count_954.trans row_count_954.symm
def row_955 : RowCertificate 955 where
  first := false
  first_correct := by
    have composite : ¬ Nat.Prime (corner 955 1) := by change ¬ Nat.Prime 3650011; exact PrattCertificates.composite_of_product (factor := 103) (cofactor := 35437) rfl (by decide) (by decide)
    exact decide_eq_false composite
  second := true
  second_correct := by
    have prime : Nat.Prime (corner 955 2) := by change Nat.Prime 3648101; exact ProjectEulerSolutions.P58PrimeNodes.p_3648101.prime
    exact decide_eq_true prime
  third := false
  third_correct := by
    have composite : ¬ Nat.Prime (corner 955 3) := by change ¬ Nat.Prime 3646191; exact PrattCertificates.composite_of_product (factor := 3) (cofactor := 1215397) rfl (by decide) (by decide)
    exact decide_eq_false composite
theorem row_count_955 : row 955 = 1 := (row_955.checked).trans rfl
theorem model_count_955 : modelRow 955 = 1 := by decide
theorem modelrow_agrees_955 : modelRow 955 = row 955 := model_count_955.trans row_count_955.symm
def row_956 : RowCertificate 956 where
  first := false
  first_correct := by
    have composite : ¬ Nat.Prime (corner 956 1) := by change ¬ Nat.Prime 3657657; exact PrattCertificates.composite_of_product (factor := 3) (cofactor := 1219219) rfl (by decide) (by decide)
    exact decide_eq_false composite
  second := false
  second_correct := by
    have composite : ¬ Nat.Prime (corner 956 2) := by change ¬ Nat.Prime 3655745; exact PrattCertificates.composite_of_product (factor := 5) (cofactor := 731149) rfl (by decide) (by decide)
    exact decide_eq_false composite
  third := false
  third_correct := by
    have composite : ¬ Nat.Prime (corner 956 3) := by change ¬ Nat.Prime 3653833; exact PrattCertificates.composite_of_product (factor := 19) (cofactor := 192307) rfl (by decide) (by decide)
    exact decide_eq_false composite
theorem row_count_956 : row 956 = 0 := (row_956.checked).trans rfl
theorem model_count_956 : modelRow 956 = 0 := by decide
theorem modelrow_agrees_956 : modelRow 956 = row 956 := model_count_956.trans row_count_956.symm
def row_957 : RowCertificate 957 where
  first := false
  first_correct := by
    have composite : ¬ Nat.Prime (corner 957 1) := by change ¬ Nat.Prime 3665311; exact PrattCertificates.composite_of_product (factor := 13) (cofactor := 281947) rfl (by decide) (by decide)
    exact decide_eq_false composite
  second := false
  second_correct := by
    have composite : ¬ Nat.Prime (corner 957 2) := by change ¬ Nat.Prime 3663397; exact PrattCertificates.composite_of_product (factor := 281) (cofactor := 13037) rfl (by decide) (by decide)
    exact decide_eq_false composite
  third := false
  third_correct := by
    have composite : ¬ Nat.Prime (corner 957 3) := by change ¬ Nat.Prime 3661483; exact PrattCertificates.composite_of_product (factor := 7) (cofactor := 523069) rfl (by decide) (by decide)
    exact decide_eq_false composite
theorem row_count_957 : row 957 = 0 := (row_957.checked).trans rfl
theorem model_count_957 : modelRow 957 = 0 := by decide
theorem modelrow_agrees_957 : modelRow 957 = row 957 := model_count_957.trans row_count_957.symm
def row_958 : RowCertificate 958 where
  first := false
  first_correct := by
    have composite : ¬ Nat.Prime (corner 958 1) := by change ¬ Nat.Prime 3672973; exact PrattCertificates.composite_of_product (factor := 31) (cofactor := 118483) rfl (by decide) (by decide)
    exact decide_eq_false composite
  second := false
  second_correct := by
    have composite : ¬ Nat.Prime (corner 958 2) := by change ¬ Nat.Prime 3671057; exact PrattCertificates.composite_of_product (factor := 13) (cofactor := 282389) rfl (by decide) (by decide)
    exact decide_eq_false composite
  third := false
  third_correct := by
    have composite : ¬ Nat.Prime (corner 958 3) := by change ¬ Nat.Prime 3669141; exact PrattCertificates.composite_of_product (factor := 3) (cofactor := 1223047) rfl (by decide) (by decide)
    exact decide_eq_false composite
theorem row_count_958 : row 958 = 0 := (row_958.checked).trans rfl
theorem model_count_958 : modelRow 958 = 0 := by decide
theorem modelrow_agrees_958 : modelRow 958 = row 958 := model_count_958.trans row_count_958.symm
def row_959 : RowCertificate 959 where
  first := false
  first_correct := by
    have composite : ¬ Nat.Prime (corner 959 1) := by change ¬ Nat.Prime 3680643; exact PrattCertificates.composite_of_product (factor := 3) (cofactor := 1226881) rfl (by decide) (by decide)
    exact decide_eq_false composite
  second := false
  second_correct := by
    have composite : ¬ Nat.Prime (corner 959 2) := by change ¬ Nat.Prime 3678725; exact PrattCertificates.composite_of_product (factor := 5) (cofactor := 735745) rfl (by decide) (by decide)
    exact decide_eq_false composite
  third := false
  third_correct := by
    have composite : ¬ Nat.Prime (corner 959 3) := by change ¬ Nat.Prime 3676807; exact PrattCertificates.composite_of_product (factor := 1669) (cofactor := 2203) rfl (by decide) (by decide)
    exact decide_eq_false composite
theorem row_count_959 : row 959 = 0 := (row_959.checked).trans rfl
theorem model_count_959 : modelRow 959 = 0 := by decide
theorem modelrow_agrees_959 : modelRow 959 = row 959 := model_count_959.trans row_count_959.symm
def row_960 : RowCertificate 960 where
  first := false
  first_correct := by
    have composite : ¬ Nat.Prime (corner 960 1) := by change ¬ Nat.Prime 3688321; exact PrattCertificates.composite_of_product (factor := 7) (cofactor := 526903) rfl (by decide) (by decide)
    exact decide_eq_false composite
  second := true
  second_correct := by
    have prime : Nat.Prime (corner 960 2) := by change Nat.Prime 3686401; exact ProjectEulerSolutions.P58PrimeNodes.p_3686401.prime
    exact decide_eq_true prime
  third := false
  third_correct := by
    have composite : ¬ Nat.Prime (corner 960 3) := by change ¬ Nat.Prime 3684481; exact PrattCertificates.composite_of_product (factor := 79) (cofactor := 46639) rfl (by decide) (by decide)
    exact decide_eq_false composite
theorem row_count_960 : row 960 = 1 := (row_960.checked).trans rfl
theorem model_count_960 : modelRow 960 = 1 := by decide
theorem modelrow_agrees_960 : modelRow 960 = row 960 := model_count_960.trans row_count_960.symm
def row_961 : RowCertificate 961 where
  first := false
  first_correct := by
    have composite : ¬ Nat.Prime (corner 961 1) := by change ¬ Nat.Prime 3696007; exact PrattCertificates.composite_of_product (factor := 7) (cofactor := 528001) rfl (by decide) (by decide)
    exact decide_eq_false composite
  second := false
  second_correct := by
    have composite : ¬ Nat.Prime (corner 961 2) := by change ¬ Nat.Prime 3694085; exact PrattCertificates.composite_of_product (factor := 5) (cofactor := 738817) rfl (by decide) (by decide)
    exact decide_eq_false composite
  third := false
  third_correct := by
    have composite : ¬ Nat.Prime (corner 961 3) := by change ¬ Nat.Prime 3692163; exact PrattCertificates.composite_of_product (factor := 3) (cofactor := 1230721) rfl (by decide) (by decide)
    exact decide_eq_false composite
theorem row_count_961 : row 961 = 0 := (row_961.checked).trans rfl
theorem model_count_961 : modelRow 961 = 0 := by decide
theorem modelrow_agrees_961 : modelRow 961 = row 961 := model_count_961.trans row_count_961.symm
def row_962 : RowCertificate 962 where
  first := false
  first_correct := by
    have composite : ¬ Nat.Prime (corner 962 1) := by change ¬ Nat.Prime 3703701; exact PrattCertificates.composite_of_product (factor := 3) (cofactor := 1234567) rfl (by decide) (by decide)
    exact decide_eq_false composite
  second := false
  second_correct := by
    have composite : ¬ Nat.Prime (corner 962 2) := by change ¬ Nat.Prime 3701777; exact PrattCertificates.composite_of_product (factor := 89) (cofactor := 41593) rfl (by decide) (by decide)
    exact decide_eq_false composite
  third := true
  third_correct := by
    have prime : Nat.Prime (corner 962 3) := by change Nat.Prime 3699853; exact ProjectEulerSolutions.P58PrimeNodes.p_3699853.prime
    exact decide_eq_true prime
theorem row_count_962 : row 962 = 1 := (row_962.checked).trans rfl
theorem model_count_962 : modelRow 962 = 1 := by decide
theorem modelrow_agrees_962 : modelRow 962 = row 962 := model_count_962.trans row_count_962.symm
def row_963 : RowCertificate 963 where
  first := false
  first_correct := by
    have composite : ¬ Nat.Prime (corner 963 1) := by change ¬ Nat.Prime 3711403; exact PrattCertificates.composite_of_product (factor := 19) (cofactor := 195337) rfl (by decide) (by decide)
    exact decide_eq_false composite
  second := false
  second_correct := by
    have composite : ¬ Nat.Prime (corner 963 2) := by change ¬ Nat.Prime 3709477; exact PrattCertificates.composite_of_product (factor := 29) (cofactor := 127913) rfl (by decide) (by decide)
    exact decide_eq_false composite
  third := false
  third_correct := by
    have composite : ¬ Nat.Prime (corner 963 3) := by change ¬ Nat.Prime 3707551; exact PrattCertificates.composite_of_product (factor := 271) (cofactor := 13681) rfl (by decide) (by decide)
    exact decide_eq_false composite
theorem row_count_963 : row 963 = 0 := (row_963.checked).trans rfl
theorem model_count_963 : modelRow 963 = 0 := by decide
theorem modelrow_agrees_963 : modelRow 963 = row 963 := model_count_963.trans row_count_963.symm
def row_964 : RowCertificate 964 where
  first := false
  first_correct := by
    have composite : ¬ Nat.Prime (corner 964 1) := by change ¬ Nat.Prime 3719113; exact PrattCertificates.composite_of_product (factor := 43) (cofactor := 86491) rfl (by decide) (by decide)
    exact decide_eq_false composite
  second := false
  second_correct := by
    have composite : ¬ Nat.Prime (corner 964 2) := by change ¬ Nat.Prime 3717185; exact PrattCertificates.composite_of_product (factor := 5) (cofactor := 743437) rfl (by decide) (by decide)
    exact decide_eq_false composite
  third := false
  third_correct := by
    have composite : ¬ Nat.Prime (corner 964 3) := by change ¬ Nat.Prime 3715257; exact PrattCertificates.composite_of_product (factor := 3) (cofactor := 1238419) rfl (by decide) (by decide)
    exact decide_eq_false composite
theorem row_count_964 : row 964 = 0 := (row_964.checked).trans rfl
theorem model_count_964 : modelRow 964 = 0 := by decide
theorem modelrow_agrees_964 : modelRow 964 = row 964 := model_count_964.trans row_count_964.symm
def row_965 : RowCertificate 965 where
  first := false
  first_correct := by
    have composite : ¬ Nat.Prime (corner 965 1) := by change ¬ Nat.Prime 3726831; exact PrattCertificates.composite_of_product (factor := 3) (cofactor := 1242277) rfl (by decide) (by decide)
    exact decide_eq_false composite
  second := false
  second_correct := by
    have composite : ¬ Nat.Prime (corner 965 2) := by change ¬ Nat.Prime 3724901; exact PrattCertificates.composite_of_product (factor := 37) (cofactor := 100673) rfl (by decide) (by decide)
    exact decide_eq_false composite
  third := false
  third_correct := by
    have composite : ¬ Nat.Prime (corner 965 3) := by change ¬ Nat.Prime 3722971; exact PrattCertificates.composite_of_product (factor := 7) (cofactor := 531853) rfl (by decide) (by decide)
    exact decide_eq_false composite
theorem row_count_965 : row 965 = 0 := (row_965.checked).trans rfl
theorem model_count_965 : modelRow 965 = 0 := by decide
theorem modelrow_agrees_965 : modelRow 965 = row 965 := model_count_965.trans row_count_965.symm
def row_966 : RowCertificate 966 where
  first := true
  first_correct := by
    have prime : Nat.Prime (corner 966 1) := by change Nat.Prime 3734557; exact ProjectEulerSolutions.P58PrimeNodes.p_3734557.prime
    exact decide_eq_true prime
  second := false
  second_correct := by
    have composite : ¬ Nat.Prime (corner 966 2) := by change ¬ Nat.Prime 3732625; exact PrattCertificates.composite_of_product (factor := 5) (cofactor := 746525) rfl (by decide) (by decide)
    exact decide_eq_false composite
  third := true
  third_correct := by
    have prime : Nat.Prime (corner 966 3) := by change Nat.Prime 3730693; exact ProjectEulerSolutions.P58PrimeNodes.p_3730693.prime
    exact decide_eq_true prime
theorem row_count_966 : row 966 = 2 := (row_966.checked).trans rfl
theorem model_count_966 : modelRow 966 = 2 := by decide
theorem modelrow_agrees_966 : modelRow 966 = row 966 := model_count_966.trans row_count_966.symm
def row_967 : RowCertificate 967 where
  first := false
  first_correct := by
    have composite : ¬ Nat.Prime (corner 967 1) := by change ¬ Nat.Prime 3742291; exact PrattCertificates.composite_of_product (factor := 7) (cofactor := 534613) rfl (by decide) (by decide)
    exact decide_eq_false composite
  second := false
  second_correct := by
    have composite : ¬ Nat.Prime (corner 967 2) := by change ¬ Nat.Prime 3740357; exact PrattCertificates.composite_of_product (factor := 17) (cofactor := 220021) rfl (by decide) (by decide)
    exact decide_eq_false composite
  third := false
  third_correct := by
    have composite : ¬ Nat.Prime (corner 967 3) := by change ¬ Nat.Prime 3738423; exact PrattCertificates.composite_of_product (factor := 3) (cofactor := 1246141) rfl (by decide) (by decide)
    exact decide_eq_false composite
theorem row_count_967 : row 967 = 0 := (row_967.checked).trans rfl
theorem model_count_967 : modelRow 967 = 0 := by decide
theorem modelrow_agrees_967 : modelRow 967 = row 967 := model_count_967.trans row_count_967.symm
def row_968 : RowCertificate 968 where
  first := false
  first_correct := by
    have composite : ¬ Nat.Prime (corner 968 1) := by change ¬ Nat.Prime 3750033; exact PrattCertificates.composite_of_product (factor := 3) (cofactor := 1250011) rfl (by decide) (by decide)
    exact decide_eq_false composite
  second := false
  second_correct := by
    have composite : ¬ Nat.Prime (corner 968 2) := by change ¬ Nat.Prime 3748097; exact PrattCertificates.composite_of_product (factor := 41) (cofactor := 91417) rfl (by decide) (by decide)
    exact decide_eq_false composite
  third := false
  third_correct := by
    have composite : ¬ Nat.Prime (corner 968 3) := by change ¬ Nat.Prime 3746161; exact PrattCertificates.composite_of_product (factor := 1753) (cofactor := 2137) rfl (by decide) (by decide)
    exact decide_eq_false composite
theorem row_count_968 : row 968 = 0 := (row_968.checked).trans rfl
theorem model_count_968 : modelRow 968 = 0 := by decide
theorem modelrow_agrees_968 : modelRow 968 = row 968 := model_count_968.trans row_count_968.symm
def row_969 : RowCertificate 969 where
  first := false
  first_correct := by
    have composite : ¬ Nat.Prime (corner 969 1) := by change ¬ Nat.Prime 3757783; exact PrattCertificates.composite_of_product (factor := 61) (cofactor := 61603) rfl (by decide) (by decide)
    exact decide_eq_false composite
  second := false
  second_correct := by
    have composite : ¬ Nat.Prime (corner 969 2) := by change ¬ Nat.Prime 3755845; exact PrattCertificates.composite_of_product (factor := 5) (cofactor := 751169) rfl (by decide) (by decide)
    exact decide_eq_false composite
  third := false
  third_correct := by
    have composite : ¬ Nat.Prime (corner 969 3) := by change ¬ Nat.Prime 3753907; exact PrattCertificates.composite_of_product (factor := 1039) (cofactor := 3613) rfl (by decide) (by decide)
    exact decide_eq_false composite
theorem row_count_969 : row 969 = 0 := (row_969.checked).trans rfl
theorem model_count_969 : modelRow 969 = 0 := by decide
theorem modelrow_agrees_969 : modelRow 969 = row 969 := model_count_969.trans row_count_969.symm
def row_970 : RowCertificate 970 where
  first := false
  first_correct := by
    have composite : ¬ Nat.Prime (corner 970 1) := by change ¬ Nat.Prime 3765541; exact PrattCertificates.composite_of_product (factor := 13) (cofactor := 289657) rfl (by decide) (by decide)
    exact decide_eq_false composite
  second := true
  second_correct := by
    have prime : Nat.Prime (corner 970 2) := by change Nat.Prime 3763601; exact ProjectEulerSolutions.P58PrimeNodes.p_3763601.prime
    exact decide_eq_true prime
  third := false
  third_correct := by
    have composite : ¬ Nat.Prime (corner 970 3) := by change ¬ Nat.Prime 3761661; exact PrattCertificates.composite_of_product (factor := 3) (cofactor := 1253887) rfl (by decide) (by decide)
    exact decide_eq_false composite
theorem row_count_970 : row 970 = 1 := (row_970.checked).trans rfl
theorem model_count_970 : modelRow 970 = 1 := by decide
theorem modelrow_agrees_970 : modelRow 970 = row 970 := model_count_970.trans row_count_970.symm
def row_971 : RowCertificate 971 where
  first := false
  first_correct := by
    have composite : ¬ Nat.Prime (corner 971 1) := by change ¬ Nat.Prime 3773307; exact PrattCertificates.composite_of_product (factor := 3) (cofactor := 1257769) rfl (by decide) (by decide)
    exact decide_eq_false composite
  second := false
  second_correct := by
    have composite : ¬ Nat.Prime (corner 971 2) := by change ¬ Nat.Prime 3771365; exact PrattCertificates.composite_of_product (factor := 5) (cofactor := 754273) rfl (by decide) (by decide)
    exact decide_eq_false composite
  third := false
  third_correct := by
    have composite : ¬ Nat.Prime (corner 971 3) := by change ¬ Nat.Prime 3769423; exact PrattCertificates.composite_of_product (factor := 7) (cofactor := 538489) rfl (by decide) (by decide)
    exact decide_eq_false composite
theorem row_count_971 : row 971 = 0 := (row_971.checked).trans rfl
theorem model_count_971 : modelRow 971 = 0 := by decide
theorem modelrow_agrees_971 : modelRow 971 = row 971 := model_count_971.trans row_count_971.symm
def row_972 : RowCertificate 972 where
  first := false
  first_correct := by
    have composite : ¬ Nat.Prime (corner 972 1) := by change ¬ Nat.Prime 3781081; exact PrattCertificates.composite_of_product (factor := 577) (cofactor := 6553) rfl (by decide) (by decide)
    exact decide_eq_false composite
  second := false
  second_correct := by
    have composite : ¬ Nat.Prime (corner 972 2) := by change ¬ Nat.Prime 3779137; exact PrattCertificates.composite_of_product (factor := 73) (cofactor := 51769) rfl (by decide) (by decide)
    exact decide_eq_false composite
  third := false
  third_correct := by
    have composite : ¬ Nat.Prime (corner 972 3) := by change ¬ Nat.Prime 3777193; exact PrattCertificates.composite_of_product (factor := 7) (cofactor := 539599) rfl (by decide) (by decide)
    exact decide_eq_false composite
theorem row_count_972 : row 972 = 0 := (row_972.checked).trans rfl
theorem model_count_972 : modelRow 972 = 0 := by decide
theorem modelrow_agrees_972 : modelRow 972 = row 972 := model_count_972.trans row_count_972.symm
def row_973 : RowCertificate 973 where
  first := false
  first_correct := by
    have composite : ¬ Nat.Prime (corner 973 1) := by change ¬ Nat.Prime 3788863; exact PrattCertificates.composite_of_product (factor := 13) (cofactor := 291451) rfl (by decide) (by decide)
    exact decide_eq_false composite
  second := false
  second_correct := by
    have composite : ¬ Nat.Prime (corner 973 2) := by change ¬ Nat.Prime 3786917; exact PrattCertificates.composite_of_product (factor := 1481) (cofactor := 2557) rfl (by decide) (by decide)
    exact decide_eq_false composite
  third := false
  third_correct := by
    have composite : ¬ Nat.Prime (corner 973 3) := by change ¬ Nat.Prime 3784971; exact PrattCertificates.composite_of_product (factor := 3) (cofactor := 1261657) rfl (by decide) (by decide)
    exact decide_eq_false composite
theorem row_count_973 : row 973 = 0 := (row_973.checked).trans rfl
theorem model_count_973 : modelRow 973 = 0 := by decide
theorem modelrow_agrees_973 : modelRow 973 = row 973 := model_count_973.trans row_count_973.symm
def row_974 : RowCertificate 974 where
  first := false
  first_correct := by
    have composite : ¬ Nat.Prime (corner 974 1) := by change ¬ Nat.Prime 3796653; exact PrattCertificates.composite_of_product (factor := 3) (cofactor := 1265551) rfl (by decide) (by decide)
    exact decide_eq_false composite
  second := false
  second_correct := by
    have composite : ¬ Nat.Prime (corner 974 2) := by change ¬ Nat.Prime 3794705; exact PrattCertificates.composite_of_product (factor := 5) (cofactor := 758941) rfl (by decide) (by decide)
    exact decide_eq_false composite
  third := false
  third_correct := by
    have composite : ¬ Nat.Prime (corner 974 3) := by change ¬ Nat.Prime 3792757; exact PrattCertificates.composite_of_product (factor := 31) (cofactor := 122347) rfl (by decide) (by decide)
    exact decide_eq_false composite
theorem row_count_974 : row 974 = 0 := (row_974.checked).trans rfl
theorem model_count_974 : modelRow 974 = 0 := by decide
theorem modelrow_agrees_974 : modelRow 974 = row 974 := model_count_974.trans row_count_974.symm
def row_975 : RowCertificate 975 where
  first := false
  first_correct := by
    have composite : ¬ Nat.Prime (corner 975 1) := by change ¬ Nat.Prime 3804451; exact PrattCertificates.composite_of_product (factor := 7) (cofactor := 543493) rfl (by decide) (by decide)
    exact decide_eq_false composite
  second := false
  second_correct := by
    have composite : ¬ Nat.Prime (corner 975 2) := by change ¬ Nat.Prime 3802501; exact PrattCertificates.composite_of_product (factor := 1453) (cofactor := 2617) rfl (by decide) (by decide)
    exact decide_eq_false composite
  third := false
  third_correct := by
    have composite : ¬ Nat.Prime (corner 975 3) := by change ¬ Nat.Prime 3800551; exact PrattCertificates.composite_of_product (factor := 19) (cofactor := 200029) rfl (by decide) (by decide)
    exact decide_eq_false composite
theorem row_count_975 : row 975 = 0 := (row_975.checked).trans rfl
theorem model_count_975 : modelRow 975 = 0 := by decide
theorem modelrow_agrees_975 : modelRow 975 = row 975 := model_count_975.trans row_count_975.symm
def row_976 : RowCertificate 976 where
  first := true
  first_correct := by
    have prime : Nat.Prime (corner 976 1) := by change Nat.Prime 3812257; exact ProjectEulerSolutions.P58PrimeNodes.p_3812257.prime
    exact decide_eq_true prime
  second := false
  second_correct := by
    have composite : ¬ Nat.Prime (corner 976 2) := by change ¬ Nat.Prime 3810305; exact PrattCertificates.composite_of_product (factor := 5) (cofactor := 762061) rfl (by decide) (by decide)
    exact decide_eq_false composite
  third := false
  third_correct := by
    have composite : ¬ Nat.Prime (corner 976 3) := by change ¬ Nat.Prime 3808353; exact PrattCertificates.composite_of_product (factor := 3) (cofactor := 1269451) rfl (by decide) (by decide)
    exact decide_eq_false composite
theorem row_count_976 : row 976 = 1 := (row_976.checked).trans rfl
theorem model_count_976 : modelRow 976 = 1 := by decide
theorem modelrow_agrees_976 : modelRow 976 = row 976 := model_count_976.trans row_count_976.symm
def row_977 : RowCertificate 977 where
  first := false
  first_correct := by
    have composite : ¬ Nat.Prime (corner 977 1) := by change ¬ Nat.Prime 3820071; exact PrattCertificates.composite_of_product (factor := 3) (cofactor := 1273357) rfl (by decide) (by decide)
    exact decide_eq_false composite
  second := false
  second_correct := by
    have composite : ¬ Nat.Prime (corner 977 2) := by change ¬ Nat.Prime 3818117; exact PrattCertificates.composite_of_product (factor := 229) (cofactor := 16673) rfl (by decide) (by decide)
    exact decide_eq_false composite
  third := false
  third_correct := by
    have composite : ¬ Nat.Prime (corner 977 3) := by change ¬ Nat.Prime 3816163; exact PrattCertificates.composite_of_product (factor := 13) (cofactor := 293551) rfl (by decide) (by decide)
    exact decide_eq_false composite
theorem row_count_977 : row 977 = 0 := (row_977.checked).trans rfl
theorem model_count_977 : modelRow 977 = 0 := by decide
theorem modelrow_agrees_977 : modelRow 977 = row 977 := model_count_977.trans row_count_977.symm
def row_978 : RowCertificate 978 where
  first := true
  first_correct := by
    have prime : Nat.Prime (corner 978 1) := by change Nat.Prime 3827893; exact ProjectEulerSolutions.P58PrimeNodes.p_3827893.prime
    exact decide_eq_true prime
  second := false
  second_correct := by
    have composite : ¬ Nat.Prime (corner 978 2) := by change ¬ Nat.Prime 3825937; exact PrattCertificates.composite_of_product (factor := 197) (cofactor := 19421) rfl (by decide) (by decide)
    exact decide_eq_false composite
  third := false
  third_correct := by
    have composite : ¬ Nat.Prime (corner 978 3) := by change ¬ Nat.Prime 3823981; exact PrattCertificates.composite_of_product (factor := 7) (cofactor := 546283) rfl (by decide) (by decide)
    exact decide_eq_false composite
theorem row_count_978 : row 978 = 1 := (row_978.checked).trans rfl
theorem model_count_978 : modelRow 978 = 1 := by decide
theorem modelrow_agrees_978 : modelRow 978 = row 978 := model_count_978.trans row_count_978.symm
def row_979 : RowCertificate 979 where
  first := false
  first_correct := by
    have composite : ¬ Nat.Prime (corner 979 1) := by change ¬ Nat.Prime 3835723; exact PrattCertificates.composite_of_product (factor := 31) (cofactor := 123733) rfl (by decide) (by decide)
    exact decide_eq_false composite
  second := false
  second_correct := by
    have composite : ¬ Nat.Prime (corner 979 2) := by change ¬ Nat.Prime 3833765; exact PrattCertificates.composite_of_product (factor := 5) (cofactor := 766753) rfl (by decide) (by decide)
    exact decide_eq_false composite
  third := false
  third_correct := by
    have composite : ¬ Nat.Prime (corner 979 3) := by change ¬ Nat.Prime 3831807; exact PrattCertificates.composite_of_product (factor := 3) (cofactor := 1277269) rfl (by decide) (by decide)
    exact decide_eq_false composite
theorem row_count_979 : row 979 = 0 := (row_979.checked).trans rfl
theorem model_count_979 : modelRow 979 = 0 := by decide
theorem modelrow_agrees_979 : modelRow 979 = row 979 := model_count_979.trans row_count_979.symm
def row_980 : RowCertificate 980 where
  first := false
  first_correct := by
    have composite : ¬ Nat.Prime (corner 980 1) := by change ¬ Nat.Prime 3843561; exact PrattCertificates.composite_of_product (factor := 3) (cofactor := 1281187) rfl (by decide) (by decide)
    exact decide_eq_false composite
  second := false
  second_correct := by
    have composite : ¬ Nat.Prime (corner 980 2) := by change ¬ Nat.Prime 3841601; exact PrattCertificates.composite_of_product (factor := 29) (cofactor := 132469) rfl (by decide) (by decide)
    exact decide_eq_false composite
  third := false
  third_correct := by
    have composite : ¬ Nat.Prime (corner 980 3) := by change ¬ Nat.Prime 3839641; exact PrattCertificates.composite_of_product (factor := 13) (cofactor := 295357) rfl (by decide) (by decide)
    exact decide_eq_false composite
theorem row_count_980 : row 980 = 0 := (row_980.checked).trans rfl
theorem model_count_980 : modelRow 980 = 0 := by decide
theorem modelrow_agrees_980 : modelRow 980 = row 980 := model_count_980.trans row_count_980.symm
def row_981 : RowCertificate 981 where
  first := false
  first_correct := by
    have composite : ¬ Nat.Prime (corner 981 1) := by change ¬ Nat.Prime 3851407; exact PrattCertificates.composite_of_product (factor := 7) (cofactor := 550201) rfl (by decide) (by decide)
    exact decide_eq_false composite
  second := false
  second_correct := by
    have composite : ¬ Nat.Prime (corner 981 2) := by change ¬ Nat.Prime 3849445; exact PrattCertificates.composite_of_product (factor := 5) (cofactor := 769889) rfl (by decide) (by decide)
    exact decide_eq_false composite
  third := false
  third_correct := by
    have composite : ¬ Nat.Prime (corner 981 3) := by change ¬ Nat.Prime 3847483; exact PrattCertificates.composite_of_product (factor := 457) (cofactor := 8419) rfl (by decide) (by decide)
    exact decide_eq_false composite
theorem row_count_981 : row 981 = 0 := (row_981.checked).trans rfl
theorem model_count_981 : modelRow 981 = 0 := by decide
theorem modelrow_agrees_981 : modelRow 981 = row 981 := model_count_981.trans row_count_981.symm
def row_982 : RowCertificate 982 where
  first := false
  first_correct := by
    have composite : ¬ Nat.Prime (corner 982 1) := by change ¬ Nat.Prime 3859261; exact PrattCertificates.composite_of_product (factor := 7) (cofactor := 551323) rfl (by decide) (by decide)
    exact decide_eq_false composite
  second := true
  second_correct := by
    have prime : Nat.Prime (corner 982 2) := by change Nat.Prime 3857297; exact ProjectEulerSolutions.P58PrimeNodes.p_3857297.prime
    exact decide_eq_true prime
  third := false
  third_correct := by
    have composite : ¬ Nat.Prime (corner 982 3) := by change ¬ Nat.Prime 3855333; exact PrattCertificates.composite_of_product (factor := 3) (cofactor := 1285111) rfl (by decide) (by decide)
    exact decide_eq_false composite
theorem row_count_982 : row 982 = 1 := (row_982.checked).trans rfl
theorem model_count_982 : modelRow 982 = 1 := by decide
theorem modelrow_agrees_982 : modelRow 982 = row 982 := model_count_982.trans row_count_982.symm
def row_983 : RowCertificate 983 where
  first := false
  first_correct := by
    have composite : ¬ Nat.Prime (corner 983 1) := by change ¬ Nat.Prime 3867123; exact PrattCertificates.composite_of_product (factor := 3) (cofactor := 1289041) rfl (by decide) (by decide)
    exact decide_eq_false composite
  second := true
  second_correct := by
    have prime : Nat.Prime (corner 983 2) := by change Nat.Prime 3865157; exact ProjectEulerSolutions.P58PrimeNodes.p_3865157.prime
    exact decide_eq_true prime
  third := false
  third_correct := by
    have composite : ¬ Nat.Prime (corner 983 3) := by change ¬ Nat.Prime 3863191; exact PrattCertificates.composite_of_product (factor := 61) (cofactor := 63331) rfl (by decide) (by decide)
    exact decide_eq_false composite
theorem row_count_983 : row 983 = 1 := (row_983.checked).trans rfl
theorem model_count_983 : modelRow 983 = 1 := by decide
theorem modelrow_agrees_983 : modelRow 983 = row 983 := model_count_983.trans row_count_983.symm
def row_984 : RowCertificate 984 where
  first := false
  first_correct := by
    have composite : ¬ Nat.Prime (corner 984 1) := by change ¬ Nat.Prime 3874993; exact PrattCertificates.composite_of_product (factor := 19) (cofactor := 203947) rfl (by decide) (by decide)
    exact decide_eq_false composite
  second := false
  second_correct := by
    have composite : ¬ Nat.Prime (corner 984 2) := by change ¬ Nat.Prime 3873025; exact PrattCertificates.composite_of_product (factor := 5) (cofactor := 774605) rfl (by decide) (by decide)
    exact decide_eq_false composite
  third := false
  third_correct := by
    have composite : ¬ Nat.Prime (corner 984 3) := by change ¬ Nat.Prime 3871057; exact PrattCertificates.composite_of_product (factor := 223) (cofactor := 17359) rfl (by decide) (by decide)
    exact decide_eq_false composite
theorem row_count_984 : row 984 = 0 := (row_984.checked).trans rfl
theorem model_count_984 : modelRow 984 = 0 := by decide
theorem modelrow_agrees_984 : modelRow 984 = row 984 := model_count_984.trans row_count_984.symm
def row_985 : RowCertificate 985 where
  first := true
  first_correct := by
    have prime : Nat.Prime (corner 985 1) := by change Nat.Prime 3882871; exact ProjectEulerSolutions.P58PrimeNodes.p_3882871.prime
    exact decide_eq_true prime
  second := true
  second_correct := by
    have prime : Nat.Prime (corner 985 2) := by change Nat.Prime 3880901; exact ProjectEulerSolutions.P58PrimeNodes.p_3880901.prime
    exact decide_eq_true prime
  third := false
  third_correct := by
    have composite : ¬ Nat.Prime (corner 985 3) := by change ¬ Nat.Prime 3878931; exact PrattCertificates.composite_of_product (factor := 3) (cofactor := 1292977) rfl (by decide) (by decide)
    exact decide_eq_false composite
theorem row_count_985 : row 985 = 2 := (row_985.checked).trans rfl
theorem model_count_985 : modelRow 985 = 2 := by decide
theorem modelrow_agrees_985 : modelRow 985 = row 985 := model_count_985.trans row_count_985.symm
def row_986 : RowCertificate 986 where
  first := false
  first_correct := by
    have composite : ¬ Nat.Prime (corner 986 1) := by change ¬ Nat.Prime 3890757; exact PrattCertificates.composite_of_product (factor := 3) (cofactor := 1296919) rfl (by decide) (by decide)
    exact decide_eq_false composite
  second := false
  second_correct := by
    have composite : ¬ Nat.Prime (corner 986 2) := by change ¬ Nat.Prime 3888785; exact PrattCertificates.composite_of_product (factor := 5) (cofactor := 777757) rfl (by decide) (by decide)
    exact decide_eq_false composite
  third := false
  third_correct := by
    have composite : ¬ Nat.Prime (corner 986 3) := by change ¬ Nat.Prime 3886813; exact PrattCertificates.composite_of_product (factor := 7) (cofactor := 555259) rfl (by decide) (by decide)
    exact decide_eq_false composite
theorem row_count_986 : row 986 = 0 := (row_986.checked).trans rfl
theorem model_count_986 : modelRow 986 = 0 := by decide
theorem modelrow_agrees_986 : modelRow 986 = row 986 := model_count_986.trans row_count_986.symm
def row_987 : RowCertificate 987 where
  first := true
  first_correct := by
    have prime : Nat.Prime (corner 987 1) := by change Nat.Prime 3898651; exact ProjectEulerSolutions.P58PrimeNodes.p_3898651.prime
    exact decide_eq_true prime
  second := true
  second_correct := by
    have prime : Nat.Prime (corner 987 2) := by change Nat.Prime 3896677; exact ProjectEulerSolutions.P58PrimeNodes.p_3896677.prime
    exact decide_eq_true prime
  third := true
  third_correct := by
    have prime : Nat.Prime (corner 987 3) := by change Nat.Prime 3894703; exact ProjectEulerSolutions.P58PrimeNodes.p_3894703.prime
    exact decide_eq_true prime
theorem row_count_987 : row 987 = 3 := (row_987.checked).trans rfl
theorem model_count_987 : modelRow 987 = 3 := by decide
theorem modelrow_agrees_987 : modelRow 987 = row 987 := model_count_987.trans row_count_987.symm
def row_988 : RowCertificate 988 where
  first := false
  first_correct := by
    have composite : ¬ Nat.Prime (corner 988 1) := by change ¬ Nat.Prime 3906553; exact PrattCertificates.composite_of_product (factor := 7) (cofactor := 558079) rfl (by decide) (by decide)
    exact decide_eq_false composite
  second := false
  second_correct := by
    have composite : ¬ Nat.Prime (corner 988 2) := by change ¬ Nat.Prime 3904577; exact PrattCertificates.composite_of_product (factor := 17) (cofactor := 229681) rfl (by decide) (by decide)
    exact decide_eq_false composite
  third := false
  third_correct := by
    have composite : ¬ Nat.Prime (corner 988 3) := by change ¬ Nat.Prime 3902601; exact PrattCertificates.composite_of_product (factor := 3) (cofactor := 1300867) rfl (by decide) (by decide)
    exact decide_eq_false composite
theorem row_count_988 : row 988 = 0 := (row_988.checked).trans rfl
theorem model_count_988 : modelRow 988 = 0 := by decide
theorem modelrow_agrees_988 : modelRow 988 = row 988 := model_count_988.trans row_count_988.symm
def row_989 : RowCertificate 989 where
  first := false
  first_correct := by
    have composite : ¬ Nat.Prime (corner 989 1) := by change ¬ Nat.Prime 3914463; exact PrattCertificates.composite_of_product (factor := 3) (cofactor := 1304821) rfl (by decide) (by decide)
    exact decide_eq_false composite
  second := false
  second_correct := by
    have composite : ¬ Nat.Prime (corner 989 2) := by change ¬ Nat.Prime 3912485; exact PrattCertificates.composite_of_product (factor := 5) (cofactor := 782497) rfl (by decide) (by decide)
    exact decide_eq_false composite
  third := true
  third_correct := by
    have prime : Nat.Prime (corner 989 3) := by change Nat.Prime 3910507; exact ProjectEulerSolutions.P58PrimeNodes.p_3910507.prime
    exact decide_eq_true prime
theorem row_count_989 : row 989 = 1 := (row_989.checked).trans rfl
theorem model_count_989 : modelRow 989 = 1 := by decide
theorem modelrow_agrees_989 : modelRow 989 = row 989 := model_count_989.trans row_count_989.symm
def row_990 : RowCertificate 990 where
  first := false
  first_correct := by
    have composite : ¬ Nat.Prime (corner 990 1) := by change ¬ Nat.Prime 3922381; exact PrattCertificates.composite_of_product (factor := 67) (cofactor := 58543) rfl (by decide) (by decide)
    exact decide_eq_false composite
  second := true
  second_correct := by
    have prime : Nat.Prime (corner 990 2) := by change Nat.Prime 3920401; exact ProjectEulerSolutions.P58PrimeNodes.p_3920401.prime
    exact decide_eq_true prime
  third := false
  third_correct := by
    have composite : ¬ Nat.Prime (corner 990 3) := by change ¬ Nat.Prime 3918421; exact PrattCertificates.composite_of_product (factor := 13) (cofactor := 301417) rfl (by decide) (by decide)
    exact decide_eq_false composite
theorem row_count_990 : row 990 = 1 := (row_990.checked).trans rfl
theorem model_count_990 : modelRow 990 = 1 := by decide
theorem modelrow_agrees_990 : modelRow 990 = row 990 := model_count_990.trans row_count_990.symm
def row_991 : RowCertificate 991 where
  first := true
  first_correct := by
    have prime : Nat.Prime (corner 991 1) := by change Nat.Prime 3930307; exact ProjectEulerSolutions.P58PrimeNodes.p_3930307.prime
    exact decide_eq_true prime
  second := false
  second_correct := by
    have composite : ¬ Nat.Prime (corner 991 2) := by change ¬ Nat.Prime 3928325; exact PrattCertificates.composite_of_product (factor := 5) (cofactor := 785665) rfl (by decide) (by decide)
    exact decide_eq_false composite
  third := false
  third_correct := by
    have composite : ¬ Nat.Prime (corner 991 3) := by change ¬ Nat.Prime 3926343; exact PrattCertificates.composite_of_product (factor := 3) (cofactor := 1308781) rfl (by decide) (by decide)
    exact decide_eq_false composite
theorem row_count_991 : row 991 = 1 := (row_991.checked).trans rfl
theorem model_count_991 : modelRow 991 = 1 := by decide
theorem modelrow_agrees_991 : modelRow 991 = row 991 := model_count_991.trans row_count_991.symm
def row_992 : RowCertificate 992 where
  first := false
  first_correct := by
    have composite : ¬ Nat.Prime (corner 992 1) := by change ¬ Nat.Prime 3938241; exact PrattCertificates.composite_of_product (factor := 3) (cofactor := 1312747) rfl (by decide) (by decide)
    exact decide_eq_false composite
  second := false
  second_correct := by
    have composite : ¬ Nat.Prime (corner 992 2) := by change ¬ Nat.Prime 3936257; exact PrattCertificates.composite_of_product (factor := 13) (cofactor := 302789) rfl (by decide) (by decide)
    exact decide_eq_false composite
  third := false
  third_correct := by
    have composite : ¬ Nat.Prime (corner 992 3) := by change ¬ Nat.Prime 3934273; exact PrattCertificates.composite_of_product (factor := 7) (cofactor := 562039) rfl (by decide) (by decide)
    exact decide_eq_false composite
theorem row_count_992 : row 992 = 0 := (row_992.checked).trans rfl
theorem model_count_992 : modelRow 992 = 0 := by decide
theorem modelrow_agrees_992 : modelRow 992 = row 992 := model_count_992.trans row_count_992.symm
def row_993 : RowCertificate 993 where
  first := true
  first_correct := by
    have prime : Nat.Prime (corner 993 1) := by change Nat.Prime 3946183; exact ProjectEulerSolutions.P58PrimeNodes.p_3946183.prime
    exact decide_eq_true prime
  second := false
  second_correct := by
    have composite : ¬ Nat.Prime (corner 993 2) := by change ¬ Nat.Prime 3944197; exact PrattCertificates.composite_of_product (factor := 433) (cofactor := 9109) rfl (by decide) (by decide)
    exact decide_eq_false composite
  third := false
  third_correct := by
    have composite : ¬ Nat.Prime (corner 993 3) := by change ¬ Nat.Prime 3942211; exact PrattCertificates.composite_of_product (factor := 7) (cofactor := 563173) rfl (by decide) (by decide)
    exact decide_eq_false composite
theorem row_count_993 : row 993 = 1 := (row_993.checked).trans rfl
theorem model_count_993 : modelRow 993 = 1 := by decide
theorem modelrow_agrees_993 : modelRow 993 = row 993 := model_count_993.trans row_count_993.symm
def row_994 : RowCertificate 994 where
  first := false
  first_correct := by
    have composite : ¬ Nat.Prime (corner 994 1) := by change ¬ Nat.Prime 3954133; exact PrattCertificates.composite_of_product (factor := 139) (cofactor := 28447) rfl (by decide) (by decide)
    exact decide_eq_false composite
  second := false
  second_correct := by
    have composite : ¬ Nat.Prime (corner 994 2) := by change ¬ Nat.Prime 3952145; exact PrattCertificates.composite_of_product (factor := 5) (cofactor := 790429) rfl (by decide) (by decide)
    exact decide_eq_false composite
  third := false
  third_correct := by
    have composite : ¬ Nat.Prime (corner 994 3) := by change ¬ Nat.Prime 3950157; exact PrattCertificates.composite_of_product (factor := 3) (cofactor := 1316719) rfl (by decide) (by decide)
    exact decide_eq_false composite
theorem row_count_994 : row 994 = 0 := (row_994.checked).trans rfl
theorem model_count_994 : modelRow 994 = 0 := by decide
theorem modelrow_agrees_994 : modelRow 994 = row 994 := model_count_994.trans row_count_994.symm
def row_995 : RowCertificate 995 where
  first := false
  first_correct := by
    have composite : ¬ Nat.Prime (corner 995 1) := by change ¬ Nat.Prime 3962091; exact PrattCertificates.composite_of_product (factor := 3) (cofactor := 1320697) rfl (by decide) (by decide)
    exact decide_eq_false composite
  second := true
  second_correct := by
    have prime : Nat.Prime (corner 995 2) := by change Nat.Prime 3960101; exact ProjectEulerSolutions.P58PrimeNodes.p_3960101.prime
    exact decide_eq_true prime
  third := false
  third_correct := by
    have composite : ¬ Nat.Prime (corner 995 3) := by change ¬ Nat.Prime 3958111; exact PrattCertificates.composite_of_product (factor := 31) (cofactor := 127681) rfl (by decide) (by decide)
    exact decide_eq_false composite
theorem row_count_995 : row 995 = 1 := (row_995.checked).trans rfl
theorem model_count_995 : modelRow 995 = 1 := by decide
theorem modelrow_agrees_995 : modelRow 995 = row 995 := model_count_995.trans row_count_995.symm
def row_996 : RowCertificate 996 where
  first := false
  first_correct := by
    have composite : ¬ Nat.Prime (corner 996 1) := by change ¬ Nat.Prime 3970057; exact PrattCertificates.composite_of_product (factor := 7) (cofactor := 567151) rfl (by decide) (by decide)
    exact decide_eq_false composite
  second := false
  second_correct := by
    have composite : ¬ Nat.Prime (corner 996 2) := by change ¬ Nat.Prime 3968065; exact PrattCertificates.composite_of_product (factor := 5) (cofactor := 793613) rfl (by decide) (by decide)
    exact decide_eq_false composite
  third := false
  third_correct := by
    have composite : ¬ Nat.Prime (corner 996 3) := by change ¬ Nat.Prime 3966073; exact PrattCertificates.composite_of_product (factor := 409) (cofactor := 9697) rfl (by decide) (by decide)
    exact decide_eq_false composite
theorem row_count_996 : row 996 = 0 := (row_996.checked).trans rfl
theorem model_count_996 : modelRow 996 = 0 := by decide
theorem modelrow_agrees_996 : modelRow 996 = row 996 := model_count_996.trans row_count_996.symm
def row_997 : RowCertificate 997 where
  first := true
  first_correct := by
    have prime : Nat.Prime (corner 997 1) := by change Nat.Prime 3978031; exact ProjectEulerSolutions.P58PrimeNodes.p_3978031.prime
    exact decide_eq_true prime
  second := false
  second_correct := by
    have composite : ¬ Nat.Prime (corner 997 2) := by change ¬ Nat.Prime 3976037; exact PrattCertificates.composite_of_product (factor := 13) (cofactor := 305849) rfl (by decide) (by decide)
    exact decide_eq_false composite
  third := false
  third_correct := by
    have composite : ¬ Nat.Prime (corner 997 3) := by change ¬ Nat.Prime 3974043; exact PrattCertificates.composite_of_product (factor := 3) (cofactor := 1324681) rfl (by decide) (by decide)
    exact decide_eq_false composite
theorem row_count_997 : row 997 = 1 := (row_997.checked).trans rfl
theorem model_count_997 : modelRow 997 = 1 := by decide
theorem modelrow_agrees_997 : modelRow 997 = row 997 := model_count_997.trans row_count_997.symm
def row_998 : RowCertificate 998 where
  first := false
  first_correct := by
    have composite : ¬ Nat.Prime (corner 998 1) := by change ¬ Nat.Prime 3986013; exact PrattCertificates.composite_of_product (factor := 3) (cofactor := 1328671) rfl (by decide) (by decide)
    exact decide_eq_false composite
  second := false
  second_correct := by
    have composite : ¬ Nat.Prime (corner 998 2) := by change ¬ Nat.Prime 3984017; exact PrattCertificates.composite_of_product (factor := 173) (cofactor := 23029) rfl (by decide) (by decide)
    exact decide_eq_false composite
  third := false
  third_correct := by
    have composite : ¬ Nat.Prime (corner 998 3) := by change ¬ Nat.Prime 3982021; exact PrattCertificates.composite_of_product (factor := 151) (cofactor := 26371) rfl (by decide) (by decide)
    exact decide_eq_false composite
theorem row_count_998 : row 998 = 0 := (row_998.checked).trans rfl
theorem model_count_998 : modelRow 998 = 0 := by decide
theorem modelrow_agrees_998 : modelRow 998 = row 998 := model_count_998.trans row_count_998.symm
def row_999 : RowCertificate 999 where
  first := false
  first_correct := by
    have composite : ¬ Nat.Prime (corner 999 1) := by change ¬ Nat.Prime 3994003; exact PrattCertificates.composite_of_product (factor := 13) (cofactor := 307231) rfl (by decide) (by decide)
    exact decide_eq_false composite
  second := false
  second_correct := by
    have composite : ¬ Nat.Prime (corner 999 2) := by change ¬ Nat.Prime 3992005; exact PrattCertificates.composite_of_product (factor := 5) (cofactor := 798401) rfl (by decide) (by decide)
    exact decide_eq_false composite
  third := false
  third_correct := by
    have composite : ¬ Nat.Prime (corner 999 3) := by change ¬ Nat.Prime 3990007; exact PrattCertificates.composite_of_product (factor := 7) (cofactor := 570001) rfl (by decide) (by decide)
    exact decide_eq_false composite
theorem row_count_999 : row 999 = 0 := (row_999.checked).trans rfl
theorem model_count_999 : modelRow 999 = 0 := by decide
theorem modelrow_agrees_999 : modelRow 999 = row 999 := model_count_999.trans row_count_999.symm
def row_1000 : RowCertificate 1000 where
  first := true
  first_correct := by
    have prime : Nat.Prime (corner 1000 1) := by change Nat.Prime 4002001; exact ProjectEulerSolutions.P58PrimeNodes.p_4002001.prime
    exact decide_eq_true prime
  second := false
  second_correct := by
    have composite : ¬ Nat.Prime (corner 1000 2) := by change ¬ Nat.Prime 4000001; exact PrattCertificates.composite_of_product (factor := 41) (cofactor := 97561) rfl (by decide) (by decide)
    exact decide_eq_false composite
  third := false
  third_correct := by
    have composite : ¬ Nat.Prime (corner 1000 3) := by change ¬ Nat.Prime 3998001; exact PrattCertificates.composite_of_product (factor := 3) (cofactor := 1332667) rfl (by decide) (by decide)
    exact decide_eq_false composite
theorem row_count_1000 : row 1000 = 1 := (row_1000.checked).trans rfl
theorem model_count_1000 : modelRow 1000 = 1 := by decide
theorem modelrow_agrees_1000 : modelRow 1000 = row 1000 := model_count_1000.trans row_count_1000.symm
def row_1001 : RowCertificate 1001 where
  first := false
  first_correct := by
    have composite : ¬ Nat.Prime (corner 1001 1) := by change ¬ Nat.Prime 4010007; exact PrattCertificates.composite_of_product (factor := 3) (cofactor := 1336669) rfl (by decide) (by decide)
    exact decide_eq_false composite
  second := false
  second_correct := by
    have composite : ¬ Nat.Prime (corner 1001 2) := by change ¬ Nat.Prime 4008005; exact PrattCertificates.composite_of_product (factor := 5) (cofactor := 801601) rfl (by decide) (by decide)
    exact decide_eq_false composite
  third := false
  third_correct := by
    have composite : ¬ Nat.Prime (corner 1001 3) := by change ¬ Nat.Prime 4006003; exact PrattCertificates.composite_of_product (factor := 97) (cofactor := 41299) rfl (by decide) (by decide)
    exact decide_eq_false composite
theorem row_count_1001 : row 1001 = 0 := (row_1001.checked).trans rfl
theorem model_count_1001 : modelRow 1001 = 0 := by decide
theorem modelrow_agrees_1001 : modelRow 1001 = row 1001 := model_count_1001.trans row_count_1001.symm
def row_1002 : RowCertificate 1002 where
  first := false
  first_correct := by
    have composite : ¬ Nat.Prime (corner 1002 1) := by change ¬ Nat.Prime 4018021; exact PrattCertificates.composite_of_product (factor := 7) (cofactor := 574003) rfl (by decide) (by decide)
    exact decide_eq_false composite
  second := false
  second_correct := by
    have composite : ¬ Nat.Prime (corner 1002 2) := by change ¬ Nat.Prime 4016017; exact PrattCertificates.composite_of_product (factor := 37) (cofactor := 108541) rfl (by decide) (by decide)
    exact decide_eq_false composite
  third := false
  third_correct := by
    have composite : ¬ Nat.Prime (corner 1002 3) := by change ¬ Nat.Prime 4014013; exact PrattCertificates.composite_of_product (factor := 103) (cofactor := 38971) rfl (by decide) (by decide)
    exact decide_eq_false composite
theorem row_count_1002 : row 1002 = 0 := (row_1002.checked).trans rfl
theorem model_count_1002 : modelRow 1002 = 0 := by decide
theorem modelrow_agrees_1002 : modelRow 1002 = row 1002 := model_count_1002.trans row_count_1002.symm
def row_1003 : RowCertificate 1003 where
  first := false
  first_correct := by
    have composite : ¬ Nat.Prime (corner 1003 1) := by change ¬ Nat.Prime 4026043; exact PrattCertificates.composite_of_product (factor := 7) (cofactor := 575149) rfl (by decide) (by decide)
    exact decide_eq_false composite
  second := true
  second_correct := by
    have prime : Nat.Prime (corner 1003 2) := by change Nat.Prime 4024037; exact ProjectEulerSolutions.P58PrimeNodes.p_4024037.prime
    exact decide_eq_true prime
  third := false
  third_correct := by
    have composite : ¬ Nat.Prime (corner 1003 3) := by change ¬ Nat.Prime 4022031; exact PrattCertificates.composite_of_product (factor := 3) (cofactor := 1340677) rfl (by decide) (by decide)
    exact decide_eq_false composite
theorem row_count_1003 : row 1003 = 1 := (row_1003.checked).trans rfl
theorem model_count_1003 : modelRow 1003 = 1 := by decide
theorem modelrow_agrees_1003 : modelRow 1003 = row 1003 := model_count_1003.trans row_count_1003.symm
def row_1004 : RowCertificate 1004 where
  first := false
  first_correct := by
    have composite : ¬ Nat.Prime (corner 1004 1) := by change ¬ Nat.Prime 4034073; exact PrattCertificates.composite_of_product (factor := 3) (cofactor := 1344691) rfl (by decide) (by decide)
    exact decide_eq_false composite
  second := false
  second_correct := by
    have composite : ¬ Nat.Prime (corner 1004 2) := by change ¬ Nat.Prime 4032065; exact PrattCertificates.composite_of_product (factor := 5) (cofactor := 806413) rfl (by decide) (by decide)
    exact decide_eq_false composite
  third := false
  third_correct := by
    have composite : ¬ Nat.Prime (corner 1004 3) := by change ¬ Nat.Prime 4030057; exact PrattCertificates.composite_of_product (factor := 109) (cofactor := 36973) rfl (by decide) (by decide)
    exact decide_eq_false composite
theorem row_count_1004 : row 1004 = 0 := (row_1004.checked).trans rfl
theorem model_count_1004 : modelRow 1004 = 0 := by decide
theorem modelrow_agrees_1004 : modelRow 1004 = row 1004 := model_count_1004.trans row_count_1004.symm
def row_1005 : RowCertificate 1005 where
  first := true
  first_correct := by
    have prime : Nat.Prime (corner 1005 1) := by change Nat.Prime 4042111; exact ProjectEulerSolutions.P58PrimeNodes.p_4042111.prime
    exact decide_eq_true prime
  second := false
  second_correct := by
    have composite : ¬ Nat.Prime (corner 1005 2) := by change ¬ Nat.Prime 4040101; exact PrattCertificates.composite_of_product (factor := 13) (cofactor := 310777) rfl (by decide) (by decide)
    exact decide_eq_false composite
  third := false
  third_correct := by
    have composite : ¬ Nat.Prime (corner 1005 3) := by change ¬ Nat.Prime 4038091; exact PrattCertificates.composite_of_product (factor := 31) (cofactor := 130261) rfl (by decide) (by decide)
    exact decide_eq_false composite
theorem row_count_1005 : row 1005 = 1 := (row_1005.checked).trans rfl
theorem model_count_1005 : modelRow 1005 = 1 := by decide
theorem modelrow_agrees_1005 : modelRow 1005 = row 1005 := model_count_1005.trans row_count_1005.symm
def row_1006 : RowCertificate 1006 where
  first := false
  first_correct := by
    have composite : ¬ Nat.Prime (corner 1006 1) := by change ¬ Nat.Prime 4050157; exact PrattCertificates.composite_of_product (factor := 127) (cofactor := 31891) rfl (by decide) (by decide)
    exact decide_eq_false composite
  second := false
  second_correct := by
    have composite : ¬ Nat.Prime (corner 1006 2) := by change ¬ Nat.Prime 4048145; exact PrattCertificates.composite_of_product (factor := 5) (cofactor := 809629) rfl (by decide) (by decide)
    exact decide_eq_false composite
  third := false
  third_correct := by
    have composite : ¬ Nat.Prime (corner 1006 3) := by change ¬ Nat.Prime 4046133; exact PrattCertificates.composite_of_product (factor := 3) (cofactor := 1348711) rfl (by decide) (by decide)
    exact decide_eq_false composite
theorem row_count_1006 : row 1006 = 0 := (row_1006.checked).trans rfl
theorem model_count_1006 : modelRow 1006 = 0 := by decide
theorem modelrow_agrees_1006 : modelRow 1006 = row 1006 := model_count_1006.trans row_count_1006.symm
def row_1007 : RowCertificate 1007 where
  first := false
  first_correct := by
    have composite : ¬ Nat.Prime (corner 1007 1) := by change ¬ Nat.Prime 4058211; exact PrattCertificates.composite_of_product (factor := 3) (cofactor := 1352737) rfl (by decide) (by decide)
    exact decide_eq_false composite
  second := false
  second_correct := by
    have composite : ¬ Nat.Prime (corner 1007 2) := by change ¬ Nat.Prime 4056197; exact PrattCertificates.composite_of_product (factor := 1877) (cofactor := 2161) rfl (by decide) (by decide)
    exact decide_eq_false composite
  third := false
  third_correct := by
    have composite : ¬ Nat.Prime (corner 1007 3) := by change ¬ Nat.Prime 4054183; exact PrattCertificates.composite_of_product (factor := 7) (cofactor := 579169) rfl (by decide) (by decide)
    exact decide_eq_false composite
theorem row_count_1007 : row 1007 = 0 := (row_1007.checked).trans rfl
theorem model_count_1007 : modelRow 1007 = 0 := by decide
theorem modelrow_agrees_1007 : modelRow 1007 = row 1007 := model_count_1007.trans row_count_1007.symm
def row_1008 : RowCertificate 1008 where
  first := true
  first_correct := by
    have prime : Nat.Prime (corner 1008 1) := by change Nat.Prime 4066273; exact ProjectEulerSolutions.P58PrimeNodes.p_4066273.prime
    exact decide_eq_true prime
  second := false
  second_correct := by
    have composite : ¬ Nat.Prime (corner 1008 2) := by change ¬ Nat.Prime 4064257; exact PrattCertificates.composite_of_product (factor := 317) (cofactor := 12821) rfl (by decide) (by decide)
    exact decide_eq_false composite
  third := true
  third_correct := by
    have prime : Nat.Prime (corner 1008 3) := by change Nat.Prime 4062241; exact ProjectEulerSolutions.P58PrimeNodes.p_4062241.prime
    exact decide_eq_true prime
theorem row_count_1008 : row 1008 = 2 := (row_1008.checked).trans rfl
theorem model_count_1008 : modelRow 1008 = 2 := by decide
theorem modelrow_agrees_1008 : modelRow 1008 = row 1008 := model_count_1008.trans row_count_1008.symm
def row_1009 : RowCertificate 1009 where
  first := false
  first_correct := by
    have composite : ¬ Nat.Prime (corner 1009 1) := by change ¬ Nat.Prime 4074343; exact PrattCertificates.composite_of_product (factor := 7) (cofactor := 582049) rfl (by decide) (by decide)
    exact decide_eq_false composite
  second := false
  second_correct := by
    have composite : ¬ Nat.Prime (corner 1009 2) := by change ¬ Nat.Prime 4072325; exact PrattCertificates.composite_of_product (factor := 5) (cofactor := 814465) rfl (by decide) (by decide)
    exact decide_eq_false composite
  third := false
  third_correct := by
    have composite : ¬ Nat.Prime (corner 1009 3) := by change ¬ Nat.Prime 4070307; exact PrattCertificates.composite_of_product (factor := 3) (cofactor := 1356769) rfl (by decide) (by decide)
    exact decide_eq_false composite
theorem row_count_1009 : row 1009 = 0 := (row_1009.checked).trans rfl
theorem model_count_1009 : modelRow 1009 = 0 := by decide
theorem modelrow_agrees_1009 : modelRow 1009 = row 1009 := model_count_1009.trans row_count_1009.symm
def row_1010 : RowCertificate 1010 where
  first := false
  first_correct := by
    have composite : ¬ Nat.Prime (corner 1010 1) := by change ¬ Nat.Prime 4082421; exact PrattCertificates.composite_of_product (factor := 3) (cofactor := 1360807) rfl (by decide) (by decide)
    exact decide_eq_false composite
  second := false
  second_correct := by
    have composite : ¬ Nat.Prime (corner 1010 2) := by change ¬ Nat.Prime 4080401; exact PrattCertificates.composite_of_product (factor := 13) (cofactor := 313877) rfl (by decide) (by decide)
    exact decide_eq_false composite
  third := false
  third_correct := by
    have composite : ¬ Nat.Prime (corner 1010 3) := by change ¬ Nat.Prime 4078381; exact PrattCertificates.composite_of_product (factor := 397) (cofactor := 10273) rfl (by decide) (by decide)
    exact decide_eq_false composite
theorem row_count_1010 : row 1010 = 0 := (row_1010.checked).trans rfl
theorem model_count_1010 : modelRow 1010 = 0 := by decide
theorem modelrow_agrees_1010 : modelRow 1010 = row 1010 := model_count_1010.trans row_count_1010.symm
def row_1011 : RowCertificate 1011 where
  first := true
  first_correct := by
    have prime : Nat.Prime (corner 1011 1) := by change Nat.Prime 4090507; exact ProjectEulerSolutions.P58PrimeNodes.p_4090507.prime
    exact decide_eq_true prime
  second := false
  second_correct := by
    have composite : ¬ Nat.Prime (corner 1011 2) := by change ¬ Nat.Prime 4088485; exact PrattCertificates.composite_of_product (factor := 5) (cofactor := 817697) rfl (by decide) (by decide)
    exact decide_eq_false composite
  third := false
  third_correct := by
    have composite : ¬ Nat.Prime (corner 1011 3) := by change ¬ Nat.Prime 4086463; exact PrattCertificates.composite_of_product (factor := 19) (cofactor := 215077) rfl (by decide) (by decide)
    exact decide_eq_false composite
theorem row_count_1011 : row 1011 = 1 := (row_1011.checked).trans rfl
theorem model_count_1011 : modelRow 1011 = 1 := by decide
theorem modelrow_agrees_1011 : modelRow 1011 = row 1011 := model_count_1011.trans row_count_1011.symm
def row_1012 : RowCertificate 1012 where
  first := false
  first_correct := by
    have composite : ¬ Nat.Prime (corner 1012 1) := by change ¬ Nat.Prime 4098601; exact PrattCertificates.composite_of_product (factor := 13) (cofactor := 315277) rfl (by decide) (by decide)
    exact decide_eq_false composite
  second := false
  second_correct := by
    have composite : ¬ Nat.Prime (corner 1012 2) := by change ¬ Nat.Prime 4096577; exact PrattCertificates.composite_of_product (factor := 61) (cofactor := 67157) rfl (by decide) (by decide)
    exact decide_eq_false composite
  third := false
  third_correct := by
    have composite : ¬ Nat.Prime (corner 1012 3) := by change ¬ Nat.Prime 4094553; exact PrattCertificates.composite_of_product (factor := 3) (cofactor := 1364851) rfl (by decide) (by decide)
    exact decide_eq_false composite
theorem row_count_1012 : row 1012 = 0 := (row_1012.checked).trans rfl
theorem model_count_1012 : modelRow 1012 = 0 := by decide
theorem modelrow_agrees_1012 : modelRow 1012 = row 1012 := model_count_1012.trans row_count_1012.symm
def row_1013 : RowCertificate 1013 where
  first := false
  first_correct := by
    have composite : ¬ Nat.Prime (corner 1013 1) := by change ¬ Nat.Prime 4106703; exact PrattCertificates.composite_of_product (factor := 3) (cofactor := 1368901) rfl (by decide) (by decide)
    exact decide_eq_false composite
  second := true
  second_correct := by
    have prime : Nat.Prime (corner 1013 2) := by change Nat.Prime 4104677; exact ProjectEulerSolutions.P58PrimeNodes.p_4104677.prime
    exact decide_eq_true prime
  third := false
  third_correct := by
    have composite : ¬ Nat.Prime (corner 1013 3) := by change ¬ Nat.Prime 4102651; exact PrattCertificates.composite_of_product (factor := 7) (cofactor := 586093) rfl (by decide) (by decide)
    exact decide_eq_false composite
theorem row_count_1013 : row 1013 = 1 := (row_1013.checked).trans rfl
theorem model_count_1013 : modelRow 1013 = 1 := by decide
theorem modelrow_agrees_1013 : modelRow 1013 = row 1013 := model_count_1013.trans row_count_1013.symm
def row_1014 : RowCertificate 1014 where
  first := false
  first_correct := by
    have composite : ¬ Nat.Prime (corner 1014 1) := by change ¬ Nat.Prime 4114813; exact PrattCertificates.composite_of_product (factor := 157) (cofactor := 26209) rfl (by decide) (by decide)
    exact decide_eq_false composite
  second := false
  second_correct := by
    have composite : ¬ Nat.Prime (corner 1014 2) := by change ¬ Nat.Prime 4112785; exact PrattCertificates.composite_of_product (factor := 5) (cofactor := 822557) rfl (by decide) (by decide)
    exact decide_eq_false composite
  third := false
  third_correct := by
    have composite : ¬ Nat.Prime (corner 1014 3) := by change ¬ Nat.Prime 4110757; exact PrattCertificates.composite_of_product (factor := 7) (cofactor := 587251) rfl (by decide) (by decide)
    exact decide_eq_false composite
theorem row_count_1014 : row 1014 = 0 := (row_1014.checked).trans rfl
theorem model_count_1014 : modelRow 1014 = 0 := by decide
theorem modelrow_agrees_1014 : modelRow 1014 = row 1014 := model_count_1014.trans row_count_1014.symm
def row_1015 : RowCertificate 1015 where
  first := false
  first_correct := by
    have composite : ¬ Nat.Prime (corner 1015 1) := by change ¬ Nat.Prime 4122931; exact PrattCertificates.composite_of_product (factor := 79) (cofactor := 52189) rfl (by decide) (by decide)
    exact decide_eq_false composite
  second := false
  second_correct := by
    have composite : ¬ Nat.Prime (corner 1015 2) := by change ¬ Nat.Prime 4120901; exact PrattCertificates.composite_of_product (factor := 101) (cofactor := 40801) rfl (by decide) (by decide)
    exact decide_eq_false composite
  third := false
  third_correct := by
    have composite : ¬ Nat.Prime (corner 1015 3) := by change ¬ Nat.Prime 4118871; exact PrattCertificates.composite_of_product (factor := 3) (cofactor := 1372957) rfl (by decide) (by decide)
    exact decide_eq_false composite
theorem row_count_1015 : row 1015 = 0 := (row_1015.checked).trans rfl
theorem model_count_1015 : modelRow 1015 = 0 := by decide
theorem modelrow_agrees_1015 : modelRow 1015 = row 1015 := model_count_1015.trans row_count_1015.symm
def row_1016 : RowCertificate 1016 where
  first := false
  first_correct := by
    have composite : ¬ Nat.Prime (corner 1016 1) := by change ¬ Nat.Prime 4131057; exact PrattCertificates.composite_of_product (factor := 3) (cofactor := 1377019) rfl (by decide) (by decide)
    exact decide_eq_false composite
  second := false
  second_correct := by
    have composite : ¬ Nat.Prime (corner 1016 2) := by change ¬ Nat.Prime 4129025; exact PrattCertificates.composite_of_product (factor := 5) (cofactor := 825805) rfl (by decide) (by decide)
    exact decide_eq_false composite
  third := false
  third_correct := by
    have composite : ¬ Nat.Prime (corner 1016 3) := by change ¬ Nat.Prime 4126993; exact PrattCertificates.composite_of_product (factor := 13) (cofactor := 317461) rfl (by decide) (by decide)
    exact decide_eq_false composite
theorem row_count_1016 : row 1016 = 0 := (row_1016.checked).trans rfl
theorem model_count_1016 : modelRow 1016 = 0 := by decide
theorem modelrow_agrees_1016 : modelRow 1016 = row 1016 := model_count_1016.trans row_count_1016.symm
def row_1017 : RowCertificate 1017 where
  first := false
  first_correct := by
    have composite : ¬ Nat.Prime (corner 1017 1) := by change ¬ Nat.Prime 4139191; exact PrattCertificates.composite_of_product (factor := 7) (cofactor := 591313) rfl (by decide) (by decide)
    exact decide_eq_false composite
  second := true
  second_correct := by
    have prime : Nat.Prime (corner 1017 2) := by change Nat.Prime 4137157; exact ProjectEulerSolutions.P58PrimeNodes.p_4137157.prime
    exact decide_eq_true prime
  third := true
  third_correct := by
    have prime : Nat.Prime (corner 1017 3) := by change Nat.Prime 4135123; exact ProjectEulerSolutions.P58PrimeNodes.p_4135123.prime
    exact decide_eq_true prime
theorem row_count_1017 : row 1017 = 2 := (row_1017.checked).trans rfl
theorem model_count_1017 : modelRow 1017 = 2 := by decide
theorem modelrow_agrees_1017 : modelRow 1017 = row 1017 := model_count_1017.trans row_count_1017.symm
def row_1018 : RowCertificate 1018 where
  first := false
  first_correct := by
    have composite : ¬ Nat.Prime (corner 1018 1) := by change ¬ Nat.Prime 4147333; exact PrattCertificates.composite_of_product (factor := 877) (cofactor := 4729) rfl (by decide) (by decide)
    exact decide_eq_false composite
  second := false
  second_correct := by
    have composite : ¬ Nat.Prime (corner 1018 2) := by change ¬ Nat.Prime 4145297; exact PrattCertificates.composite_of_product (factor := 13) (cofactor := 318869) rfl (by decide) (by decide)
    exact decide_eq_false composite
  third := false
  third_correct := by
    have composite : ¬ Nat.Prime (corner 1018 3) := by change ¬ Nat.Prime 4143261; exact PrattCertificates.composite_of_product (factor := 3) (cofactor := 1381087) rfl (by decide) (by decide)
    exact decide_eq_false composite
theorem row_count_1018 : row 1018 = 0 := (row_1018.checked).trans rfl
theorem model_count_1018 : modelRow 1018 = 0 := by decide
theorem modelrow_agrees_1018 : modelRow 1018 = row 1018 := model_count_1018.trans row_count_1018.symm
def row_1019 : RowCertificate 1019 where
  first := false
  first_correct := by
    have composite : ¬ Nat.Prime (corner 1019 1) := by change ¬ Nat.Prime 4155483; exact PrattCertificates.composite_of_product (factor := 3) (cofactor := 1385161) rfl (by decide) (by decide)
    exact decide_eq_false composite
  second := false
  second_correct := by
    have composite : ¬ Nat.Prime (corner 1019 2) := by change ¬ Nat.Prime 4153445; exact PrattCertificates.composite_of_product (factor := 5) (cofactor := 830689) rfl (by decide) (by decide)
    exact decide_eq_false composite
  third := false
  third_correct := by
    have composite : ¬ Nat.Prime (corner 1019 3) := by change ¬ Nat.Prime 4151407; exact PrattCertificates.composite_of_product (factor := 13) (cofactor := 319339) rfl (by decide) (by decide)
    exact decide_eq_false composite
theorem row_count_1019 : row 1019 = 0 := (row_1019.checked).trans rfl
theorem model_count_1019 : modelRow 1019 = 0 := by decide
theorem modelrow_agrees_1019 : modelRow 1019 = row 1019 := model_count_1019.trans row_count_1019.symm
def row_1020 : RowCertificate 1020 where
  first := false
  first_correct := by
    have composite : ¬ Nat.Prime (corner 1020 1) := by change ¬ Nat.Prime 4163641; exact PrattCertificates.composite_of_product (factor := 19) (cofactor := 219139) rfl (by decide) (by decide)
    exact decide_eq_false composite
  second := false
  second_correct := by
    have composite : ¬ Nat.Prime (corner 1020 2) := by change ¬ Nat.Prime 4161601; exact PrattCertificates.composite_of_product (factor := 257) (cofactor := 16193) rfl (by decide) (by decide)
    exact decide_eq_false composite
  third := false
  third_correct := by
    have composite : ¬ Nat.Prime (corner 1020 3) := by change ¬ Nat.Prime 4159561; exact PrattCertificates.composite_of_product (factor := 7) (cofactor := 594223) rfl (by decide) (by decide)
    exact decide_eq_false composite
theorem row_count_1020 : row 1020 = 0 := (row_1020.checked).trans rfl
theorem model_count_1020 : modelRow 1020 = 0 := by decide
theorem modelrow_agrees_1020 : modelRow 1020 = row 1020 := model_count_1020.trans row_count_1020.symm
def row_1021 : RowCertificate 1021 where
  first := false
  first_correct := by
    have composite : ¬ Nat.Prime (corner 1021 1) := by change ¬ Nat.Prime 4171807; exact PrattCertificates.composite_of_product (factor := 139) (cofactor := 30013) rfl (by decide) (by decide)
    exact decide_eq_false composite
  second := false
  second_correct := by
    have composite : ¬ Nat.Prime (corner 1021 2) := by change ¬ Nat.Prime 4169765; exact PrattCertificates.composite_of_product (factor := 5) (cofactor := 833953) rfl (by decide) (by decide)
    exact decide_eq_false composite
  third := false
  third_correct := by
    have composite : ¬ Nat.Prime (corner 1021 3) := by change ¬ Nat.Prime 4167723; exact PrattCertificates.composite_of_product (factor := 3) (cofactor := 1389241) rfl (by decide) (by decide)
    exact decide_eq_false composite
theorem row_count_1021 : row 1021 = 0 := (row_1021.checked).trans rfl
theorem model_count_1021 : modelRow 1021 = 0 := by decide
theorem modelrow_agrees_1021 : modelRow 1021 = row 1021 := model_count_1021.trans row_count_1021.symm
def row_1022 : RowCertificate 1022 where
  first := false
  first_correct := by
    have composite : ¬ Nat.Prime (corner 1022 1) := by change ¬ Nat.Prime 4179981; exact PrattCertificates.composite_of_product (factor := 3) (cofactor := 1393327) rfl (by decide) (by decide)
    exact decide_eq_false composite
  second := false
  second_correct := by
    have composite : ¬ Nat.Prime (corner 1022 2) := by change ¬ Nat.Prime 4177937; exact PrattCertificates.composite_of_product (factor := 17) (cofactor := 245761) rfl (by decide) (by decide)
    exact decide_eq_false composite
  third := true
  third_correct := by
    have prime : Nat.Prime (corner 1022 3) := by change Nat.Prime 4175893; exact ProjectEulerSolutions.P58PrimeNodes.p_4175893.prime
    exact decide_eq_true prime
theorem row_count_1022 : row 1022 = 1 := (row_1022.checked).trans rfl
theorem model_count_1022 : modelRow 1022 = 1 := by decide
theorem modelrow_agrees_1022 : modelRow 1022 = row 1022 := model_count_1022.trans row_count_1022.symm
def row_1023 : RowCertificate 1023 where
  first := false
  first_correct := by
    have composite : ¬ Nat.Prime (corner 1023 1) := by change ¬ Nat.Prime 4188163; exact PrattCertificates.composite_of_product (factor := 7) (cofactor := 598309) rfl (by decide) (by decide)
    exact decide_eq_false composite
  second := false
  second_correct := by
    have composite : ¬ Nat.Prime (corner 1023 2) := by change ¬ Nat.Prime 4186117; exact PrattCertificates.composite_of_product (factor := 13) (cofactor := 322009) rfl (by decide) (by decide)
    exact decide_eq_false composite
  third := false
  third_correct := by
    have composite : ¬ Nat.Prime (corner 1023 3) := by change ¬ Nat.Prime 4184071; exact PrattCertificates.composite_of_product (factor := 37) (cofactor := 113083) rfl (by decide) (by decide)
    exact decide_eq_false composite
theorem row_count_1023 : row 1023 = 0 := (row_1023.checked).trans rfl
theorem model_count_1023 : modelRow 1023 = 0 := by decide
theorem modelrow_agrees_1023 : modelRow 1023 = row 1023 := model_count_1023.trans row_count_1023.symm
def row_1024 : RowCertificate 1024 where
  first := false
  first_correct := by
    have composite : ¬ Nat.Prime (corner 1024 1) := by change ¬ Nat.Prime 4196353; exact PrattCertificates.composite_of_product (factor := 7) (cofactor := 599479) rfl (by decide) (by decide)
    exact decide_eq_false composite
  second := false
  second_correct := by
    have composite : ¬ Nat.Prime (corner 1024 2) := by change ¬ Nat.Prime 4194305; exact PrattCertificates.composite_of_product (factor := 5) (cofactor := 838861) rfl (by decide) (by decide)
    exact decide_eq_false composite
  third := false
  third_correct := by
    have composite : ¬ Nat.Prime (corner 1024 3) := by change ¬ Nat.Prime 4192257; exact PrattCertificates.composite_of_product (factor := 3) (cofactor := 1397419) rfl (by decide) (by decide)
    exact decide_eq_false composite
theorem row_count_1024 : row 1024 = 0 := (row_1024.checked).trans rfl
theorem model_count_1024 : modelRow 1024 = 0 := by decide
theorem modelrow_agrees_1024 : modelRow 1024 = row 1024 := model_count_1024.trans row_count_1024.symm
private def atomic_897 : CountSegment 1 10 13120 896 897 507 508 := CountSegment.single (first := 896) (initial := 507) row_count_897 (by decide)
private theorem atomic_897_agreement : RowAgreement 896 897 := RowAgreement.single (first := 896) modelrow_agrees_897
private def atomic_898 : CountSegment 1 10 13120 897 898 508 508 := CountSegment.single (first := 897) (initial := 508) row_count_898 (by decide)
private theorem atomic_898_agreement : RowAgreement 897 898 := RowAgreement.single (first := 897) modelrow_agrees_898
private def atomic_899 : CountSegment 1 10 13120 898 899 508 508 := CountSegment.single (first := 898) (initial := 508) row_count_899 (by decide)
private theorem atomic_899_agreement : RowAgreement 898 899 := RowAgreement.single (first := 898) modelrow_agrees_899
private def atomic_900 : CountSegment 1 10 13120 899 900 508 509 := CountSegment.single (first := 899) (initial := 508) row_count_900 (by decide)
private theorem atomic_900_agreement : RowAgreement 899 900 := RowAgreement.single (first := 899) modelrow_agrees_900
private def atomic_901 : CountSegment 1 10 13120 900 901 509 509 := CountSegment.single (first := 900) (initial := 509) row_count_901 (by decide)
private theorem atomic_901_agreement : RowAgreement 900 901 := RowAgreement.single (first := 900) modelrow_agrees_901
private def atomic_902 : CountSegment 1 10 13120 901 902 509 509 := CountSegment.single (first := 901) (initial := 509) row_count_902 (by decide)
private theorem atomic_902_agreement : RowAgreement 901 902 := RowAgreement.single (first := 901) modelrow_agrees_902
private def atomic_903 : CountSegment 1 10 13120 902 903 509 510 := CountSegment.single (first := 902) (initial := 509) row_count_903 (by decide)
private theorem atomic_903_agreement : RowAgreement 902 903 := RowAgreement.single (first := 902) modelrow_agrees_903
private def atomic_904 : CountSegment 1 10 13120 903 904 510 510 := CountSegment.single (first := 903) (initial := 510) row_count_904 (by decide)
private theorem atomic_904_agreement : RowAgreement 903 904 := RowAgreement.single (first := 903) modelrow_agrees_904
private def atomic_905 : CountSegment 1 10 13120 904 905 510 511 := CountSegment.single (first := 904) (initial := 510) row_count_905 (by decide)
private theorem atomic_905_agreement : RowAgreement 904 905 := RowAgreement.single (first := 904) modelrow_agrees_905
private def atomic_906 : CountSegment 1 10 13120 905 906 511 512 := CountSegment.single (first := 905) (initial := 511) row_count_906 (by decide)
private theorem atomic_906_agreement : RowAgreement 905 906 := RowAgreement.single (first := 905) modelrow_agrees_906
private def atomic_907 : CountSegment 1 10 13120 906 907 512 513 := CountSegment.single (first := 906) (initial := 512) row_count_907 (by decide)
private theorem atomic_907_agreement : RowAgreement 906 907 := RowAgreement.single (first := 906) modelrow_agrees_907
private def atomic_908 : CountSegment 1 10 13120 907 908 513 514 := CountSegment.single (first := 907) (initial := 513) row_count_908 (by decide)
private theorem atomic_908_agreement : RowAgreement 907 908 := RowAgreement.single (first := 907) modelrow_agrees_908
private def atomic_909 : CountSegment 1 10 13120 908 909 514 515 := CountSegment.single (first := 908) (initial := 514) row_count_909 (by decide)
private theorem atomic_909_agreement : RowAgreement 908 909 := RowAgreement.single (first := 908) modelrow_agrees_909
private def atomic_910 : CountSegment 1 10 13120 909 910 515 515 := CountSegment.single (first := 909) (initial := 515) row_count_910 (by decide)
private theorem atomic_910_agreement : RowAgreement 909 910 := RowAgreement.single (first := 909) modelrow_agrees_910
private def atomic_911 : CountSegment 1 10 13120 910 911 515 515 := CountSegment.single (first := 910) (initial := 515) row_count_911 (by decide)
private theorem atomic_911_agreement : RowAgreement 910 911 := RowAgreement.single (first := 910) modelrow_agrees_911
private def atomic_912 : CountSegment 1 10 13120 911 912 515 516 := CountSegment.single (first := 911) (initial := 515) row_count_912 (by decide)
private theorem atomic_912_agreement : RowAgreement 911 912 := RowAgreement.single (first := 911) modelrow_agrees_912
private def atomic_913 : CountSegment 1 10 13120 912 913 516 517 := CountSegment.single (first := 912) (initial := 516) row_count_913 (by decide)
private theorem atomic_913_agreement : RowAgreement 912 913 := RowAgreement.single (first := 912) modelrow_agrees_913
private def atomic_914 : CountSegment 1 10 13120 913 914 517 517 := CountSegment.single (first := 913) (initial := 517) row_count_914 (by decide)
private theorem atomic_914_agreement : RowAgreement 913 914 := RowAgreement.single (first := 913) modelrow_agrees_914
private def atomic_915 : CountSegment 1 10 13120 914 915 517 517 := CountSegment.single (first := 914) (initial := 517) row_count_915 (by decide)
private theorem atomic_915_agreement : RowAgreement 914 915 := RowAgreement.single (first := 914) modelrow_agrees_915
private def atomic_916 : CountSegment 1 10 13120 915 916 517 518 := CountSegment.single (first := 915) (initial := 517) row_count_916 (by decide)
private theorem atomic_916_agreement : RowAgreement 915 916 := RowAgreement.single (first := 915) modelrow_agrees_916
private def atomic_917 : CountSegment 1 10 13120 916 917 518 518 := CountSegment.single (first := 916) (initial := 518) row_count_917 (by decide)
private theorem atomic_917_agreement : RowAgreement 916 917 := RowAgreement.single (first := 916) modelrow_agrees_917
private def atomic_918 : CountSegment 1 10 13120 917 918 518 518 := CountSegment.single (first := 917) (initial := 518) row_count_918 (by decide)
private theorem atomic_918_agreement : RowAgreement 917 918 := RowAgreement.single (first := 917) modelrow_agrees_918
private def atomic_919 : CountSegment 1 10 13120 918 919 518 518 := CountSegment.single (first := 918) (initial := 518) row_count_919 (by decide)
private theorem atomic_919_agreement : RowAgreement 918 919 := RowAgreement.single (first := 918) modelrow_agrees_919
private def atomic_920 : CountSegment 1 10 13120 919 920 518 518 := CountSegment.single (first := 919) (initial := 518) row_count_920 (by decide)
private theorem atomic_920_agreement : RowAgreement 919 920 := RowAgreement.single (first := 919) modelrow_agrees_920
private def atomic_921 : CountSegment 1 10 13120 920 921 518 519 := CountSegment.single (first := 920) (initial := 518) row_count_921 (by decide)
private theorem atomic_921_agreement : RowAgreement 920 921 := RowAgreement.single (first := 920) modelrow_agrees_921
private def atomic_922 : CountSegment 1 10 13120 921 922 519 519 := CountSegment.single (first := 921) (initial := 519) row_count_922 (by decide)
private theorem atomic_922_agreement : RowAgreement 921 922 := RowAgreement.single (first := 921) modelrow_agrees_922
private def atomic_923 : CountSegment 1 10 13120 922 923 519 519 := CountSegment.single (first := 922) (initial := 519) row_count_923 (by decide)
private theorem atomic_923_agreement : RowAgreement 922 923 := RowAgreement.single (first := 922) modelrow_agrees_923
private def atomic_924 : CountSegment 1 10 13120 923 924 519 521 := CountSegment.single (first := 923) (initial := 519) row_count_924 (by decide)
private theorem atomic_924_agreement : RowAgreement 923 924 := RowAgreement.single (first := 923) modelrow_agrees_924
private def atomic_925 : CountSegment 1 10 13120 924 925 521 522 := CountSegment.single (first := 924) (initial := 521) row_count_925 (by decide)
private theorem atomic_925_agreement : RowAgreement 924 925 := RowAgreement.single (first := 924) modelrow_agrees_925
private def atomic_926 : CountSegment 1 10 13120 925 926 522 522 := CountSegment.single (first := 925) (initial := 522) row_count_926 (by decide)
private theorem atomic_926_agreement : RowAgreement 925 926 := RowAgreement.single (first := 925) modelrow_agrees_926
private def atomic_927 : CountSegment 1 10 13120 926 927 522 523 := CountSegment.single (first := 926) (initial := 522) row_count_927 (by decide)
private theorem atomic_927_agreement : RowAgreement 926 927 := RowAgreement.single (first := 926) modelrow_agrees_927
private def atomic_928 : CountSegment 1 10 13120 927 928 523 523 := CountSegment.single (first := 927) (initial := 523) row_count_928 (by decide)
private theorem atomic_928_agreement : RowAgreement 927 928 := RowAgreement.single (first := 927) modelrow_agrees_928
private def atomic_929 : CountSegment 1 10 13120 928 929 523 523 := CountSegment.single (first := 928) (initial := 523) row_count_929 (by decide)
private theorem atomic_929_agreement : RowAgreement 928 929 := RowAgreement.single (first := 928) modelrow_agrees_929
private def atomic_930 : CountSegment 1 10 13120 929 930 523 524 := CountSegment.single (first := 929) (initial := 523) row_count_930 (by decide)
private theorem atomic_930_agreement : RowAgreement 929 930 := RowAgreement.single (first := 929) modelrow_agrees_930
private def atomic_931 : CountSegment 1 10 13120 930 931 524 524 := CountSegment.single (first := 930) (initial := 524) row_count_931 (by decide)
private theorem atomic_931_agreement : RowAgreement 930 931 := RowAgreement.single (first := 930) modelrow_agrees_931
private def atomic_932 : CountSegment 1 10 13120 931 932 524 525 := CountSegment.single (first := 931) (initial := 524) row_count_932 (by decide)
private theorem atomic_932_agreement : RowAgreement 931 932 := RowAgreement.single (first := 931) modelrow_agrees_932
private def atomic_933 : CountSegment 1 10 13120 932 933 525 525 := CountSegment.single (first := 932) (initial := 525) row_count_933 (by decide)
private theorem atomic_933_agreement : RowAgreement 932 933 := RowAgreement.single (first := 932) modelrow_agrees_933
private def atomic_934 : CountSegment 1 10 13120 933 934 525 525 := CountSegment.single (first := 933) (initial := 525) row_count_934 (by decide)
private theorem atomic_934_agreement : RowAgreement 933 934 := RowAgreement.single (first := 933) modelrow_agrees_934
private def atomic_935 : CountSegment 1 10 13120 934 935 525 526 := CountSegment.single (first := 934) (initial := 525) row_count_935 (by decide)
private theorem atomic_935_agreement : RowAgreement 934 935 := RowAgreement.single (first := 934) modelrow_agrees_935
private def atomic_936 : CountSegment 1 10 13120 935 936 526 526 := CountSegment.single (first := 935) (initial := 526) row_count_936 (by decide)
private theorem atomic_936_agreement : RowAgreement 935 936 := RowAgreement.single (first := 935) modelrow_agrees_936
private def atomic_937 : CountSegment 1 10 13120 936 937 526 526 := CountSegment.single (first := 936) (initial := 526) row_count_937 (by decide)
private theorem atomic_937_agreement : RowAgreement 936 937 := RowAgreement.single (first := 936) modelrow_agrees_937
private def atomic_938 : CountSegment 1 10 13120 937 938 526 527 := CountSegment.single (first := 937) (initial := 526) row_count_938 (by decide)
private theorem atomic_938_agreement : RowAgreement 937 938 := RowAgreement.single (first := 937) modelrow_agrees_938
private def atomic_939 : CountSegment 1 10 13120 938 939 527 527 := CountSegment.single (first := 938) (initial := 527) row_count_939 (by decide)
private theorem atomic_939_agreement : RowAgreement 938 939 := RowAgreement.single (first := 938) modelrow_agrees_939
private def atomic_940 : CountSegment 1 10 13120 939 940 527 527 := CountSegment.single (first := 939) (initial := 527) row_count_940 (by decide)
private theorem atomic_940_agreement : RowAgreement 939 940 := RowAgreement.single (first := 939) modelrow_agrees_940
private def atomic_941 : CountSegment 1 10 13120 940 941 527 527 := CountSegment.single (first := 940) (initial := 527) row_count_941 (by decide)
private theorem atomic_941_agreement : RowAgreement 940 941 := RowAgreement.single (first := 940) modelrow_agrees_941
private def atomic_942 : CountSegment 1 10 13120 941 942 527 528 := CountSegment.single (first := 941) (initial := 527) row_count_942 (by decide)
private theorem atomic_942_agreement : RowAgreement 941 942 := RowAgreement.single (first := 941) modelrow_agrees_942
private def atomic_943 : CountSegment 1 10 13120 942 943 528 529 := CountSegment.single (first := 942) (initial := 528) row_count_943 (by decide)
private theorem atomic_943_agreement : RowAgreement 942 943 := RowAgreement.single (first := 942) modelrow_agrees_943
private def atomic_944 : CountSegment 1 10 13120 943 944 529 529 := CountSegment.single (first := 943) (initial := 529) row_count_944 (by decide)
private theorem atomic_944_agreement : RowAgreement 943 944 := RowAgreement.single (first := 943) modelrow_agrees_944
private def atomic_945 : CountSegment 1 10 13120 944 945 529 529 := CountSegment.single (first := 944) (initial := 529) row_count_945 (by decide)
private theorem atomic_945_agreement : RowAgreement 944 945 := RowAgreement.single (first := 944) modelrow_agrees_945
private def atomic_946 : CountSegment 1 10 13120 945 946 529 529 := CountSegment.single (first := 945) (initial := 529) row_count_946 (by decide)
private theorem atomic_946_agreement : RowAgreement 945 946 := RowAgreement.single (first := 945) modelrow_agrees_946
private def atomic_947 : CountSegment 1 10 13120 946 947 529 530 := CountSegment.single (first := 946) (initial := 529) row_count_947 (by decide)
private theorem atomic_947_agreement : RowAgreement 946 947 := RowAgreement.single (first := 946) modelrow_agrees_947
private def atomic_948 : CountSegment 1 10 13120 947 948 530 531 := CountSegment.single (first := 947) (initial := 530) row_count_948 (by decide)
private theorem atomic_948_agreement : RowAgreement 947 948 := RowAgreement.single (first := 947) modelrow_agrees_948
private def atomic_949 : CountSegment 1 10 13120 948 949 531 531 := CountSegment.single (first := 948) (initial := 531) row_count_949 (by decide)
private theorem atomic_949_agreement : RowAgreement 948 949 := RowAgreement.single (first := 948) modelrow_agrees_949
private def atomic_950 : CountSegment 1 10 13120 949 950 531 531 := CountSegment.single (first := 949) (initial := 531) row_count_950 (by decide)
private theorem atomic_950_agreement : RowAgreement 949 950 := RowAgreement.single (first := 949) modelrow_agrees_950
private def atomic_951 : CountSegment 1 10 13120 950 951 531 532 := CountSegment.single (first := 950) (initial := 531) row_count_951 (by decide)
private theorem atomic_951_agreement : RowAgreement 950 951 := RowAgreement.single (first := 950) modelrow_agrees_951
private def atomic_952 : CountSegment 1 10 13120 951 952 532 532 := CountSegment.single (first := 951) (initial := 532) row_count_952 (by decide)
private theorem atomic_952_agreement : RowAgreement 951 952 := RowAgreement.single (first := 951) modelrow_agrees_952
private def atomic_953 : CountSegment 1 10 13120 952 953 532 532 := CountSegment.single (first := 952) (initial := 532) row_count_953 (by decide)
private theorem atomic_953_agreement : RowAgreement 952 953 := RowAgreement.single (first := 952) modelrow_agrees_953
private def atomic_954 : CountSegment 1 10 13120 953 954 532 532 := CountSegment.single (first := 953) (initial := 532) row_count_954 (by decide)
private theorem atomic_954_agreement : RowAgreement 953 954 := RowAgreement.single (first := 953) modelrow_agrees_954
private def atomic_955 : CountSegment 1 10 13120 954 955 532 533 := CountSegment.single (first := 954) (initial := 532) row_count_955 (by decide)
private theorem atomic_955_agreement : RowAgreement 954 955 := RowAgreement.single (first := 954) modelrow_agrees_955
private def atomic_956 : CountSegment 1 10 13120 955 956 533 533 := CountSegment.single (first := 955) (initial := 533) row_count_956 (by decide)
private theorem atomic_956_agreement : RowAgreement 955 956 := RowAgreement.single (first := 955) modelrow_agrees_956
private def atomic_957 : CountSegment 1 10 13120 956 957 533 533 := CountSegment.single (first := 956) (initial := 533) row_count_957 (by decide)
private theorem atomic_957_agreement : RowAgreement 956 957 := RowAgreement.single (first := 956) modelrow_agrees_957
private def atomic_958 : CountSegment 1 10 13120 957 958 533 533 := CountSegment.single (first := 957) (initial := 533) row_count_958 (by decide)
private theorem atomic_958_agreement : RowAgreement 957 958 := RowAgreement.single (first := 957) modelrow_agrees_958
private def atomic_959 : CountSegment 1 10 13120 958 959 533 533 := CountSegment.single (first := 958) (initial := 533) row_count_959 (by decide)
private theorem atomic_959_agreement : RowAgreement 958 959 := RowAgreement.single (first := 958) modelrow_agrees_959
private def atomic_960 : CountSegment 1 10 13120 959 960 533 534 := CountSegment.single (first := 959) (initial := 533) row_count_960 (by decide)
private theorem atomic_960_agreement : RowAgreement 959 960 := RowAgreement.single (first := 959) modelrow_agrees_960
private def atomic_961 : CountSegment 1 10 13120 960 961 534 534 := CountSegment.single (first := 960) (initial := 534) row_count_961 (by decide)
private theorem atomic_961_agreement : RowAgreement 960 961 := RowAgreement.single (first := 960) modelrow_agrees_961
private def atomic_962 : CountSegment 1 10 13120 961 962 534 535 := CountSegment.single (first := 961) (initial := 534) row_count_962 (by decide)
private theorem atomic_962_agreement : RowAgreement 961 962 := RowAgreement.single (first := 961) modelrow_agrees_962
private def atomic_963 : CountSegment 1 10 13120 962 963 535 535 := CountSegment.single (first := 962) (initial := 535) row_count_963 (by decide)
private theorem atomic_963_agreement : RowAgreement 962 963 := RowAgreement.single (first := 962) modelrow_agrees_963
private def atomic_964 : CountSegment 1 10 13120 963 964 535 535 := CountSegment.single (first := 963) (initial := 535) row_count_964 (by decide)
private theorem atomic_964_agreement : RowAgreement 963 964 := RowAgreement.single (first := 963) modelrow_agrees_964
private def atomic_965 : CountSegment 1 10 13120 964 965 535 535 := CountSegment.single (first := 964) (initial := 535) row_count_965 (by decide)
private theorem atomic_965_agreement : RowAgreement 964 965 := RowAgreement.single (first := 964) modelrow_agrees_965
private def atomic_966 : CountSegment 1 10 13120 965 966 535 537 := CountSegment.single (first := 965) (initial := 535) row_count_966 (by decide)
private theorem atomic_966_agreement : RowAgreement 965 966 := RowAgreement.single (first := 965) modelrow_agrees_966
private def atomic_967 : CountSegment 1 10 13120 966 967 537 537 := CountSegment.single (first := 966) (initial := 537) row_count_967 (by decide)
private theorem atomic_967_agreement : RowAgreement 966 967 := RowAgreement.single (first := 966) modelrow_agrees_967
private def atomic_968 : CountSegment 1 10 13120 967 968 537 537 := CountSegment.single (first := 967) (initial := 537) row_count_968 (by decide)
private theorem atomic_968_agreement : RowAgreement 967 968 := RowAgreement.single (first := 967) modelrow_agrees_968
private def atomic_969 : CountSegment 1 10 13120 968 969 537 537 := CountSegment.single (first := 968) (initial := 537) row_count_969 (by decide)
private theorem atomic_969_agreement : RowAgreement 968 969 := RowAgreement.single (first := 968) modelrow_agrees_969
private def atomic_970 : CountSegment 1 10 13120 969 970 537 538 := CountSegment.single (first := 969) (initial := 537) row_count_970 (by decide)
private theorem atomic_970_agreement : RowAgreement 969 970 := RowAgreement.single (first := 969) modelrow_agrees_970
private def atomic_971 : CountSegment 1 10 13120 970 971 538 538 := CountSegment.single (first := 970) (initial := 538) row_count_971 (by decide)
private theorem atomic_971_agreement : RowAgreement 970 971 := RowAgreement.single (first := 970) modelrow_agrees_971
private def atomic_972 : CountSegment 1 10 13120 971 972 538 538 := CountSegment.single (first := 971) (initial := 538) row_count_972 (by decide)
private theorem atomic_972_agreement : RowAgreement 971 972 := RowAgreement.single (first := 971) modelrow_agrees_972
private def atomic_973 : CountSegment 1 10 13120 972 973 538 538 := CountSegment.single (first := 972) (initial := 538) row_count_973 (by decide)
private theorem atomic_973_agreement : RowAgreement 972 973 := RowAgreement.single (first := 972) modelrow_agrees_973
private def atomic_974 : CountSegment 1 10 13120 973 974 538 538 := CountSegment.single (first := 973) (initial := 538) row_count_974 (by decide)
private theorem atomic_974_agreement : RowAgreement 973 974 := RowAgreement.single (first := 973) modelrow_agrees_974
private def atomic_975 : CountSegment 1 10 13120 974 975 538 538 := CountSegment.single (first := 974) (initial := 538) row_count_975 (by decide)
private theorem atomic_975_agreement : RowAgreement 974 975 := RowAgreement.single (first := 974) modelrow_agrees_975
private def atomic_976 : CountSegment 1 10 13120 975 976 538 539 := CountSegment.single (first := 975) (initial := 538) row_count_976 (by decide)
private theorem atomic_976_agreement : RowAgreement 975 976 := RowAgreement.single (first := 975) modelrow_agrees_976
private def atomic_977 : CountSegment 1 10 13120 976 977 539 539 := CountSegment.single (first := 976) (initial := 539) row_count_977 (by decide)
private theorem atomic_977_agreement : RowAgreement 976 977 := RowAgreement.single (first := 976) modelrow_agrees_977
private def atomic_978 : CountSegment 1 10 13120 977 978 539 540 := CountSegment.single (first := 977) (initial := 539) row_count_978 (by decide)
private theorem atomic_978_agreement : RowAgreement 977 978 := RowAgreement.single (first := 977) modelrow_agrees_978
private def atomic_979 : CountSegment 1 10 13120 978 979 540 540 := CountSegment.single (first := 978) (initial := 540) row_count_979 (by decide)
private theorem atomic_979_agreement : RowAgreement 978 979 := RowAgreement.single (first := 978) modelrow_agrees_979
private def atomic_980 : CountSegment 1 10 13120 979 980 540 540 := CountSegment.single (first := 979) (initial := 540) row_count_980 (by decide)
private theorem atomic_980_agreement : RowAgreement 979 980 := RowAgreement.single (first := 979) modelrow_agrees_980
private def atomic_981 : CountSegment 1 10 13120 980 981 540 540 := CountSegment.single (first := 980) (initial := 540) row_count_981 (by decide)
private theorem atomic_981_agreement : RowAgreement 980 981 := RowAgreement.single (first := 980) modelrow_agrees_981
private def atomic_982 : CountSegment 1 10 13120 981 982 540 541 := CountSegment.single (first := 981) (initial := 540) row_count_982 (by decide)
private theorem atomic_982_agreement : RowAgreement 981 982 := RowAgreement.single (first := 981) modelrow_agrees_982
private def atomic_983 : CountSegment 1 10 13120 982 983 541 542 := CountSegment.single (first := 982) (initial := 541) row_count_983 (by decide)
private theorem atomic_983_agreement : RowAgreement 982 983 := RowAgreement.single (first := 982) modelrow_agrees_983
private def atomic_984 : CountSegment 1 10 13120 983 984 542 542 := CountSegment.single (first := 983) (initial := 542) row_count_984 (by decide)
private theorem atomic_984_agreement : RowAgreement 983 984 := RowAgreement.single (first := 983) modelrow_agrees_984
private def atomic_985 : CountSegment 1 10 13120 984 985 542 544 := CountSegment.single (first := 984) (initial := 542) row_count_985 (by decide)
private theorem atomic_985_agreement : RowAgreement 984 985 := RowAgreement.single (first := 984) modelrow_agrees_985
private def atomic_986 : CountSegment 1 10 13120 985 986 544 544 := CountSegment.single (first := 985) (initial := 544) row_count_986 (by decide)
private theorem atomic_986_agreement : RowAgreement 985 986 := RowAgreement.single (first := 985) modelrow_agrees_986
private def atomic_987 : CountSegment 1 10 13120 986 987 544 547 := CountSegment.single (first := 986) (initial := 544) row_count_987 (by decide)
private theorem atomic_987_agreement : RowAgreement 986 987 := RowAgreement.single (first := 986) modelrow_agrees_987
private def atomic_988 : CountSegment 1 10 13120 987 988 547 547 := CountSegment.single (first := 987) (initial := 547) row_count_988 (by decide)
private theorem atomic_988_agreement : RowAgreement 987 988 := RowAgreement.single (first := 987) modelrow_agrees_988
private def atomic_989 : CountSegment 1 10 13120 988 989 547 548 := CountSegment.single (first := 988) (initial := 547) row_count_989 (by decide)
private theorem atomic_989_agreement : RowAgreement 988 989 := RowAgreement.single (first := 988) modelrow_agrees_989
private def atomic_990 : CountSegment 1 10 13120 989 990 548 549 := CountSegment.single (first := 989) (initial := 548) row_count_990 (by decide)
private theorem atomic_990_agreement : RowAgreement 989 990 := RowAgreement.single (first := 989) modelrow_agrees_990
private def atomic_991 : CountSegment 1 10 13120 990 991 549 550 := CountSegment.single (first := 990) (initial := 549) row_count_991 (by decide)
private theorem atomic_991_agreement : RowAgreement 990 991 := RowAgreement.single (first := 990) modelrow_agrees_991
private def atomic_992 : CountSegment 1 10 13120 991 992 550 550 := CountSegment.single (first := 991) (initial := 550) row_count_992 (by decide)
private theorem atomic_992_agreement : RowAgreement 991 992 := RowAgreement.single (first := 991) modelrow_agrees_992
private def atomic_993 : CountSegment 1 10 13120 992 993 550 551 := CountSegment.single (first := 992) (initial := 550) row_count_993 (by decide)
private theorem atomic_993_agreement : RowAgreement 992 993 := RowAgreement.single (first := 992) modelrow_agrees_993
private def atomic_994 : CountSegment 1 10 13120 993 994 551 551 := CountSegment.single (first := 993) (initial := 551) row_count_994 (by decide)
private theorem atomic_994_agreement : RowAgreement 993 994 := RowAgreement.single (first := 993) modelrow_agrees_994
private def atomic_995 : CountSegment 1 10 13120 994 995 551 552 := CountSegment.single (first := 994) (initial := 551) row_count_995 (by decide)
private theorem atomic_995_agreement : RowAgreement 994 995 := RowAgreement.single (first := 994) modelrow_agrees_995
private def atomic_996 : CountSegment 1 10 13120 995 996 552 552 := CountSegment.single (first := 995) (initial := 552) row_count_996 (by decide)
private theorem atomic_996_agreement : RowAgreement 995 996 := RowAgreement.single (first := 995) modelrow_agrees_996
private def atomic_997 : CountSegment 1 10 13120 996 997 552 553 := CountSegment.single (first := 996) (initial := 552) row_count_997 (by decide)
private theorem atomic_997_agreement : RowAgreement 996 997 := RowAgreement.single (first := 996) modelrow_agrees_997
private def atomic_998 : CountSegment 1 10 13120 997 998 553 553 := CountSegment.single (first := 997) (initial := 553) row_count_998 (by decide)
private theorem atomic_998_agreement : RowAgreement 997 998 := RowAgreement.single (first := 997) modelrow_agrees_998
private def atomic_999 : CountSegment 1 10 13120 998 999 553 553 := CountSegment.single (first := 998) (initial := 553) row_count_999 (by decide)
private theorem atomic_999_agreement : RowAgreement 998 999 := RowAgreement.single (first := 998) modelrow_agrees_999
private def atomic_1000 : CountSegment 1 10 13120 999 1000 553 554 := CountSegment.single (first := 999) (initial := 553) row_count_1000 (by decide)
private theorem atomic_1000_agreement : RowAgreement 999 1000 := RowAgreement.single (first := 999) modelrow_agrees_1000
private def atomic_1001 : CountSegment 1 10 13120 1000 1001 554 554 := CountSegment.single (first := 1000) (initial := 554) row_count_1001 (by decide)
private theorem atomic_1001_agreement : RowAgreement 1000 1001 := RowAgreement.single (first := 1000) modelrow_agrees_1001
private def atomic_1002 : CountSegment 1 10 13120 1001 1002 554 554 := CountSegment.single (first := 1001) (initial := 554) row_count_1002 (by decide)
private theorem atomic_1002_agreement : RowAgreement 1001 1002 := RowAgreement.single (first := 1001) modelrow_agrees_1002
private def atomic_1003 : CountSegment 1 10 13120 1002 1003 554 555 := CountSegment.single (first := 1002) (initial := 554) row_count_1003 (by decide)
private theorem atomic_1003_agreement : RowAgreement 1002 1003 := RowAgreement.single (first := 1002) modelrow_agrees_1003
private def atomic_1004 : CountSegment 1 10 13120 1003 1004 555 555 := CountSegment.single (first := 1003) (initial := 555) row_count_1004 (by decide)
private theorem atomic_1004_agreement : RowAgreement 1003 1004 := RowAgreement.single (first := 1003) modelrow_agrees_1004
private def atomic_1005 : CountSegment 1 10 13120 1004 1005 555 556 := CountSegment.single (first := 1004) (initial := 555) row_count_1005 (by decide)
private theorem atomic_1005_agreement : RowAgreement 1004 1005 := RowAgreement.single (first := 1004) modelrow_agrees_1005
private def atomic_1006 : CountSegment 1 10 13120 1005 1006 556 556 := CountSegment.single (first := 1005) (initial := 556) row_count_1006 (by decide)
private theorem atomic_1006_agreement : RowAgreement 1005 1006 := RowAgreement.single (first := 1005) modelrow_agrees_1006
private def atomic_1007 : CountSegment 1 10 13120 1006 1007 556 556 := CountSegment.single (first := 1006) (initial := 556) row_count_1007 (by decide)
private theorem atomic_1007_agreement : RowAgreement 1006 1007 := RowAgreement.single (first := 1006) modelrow_agrees_1007
private def atomic_1008 : CountSegment 1 10 13120 1007 1008 556 558 := CountSegment.single (first := 1007) (initial := 556) row_count_1008 (by decide)
private theorem atomic_1008_agreement : RowAgreement 1007 1008 := RowAgreement.single (first := 1007) modelrow_agrees_1008
private def atomic_1009 : CountSegment 1 10 13120 1008 1009 558 558 := CountSegment.single (first := 1008) (initial := 558) row_count_1009 (by decide)
private theorem atomic_1009_agreement : RowAgreement 1008 1009 := RowAgreement.single (first := 1008) modelrow_agrees_1009
private def atomic_1010 : CountSegment 1 10 13120 1009 1010 558 558 := CountSegment.single (first := 1009) (initial := 558) row_count_1010 (by decide)
private theorem atomic_1010_agreement : RowAgreement 1009 1010 := RowAgreement.single (first := 1009) modelrow_agrees_1010
private def atomic_1011 : CountSegment 1 10 13120 1010 1011 558 559 := CountSegment.single (first := 1010) (initial := 558) row_count_1011 (by decide)
private theorem atomic_1011_agreement : RowAgreement 1010 1011 := RowAgreement.single (first := 1010) modelrow_agrees_1011
private def atomic_1012 : CountSegment 1 10 13120 1011 1012 559 559 := CountSegment.single (first := 1011) (initial := 559) row_count_1012 (by decide)
private theorem atomic_1012_agreement : RowAgreement 1011 1012 := RowAgreement.single (first := 1011) modelrow_agrees_1012
private def atomic_1013 : CountSegment 1 10 13120 1012 1013 559 560 := CountSegment.single (first := 1012) (initial := 559) row_count_1013 (by decide)
private theorem atomic_1013_agreement : RowAgreement 1012 1013 := RowAgreement.single (first := 1012) modelrow_agrees_1013
private def atomic_1014 : CountSegment 1 10 13120 1013 1014 560 560 := CountSegment.single (first := 1013) (initial := 560) row_count_1014 (by decide)
private theorem atomic_1014_agreement : RowAgreement 1013 1014 := RowAgreement.single (first := 1013) modelrow_agrees_1014
private def atomic_1015 : CountSegment 1 10 13120 1014 1015 560 560 := CountSegment.single (first := 1014) (initial := 560) row_count_1015 (by decide)
private theorem atomic_1015_agreement : RowAgreement 1014 1015 := RowAgreement.single (first := 1014) modelrow_agrees_1015
private def atomic_1016 : CountSegment 1 10 13120 1015 1016 560 560 := CountSegment.single (first := 1015) (initial := 560) row_count_1016 (by decide)
private theorem atomic_1016_agreement : RowAgreement 1015 1016 := RowAgreement.single (first := 1015) modelrow_agrees_1016
private def atomic_1017 : CountSegment 1 10 13120 1016 1017 560 562 := CountSegment.single (first := 1016) (initial := 560) row_count_1017 (by decide)
private theorem atomic_1017_agreement : RowAgreement 1016 1017 := RowAgreement.single (first := 1016) modelrow_agrees_1017
private def atomic_1018 : CountSegment 1 10 13120 1017 1018 562 562 := CountSegment.single (first := 1017) (initial := 562) row_count_1018 (by decide)
private theorem atomic_1018_agreement : RowAgreement 1017 1018 := RowAgreement.single (first := 1017) modelrow_agrees_1018
private def atomic_1019 : CountSegment 1 10 13120 1018 1019 562 562 := CountSegment.single (first := 1018) (initial := 562) row_count_1019 (by decide)
private theorem atomic_1019_agreement : RowAgreement 1018 1019 := RowAgreement.single (first := 1018) modelrow_agrees_1019
private def atomic_1020 : CountSegment 1 10 13120 1019 1020 562 562 := CountSegment.single (first := 1019) (initial := 562) row_count_1020 (by decide)
private theorem atomic_1020_agreement : RowAgreement 1019 1020 := RowAgreement.single (first := 1019) modelrow_agrees_1020
private def atomic_1021 : CountSegment 1 10 13120 1020 1021 562 562 := CountSegment.single (first := 1020) (initial := 562) row_count_1021 (by decide)
private theorem atomic_1021_agreement : RowAgreement 1020 1021 := RowAgreement.single (first := 1020) modelrow_agrees_1021
private def atomic_1022 : CountSegment 1 10 13120 1021 1022 562 563 := CountSegment.single (first := 1021) (initial := 562) row_count_1022 (by decide)
private theorem atomic_1022_agreement : RowAgreement 1021 1022 := RowAgreement.single (first := 1021) modelrow_agrees_1022
private def atomic_1023 : CountSegment 1 10 13120 1022 1023 563 563 := CountSegment.single (first := 1022) (initial := 563) row_count_1023 (by decide)
private theorem atomic_1023_agreement : RowAgreement 1022 1023 := RowAgreement.single (first := 1022) modelrow_agrees_1023
private def atomic_1024 : CountSegment 1 10 13120 1023 1024 563 563 := CountSegment.single (first := 1023) (initial := 563) row_count_1024 (by decide)
private theorem atomic_1024_agreement : RowAgreement 1023 1024 := RowAgreement.single (first := 1023) modelrow_agrees_1024
private def within_007_000 : CountSegment 1 10 13120 896 898 507 508 := CountSegment.join atomic_897 atomic_898
private theorem within_007_000_agreement : RowAgreement 896 898 := RowAgreement.join atomic_897_agreement atomic_898_agreement
private def within_007_001 : CountSegment 1 10 13120 898 900 508 509 := CountSegment.join atomic_899 atomic_900
private theorem within_007_001_agreement : RowAgreement 898 900 := RowAgreement.join atomic_899_agreement atomic_900_agreement
private def within_007_002 : CountSegment 1 10 13120 896 900 507 509 := CountSegment.join within_007_000 within_007_001
private theorem within_007_002_agreement : RowAgreement 896 900 := RowAgreement.join within_007_000_agreement within_007_001_agreement
private def within_007_003 : CountSegment 1 10 13120 900 902 509 509 := CountSegment.join atomic_901 atomic_902
private theorem within_007_003_agreement : RowAgreement 900 902 := RowAgreement.join atomic_901_agreement atomic_902_agreement
private def within_007_004 : CountSegment 1 10 13120 902 904 509 510 := CountSegment.join atomic_903 atomic_904
private theorem within_007_004_agreement : RowAgreement 902 904 := RowAgreement.join atomic_903_agreement atomic_904_agreement
private def within_007_005 : CountSegment 1 10 13120 900 904 509 510 := CountSegment.join within_007_003 within_007_004
private theorem within_007_005_agreement : RowAgreement 900 904 := RowAgreement.join within_007_003_agreement within_007_004_agreement
private def within_007_006 : CountSegment 1 10 13120 896 904 507 510 := CountSegment.join within_007_002 within_007_005
private theorem within_007_006_agreement : RowAgreement 896 904 := RowAgreement.join within_007_002_agreement within_007_005_agreement
private def within_007_007 : CountSegment 1 10 13120 904 906 510 512 := CountSegment.join atomic_905 atomic_906
private theorem within_007_007_agreement : RowAgreement 904 906 := RowAgreement.join atomic_905_agreement atomic_906_agreement
private def within_007_008 : CountSegment 1 10 13120 906 908 512 514 := CountSegment.join atomic_907 atomic_908
private theorem within_007_008_agreement : RowAgreement 906 908 := RowAgreement.join atomic_907_agreement atomic_908_agreement
private def within_007_009 : CountSegment 1 10 13120 904 908 510 514 := CountSegment.join within_007_007 within_007_008
private theorem within_007_009_agreement : RowAgreement 904 908 := RowAgreement.join within_007_007_agreement within_007_008_agreement
private def within_007_010 : CountSegment 1 10 13120 908 910 514 515 := CountSegment.join atomic_909 atomic_910
private theorem within_007_010_agreement : RowAgreement 908 910 := RowAgreement.join atomic_909_agreement atomic_910_agreement
private def within_007_011 : CountSegment 1 10 13120 910 912 515 516 := CountSegment.join atomic_911 atomic_912
private theorem within_007_011_agreement : RowAgreement 910 912 := RowAgreement.join atomic_911_agreement atomic_912_agreement
private def within_007_012 : CountSegment 1 10 13120 908 912 514 516 := CountSegment.join within_007_010 within_007_011
private theorem within_007_012_agreement : RowAgreement 908 912 := RowAgreement.join within_007_010_agreement within_007_011_agreement
private def within_007_013 : CountSegment 1 10 13120 904 912 510 516 := CountSegment.join within_007_009 within_007_012
private theorem within_007_013_agreement : RowAgreement 904 912 := RowAgreement.join within_007_009_agreement within_007_012_agreement
private def within_007_014 : CountSegment 1 10 13120 896 912 507 516 := CountSegment.join within_007_006 within_007_013
private theorem within_007_014_agreement : RowAgreement 896 912 := RowAgreement.join within_007_006_agreement within_007_013_agreement
private def within_007_015 : CountSegment 1 10 13120 912 914 516 517 := CountSegment.join atomic_913 atomic_914
private theorem within_007_015_agreement : RowAgreement 912 914 := RowAgreement.join atomic_913_agreement atomic_914_agreement
private def within_007_016 : CountSegment 1 10 13120 914 916 517 518 := CountSegment.join atomic_915 atomic_916
private theorem within_007_016_agreement : RowAgreement 914 916 := RowAgreement.join atomic_915_agreement atomic_916_agreement
private def within_007_017 : CountSegment 1 10 13120 912 916 516 518 := CountSegment.join within_007_015 within_007_016
private theorem within_007_017_agreement : RowAgreement 912 916 := RowAgreement.join within_007_015_agreement within_007_016_agreement
private def within_007_018 : CountSegment 1 10 13120 916 918 518 518 := CountSegment.join atomic_917 atomic_918
private theorem within_007_018_agreement : RowAgreement 916 918 := RowAgreement.join atomic_917_agreement atomic_918_agreement
private def within_007_019 : CountSegment 1 10 13120 918 920 518 518 := CountSegment.join atomic_919 atomic_920
private theorem within_007_019_agreement : RowAgreement 918 920 := RowAgreement.join atomic_919_agreement atomic_920_agreement
private def within_007_020 : CountSegment 1 10 13120 916 920 518 518 := CountSegment.join within_007_018 within_007_019
private theorem within_007_020_agreement : RowAgreement 916 920 := RowAgreement.join within_007_018_agreement within_007_019_agreement
private def within_007_021 : CountSegment 1 10 13120 912 920 516 518 := CountSegment.join within_007_017 within_007_020
private theorem within_007_021_agreement : RowAgreement 912 920 := RowAgreement.join within_007_017_agreement within_007_020_agreement
private def within_007_022 : CountSegment 1 10 13120 920 922 518 519 := CountSegment.join atomic_921 atomic_922
private theorem within_007_022_agreement : RowAgreement 920 922 := RowAgreement.join atomic_921_agreement atomic_922_agreement
private def within_007_023 : CountSegment 1 10 13120 922 924 519 521 := CountSegment.join atomic_923 atomic_924
private theorem within_007_023_agreement : RowAgreement 922 924 := RowAgreement.join atomic_923_agreement atomic_924_agreement
private def within_007_024 : CountSegment 1 10 13120 920 924 518 521 := CountSegment.join within_007_022 within_007_023
private theorem within_007_024_agreement : RowAgreement 920 924 := RowAgreement.join within_007_022_agreement within_007_023_agreement
private def within_007_025 : CountSegment 1 10 13120 924 926 521 522 := CountSegment.join atomic_925 atomic_926
private theorem within_007_025_agreement : RowAgreement 924 926 := RowAgreement.join atomic_925_agreement atomic_926_agreement
private def within_007_026 : CountSegment 1 10 13120 926 928 522 523 := CountSegment.join atomic_927 atomic_928
private theorem within_007_026_agreement : RowAgreement 926 928 := RowAgreement.join atomic_927_agreement atomic_928_agreement
private def within_007_027 : CountSegment 1 10 13120 924 928 521 523 := CountSegment.join within_007_025 within_007_026
private theorem within_007_027_agreement : RowAgreement 924 928 := RowAgreement.join within_007_025_agreement within_007_026_agreement
private def within_007_028 : CountSegment 1 10 13120 920 928 518 523 := CountSegment.join within_007_024 within_007_027
private theorem within_007_028_agreement : RowAgreement 920 928 := RowAgreement.join within_007_024_agreement within_007_027_agreement
private def within_007_029 : CountSegment 1 10 13120 912 928 516 523 := CountSegment.join within_007_021 within_007_028
private theorem within_007_029_agreement : RowAgreement 912 928 := RowAgreement.join within_007_021_agreement within_007_028_agreement
private def within_007_030 : CountSegment 1 10 13120 896 928 507 523 := CountSegment.join within_007_014 within_007_029
private theorem within_007_030_agreement : RowAgreement 896 928 := RowAgreement.join within_007_014_agreement within_007_029_agreement
private def within_007_031 : CountSegment 1 10 13120 928 930 523 524 := CountSegment.join atomic_929 atomic_930
private theorem within_007_031_agreement : RowAgreement 928 930 := RowAgreement.join atomic_929_agreement atomic_930_agreement
private def within_007_032 : CountSegment 1 10 13120 930 932 524 525 := CountSegment.join atomic_931 atomic_932
private theorem within_007_032_agreement : RowAgreement 930 932 := RowAgreement.join atomic_931_agreement atomic_932_agreement
private def within_007_033 : CountSegment 1 10 13120 928 932 523 525 := CountSegment.join within_007_031 within_007_032
private theorem within_007_033_agreement : RowAgreement 928 932 := RowAgreement.join within_007_031_agreement within_007_032_agreement
private def within_007_034 : CountSegment 1 10 13120 932 934 525 525 := CountSegment.join atomic_933 atomic_934
private theorem within_007_034_agreement : RowAgreement 932 934 := RowAgreement.join atomic_933_agreement atomic_934_agreement
private def within_007_035 : CountSegment 1 10 13120 934 936 525 526 := CountSegment.join atomic_935 atomic_936
private theorem within_007_035_agreement : RowAgreement 934 936 := RowAgreement.join atomic_935_agreement atomic_936_agreement
private def within_007_036 : CountSegment 1 10 13120 932 936 525 526 := CountSegment.join within_007_034 within_007_035
private theorem within_007_036_agreement : RowAgreement 932 936 := RowAgreement.join within_007_034_agreement within_007_035_agreement
private def within_007_037 : CountSegment 1 10 13120 928 936 523 526 := CountSegment.join within_007_033 within_007_036
private theorem within_007_037_agreement : RowAgreement 928 936 := RowAgreement.join within_007_033_agreement within_007_036_agreement
private def within_007_038 : CountSegment 1 10 13120 936 938 526 527 := CountSegment.join atomic_937 atomic_938
private theorem within_007_038_agreement : RowAgreement 936 938 := RowAgreement.join atomic_937_agreement atomic_938_agreement
private def within_007_039 : CountSegment 1 10 13120 938 940 527 527 := CountSegment.join atomic_939 atomic_940
private theorem within_007_039_agreement : RowAgreement 938 940 := RowAgreement.join atomic_939_agreement atomic_940_agreement
private def within_007_040 : CountSegment 1 10 13120 936 940 526 527 := CountSegment.join within_007_038 within_007_039
private theorem within_007_040_agreement : RowAgreement 936 940 := RowAgreement.join within_007_038_agreement within_007_039_agreement
private def within_007_041 : CountSegment 1 10 13120 940 942 527 528 := CountSegment.join atomic_941 atomic_942
private theorem within_007_041_agreement : RowAgreement 940 942 := RowAgreement.join atomic_941_agreement atomic_942_agreement
private def within_007_042 : CountSegment 1 10 13120 942 944 528 529 := CountSegment.join atomic_943 atomic_944
private theorem within_007_042_agreement : RowAgreement 942 944 := RowAgreement.join atomic_943_agreement atomic_944_agreement
private def within_007_043 : CountSegment 1 10 13120 940 944 527 529 := CountSegment.join within_007_041 within_007_042
private theorem within_007_043_agreement : RowAgreement 940 944 := RowAgreement.join within_007_041_agreement within_007_042_agreement
private def within_007_044 : CountSegment 1 10 13120 936 944 526 529 := CountSegment.join within_007_040 within_007_043
private theorem within_007_044_agreement : RowAgreement 936 944 := RowAgreement.join within_007_040_agreement within_007_043_agreement
private def within_007_045 : CountSegment 1 10 13120 928 944 523 529 := CountSegment.join within_007_037 within_007_044
private theorem within_007_045_agreement : RowAgreement 928 944 := RowAgreement.join within_007_037_agreement within_007_044_agreement
private def within_007_046 : CountSegment 1 10 13120 944 946 529 529 := CountSegment.join atomic_945 atomic_946
private theorem within_007_046_agreement : RowAgreement 944 946 := RowAgreement.join atomic_945_agreement atomic_946_agreement
private def within_007_047 : CountSegment 1 10 13120 946 948 529 531 := CountSegment.join atomic_947 atomic_948
private theorem within_007_047_agreement : RowAgreement 946 948 := RowAgreement.join atomic_947_agreement atomic_948_agreement
private def within_007_048 : CountSegment 1 10 13120 944 948 529 531 := CountSegment.join within_007_046 within_007_047
private theorem within_007_048_agreement : RowAgreement 944 948 := RowAgreement.join within_007_046_agreement within_007_047_agreement
private def within_007_049 : CountSegment 1 10 13120 948 950 531 531 := CountSegment.join atomic_949 atomic_950
private theorem within_007_049_agreement : RowAgreement 948 950 := RowAgreement.join atomic_949_agreement atomic_950_agreement
private def within_007_050 : CountSegment 1 10 13120 950 952 531 532 := CountSegment.join atomic_951 atomic_952
private theorem within_007_050_agreement : RowAgreement 950 952 := RowAgreement.join atomic_951_agreement atomic_952_agreement
private def within_007_051 : CountSegment 1 10 13120 948 952 531 532 := CountSegment.join within_007_049 within_007_050
private theorem within_007_051_agreement : RowAgreement 948 952 := RowAgreement.join within_007_049_agreement within_007_050_agreement
private def within_007_052 : CountSegment 1 10 13120 944 952 529 532 := CountSegment.join within_007_048 within_007_051
private theorem within_007_052_agreement : RowAgreement 944 952 := RowAgreement.join within_007_048_agreement within_007_051_agreement
private def within_007_053 : CountSegment 1 10 13120 952 954 532 532 := CountSegment.join atomic_953 atomic_954
private theorem within_007_053_agreement : RowAgreement 952 954 := RowAgreement.join atomic_953_agreement atomic_954_agreement
private def within_007_054 : CountSegment 1 10 13120 954 956 532 533 := CountSegment.join atomic_955 atomic_956
private theorem within_007_054_agreement : RowAgreement 954 956 := RowAgreement.join atomic_955_agreement atomic_956_agreement
private def within_007_055 : CountSegment 1 10 13120 952 956 532 533 := CountSegment.join within_007_053 within_007_054
private theorem within_007_055_agreement : RowAgreement 952 956 := RowAgreement.join within_007_053_agreement within_007_054_agreement
private def within_007_056 : CountSegment 1 10 13120 956 958 533 533 := CountSegment.join atomic_957 atomic_958
private theorem within_007_056_agreement : RowAgreement 956 958 := RowAgreement.join atomic_957_agreement atomic_958_agreement
private def within_007_057 : CountSegment 1 10 13120 958 960 533 534 := CountSegment.join atomic_959 atomic_960
private theorem within_007_057_agreement : RowAgreement 958 960 := RowAgreement.join atomic_959_agreement atomic_960_agreement
private def within_007_058 : CountSegment 1 10 13120 956 960 533 534 := CountSegment.join within_007_056 within_007_057
private theorem within_007_058_agreement : RowAgreement 956 960 := RowAgreement.join within_007_056_agreement within_007_057_agreement
private def within_007_059 : CountSegment 1 10 13120 952 960 532 534 := CountSegment.join within_007_055 within_007_058
private theorem within_007_059_agreement : RowAgreement 952 960 := RowAgreement.join within_007_055_agreement within_007_058_agreement
private def within_007_060 : CountSegment 1 10 13120 944 960 529 534 := CountSegment.join within_007_052 within_007_059
private theorem within_007_060_agreement : RowAgreement 944 960 := RowAgreement.join within_007_052_agreement within_007_059_agreement
private def within_007_061 : CountSegment 1 10 13120 928 960 523 534 := CountSegment.join within_007_045 within_007_060
private theorem within_007_061_agreement : RowAgreement 928 960 := RowAgreement.join within_007_045_agreement within_007_060_agreement
private def within_007_062 : CountSegment 1 10 13120 896 960 507 534 := CountSegment.join within_007_030 within_007_061
private theorem within_007_062_agreement : RowAgreement 896 960 := RowAgreement.join within_007_030_agreement within_007_061_agreement
private def within_007_063 : CountSegment 1 10 13120 960 962 534 535 := CountSegment.join atomic_961 atomic_962
private theorem within_007_063_agreement : RowAgreement 960 962 := RowAgreement.join atomic_961_agreement atomic_962_agreement
private def within_007_064 : CountSegment 1 10 13120 962 964 535 535 := CountSegment.join atomic_963 atomic_964
private theorem within_007_064_agreement : RowAgreement 962 964 := RowAgreement.join atomic_963_agreement atomic_964_agreement
private def within_007_065 : CountSegment 1 10 13120 960 964 534 535 := CountSegment.join within_007_063 within_007_064
private theorem within_007_065_agreement : RowAgreement 960 964 := RowAgreement.join within_007_063_agreement within_007_064_agreement
private def within_007_066 : CountSegment 1 10 13120 964 966 535 537 := CountSegment.join atomic_965 atomic_966
private theorem within_007_066_agreement : RowAgreement 964 966 := RowAgreement.join atomic_965_agreement atomic_966_agreement
private def within_007_067 : CountSegment 1 10 13120 966 968 537 537 := CountSegment.join atomic_967 atomic_968
private theorem within_007_067_agreement : RowAgreement 966 968 := RowAgreement.join atomic_967_agreement atomic_968_agreement
private def within_007_068 : CountSegment 1 10 13120 964 968 535 537 := CountSegment.join within_007_066 within_007_067
private theorem within_007_068_agreement : RowAgreement 964 968 := RowAgreement.join within_007_066_agreement within_007_067_agreement
private def within_007_069 : CountSegment 1 10 13120 960 968 534 537 := CountSegment.join within_007_065 within_007_068
private theorem within_007_069_agreement : RowAgreement 960 968 := RowAgreement.join within_007_065_agreement within_007_068_agreement
private def within_007_070 : CountSegment 1 10 13120 968 970 537 538 := CountSegment.join atomic_969 atomic_970
private theorem within_007_070_agreement : RowAgreement 968 970 := RowAgreement.join atomic_969_agreement atomic_970_agreement
private def within_007_071 : CountSegment 1 10 13120 970 972 538 538 := CountSegment.join atomic_971 atomic_972
private theorem within_007_071_agreement : RowAgreement 970 972 := RowAgreement.join atomic_971_agreement atomic_972_agreement
private def within_007_072 : CountSegment 1 10 13120 968 972 537 538 := CountSegment.join within_007_070 within_007_071
private theorem within_007_072_agreement : RowAgreement 968 972 := RowAgreement.join within_007_070_agreement within_007_071_agreement
private def within_007_073 : CountSegment 1 10 13120 972 974 538 538 := CountSegment.join atomic_973 atomic_974
private theorem within_007_073_agreement : RowAgreement 972 974 := RowAgreement.join atomic_973_agreement atomic_974_agreement
private def within_007_074 : CountSegment 1 10 13120 974 976 538 539 := CountSegment.join atomic_975 atomic_976
private theorem within_007_074_agreement : RowAgreement 974 976 := RowAgreement.join atomic_975_agreement atomic_976_agreement
private def within_007_075 : CountSegment 1 10 13120 972 976 538 539 := CountSegment.join within_007_073 within_007_074
private theorem within_007_075_agreement : RowAgreement 972 976 := RowAgreement.join within_007_073_agreement within_007_074_agreement
private def within_007_076 : CountSegment 1 10 13120 968 976 537 539 := CountSegment.join within_007_072 within_007_075
private theorem within_007_076_agreement : RowAgreement 968 976 := RowAgreement.join within_007_072_agreement within_007_075_agreement
private def within_007_077 : CountSegment 1 10 13120 960 976 534 539 := CountSegment.join within_007_069 within_007_076
private theorem within_007_077_agreement : RowAgreement 960 976 := RowAgreement.join within_007_069_agreement within_007_076_agreement
private def within_007_078 : CountSegment 1 10 13120 976 978 539 540 := CountSegment.join atomic_977 atomic_978
private theorem within_007_078_agreement : RowAgreement 976 978 := RowAgreement.join atomic_977_agreement atomic_978_agreement
private def within_007_079 : CountSegment 1 10 13120 978 980 540 540 := CountSegment.join atomic_979 atomic_980
private theorem within_007_079_agreement : RowAgreement 978 980 := RowAgreement.join atomic_979_agreement atomic_980_agreement
private def within_007_080 : CountSegment 1 10 13120 976 980 539 540 := CountSegment.join within_007_078 within_007_079
private theorem within_007_080_agreement : RowAgreement 976 980 := RowAgreement.join within_007_078_agreement within_007_079_agreement
private def within_007_081 : CountSegment 1 10 13120 980 982 540 541 := CountSegment.join atomic_981 atomic_982
private theorem within_007_081_agreement : RowAgreement 980 982 := RowAgreement.join atomic_981_agreement atomic_982_agreement
private def within_007_082 : CountSegment 1 10 13120 982 984 541 542 := CountSegment.join atomic_983 atomic_984
private theorem within_007_082_agreement : RowAgreement 982 984 := RowAgreement.join atomic_983_agreement atomic_984_agreement
private def within_007_083 : CountSegment 1 10 13120 980 984 540 542 := CountSegment.join within_007_081 within_007_082
private theorem within_007_083_agreement : RowAgreement 980 984 := RowAgreement.join within_007_081_agreement within_007_082_agreement
private def within_007_084 : CountSegment 1 10 13120 976 984 539 542 := CountSegment.join within_007_080 within_007_083
private theorem within_007_084_agreement : RowAgreement 976 984 := RowAgreement.join within_007_080_agreement within_007_083_agreement
private def within_007_085 : CountSegment 1 10 13120 984 986 542 544 := CountSegment.join atomic_985 atomic_986
private theorem within_007_085_agreement : RowAgreement 984 986 := RowAgreement.join atomic_985_agreement atomic_986_agreement
private def within_007_086 : CountSegment 1 10 13120 986 988 544 547 := CountSegment.join atomic_987 atomic_988
private theorem within_007_086_agreement : RowAgreement 986 988 := RowAgreement.join atomic_987_agreement atomic_988_agreement
private def within_007_087 : CountSegment 1 10 13120 984 988 542 547 := CountSegment.join within_007_085 within_007_086
private theorem within_007_087_agreement : RowAgreement 984 988 := RowAgreement.join within_007_085_agreement within_007_086_agreement
private def within_007_088 : CountSegment 1 10 13120 988 990 547 549 := CountSegment.join atomic_989 atomic_990
private theorem within_007_088_agreement : RowAgreement 988 990 := RowAgreement.join atomic_989_agreement atomic_990_agreement
private def within_007_089 : CountSegment 1 10 13120 990 992 549 550 := CountSegment.join atomic_991 atomic_992
private theorem within_007_089_agreement : RowAgreement 990 992 := RowAgreement.join atomic_991_agreement atomic_992_agreement
private def within_007_090 : CountSegment 1 10 13120 988 992 547 550 := CountSegment.join within_007_088 within_007_089
private theorem within_007_090_agreement : RowAgreement 988 992 := RowAgreement.join within_007_088_agreement within_007_089_agreement
private def within_007_091 : CountSegment 1 10 13120 984 992 542 550 := CountSegment.join within_007_087 within_007_090
private theorem within_007_091_agreement : RowAgreement 984 992 := RowAgreement.join within_007_087_agreement within_007_090_agreement
private def within_007_092 : CountSegment 1 10 13120 976 992 539 550 := CountSegment.join within_007_084 within_007_091
private theorem within_007_092_agreement : RowAgreement 976 992 := RowAgreement.join within_007_084_agreement within_007_091_agreement
private def within_007_093 : CountSegment 1 10 13120 960 992 534 550 := CountSegment.join within_007_077 within_007_092
private theorem within_007_093_agreement : RowAgreement 960 992 := RowAgreement.join within_007_077_agreement within_007_092_agreement
private def within_007_094 : CountSegment 1 10 13120 992 994 550 551 := CountSegment.join atomic_993 atomic_994
private theorem within_007_094_agreement : RowAgreement 992 994 := RowAgreement.join atomic_993_agreement atomic_994_agreement
private def within_007_095 : CountSegment 1 10 13120 994 996 551 552 := CountSegment.join atomic_995 atomic_996
private theorem within_007_095_agreement : RowAgreement 994 996 := RowAgreement.join atomic_995_agreement atomic_996_agreement
private def within_007_096 : CountSegment 1 10 13120 992 996 550 552 := CountSegment.join within_007_094 within_007_095
private theorem within_007_096_agreement : RowAgreement 992 996 := RowAgreement.join within_007_094_agreement within_007_095_agreement
private def within_007_097 : CountSegment 1 10 13120 996 998 552 553 := CountSegment.join atomic_997 atomic_998
private theorem within_007_097_agreement : RowAgreement 996 998 := RowAgreement.join atomic_997_agreement atomic_998_agreement
private def within_007_098 : CountSegment 1 10 13120 998 1000 553 554 := CountSegment.join atomic_999 atomic_1000
private theorem within_007_098_agreement : RowAgreement 998 1000 := RowAgreement.join atomic_999_agreement atomic_1000_agreement
private def within_007_099 : CountSegment 1 10 13120 996 1000 552 554 := CountSegment.join within_007_097 within_007_098
private theorem within_007_099_agreement : RowAgreement 996 1000 := RowAgreement.join within_007_097_agreement within_007_098_agreement
private def within_007_100 : CountSegment 1 10 13120 992 1000 550 554 := CountSegment.join within_007_096 within_007_099
private theorem within_007_100_agreement : RowAgreement 992 1000 := RowAgreement.join within_007_096_agreement within_007_099_agreement
private def within_007_101 : CountSegment 1 10 13120 1000 1002 554 554 := CountSegment.join atomic_1001 atomic_1002
private theorem within_007_101_agreement : RowAgreement 1000 1002 := RowAgreement.join atomic_1001_agreement atomic_1002_agreement
private def within_007_102 : CountSegment 1 10 13120 1002 1004 554 555 := CountSegment.join atomic_1003 atomic_1004
private theorem within_007_102_agreement : RowAgreement 1002 1004 := RowAgreement.join atomic_1003_agreement atomic_1004_agreement
private def within_007_103 : CountSegment 1 10 13120 1000 1004 554 555 := CountSegment.join within_007_101 within_007_102
private theorem within_007_103_agreement : RowAgreement 1000 1004 := RowAgreement.join within_007_101_agreement within_007_102_agreement
private def within_007_104 : CountSegment 1 10 13120 1004 1006 555 556 := CountSegment.join atomic_1005 atomic_1006
private theorem within_007_104_agreement : RowAgreement 1004 1006 := RowAgreement.join atomic_1005_agreement atomic_1006_agreement
private def within_007_105 : CountSegment 1 10 13120 1006 1008 556 558 := CountSegment.join atomic_1007 atomic_1008
private theorem within_007_105_agreement : RowAgreement 1006 1008 := RowAgreement.join atomic_1007_agreement atomic_1008_agreement
private def within_007_106 : CountSegment 1 10 13120 1004 1008 555 558 := CountSegment.join within_007_104 within_007_105
private theorem within_007_106_agreement : RowAgreement 1004 1008 := RowAgreement.join within_007_104_agreement within_007_105_agreement
private def within_007_107 : CountSegment 1 10 13120 1000 1008 554 558 := CountSegment.join within_007_103 within_007_106
private theorem within_007_107_agreement : RowAgreement 1000 1008 := RowAgreement.join within_007_103_agreement within_007_106_agreement
private def within_007_108 : CountSegment 1 10 13120 992 1008 550 558 := CountSegment.join within_007_100 within_007_107
private theorem within_007_108_agreement : RowAgreement 992 1008 := RowAgreement.join within_007_100_agreement within_007_107_agreement
private def within_007_109 : CountSegment 1 10 13120 1008 1010 558 558 := CountSegment.join atomic_1009 atomic_1010
private theorem within_007_109_agreement : RowAgreement 1008 1010 := RowAgreement.join atomic_1009_agreement atomic_1010_agreement
private def within_007_110 : CountSegment 1 10 13120 1010 1012 558 559 := CountSegment.join atomic_1011 atomic_1012
private theorem within_007_110_agreement : RowAgreement 1010 1012 := RowAgreement.join atomic_1011_agreement atomic_1012_agreement
private def within_007_111 : CountSegment 1 10 13120 1008 1012 558 559 := CountSegment.join within_007_109 within_007_110
private theorem within_007_111_agreement : RowAgreement 1008 1012 := RowAgreement.join within_007_109_agreement within_007_110_agreement
private def within_007_112 : CountSegment 1 10 13120 1012 1014 559 560 := CountSegment.join atomic_1013 atomic_1014
private theorem within_007_112_agreement : RowAgreement 1012 1014 := RowAgreement.join atomic_1013_agreement atomic_1014_agreement
private def within_007_113 : CountSegment 1 10 13120 1014 1016 560 560 := CountSegment.join atomic_1015 atomic_1016
private theorem within_007_113_agreement : RowAgreement 1014 1016 := RowAgreement.join atomic_1015_agreement atomic_1016_agreement
private def within_007_114 : CountSegment 1 10 13120 1012 1016 559 560 := CountSegment.join within_007_112 within_007_113
private theorem within_007_114_agreement : RowAgreement 1012 1016 := RowAgreement.join within_007_112_agreement within_007_113_agreement
private def within_007_115 : CountSegment 1 10 13120 1008 1016 558 560 := CountSegment.join within_007_111 within_007_114
private theorem within_007_115_agreement : RowAgreement 1008 1016 := RowAgreement.join within_007_111_agreement within_007_114_agreement
private def within_007_116 : CountSegment 1 10 13120 1016 1018 560 562 := CountSegment.join atomic_1017 atomic_1018
private theorem within_007_116_agreement : RowAgreement 1016 1018 := RowAgreement.join atomic_1017_agreement atomic_1018_agreement
private def within_007_117 : CountSegment 1 10 13120 1018 1020 562 562 := CountSegment.join atomic_1019 atomic_1020
private theorem within_007_117_agreement : RowAgreement 1018 1020 := RowAgreement.join atomic_1019_agreement atomic_1020_agreement
private def within_007_118 : CountSegment 1 10 13120 1016 1020 560 562 := CountSegment.join within_007_116 within_007_117
private theorem within_007_118_agreement : RowAgreement 1016 1020 := RowAgreement.join within_007_116_agreement within_007_117_agreement
private def within_007_119 : CountSegment 1 10 13120 1020 1022 562 563 := CountSegment.join atomic_1021 atomic_1022
private theorem within_007_119_agreement : RowAgreement 1020 1022 := RowAgreement.join atomic_1021_agreement atomic_1022_agreement
private def within_007_120 : CountSegment 1 10 13120 1022 1024 563 563 := CountSegment.join atomic_1023 atomic_1024
private theorem within_007_120_agreement : RowAgreement 1022 1024 := RowAgreement.join atomic_1023_agreement atomic_1024_agreement
private def within_007_121 : CountSegment 1 10 13120 1020 1024 562 563 := CountSegment.join within_007_119 within_007_120
private theorem within_007_121_agreement : RowAgreement 1020 1024 := RowAgreement.join within_007_119_agreement within_007_120_agreement
private def within_007_122 : CountSegment 1 10 13120 1016 1024 560 563 := CountSegment.join within_007_118 within_007_121
private theorem within_007_122_agreement : RowAgreement 1016 1024 := RowAgreement.join within_007_118_agreement within_007_121_agreement
private def within_007_123 : CountSegment 1 10 13120 1008 1024 558 563 := CountSegment.join within_007_115 within_007_122
private theorem within_007_123_agreement : RowAgreement 1008 1024 := RowAgreement.join within_007_115_agreement within_007_122_agreement
private def within_007_124 : CountSegment 1 10 13120 992 1024 550 563 := CountSegment.join within_007_108 within_007_123
private theorem within_007_124_agreement : RowAgreement 992 1024 := RowAgreement.join within_007_108_agreement within_007_123_agreement
private def within_007_125 : CountSegment 1 10 13120 960 1024 534 563 := CountSegment.join within_007_093 within_007_124
private theorem within_007_125_agreement : RowAgreement 960 1024 := RowAgreement.join within_007_093_agreement within_007_124_agreement
private def within_007_126 : CountSegment 1 10 13120 896 1024 507 563 := CountSegment.join within_007_062 within_007_125
private theorem within_007_126_agreement : RowAgreement 896 1024 := RowAgreement.join within_007_062_agreement within_007_125_agreement
def segment_007 : CountSegment 1 10 13120 896 1024 507 563 := within_007_126
theorem agreement_007 : RowAgreement 896 1024 := within_007_126_agreement
end ProjectEulerSolutions.P58Numerical
