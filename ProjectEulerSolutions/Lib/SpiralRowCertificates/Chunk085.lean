import ProjectEulerSolutions.Lib.SpiralPrimeSegments
import ProjectEulerSolutions.Lib.SpiralPrimeCertificates.All
import ProjectEulerSolutions.Lib.MillerRabin
import ProjectEulerSolutions.Lib.SpiralMillerAgreement
/-! Finite spiral row and counter certificates. Python execution is checked separately. -/
namespace ProjectEulerSolutions.P58Numerical
open SpiralPrimes
set_option Elab.async false
set_option maxRecDepth 4096
def row_10881 : RowCertificate 10881 where
  first := false
  first_correct := by
    have composite : ¬ Nat.Prime (corner 10881 1) := by change ¬ Nat.Prime 473606407; exact PrattCertificates.composite_of_product (factor := 19) (cofactor := 24926653) rfl (by decide) (by decide)
    exact decide_eq_false composite
  second := false
  second_correct := by
    have composite : ¬ Nat.Prime (corner 10881 2) := by change ¬ Nat.Prime 473584645; exact PrattCertificates.composite_of_product (factor := 5) (cofactor := 94716929) rfl (by decide) (by decide)
    exact decide_eq_false composite
  third := true
  third_correct := by
    have prime : Nat.Prime (corner 10881 3) := by change Nat.Prime 473562883; exact ProjectEulerSolutions.P58PrimeNodes.p_473562883.prime
    exact decide_eq_true prime
theorem row_count_10881 : row 10881 = 1 := (row_10881.checked).trans rfl
theorem model_count_10881 : modelRow 10881 = 1 := by decide
theorem modelrow_agrees_10881 : modelRow 10881 = row 10881 := model_count_10881.trans row_count_10881.symm
def row_10882 : RowCertificate 10882 where
  first := false
  first_correct := by
    have composite : ¬ Nat.Prime (corner 10882 1) := by change ¬ Nat.Prime 473693461; exact PrattCertificates.composite_of_product (factor := 43) (cofactor := 11016127) rfl (by decide) (by decide)
    exact decide_eq_false composite
  second := false
  second_correct := by
    have composite : ¬ Nat.Prime (corner 10882 2) := by change ¬ Nat.Prime 473671697; exact PrattCertificates.composite_of_product (factor := 17) (cofactor := 27863041) rfl (by decide) (by decide)
    exact decide_eq_false composite
  third := false
  third_correct := by
    have composite : ¬ Nat.Prime (corner 10882 3) := by change ¬ Nat.Prime 473649933; exact PrattCertificates.composite_of_product (factor := 3) (cofactor := 157883311) rfl (by decide) (by decide)
    exact decide_eq_false composite
theorem row_count_10882 : row 10882 = 0 := (row_10882.checked).trans rfl
theorem model_count_10882 : modelRow 10882 = 0 := by decide
theorem modelrow_agrees_10882 : modelRow 10882 = row 10882 := model_count_10882.trans row_count_10882.symm
def row_10883 : RowCertificate 10883 where
  first := false
  first_correct := by
    have composite : ¬ Nat.Prime (corner 10883 1) := by change ¬ Nat.Prime 473780523; exact PrattCertificates.composite_of_product (factor := 3) (cofactor := 157926841) rfl (by decide) (by decide)
    exact decide_eq_false composite
  second := false
  second_correct := by
    have composite : ¬ Nat.Prime (corner 10883 2) := by change ¬ Nat.Prime 473758757; exact PrattCertificates.composite_of_product (factor := 61) (cofactor := 7766537) rfl (by decide) (by decide)
    exact decide_eq_false composite
  third := false
  third_correct := by
    have composite : ¬ Nat.Prime (corner 10883 3) := by change ¬ Nat.Prime 473736991; exact PrattCertificates.composite_of_product (factor := 7) (cofactor := 67676713) rfl (by decide) (by decide)
    exact decide_eq_false composite
theorem row_count_10883 : row 10883 = 0 := (row_10883.checked).trans rfl
theorem model_count_10883 : modelRow 10883 = 0 := by decide
theorem modelrow_agrees_10883 : modelRow 10883 = row 10883 := model_count_10883.trans row_count_10883.symm
def row_10884 : RowCertificate 10884 where
  first := false
  first_correct := by
    have composite : ¬ Nat.Prime (corner 10884 1) := by change ¬ Nat.Prime 473867593; exact PrattCertificates.composite_of_product (factor := 181) (cofactor := 2618053) rfl (by decide) (by decide)
    exact decide_eq_false composite
  second := false
  second_correct := by
    have composite : ¬ Nat.Prime (corner 10884 2) := by change ¬ Nat.Prime 473845825; exact PrattCertificates.composite_of_product (factor := 5) (cofactor := 94769165) rfl (by decide) (by decide)
    exact decide_eq_false composite
  third := false
  third_correct := by
    have composite : ¬ Nat.Prime (corner 10884 3) := by change ¬ Nat.Prime 473824057; exact PrattCertificates.composite_of_product (factor := 7) (cofactor := 67689151) rfl (by decide) (by decide)
    exact decide_eq_false composite
theorem row_count_10884 : row 10884 = 0 := (row_10884.checked).trans rfl
theorem model_count_10884 : modelRow 10884 = 0 := by decide
theorem modelrow_agrees_10884 : modelRow 10884 = row 10884 := model_count_10884.trans row_count_10884.symm
def row_10885 : RowCertificate 10885 where
  first := true
  first_correct := by
    have prime : Nat.Prime (corner 10885 1) := by change Nat.Prime 473954671; exact ProjectEulerSolutions.P58PrimeNodes.p_473954671.prime
    exact decide_eq_true prime
  second := false
  second_correct := by
    have composite : ¬ Nat.Prime (corner 10885 2) := by change ¬ Nat.Prime 473932901; exact PrattCertificates.composite_of_product (factor := 13) (cofactor := 36456377) rfl (by decide) (by decide)
    exact decide_eq_false composite
  third := false
  third_correct := by
    have composite : ¬ Nat.Prime (corner 10885 3) := by change ¬ Nat.Prime 473911131; exact PrattCertificates.composite_of_product (factor := 3) (cofactor := 157970377) rfl (by decide) (by decide)
    exact decide_eq_false composite
theorem row_count_10885 : row 10885 = 1 := (row_10885.checked).trans rfl
theorem model_count_10885 : modelRow 10885 = 1 := by decide
theorem modelrow_agrees_10885 : modelRow 10885 = row 10885 := model_count_10885.trans row_count_10885.symm
def row_10886 : RowCertificate 10886 where
  first := false
  first_correct := by
    have composite : ¬ Nat.Prime (corner 10886 1) := by change ¬ Nat.Prime 474041757; exact PrattCertificates.composite_of_product (factor := 3) (cofactor := 158013919) rfl (by decide) (by decide)
    exact decide_eq_false composite
  second := false
  second_correct := by
    have composite : ¬ Nat.Prime (corner 10886 2) := by change ¬ Nat.Prime 474019985; exact PrattCertificates.composite_of_product (factor := 5) (cofactor := 94803997) rfl (by decide) (by decide)
    exact decide_eq_false composite
  third := false
  third_correct := by
    have composite : ¬ Nat.Prime (corner 10886 3) := by change ¬ Nat.Prime 473998213; exact PrattCertificates.composite_of_product (factor := 13) (cofactor := 36461401) rfl (by decide) (by decide)
    exact decide_eq_false composite
theorem row_count_10886 : row 10886 = 0 := (row_10886.checked).trans rfl
theorem model_count_10886 : modelRow 10886 = 0 := by decide
theorem modelrow_agrees_10886 : modelRow 10886 = row 10886 := model_count_10886.trans row_count_10886.symm
def row_10887 : RowCertificate 10887 where
  first := false
  first_correct := by
    have composite : ¬ Nat.Prime (corner 10887 1) := by change ¬ Nat.Prime 474128851; exact PrattCertificates.composite_of_product (factor := 7) (cofactor := 67732693) rfl (by decide) (by decide)
    exact decide_eq_false composite
  second := false
  second_correct := by
    have composite : ¬ Nat.Prime (corner 10887 2) := by change ¬ Nat.Prime 474107077; exact PrattCertificates.composite_of_product (factor := 349) (cofactor := 1358473) rfl (by decide) (by decide)
    exact decide_eq_false composite
  third := true
  third_correct := by
    have prime : Nat.Prime (corner 10887 3) := by change Nat.Prime 474085303; exact ProjectEulerSolutions.P58PrimeNodes.p_474085303.prime
    exact decide_eq_true prime
theorem row_count_10887 : row 10887 = 1 := (row_10887.checked).trans rfl
theorem model_count_10887 : modelRow 10887 = 1 := by decide
theorem modelrow_agrees_10887 : modelRow 10887 = row 10887 := model_count_10887.trans row_count_10887.symm
def row_10888 : RowCertificate 10888 where
  first := false
  first_correct := by
    have composite : ¬ Nat.Prime (corner 10888 1) := by change ¬ Nat.Prime 474215953; exact PrattCertificates.composite_of_product (factor := 151) (cofactor := 3140503) rfl (by decide) (by decide)
    exact decide_eq_false composite
  second := true
  second_correct := by
    have prime : Nat.Prime (corner 10888 2) := by change Nat.Prime 474194177; exact ProjectEulerSolutions.P58PrimeNodes.p_474194177.prime
    exact decide_eq_true prime
  third := false
  third_correct := by
    have composite : ¬ Nat.Prime (corner 10888 3) := by change ¬ Nat.Prime 474172401; exact PrattCertificates.composite_of_product (factor := 3) (cofactor := 158057467) rfl (by decide) (by decide)
    exact decide_eq_false composite
theorem row_count_10888 : row 10888 = 1 := (row_10888.checked).trans rfl
theorem model_count_10888 : modelRow 10888 = 1 := by decide
theorem modelrow_agrees_10888 : modelRow 10888 = row 10888 := model_count_10888.trans row_count_10888.symm
def row_10889 : RowCertificate 10889 where
  first := false
  first_correct := by
    have composite : ¬ Nat.Prime (corner 10889 1) := by change ¬ Nat.Prime 474303063; exact PrattCertificates.composite_of_product (factor := 3) (cofactor := 158101021) rfl (by decide) (by decide)
    exact decide_eq_false composite
  second := false
  second_correct := by
    have composite : ¬ Nat.Prime (corner 10889 2) := by change ¬ Nat.Prime 474281285; exact PrattCertificates.composite_of_product (factor := 5) (cofactor := 94856257) rfl (by decide) (by decide)
    exact decide_eq_false composite
  third := false
  third_correct := by
    have composite : ¬ Nat.Prime (corner 10889 3) := by change ¬ Nat.Prime 474259507; exact PrattCertificates.composite_of_product (factor := 16111) (cofactor := 29437) rfl (by decide) (by decide)
    exact decide_eq_false composite
theorem row_count_10889 : row 10889 = 0 := (row_10889.checked).trans rfl
theorem model_count_10889 : modelRow 10889 = 0 := by decide
theorem modelrow_agrees_10889 : modelRow 10889 = row 10889 := model_count_10889.trans row_count_10889.symm
def row_10890 : RowCertificate 10890 where
  first := false
  first_correct := by
    have composite : ¬ Nat.Prime (corner 10890 1) := by change ¬ Nat.Prime 474390181; exact PrattCertificates.composite_of_product (factor := 79) (cofactor := 6004939) rfl (by decide) (by decide)
    exact decide_eq_false composite
  second := false
  second_correct := by
    have composite : ¬ Nat.Prime (corner 10890 2) := by change ¬ Nat.Prime 474368401; exact PrattCertificates.composite_of_product (factor := 13) (cofactor := 36489877) rfl (by decide) (by decide)
    exact decide_eq_false composite
  third := false
  third_correct := by
    have composite : ¬ Nat.Prime (corner 10890 3) := by change ¬ Nat.Prime 474346621; exact PrattCertificates.composite_of_product (factor := 7) (cofactor := 67763803) rfl (by decide) (by decide)
    exact decide_eq_false composite
theorem row_count_10890 : row 10890 = 0 := (row_10890.checked).trans rfl
theorem model_count_10890 : modelRow 10890 = 0 := by decide
theorem modelrow_agrees_10890 : modelRow 10890 = row 10890 := model_count_10890.trans row_count_10890.symm
def row_10891 : RowCertificate 10891 where
  first := false
  first_correct := by
    have composite : ¬ Nat.Prime (corner 10891 1) := by change ¬ Nat.Prime 474477307; exact PrattCertificates.composite_of_product (factor := 37) (cofactor := 12823711) rfl (by decide) (by decide)
    exact decide_eq_false composite
  second := false
  second_correct := by
    have composite : ¬ Nat.Prime (corner 10891 2) := by change ¬ Nat.Prime 474455525; exact PrattCertificates.composite_of_product (factor := 5) (cofactor := 94891105) rfl (by decide) (by decide)
    exact decide_eq_false composite
  third := false
  third_correct := by
    have composite : ¬ Nat.Prime (corner 10891 3) := by change ¬ Nat.Prime 474433743; exact PrattCertificates.composite_of_product (factor := 3) (cofactor := 158144581) rfl (by decide) (by decide)
    exact decide_eq_false composite
theorem row_count_10891 : row 10891 = 0 := (row_10891.checked).trans rfl
theorem model_count_10891 : modelRow 10891 = 0 := by decide
theorem modelrow_agrees_10891 : modelRow 10891 = row 10891 := model_count_10891.trans row_count_10891.symm
def row_10892 : RowCertificate 10892 where
  first := false
  first_correct := by
    have composite : ¬ Nat.Prime (corner 10892 1) := by change ¬ Nat.Prime 474564441; exact PrattCertificates.composite_of_product (factor := 3) (cofactor := 158188147) rfl (by decide) (by decide)
    exact decide_eq_false composite
  second := true
  second_correct := by
    have prime : Nat.Prime (corner 10892 2) := by change Nat.Prime 474542657; exact ProjectEulerSolutions.P58PrimeNodes.p_474542657.prime
    exact decide_eq_true prime
  third := false
  third_correct := by
    have composite : ¬ Nat.Prime (corner 10892 3) := by change ¬ Nat.Prime 474520873; exact PrattCertificates.composite_of_product (factor := 163) (cofactor := 2911171) rfl (by decide) (by decide)
    exact decide_eq_false composite
theorem row_count_10892 : row 10892 = 1 := (row_10892.checked).trans rfl
theorem model_count_10892 : modelRow 10892 = 1 := by decide
theorem modelrow_agrees_10892 : modelRow 10892 = row 10892 := model_count_10892.trans row_count_10892.symm
def row_10893 : RowCertificate 10893 where
  first := false
  first_correct := by
    have composite : ¬ Nat.Prime (corner 10893 1) := by change ¬ Nat.Prime 474651583; exact PrattCertificates.composite_of_product (factor := 7) (cofactor := 67807369) rfl (by decide) (by decide)
    exact decide_eq_false composite
  second := false
  second_correct := by
    have composite : ¬ Nat.Prime (corner 10893 2) := by change ¬ Nat.Prime 474629797; exact PrattCertificates.composite_of_product (factor := 2389) (cofactor := 198673) rfl (by decide) (by decide)
    exact decide_eq_false composite
  third := false
  third_correct := by
    have composite : ¬ Nat.Prime (corner 10893 3) := by change ¬ Nat.Prime 474608011; exact PrattCertificates.composite_of_product (factor := 19) (cofactor := 24979369) rfl (by decide) (by decide)
    exact decide_eq_false composite
theorem row_count_10893 : row 10893 = 0 := (row_10893.checked).trans rfl
theorem model_count_10893 : modelRow 10893 = 0 := by decide
theorem modelrow_agrees_10893 : modelRow 10893 = row 10893 := model_count_10893.trans row_count_10893.symm
def row_10894 : RowCertificate 10894 where
  first := false
  first_correct := by
    have composite : ¬ Nat.Prime (corner 10894 1) := by change ¬ Nat.Prime 474738733; exact PrattCertificates.composite_of_product (factor := 7) (cofactor := 67819819) rfl (by decide) (by decide)
    exact decide_eq_false composite
  second := false
  second_correct := by
    have composite : ¬ Nat.Prime (corner 10894 2) := by change ¬ Nat.Prime 474716945; exact PrattCertificates.composite_of_product (factor := 5) (cofactor := 94943389) rfl (by decide) (by decide)
    exact decide_eq_false composite
  third := false
  third_correct := by
    have composite : ¬ Nat.Prime (corner 10894 3) := by change ¬ Nat.Prime 474695157; exact PrattCertificates.composite_of_product (factor := 3) (cofactor := 158231719) rfl (by decide) (by decide)
    exact decide_eq_false composite
theorem row_count_10894 : row 10894 = 0 := (row_10894.checked).trans rfl
theorem model_count_10894 : modelRow 10894 = 0 := by decide
theorem modelrow_agrees_10894 : modelRow 10894 = row 10894 := model_count_10894.trans row_count_10894.symm
def row_10895 : RowCertificate 10895 where
  first := false
  first_correct := by
    have composite : ¬ Nat.Prime (corner 10895 1) := by change ¬ Nat.Prime 474825891; exact PrattCertificates.composite_of_product (factor := 3) (cofactor := 158275297) rfl (by decide) (by decide)
    exact decide_eq_false composite
  second := false
  second_correct := by
    have composite : ¬ Nat.Prime (corner 10895 2) := by change ¬ Nat.Prime 474804101; exact PrattCertificates.composite_of_product (factor := 17) (cofactor := 27929653) rfl (by decide) (by decide)
    exact decide_eq_false composite
  third := false
  third_correct := by
    have composite : ¬ Nat.Prime (corner 10895 3) := by change ¬ Nat.Prime 474782311; exact PrattCertificates.composite_of_product (factor := 97) (cofactor := 4894663) rfl (by decide) (by decide)
    exact decide_eq_false composite
theorem row_count_10895 : row 10895 = 0 := (row_10895.checked).trans rfl
theorem model_count_10895 : modelRow 10895 = 0 := by decide
theorem modelrow_agrees_10895 : modelRow 10895 = row 10895 := model_count_10895.trans row_count_10895.symm
def row_10896 : RowCertificate 10896 where
  first := true
  first_correct := by
    have prime : Nat.Prime (corner 10896 1) := by change Nat.Prime 474913057; exact ProjectEulerSolutions.P58PrimeNodes.p_474913057.prime
    exact decide_eq_true prime
  second := false
  second_correct := by
    have composite : ¬ Nat.Prime (corner 10896 2) := by change ¬ Nat.Prime 474891265; exact PrattCertificates.composite_of_product (factor := 5) (cofactor := 94978253) rfl (by decide) (by decide)
    exact decide_eq_false composite
  third := false
  third_correct := by
    have composite : ¬ Nat.Prime (corner 10896 3) := by change ¬ Nat.Prime 474869473; exact PrattCertificates.composite_of_product (factor := 13) (cofactor := 36528421) rfl (by decide) (by decide)
    exact decide_eq_false composite
theorem row_count_10896 : row 10896 = 1 := (row_10896.checked).trans rfl
theorem model_count_10896 : modelRow 10896 = 1 := by decide
theorem modelrow_agrees_10896 : modelRow 10896 = row 10896 := model_count_10896.trans row_count_10896.symm
def row_10897 : RowCertificate 10897 where
  first := false
  first_correct := by
    have composite : ¬ Nat.Prime (corner 10897 1) := by change ¬ Nat.Prime 475000231; exact PrattCertificates.composite_of_product (factor := 43) (cofactor := 11046517) rfl (by decide) (by decide)
    exact decide_eq_false composite
  second := false
  second_correct := by
    have composite : ¬ Nat.Prime (corner 10897 2) := by change ¬ Nat.Prime 474978437; exact PrattCertificates.composite_of_product (factor := 113) (cofactor := 4203349) rfl (by decide) (by decide)
    exact decide_eq_false composite
  third := false
  third_correct := by
    have composite : ¬ Nat.Prime (corner 10897 3) := by change ¬ Nat.Prime 474956643; exact PrattCertificates.composite_of_product (factor := 3) (cofactor := 158318881) rfl (by decide) (by decide)
    exact decide_eq_false composite
theorem row_count_10897 : row 10897 = 0 := (row_10897.checked).trans rfl
theorem model_count_10897 : modelRow 10897 = 0 := by decide
theorem modelrow_agrees_10897 : modelRow 10897 = row 10897 := model_count_10897.trans row_count_10897.symm
def row_10898 : RowCertificate 10898 where
  first := false
  first_correct := by
    have composite : ¬ Nat.Prime (corner 10898 1) := by change ¬ Nat.Prime 475087413; exact PrattCertificates.composite_of_product (factor := 3) (cofactor := 158362471) rfl (by decide) (by decide)
    exact decide_eq_false composite
  second := false
  second_correct := by
    have composite : ¬ Nat.Prime (corner 10898 2) := by change ¬ Nat.Prime 475065617; exact PrattCertificates.composite_of_product (factor := 13) (cofactor := 36543509) rfl (by decide) (by decide)
    exact decide_eq_false composite
  third := false
  third_correct := by
    have composite : ¬ Nat.Prime (corner 10898 3) := by change ¬ Nat.Prime 475043821; exact PrattCertificates.composite_of_product (factor := 7) (cofactor := 67863403) rfl (by decide) (by decide)
    exact decide_eq_false composite
theorem row_count_10898 : row 10898 = 0 := (row_10898.checked).trans rfl
theorem model_count_10898 : modelRow 10898 = 0 := by decide
theorem modelrow_agrees_10898 : modelRow 10898 = row 10898 := model_count_10898.trans row_count_10898.symm
def row_10899 : RowCertificate 10899 where
  first := false
  first_correct := by
    have composite : ¬ Nat.Prime (corner 10899 1) := by change ¬ Nat.Prime 475174603; exact PrattCertificates.composite_of_product (factor := 31) (cofactor := 15328213) rfl (by decide) (by decide)
    exact decide_eq_false composite
  second := false
  second_correct := by
    have composite : ¬ Nat.Prime (corner 10899 2) := by change ¬ Nat.Prime 475152805; exact PrattCertificates.composite_of_product (factor := 5) (cofactor := 95030561) rfl (by decide) (by decide)
    exact decide_eq_false composite
  third := false
  third_correct := by
    have composite : ¬ Nat.Prime (corner 10899 3) := by change ¬ Nat.Prime 475131007; exact PrattCertificates.composite_of_product (factor := 13) (cofactor := 36548539) rfl (by decide) (by decide)
    exact decide_eq_false composite
theorem row_count_10899 : row 10899 = 0 := (row_10899.checked).trans rfl
theorem model_count_10899 : modelRow 10899 = 0 := by decide
theorem modelrow_agrees_10899 : modelRow 10899 = row 10899 := model_count_10899.trans row_count_10899.symm
def row_10900 : RowCertificate 10900 where
  first := false
  first_correct := by
    have composite : ¬ Nat.Prime (corner 10900 1) := by change ¬ Nat.Prime 475261801; exact PrattCertificates.composite_of_product (factor := 7) (cofactor := 67894543) rfl (by decide) (by decide)
    exact decide_eq_false composite
  second := false
  second_correct := by
    have composite : ¬ Nat.Prime (corner 10900 2) := by change ¬ Nat.Prime 475240001; exact PrattCertificates.composite_of_product (factor := 73) (cofactor := 6510137) rfl (by decide) (by decide)
    exact decide_eq_false composite
  third := false
  third_correct := by
    have composite : ¬ Nat.Prime (corner 10900 3) := by change ¬ Nat.Prime 475218201; exact PrattCertificates.composite_of_product (factor := 3) (cofactor := 158406067) rfl (by decide) (by decide)
    exact decide_eq_false composite
theorem row_count_10900 : row 10900 = 0 := (row_10900.checked).trans rfl
theorem model_count_10900 : modelRow 10900 = 0 := by decide
theorem modelrow_agrees_10900 : modelRow 10900 = row 10900 := model_count_10900.trans row_count_10900.symm
def row_10901 : RowCertificate 10901 where
  first := false
  first_correct := by
    have composite : ¬ Nat.Prime (corner 10901 1) := by change ¬ Nat.Prime 475349007; exact PrattCertificates.composite_of_product (factor := 3) (cofactor := 158449669) rfl (by decide) (by decide)
    exact decide_eq_false composite
  second := false
  second_correct := by
    have composite : ¬ Nat.Prime (corner 10901 2) := by change ¬ Nat.Prime 475327205; exact PrattCertificates.composite_of_product (factor := 5) (cofactor := 95065441) rfl (by decide) (by decide)
    exact decide_eq_false composite
  third := false
  third_correct := by
    have composite : ¬ Nat.Prime (corner 10901 3) := by change ¬ Nat.Prime 475305403; exact PrattCertificates.composite_of_product (factor := 2251) (cofactor := 211153) rfl (by decide) (by decide)
    exact decide_eq_false composite
theorem row_count_10901 : row 10901 = 0 := (row_10901.checked).trans rfl
theorem model_count_10901 : modelRow 10901 = 0 := by decide
theorem modelrow_agrees_10901 : modelRow 10901 = row 10901 := model_count_10901.trans row_count_10901.symm
def row_10902 : RowCertificate 10902 where
  first := false
  first_correct := by
    have composite : ¬ Nat.Prime (corner 10902 1) := by change ¬ Nat.Prime 475436221; exact PrattCertificates.composite_of_product (factor := 13) (cofactor := 36572017) rfl (by decide) (by decide)
    exact decide_eq_false composite
  second := true
  second_correct := by
    have prime : Nat.Prime (corner 10902 2) := by change Nat.Prime 475414417; exact ProjectEulerSolutions.P58PrimeNodes.p_475414417.prime
    exact decide_eq_true prime
  third := false
  third_correct := by
    have composite : ¬ Nat.Prime (corner 10902 3) := by change ¬ Nat.Prime 475392613; exact PrattCertificates.composite_of_product (factor := 37) (cofactor := 12848449) rfl (by decide) (by decide)
    exact decide_eq_false composite
theorem row_count_10902 : row 10902 = 1 := (row_10902.checked).trans rfl
theorem model_count_10902 : modelRow 10902 = 1 := by decide
theorem modelrow_agrees_10902 : modelRow 10902 = row 10902 := model_count_10902.trans row_count_10902.symm
def row_10903 : RowCertificate 10903 where
  first := true
  first_correct := by
    have prime : Nat.Prime (corner 10903 1) := by change Nat.Prime 475523443; exact ProjectEulerSolutions.P58PrimeNodes.p_475523443.prime
    exact decide_eq_true prime
  second := false
  second_correct := by
    have composite : ¬ Nat.Prime (corner 10903 2) := by change ¬ Nat.Prime 475501637; exact PrattCertificates.composite_of_product (factor := 13) (cofactor := 36577049) rfl (by decide) (by decide)
    exact decide_eq_false composite
  third := false
  third_correct := by
    have composite : ¬ Nat.Prime (corner 10903 3) := by change ¬ Nat.Prime 475479831; exact PrattCertificates.composite_of_product (factor := 3) (cofactor := 158493277) rfl (by decide) (by decide)
    exact decide_eq_false composite
theorem row_count_10903 : row 10903 = 1 := (row_10903.checked).trans rfl
theorem model_count_10903 : modelRow 10903 = 1 := by decide
theorem modelrow_agrees_10903 : modelRow 10903 = row 10903 := model_count_10903.trans row_count_10903.symm
def row_10904 : RowCertificate 10904 where
  first := false
  first_correct := by
    have composite : ¬ Nat.Prime (corner 10904 1) := by change ¬ Nat.Prime 475610673; exact PrattCertificates.composite_of_product (factor := 3) (cofactor := 158536891) rfl (by decide) (by decide)
    exact decide_eq_false composite
  second := false
  second_correct := by
    have composite : ¬ Nat.Prime (corner 10904 2) := by change ¬ Nat.Prime 475588865; exact PrattCertificates.composite_of_product (factor := 5) (cofactor := 95117773) rfl (by decide) (by decide)
    exact decide_eq_false composite
  third := false
  third_correct := by
    have composite : ¬ Nat.Prime (corner 10904 3) := by change ¬ Nat.Prime 475567057; exact PrattCertificates.composite_of_product (factor := 7) (cofactor := 67938151) rfl (by decide) (by decide)
    exact decide_eq_false composite
theorem row_count_10904 : row 10904 = 0 := (row_10904.checked).trans rfl
theorem model_count_10904 : modelRow 10904 = 0 := by decide
theorem modelrow_agrees_10904 : modelRow 10904 = row 10904 := model_count_10904.trans row_count_10904.symm
def row_10905 : RowCertificate 10905 where
  first := false
  first_correct := by
    have composite : ¬ Nat.Prime (corner 10905 1) := by change ¬ Nat.Prime 475697911; exact PrattCertificates.composite_of_product (factor := 13) (cofactor := 36592147) rfl (by decide) (by decide)
    exact decide_eq_false composite
  second := true
  second_correct := by
    have prime : Nat.Prime (corner 10905 2) := by change Nat.Prime 475676101; exact ProjectEulerSolutions.P58PrimeNodes.p_475676101.prime
    exact decide_eq_true prime
  third := false
  third_correct := by
    have composite : ¬ Nat.Prime (corner 10905 3) := by change ¬ Nat.Prime 475654291; exact PrattCertificates.composite_of_product (factor := 7) (cofactor := 67950613) rfl (by decide) (by decide)
    exact decide_eq_false composite
theorem row_count_10905 : row 10905 = 1 := (row_10905.checked).trans rfl
theorem model_count_10905 : modelRow 10905 = 1 := by decide
theorem modelrow_agrees_10905 : modelRow 10905 = row 10905 := model_count_10905.trans row_count_10905.symm
def row_10906 : RowCertificate 10906 where
  first := false
  first_correct := by
    have composite : ¬ Nat.Prime (corner 10906 1) := by change ¬ Nat.Prime 475785157; exact PrattCertificates.composite_of_product (factor := 67) (cofactor := 7101271) rfl (by decide) (by decide)
    exact decide_eq_false composite
  second := false
  second_correct := by
    have composite : ¬ Nat.Prime (corner 10906 2) := by change ¬ Nat.Prime 475763345; exact PrattCertificates.composite_of_product (factor := 5) (cofactor := 95152669) rfl (by decide) (by decide)
    exact decide_eq_false composite
  third := false
  third_correct := by
    have composite : ¬ Nat.Prime (corner 10906 3) := by change ¬ Nat.Prime 475741533; exact PrattCertificates.composite_of_product (factor := 3) (cofactor := 158580511) rfl (by decide) (by decide)
    exact decide_eq_false composite
theorem row_count_10906 : row 10906 = 0 := (row_10906.checked).trans rfl
theorem model_count_10906 : modelRow 10906 = 0 := by decide
theorem modelrow_agrees_10906 : modelRow 10906 = row 10906 := model_count_10906.trans row_count_10906.symm
def row_10907 : RowCertificate 10907 where
  first := false
  first_correct := by
    have composite : ¬ Nat.Prime (corner 10907 1) := by change ¬ Nat.Prime 475872411; exact PrattCertificates.composite_of_product (factor := 3) (cofactor := 158624137) rfl (by decide) (by decide)
    exact decide_eq_false composite
  second := true
  second_correct := by
    have prime : Nat.Prime (corner 10907 2) := by change Nat.Prime 475850597; exact ProjectEulerSolutions.P58PrimeNodes.p_475850597.prime
    exact decide_eq_true prime
  third := false
  third_correct := by
    have composite : ¬ Nat.Prime (corner 10907 3) := by change ¬ Nat.Prime 475828783; exact PrattCertificates.composite_of_product (factor := 1471) (cofactor := 323473) rfl (by decide) (by decide)
    exact decide_eq_false composite
theorem row_count_10907 : row 10907 = 1 := (row_10907.checked).trans rfl
theorem model_count_10907 : modelRow 10907 = 1 := by decide
theorem modelrow_agrees_10907 : modelRow 10907 = row 10907 := model_count_10907.trans row_count_10907.symm
def row_10908 : RowCertificate 10908 where
  first := false
  first_correct := by
    have composite : ¬ Nat.Prime (corner 10908 1) := by change ¬ Nat.Prime 475959673; exact PrattCertificates.composite_of_product (factor := 7) (cofactor := 67994239) rfl (by decide) (by decide)
    exact decide_eq_false composite
  second := false
  second_correct := by
    have composite : ¬ Nat.Prime (corner 10908 2) := by change ¬ Nat.Prime 475937857; exact PrattCertificates.composite_of_product (factor := 3449) (cofactor := 137993) rfl (by decide) (by decide)
    exact decide_eq_false composite
  third := true
  third_correct := by
    have prime : Nat.Prime (corner 10908 3) := by change Nat.Prime 475916041; exact ProjectEulerSolutions.P58PrimeNodes.p_475916041.prime
    exact decide_eq_true prime
theorem row_count_10908 : row 10908 = 1 := (row_10908.checked).trans rfl
theorem model_count_10908 : modelRow 10908 = 1 := by decide
theorem modelrow_agrees_10908 : modelRow 10908 = row 10908 := model_count_10908.trans row_count_10908.symm
def row_10909 : RowCertificate 10909 where
  first := false
  first_correct := by
    have composite : ¬ Nat.Prime (corner 10909 1) := by change ¬ Nat.Prime 476046943; exact PrattCertificates.composite_of_product (factor := 31) (cofactor := 15356353) rfl (by decide) (by decide)
    exact decide_eq_false composite
  second := false
  second_correct := by
    have composite : ¬ Nat.Prime (corner 10909 2) := by change ¬ Nat.Prime 476025125; exact PrattCertificates.composite_of_product (factor := 5) (cofactor := 95205025) rfl (by decide) (by decide)
    exact decide_eq_false composite
  third := false
  third_correct := by
    have composite : ¬ Nat.Prime (corner 10909 3) := by change ¬ Nat.Prime 476003307; exact PrattCertificates.composite_of_product (factor := 3) (cofactor := 158667769) rfl (by decide) (by decide)
    exact decide_eq_false composite
theorem row_count_10909 : row 10909 = 0 := (row_10909.checked).trans rfl
theorem model_count_10909 : modelRow 10909 = 0 := by decide
theorem modelrow_agrees_10909 : modelRow 10909 = row 10909 := model_count_10909.trans row_count_10909.symm
def row_10910 : RowCertificate 10910 where
  first := false
  first_correct := by
    have composite : ¬ Nat.Prime (corner 10910 1) := by change ¬ Nat.Prime 476134221; exact PrattCertificates.composite_of_product (factor := 3) (cofactor := 158711407) rfl (by decide) (by decide)
    exact decide_eq_false composite
  second := false
  second_correct := by
    have composite : ¬ Nat.Prime (corner 10910 2) := by change ¬ Nat.Prime 476112401; exact PrattCertificates.composite_of_product (factor := 29) (cofactor := 16417669) rfl (by decide) (by decide)
    exact decide_eq_false composite
  third := false
  third_correct := by
    have composite : ¬ Nat.Prime (corner 10910 3) := by change ¬ Nat.Prime 476090581; exact PrattCertificates.composite_of_product (factor := 19) (cofactor := 25057399) rfl (by decide) (by decide)
    exact decide_eq_false composite
theorem row_count_10910 : row 10910 = 0 := (row_10910.checked).trans rfl
theorem model_count_10910 : modelRow 10910 = 0 := by decide
theorem modelrow_agrees_10910 : modelRow 10910 = row 10910 := model_count_10910.trans row_count_10910.symm
def row_10911 : RowCertificate 10911 where
  first := false
  first_correct := by
    have composite : ¬ Nat.Prime (corner 10911 1) := by change ¬ Nat.Prime 476221507; exact PrattCertificates.composite_of_product (factor := 3511) (cofactor := 135637) rfl (by decide) (by decide)
    exact decide_eq_false composite
  second := false
  second_correct := by
    have composite : ¬ Nat.Prime (corner 10911 2) := by change ¬ Nat.Prime 476199685; exact PrattCertificates.composite_of_product (factor := 5) (cofactor := 95239937) rfl (by decide) (by decide)
    exact decide_eq_false composite
  third := false
  third_correct := by
    have composite : ¬ Nat.Prime (corner 10911 3) := by change ¬ Nat.Prime 476177863; exact PrattCertificates.composite_of_product (factor := 7) (cofactor := 68025409) rfl (by decide) (by decide)
    exact decide_eq_false composite
theorem row_count_10911 : row 10911 = 0 := (row_10911.checked).trans rfl
theorem model_count_10911 : modelRow 10911 = 0 := by decide
theorem modelrow_agrees_10911 : modelRow 10911 = row 10911 := model_count_10911.trans row_count_10911.symm
def row_10912 : RowCertificate 10912 where
  first := false
  first_correct := by
    have composite : ¬ Nat.Prime (corner 10912 1) := by change ¬ Nat.Prime 476308801; exact PrattCertificates.composite_of_product (factor := 61) (cofactor := 7808341) rfl (by decide) (by decide)
    exact decide_eq_false composite
  second := false
  second_correct := by
    have composite : ¬ Nat.Prime (corner 10912 2) := by change ¬ Nat.Prime 476286977; exact PrattCertificates.composite_of_product (factor := 17) (cofactor := 28016881) rfl (by decide) (by decide)
    exact decide_eq_false composite
  third := false
  third_correct := by
    have composite : ¬ Nat.Prime (corner 10912 3) := by change ¬ Nat.Prime 476265153; exact PrattCertificates.composite_of_product (factor := 3) (cofactor := 158755051) rfl (by decide) (by decide)
    exact decide_eq_false composite
theorem row_count_10912 : row 10912 = 0 := (row_10912.checked).trans rfl
theorem model_count_10912 : modelRow 10912 = 0 := by decide
theorem modelrow_agrees_10912 : modelRow 10912 = row 10912 := model_count_10912.trans row_count_10912.symm
def row_10913 : RowCertificate 10913 where
  first := false
  first_correct := by
    have composite : ¬ Nat.Prime (corner 10913 1) := by change ¬ Nat.Prime 476396103; exact PrattCertificates.composite_of_product (factor := 3) (cofactor := 158798701) rfl (by decide) (by decide)
    exact decide_eq_false composite
  second := false
  second_correct := by
    have composite : ¬ Nat.Prime (corner 10913 2) := by change ¬ Nat.Prime 476374277; exact PrattCertificates.composite_of_product (factor := 101) (cofactor := 4716577) rfl (by decide) (by decide)
    exact decide_eq_false composite
  third := true
  third_correct := by
    have prime : Nat.Prime (corner 10913 3) := by change Nat.Prime 476352451; exact ProjectEulerSolutions.P58PrimeNodes.p_476352451.prime
    exact decide_eq_true prime
theorem row_count_10913 : row 10913 = 1 := (row_10913.checked).trans rfl
theorem model_count_10913 : modelRow 10913 = 1 := by decide
theorem modelrow_agrees_10913 : modelRow 10913 = row 10913 := model_count_10913.trans row_count_10913.symm
def row_10914 : RowCertificate 10914 where
  first := false
  first_correct := by
    have composite : ¬ Nat.Prime (corner 10914 1) := by change ¬ Nat.Prime 476483413; exact PrattCertificates.composite_of_product (factor := 7) (cofactor := 68069059) rfl (by decide) (by decide)
    exact decide_eq_false composite
  second := false
  second_correct := by
    have composite : ¬ Nat.Prime (corner 10914 2) := by change ¬ Nat.Prime 476461585; exact PrattCertificates.composite_of_product (factor := 5) (cofactor := 95292317) rfl (by decide) (by decide)
    exact decide_eq_false composite
  third := false
  third_correct := by
    have composite : ¬ Nat.Prime (corner 10914 3) := by change ¬ Nat.Prime 476439757; exact PrattCertificates.composite_of_product (factor := 79) (cofactor := 6030883) rfl (by decide) (by decide)
    exact decide_eq_false composite
theorem row_count_10914 : row 10914 = 0 := (row_10914.checked).trans rfl
theorem model_count_10914 : modelRow 10914 = 0 := by decide
theorem modelrow_agrees_10914 : modelRow 10914 = row 10914 := model_count_10914.trans row_count_10914.symm
def row_10915 : RowCertificate 10915 where
  first := false
  first_correct := by
    have composite : ¬ Nat.Prime (corner 10915 1) := by change ¬ Nat.Prime 476570731; exact PrattCertificates.composite_of_product (factor := 7) (cofactor := 68081533) rfl (by decide) (by decide)
    exact decide_eq_false composite
  second := true
  second_correct := by
    have prime : Nat.Prime (corner 10915 2) := by change Nat.Prime 476548901; exact ProjectEulerSolutions.P58PrimeNodes.p_476548901.prime
    exact decide_eq_true prime
  third := false
  third_correct := by
    have composite : ¬ Nat.Prime (corner 10915 3) := by change ¬ Nat.Prime 476527071; exact PrattCertificates.composite_of_product (factor := 3) (cofactor := 158842357) rfl (by decide) (by decide)
    exact decide_eq_false composite
theorem row_count_10915 : row 10915 = 1 := (row_10915.checked).trans rfl
theorem model_count_10915 : modelRow 10915 = 1 := by decide
theorem modelrow_agrees_10915 : modelRow 10915 = row 10915 := model_count_10915.trans row_count_10915.symm
def row_10916 : RowCertificate 10916 where
  first := false
  first_correct := by
    have composite : ¬ Nat.Prime (corner 10916 1) := by change ¬ Nat.Prime 476658057; exact PrattCertificates.composite_of_product (factor := 3) (cofactor := 158886019) rfl (by decide) (by decide)
    exact decide_eq_false composite
  second := false
  second_correct := by
    have composite : ¬ Nat.Prime (corner 10916 2) := by change ¬ Nat.Prime 476636225; exact PrattCertificates.composite_of_product (factor := 5) (cofactor := 95327245) rfl (by decide) (by decide)
    exact decide_eq_false composite
  third := true
  third_correct := by
    have prime : Nat.Prime (corner 10916 3) := by change Nat.Prime 476614393; exact ProjectEulerSolutions.P58PrimeNodes.p_476614393.prime
    exact decide_eq_true prime
theorem row_count_10916 : row 10916 = 1 := (row_10916.checked).trans rfl
theorem model_count_10916 : modelRow 10916 = 1 := by decide
theorem modelrow_agrees_10916 : modelRow 10916 = row 10916 := model_count_10916.trans row_count_10916.symm
def row_10917 : RowCertificate 10917 where
  first := false
  first_correct := by
    have composite : ¬ Nat.Prime (corner 10917 1) := by change ¬ Nat.Prime 476745391; exact PrattCertificates.composite_of_product (factor := 4639) (cofactor := 102769) rfl (by decide) (by decide)
    exact decide_eq_false composite
  second := false
  second_correct := by
    have composite : ¬ Nat.Prime (corner 10917 2) := by change ¬ Nat.Prime 476723557; exact PrattCertificates.composite_of_product (factor := 5477) (cofactor := 87041) rfl (by decide) (by decide)
    exact decide_eq_false composite
  third := true
  third_correct := by
    have prime : Nat.Prime (corner 10917 3) := by change Nat.Prime 476701723; exact ProjectEulerSolutions.P58PrimeNodes.p_476701723.prime
    exact decide_eq_true prime
theorem row_count_10917 : row 10917 = 1 := (row_10917.checked).trans rfl
theorem model_count_10917 : modelRow 10917 = 1 := by decide
theorem modelrow_agrees_10917 : modelRow 10917 = row 10917 := model_count_10917.trans row_count_10917.symm
def row_10918 : RowCertificate 10918 where
  first := false
  first_correct := by
    have composite : ¬ Nat.Prime (corner 10918 1) := by change ¬ Nat.Prime 476832733; exact PrattCertificates.composite_of_product (factor := 13) (cofactor := 36679441) rfl (by decide) (by decide)
    exact decide_eq_false composite
  second := false
  second_correct := by
    have composite : ¬ Nat.Prime (corner 10918 2) := by change ¬ Nat.Prime 476810897; exact PrattCertificates.composite_of_product (factor := 37) (cofactor := 12886781) rfl (by decide) (by decide)
    exact decide_eq_false composite
  third := false
  third_correct := by
    have composite : ¬ Nat.Prime (corner 10918 3) := by change ¬ Nat.Prime 476789061; exact PrattCertificates.composite_of_product (factor := 3) (cofactor := 158929687) rfl (by decide) (by decide)
    exact decide_eq_false composite
theorem row_count_10918 : row 10918 = 0 := (row_10918.checked).trans rfl
theorem model_count_10918 : modelRow 10918 = 0 := by decide
theorem modelrow_agrees_10918 : modelRow 10918 = row 10918 := model_count_10918.trans row_count_10918.symm
def row_10919 : RowCertificate 10919 where
  first := false
  first_correct := by
    have composite : ¬ Nat.Prime (corner 10919 1) := by change ¬ Nat.Prime 476920083; exact PrattCertificates.composite_of_product (factor := 3) (cofactor := 158973361) rfl (by decide) (by decide)
    exact decide_eq_false composite
  second := false
  second_correct := by
    have composite : ¬ Nat.Prime (corner 10919 2) := by change ¬ Nat.Prime 476898245; exact PrattCertificates.composite_of_product (factor := 5) (cofactor := 95379649) rfl (by decide) (by decide)
    exact decide_eq_false composite
  third := false
  third_correct := by
    have composite : ¬ Nat.Prime (corner 10919 3) := by change ¬ Nat.Prime 476876407; exact PrattCertificates.composite_of_product (factor := 7) (cofactor := 68125201) rfl (by decide) (by decide)
    exact decide_eq_false composite
theorem row_count_10919 : row 10919 = 0 := (row_10919.checked).trans rfl
theorem model_count_10919 : modelRow 10919 = 0 := by decide
theorem modelrow_agrees_10919 : modelRow 10919 = row 10919 := model_count_10919.trans row_count_10919.symm
def row_10920 : RowCertificate 10920 where
  first := false
  first_correct := by
    have composite : ¬ Nat.Prime (corner 10920 1) := by change ¬ Nat.Prime 477007441; exact PrattCertificates.composite_of_product (factor := 37) (cofactor := 12892093) rfl (by decide) (by decide)
    exact decide_eq_false composite
  second := true
  second_correct := by
    have prime : Nat.Prime (corner 10920 2) := by change Nat.Prime 476985601; exact ProjectEulerSolutions.P58PrimeNodes.p_476985601.prime
    exact decide_eq_true prime
  third := false
  third_correct := by
    have composite : ¬ Nat.Prime (corner 10920 3) := by change ¬ Nat.Prime 476963761; exact PrattCertificates.composite_of_product (factor := 547) (cofactor := 871963) rfl (by decide) (by decide)
    exact decide_eq_false composite
theorem row_count_10920 : row 10920 = 1 := (row_10920.checked).trans rfl
theorem model_count_10920 : modelRow 10920 = 1 := by decide
theorem modelrow_agrees_10920 : modelRow 10920 = row 10920 := model_count_10920.trans row_count_10920.symm
def row_10921 : RowCertificate 10921 where
  first := false
  first_correct := by
    have composite : ¬ Nat.Prime (corner 10921 1) := by change ¬ Nat.Prime 477094807; exact PrattCertificates.composite_of_product (factor := 7) (cofactor := 68156401) rfl (by decide) (by decide)
    exact decide_eq_false composite
  second := false
  second_correct := by
    have composite : ¬ Nat.Prime (corner 10921 2) := by change ¬ Nat.Prime 477072965; exact PrattCertificates.composite_of_product (factor := 5) (cofactor := 95414593) rfl (by decide) (by decide)
    exact decide_eq_false composite
  third := false
  third_correct := by
    have composite : ¬ Nat.Prime (corner 10921 3) := by change ¬ Nat.Prime 477051123; exact PrattCertificates.composite_of_product (factor := 3) (cofactor := 159017041) rfl (by decide) (by decide)
    exact decide_eq_false composite
theorem row_count_10921 : row 10921 = 0 := (row_10921.checked).trans rfl
theorem model_count_10921 : modelRow 10921 = 0 := by decide
theorem modelrow_agrees_10921 : modelRow 10921 = row 10921 := model_count_10921.trans row_count_10921.symm
def row_10922 : RowCertificate 10922 where
  first := false
  first_correct := by
    have composite : ¬ Nat.Prime (corner 10922 1) := by change ¬ Nat.Prime 477182181; exact PrattCertificates.composite_of_product (factor := 3) (cofactor := 159060727) rfl (by decide) (by decide)
    exact decide_eq_false composite
  second := false
  second_correct := by
    have composite : ¬ Nat.Prime (corner 10922 2) := by change ¬ Nat.Prime 477160337; exact PrattCertificates.composite_of_product (factor := 41) (cofactor := 11638057) rfl (by decide) (by decide)
    exact decide_eq_false composite
  third := false
  third_correct := by
    have composite : ¬ Nat.Prime (corner 10922 3) := by change ¬ Nat.Prime 477138493; exact PrattCertificates.composite_of_product (factor := 13) (cofactor := 36702961) rfl (by decide) (by decide)
    exact decide_eq_false composite
theorem row_count_10922 : row 10922 = 0 := (row_10922.checked).trans rfl
theorem model_count_10922 : modelRow 10922 = 0 := by decide
theorem modelrow_agrees_10922 : modelRow 10922 = row 10922 := model_count_10922.trans row_count_10922.symm
def row_10923 : RowCertificate 10923 where
  first := false
  first_correct := by
    have composite : ¬ Nat.Prime (corner 10923 1) := by change ¬ Nat.Prime 477269563; exact PrattCertificates.composite_of_product (factor := 16693) (cofactor := 28591) rfl (by decide) (by decide)
    exact decide_eq_false composite
  second := false
  second_correct := by
    have composite : ¬ Nat.Prime (corner 10923 2) := by change ¬ Nat.Prime 477247717; exact PrattCertificates.composite_of_product (factor := 2357) (cofactor := 202481) rfl (by decide) (by decide)
    exact decide_eq_false composite
  third := false
  third_correct := by
    have composite : ¬ Nat.Prime (corner 10923 3) := by change ¬ Nat.Prime 477225871; exact PrattCertificates.composite_of_product (factor := 109) (cofactor := 4378219) rfl (by decide) (by decide)
    exact decide_eq_false composite
theorem row_count_10923 : row 10923 = 0 := (row_10923.checked).trans rfl
theorem model_count_10923 : modelRow 10923 = 0 := by decide
theorem modelrow_agrees_10923 : modelRow 10923 = row 10923 := model_count_10923.trans row_count_10923.symm
def row_10924 : RowCertificate 10924 where
  first := false
  first_correct := by
    have composite : ¬ Nat.Prime (corner 10924 1) := by change ¬ Nat.Prime 477356953; exact PrattCertificates.composite_of_product (factor := 433) (cofactor := 1102441) rfl (by decide) (by decide)
    exact decide_eq_false composite
  second := false
  second_correct := by
    have composite : ¬ Nat.Prime (corner 10924 2) := by change ¬ Nat.Prime 477335105; exact PrattCertificates.composite_of_product (factor := 5) (cofactor := 95467021) rfl (by decide) (by decide)
    exact decide_eq_false composite
  third := false
  third_correct := by
    have composite : ¬ Nat.Prime (corner 10924 3) := by change ¬ Nat.Prime 477313257; exact PrattCertificates.composite_of_product (factor := 3) (cofactor := 159104419) rfl (by decide) (by decide)
    exact decide_eq_false composite
theorem row_count_10924 : row 10924 = 0 := (row_10924.checked).trans rfl
theorem model_count_10924 : modelRow 10924 = 0 := by decide
theorem modelrow_agrees_10924 : modelRow 10924 = row 10924 := model_count_10924.trans row_count_10924.symm
def row_10925 : RowCertificate 10925 where
  first := false
  first_correct := by
    have composite : ¬ Nat.Prime (corner 10925 1) := by change ¬ Nat.Prime 477444351; exact PrattCertificates.composite_of_product (factor := 3) (cofactor := 159148117) rfl (by decide) (by decide)
    exact decide_eq_false composite
  second := false
  second_correct := by
    have composite : ¬ Nat.Prime (corner 10925 2) := by change ¬ Nat.Prime 477422501; exact PrattCertificates.composite_of_product (factor := 3581) (cofactor := 133321) rfl (by decide) (by decide)
    exact decide_eq_false composite
  third := false
  third_correct := by
    have composite : ¬ Nat.Prime (corner 10925 3) := by change ¬ Nat.Prime 477400651; exact PrattCertificates.composite_of_product (factor := 7) (cofactor := 68200093) rfl (by decide) (by decide)
    exact decide_eq_false composite
theorem row_count_10925 : row 10925 = 0 := (row_10925.checked).trans rfl
theorem model_count_10925 : modelRow 10925 = 0 := by decide
theorem modelrow_agrees_10925 : modelRow 10925 = row 10925 := model_count_10925.trans row_count_10925.symm
def row_10926 : RowCertificate 10926 where
  first := false
  first_correct := by
    have composite : ¬ Nat.Prime (corner 10926 1) := by change ¬ Nat.Prime 477531757; exact PrattCertificates.composite_of_product (factor := 181) (cofactor := 2638297) rfl (by decide) (by decide)
    exact decide_eq_false composite
  second := false
  second_correct := by
    have composite : ¬ Nat.Prime (corner 10926 2) := by change ¬ Nat.Prime 477509905; exact PrattCertificates.composite_of_product (factor := 5) (cofactor := 95501981) rfl (by decide) (by decide)
    exact decide_eq_false composite
  third := false
  third_correct := by
    have composite : ¬ Nat.Prime (corner 10926 3) := by change ¬ Nat.Prime 477488053; exact PrattCertificates.composite_of_product (factor := 7) (cofactor := 68212579) rfl (by decide) (by decide)
    exact decide_eq_false composite
theorem row_count_10926 : row 10926 = 0 := (row_10926.checked).trans rfl
theorem model_count_10926 : modelRow 10926 = 0 := by decide
theorem modelrow_agrees_10926 : modelRow 10926 = row 10926 := model_count_10926.trans row_count_10926.symm
def row_10927 : RowCertificate 10927 where
  first := false
  first_correct := by
    have composite : ¬ Nat.Prime (corner 10927 1) := by change ¬ Nat.Prime 477619171; exact PrattCertificates.composite_of_product (factor := 1933) (cofactor := 247087) rfl (by decide) (by decide)
    exact decide_eq_false composite
  second := false
  second_correct := by
    have composite : ¬ Nat.Prime (corner 10927 2) := by change ¬ Nat.Prime 477597317; exact PrattCertificates.composite_of_product (factor := 29) (cofactor := 16468873) rfl (by decide) (by decide)
    exact decide_eq_false composite
  third := false
  third_correct := by
    have composite : ¬ Nat.Prime (corner 10927 3) := by change ¬ Nat.Prime 477575463; exact PrattCertificates.composite_of_product (factor := 3) (cofactor := 159191821) rfl (by decide) (by decide)
    exact decide_eq_false composite
theorem row_count_10927 : row 10927 = 0 := (row_10927.checked).trans rfl
theorem model_count_10927 : modelRow 10927 = 0 := by decide
theorem modelrow_agrees_10927 : modelRow 10927 = row 10927 := model_count_10927.trans row_count_10927.symm
def row_10928 : RowCertificate 10928 where
  first := false
  first_correct := by
    have composite : ¬ Nat.Prime (corner 10928 1) := by change ¬ Nat.Prime 477706593; exact PrattCertificates.composite_of_product (factor := 3) (cofactor := 159235531) rfl (by decide) (by decide)
    exact decide_eq_false composite
  second := true
  second_correct := by
    have prime : Nat.Prime (corner 10928 2) := by change Nat.Prime 477684737; exact ProjectEulerSolutions.P58PrimeNodes.p_477684737.prime
    exact decide_eq_true prime
  third := true
  third_correct := by
    have prime : Nat.Prime (corner 10928 3) := by change Nat.Prime 477662881; exact ProjectEulerSolutions.P58PrimeNodes.p_477662881.prime
    exact decide_eq_true prime
theorem row_count_10928 : row 10928 = 2 := (row_10928.checked).trans rfl
theorem model_count_10928 : modelRow 10928 = 2 := by decide
theorem modelrow_agrees_10928 : modelRow 10928 = row 10928 := model_count_10928.trans row_count_10928.symm
def row_10929 : RowCertificate 10929 where
  first := false
  first_correct := by
    have composite : ¬ Nat.Prime (corner 10929 1) := by change ¬ Nat.Prime 477794023; exact PrattCertificates.composite_of_product (factor := 7) (cofactor := 68256289) rfl (by decide) (by decide)
    exact decide_eq_false composite
  second := false
  second_correct := by
    have composite : ¬ Nat.Prime (corner 10929 2) := by change ¬ Nat.Prime 477772165; exact PrattCertificates.composite_of_product (factor := 5) (cofactor := 95554433) rfl (by decide) (by decide)
    exact decide_eq_false composite
  third := false
  third_correct := by
    have composite : ¬ Nat.Prime (corner 10929 3) := by change ¬ Nat.Prime 477750307; exact PrattCertificates.composite_of_product (factor := 19) (cofactor := 25144753) rfl (by decide) (by decide)
    exact decide_eq_false composite
theorem row_count_10929 : row 10929 = 0 := (row_10929.checked).trans rfl
theorem model_count_10929 : modelRow 10929 = 0 := by decide
theorem modelrow_agrees_10929 : modelRow 10929 = row 10929 := model_count_10929.trans row_count_10929.symm
def row_10930 : RowCertificate 10930 where
  first := false
  first_correct := by
    have composite : ¬ Nat.Prime (corner 10930 1) := by change ¬ Nat.Prime 477881461; exact PrattCertificates.composite_of_product (factor := 31) (cofactor := 15415531) rfl (by decide) (by decide)
    exact decide_eq_false composite
  second := false
  second_correct := by
    have composite : ¬ Nat.Prime (corner 10930 2) := by change ¬ Nat.Prime 477859601; exact PrattCertificates.composite_of_product (factor := 89) (cofactor := 5369209) rfl (by decide) (by decide)
    exact decide_eq_false composite
  third := false
  third_correct := by
    have composite : ¬ Nat.Prime (corner 10930 3) := by change ¬ Nat.Prime 477837741; exact PrattCertificates.composite_of_product (factor := 3) (cofactor := 159279247) rfl (by decide) (by decide)
    exact decide_eq_false composite
theorem row_count_10930 : row 10930 = 0 := (row_10930.checked).trans rfl
theorem model_count_10930 : modelRow 10930 = 0 := by decide
theorem modelrow_agrees_10930 : modelRow 10930 = row 10930 := model_count_10930.trans row_count_10930.symm
def row_10931 : RowCertificate 10931 where
  first := false
  first_correct := by
    have composite : ¬ Nat.Prime (corner 10931 1) := by change ¬ Nat.Prime 477968907; exact PrattCertificates.composite_of_product (factor := 3) (cofactor := 159322969) rfl (by decide) (by decide)
    exact decide_eq_false composite
  second := false
  second_correct := by
    have composite : ¬ Nat.Prime (corner 10931 2) := by change ¬ Nat.Prime 477947045; exact PrattCertificates.composite_of_product (factor := 5) (cofactor := 95589409) rfl (by decide) (by decide)
    exact decide_eq_false composite
  third := false
  third_correct := by
    have composite : ¬ Nat.Prime (corner 10931 3) := by change ¬ Nat.Prime 477925183; exact PrattCertificates.composite_of_product (factor := 19) (cofactor := 25153957) rfl (by decide) (by decide)
    exact decide_eq_false composite
theorem row_count_10931 : row 10931 = 0 := (row_10931.checked).trans rfl
theorem model_count_10931 : modelRow 10931 = 0 := by decide
theorem modelrow_agrees_10931 : modelRow 10931 = row 10931 := model_count_10931.trans row_count_10931.symm
def row_10932 : RowCertificate 10932 where
  first := false
  first_correct := by
    have composite : ¬ Nat.Prime (corner 10932 1) := by change ¬ Nat.Prime 478056361; exact PrattCertificates.composite_of_product (factor := 349) (cofactor := 1369789) rfl (by decide) (by decide)
    exact decide_eq_false composite
  second := false
  second_correct := by
    have composite : ¬ Nat.Prime (corner 10932 2) := by change ¬ Nat.Prime 478034497; exact PrattCertificates.composite_of_product (factor := 593) (cofactor := 806129) rfl (by decide) (by decide)
    exact decide_eq_false composite
  third := false
  third_correct := by
    have composite : ¬ Nat.Prime (corner 10932 3) := by change ¬ Nat.Prime 478012633; exact PrattCertificates.composite_of_product (factor := 7) (cofactor := 68287519) rfl (by decide) (by decide)
    exact decide_eq_false composite
theorem row_count_10932 : row 10932 = 0 := (row_10932.checked).trans rfl
theorem model_count_10932 : modelRow 10932 = 0 := by decide
theorem modelrow_agrees_10932 : modelRow 10932 = row 10932 := model_count_10932.trans row_count_10932.symm
def row_10933 : RowCertificate 10933 where
  first := true
  first_correct := by
    have prime : Nat.Prime (corner 10933 1) := by change Nat.Prime 478143823; exact ProjectEulerSolutions.P58PrimeNodes.p_478143823.prime
    exact decide_eq_true prime
  second := false
  second_correct := by
    have composite : ¬ Nat.Prime (corner 10933 2) := by change ¬ Nat.Prime 478121957; exact PrattCertificates.composite_of_product (factor := 17) (cofactor := 28124821) rfl (by decide) (by decide)
    exact decide_eq_false composite
  third := false
  third_correct := by
    have composite : ¬ Nat.Prime (corner 10933 3) := by change ¬ Nat.Prime 478100091; exact PrattCertificates.composite_of_product (factor := 3) (cofactor := 159366697) rfl (by decide) (by decide)
    exact decide_eq_false composite
theorem row_count_10933 : row 10933 = 1 := (row_10933.checked).trans rfl
theorem model_count_10933 : modelRow 10933 = 1 := by decide
theorem modelrow_agrees_10933 : modelRow 10933 = row 10933 := model_count_10933.trans row_count_10933.symm
def row_10934 : RowCertificate 10934 where
  first := false
  first_correct := by
    have composite : ¬ Nat.Prime (corner 10934 1) := by change ¬ Nat.Prime 478231293; exact PrattCertificates.composite_of_product (factor := 3) (cofactor := 159410431) rfl (by decide) (by decide)
    exact decide_eq_false composite
  second := false
  second_correct := by
    have composite : ¬ Nat.Prime (corner 10934 2) := by change ¬ Nat.Prime 478209425; exact PrattCertificates.composite_of_product (factor := 5) (cofactor := 95641885) rfl (by decide) (by decide)
    exact decide_eq_false composite
  third := false
  third_correct := by
    have composite : ¬ Nat.Prime (corner 10934 3) := by change ¬ Nat.Prime 478187557; exact PrattCertificates.composite_of_product (factor := 5347) (cofactor := 89431) rfl (by decide) (by decide)
    exact decide_eq_false composite
theorem row_count_10934 : row 10934 = 0 := (row_10934.checked).trans rfl
theorem model_count_10934 : modelRow 10934 = 0 := by decide
theorem modelrow_agrees_10934 : modelRow 10934 = row 10934 := model_count_10934.trans row_count_10934.symm
def row_10935 : RowCertificate 10935 where
  first := false
  first_correct := by
    have composite : ¬ Nat.Prime (corner 10935 1) := by change ¬ Nat.Prime 478318771; exact PrattCertificates.composite_of_product (factor := 7) (cofactor := 68331253) rfl (by decide) (by decide)
    exact decide_eq_false composite
  second := true
  second_correct := by
    have prime : Nat.Prime (corner 10935 2) := by change Nat.Prime 478296901; exact ProjectEulerSolutions.P58PrimeNodes.p_478296901.prime
    exact decide_eq_true prime
  third := false
  third_correct := by
    have composite : ¬ Nat.Prime (corner 10935 3) := by change ¬ Nat.Prime 478275031; exact PrattCertificates.composite_of_product (factor := 13) (cofactor := 36790387) rfl (by decide) (by decide)
    exact decide_eq_false composite
theorem row_count_10935 : row 10935 = 1 := (row_10935.checked).trans rfl
theorem model_count_10935 : modelRow 10935 = 1 := by decide
theorem modelrow_agrees_10935 : modelRow 10935 = row 10935 := model_count_10935.trans row_count_10935.symm
def row_10936 : RowCertificate 10936 where
  first := false
  first_correct := by
    have composite : ¬ Nat.Prime (corner 10936 1) := by change ¬ Nat.Prime 478406257; exact PrattCertificates.composite_of_product (factor := 7) (cofactor := 68343751) rfl (by decide) (by decide)
    exact decide_eq_false composite
  second := false
  second_correct := by
    have composite : ¬ Nat.Prime (corner 10936 2) := by change ¬ Nat.Prime 478384385; exact PrattCertificates.composite_of_product (factor := 5) (cofactor := 95676877) rfl (by decide) (by decide)
    exact decide_eq_false composite
  third := false
  third_correct := by
    have composite : ¬ Nat.Prime (corner 10936 3) := by change ¬ Nat.Prime 478362513; exact PrattCertificates.composite_of_product (factor := 3) (cofactor := 159454171) rfl (by decide) (by decide)
    exact decide_eq_false composite
theorem row_count_10936 : row 10936 = 0 := (row_10936.checked).trans rfl
theorem model_count_10936 : modelRow 10936 = 0 := by decide
theorem modelrow_agrees_10936 : modelRow 10936 = row 10936 := model_count_10936.trans row_count_10936.symm
def row_10937 : RowCertificate 10937 where
  first := false
  first_correct := by
    have composite : ¬ Nat.Prime (corner 10937 1) := by change ¬ Nat.Prime 478493751; exact PrattCertificates.composite_of_product (factor := 3) (cofactor := 159497917) rfl (by decide) (by decide)
    exact decide_eq_false composite
  second := false
  second_correct := by
    have composite : ¬ Nat.Prime (corner 10937 2) := by change ¬ Nat.Prime 478471877; exact PrattCertificates.composite_of_product (factor := 13) (cofactor := 36805529) rfl (by decide) (by decide)
    exact decide_eq_false composite
  third := false
  third_correct := by
    have composite : ¬ Nat.Prime (corner 10937 3) := by change ¬ Nat.Prime 478450003; exact PrattCertificates.composite_of_product (factor := 937) (cofactor := 510619) rfl (by decide) (by decide)
    exact decide_eq_false composite
theorem row_count_10937 : row 10937 = 0 := (row_10937.checked).trans rfl
theorem model_count_10937 : modelRow 10937 = 0 := by decide
theorem modelrow_agrees_10937 : modelRow 10937 = row 10937 := model_count_10937.trans row_count_10937.symm
def row_10938 : RowCertificate 10938 where
  first := false
  first_correct := by
    have composite : ¬ Nat.Prime (corner 10938 1) := by change ¬ Nat.Prime 478581253; exact PrattCertificates.composite_of_product (factor := 19) (cofactor := 25188487) rfl (by decide) (by decide)
    exact decide_eq_false composite
  second := false
  second_correct := by
    have composite : ¬ Nat.Prime (corner 10938 2) := by change ¬ Nat.Prime 478559377; exact PrattCertificates.composite_of_product (factor := 109) (cofactor := 4390453) rfl (by decide) (by decide)
    exact decide_eq_false composite
  third := false
  third_correct := by
    have composite : ¬ Nat.Prime (corner 10938 3) := by change ¬ Nat.Prime 478537501; exact PrattCertificates.composite_of_product (factor := 13) (cofactor := 36810577) rfl (by decide) (by decide)
    exact decide_eq_false composite
theorem row_count_10938 : row 10938 = 0 := (row_10938.checked).trans rfl
theorem model_count_10938 : modelRow 10938 = 0 := by decide
theorem modelrow_agrees_10938 : modelRow 10938 = row 10938 := model_count_10938.trans row_count_10938.symm
def row_10939 : RowCertificate 10939 where
  first := true
  first_correct := by
    have prime : Nat.Prime (corner 10939 1) := by change Nat.Prime 478668763; exact ProjectEulerSolutions.P58PrimeNodes.p_478668763.prime
    exact decide_eq_true prime
  second := false
  second_correct := by
    have composite : ¬ Nat.Prime (corner 10939 2) := by change ¬ Nat.Prime 478646885; exact PrattCertificates.composite_of_product (factor := 5) (cofactor := 95729377) rfl (by decide) (by decide)
    exact decide_eq_false composite
  third := false
  third_correct := by
    have composite : ¬ Nat.Prime (corner 10939 3) := by change ¬ Nat.Prime 478625007; exact PrattCertificates.composite_of_product (factor := 3) (cofactor := 159541669) rfl (by decide) (by decide)
    exact decide_eq_false composite
theorem row_count_10939 : row 10939 = 1 := (row_10939.checked).trans rfl
theorem model_count_10939 : modelRow 10939 = 1 := by decide
theorem modelrow_agrees_10939 : modelRow 10939 = row 10939 := model_count_10939.trans row_count_10939.symm
def row_10940 : RowCertificate 10940 where
  first := false
  first_correct := by
    have composite : ¬ Nat.Prime (corner 10940 1) := by change ¬ Nat.Prime 478756281; exact PrattCertificates.composite_of_product (factor := 3) (cofactor := 159585427) rfl (by decide) (by decide)
    exact decide_eq_false composite
  second := true
  second_correct := by
    have prime : Nat.Prime (corner 10940 2) := by change Nat.Prime 478734401; exact ProjectEulerSolutions.P58PrimeNodes.p_478734401.prime
    exact decide_eq_true prime
  third := false
  third_correct := by
    have composite : ¬ Nat.Prime (corner 10940 3) := by change ¬ Nat.Prime 478712521; exact PrattCertificates.composite_of_product (factor := 7) (cofactor := 68387503) rfl (by decide) (by decide)
    exact decide_eq_false composite
theorem row_count_10940 : row 10940 = 1 := (row_10940.checked).trans rfl
theorem model_count_10940 : modelRow 10940 = 1 := by decide
theorem modelrow_agrees_10940 : modelRow 10940 = row 10940 := model_count_10940.trans row_count_10940.symm
def row_10941 : RowCertificate 10941 where
  first := false
  first_correct := by
    have composite : ¬ Nat.Prime (corner 10941 1) := by change ¬ Nat.Prime 478843807; exact PrattCertificates.composite_of_product (factor := 13) (cofactor := 36834139) rfl (by decide) (by decide)
    exact decide_eq_false composite
  second := false
  second_correct := by
    have composite : ¬ Nat.Prime (corner 10941 2) := by change ¬ Nat.Prime 478821925; exact PrattCertificates.composite_of_product (factor := 5) (cofactor := 95764385) rfl (by decide) (by decide)
    exact decide_eq_false composite
  third := false
  third_correct := by
    have composite : ¬ Nat.Prime (corner 10941 3) := by change ¬ Nat.Prime 478800043; exact PrattCertificates.composite_of_product (factor := 2593) (cofactor := 184651) rfl (by decide) (by decide)
    exact decide_eq_false composite
theorem row_count_10941 : row 10941 = 0 := (row_10941.checked).trans rfl
theorem model_count_10941 : modelRow 10941 = 0 := by decide
theorem modelrow_agrees_10941 : modelRow 10941 = row 10941 := model_count_10941.trans row_count_10941.symm
def row_10942 : RowCertificate 10942 where
  first := false
  first_correct := by
    have composite : ¬ Nat.Prime (corner 10942 1) := by change ¬ Nat.Prime 478931341; exact PrattCertificates.composite_of_product (factor := 7) (cofactor := 68418763) rfl (by decide) (by decide)
    exact decide_eq_false composite
  second := false
  second_correct := by
    have composite : ¬ Nat.Prime (corner 10942 2) := by change ¬ Nat.Prime 478909457; exact PrattCertificates.composite_of_product (factor := 13) (cofactor := 36839189) rfl (by decide) (by decide)
    exact decide_eq_false composite
  third := false
  third_correct := by
    have composite : ¬ Nat.Prime (corner 10942 3) := by change ¬ Nat.Prime 478887573; exact PrattCertificates.composite_of_product (factor := 3) (cofactor := 159629191) rfl (by decide) (by decide)
    exact decide_eq_false composite
theorem row_count_10942 : row 10942 = 0 := (row_10942.checked).trans rfl
theorem model_count_10942 : modelRow 10942 = 0 := by decide
theorem modelrow_agrees_10942 : modelRow 10942 = row 10942 := model_count_10942.trans row_count_10942.symm
def row_10943 : RowCertificate 10943 where
  first := false
  first_correct := by
    have composite : ¬ Nat.Prime (corner 10943 1) := by change ¬ Nat.Prime 479018883; exact PrattCertificates.composite_of_product (factor := 3) (cofactor := 159672961) rfl (by decide) (by decide)
    exact decide_eq_false composite
  second := true
  second_correct := by
    have prime : Nat.Prime (corner 10943 2) := by change Nat.Prime 478996997; exact ProjectEulerSolutions.P58PrimeNodes.p_478996997.prime
    exact decide_eq_true prime
  third := false
  third_correct := by
    have composite : ¬ Nat.Prime (corner 10943 3) := by change ¬ Nat.Prime 478975111; exact PrattCertificates.composite_of_product (factor := 61) (cofactor := 7852051) rfl (by decide) (by decide)
    exact decide_eq_false composite
theorem row_count_10943 : row 10943 = 1 := (row_10943.checked).trans rfl
theorem model_count_10943 : modelRow 10943 = 1 := by decide
theorem modelrow_agrees_10943 : modelRow 10943 = row 10943 := model_count_10943.trans row_count_10943.symm
def row_10944 : RowCertificate 10944 where
  first := false
  first_correct := by
    have composite : ¬ Nat.Prime (corner 10944 1) := by change ¬ Nat.Prime 479106433; exact PrattCertificates.composite_of_product (factor := 13) (cofactor := 36854341) rfl (by decide) (by decide)
    exact decide_eq_false composite
  second := false
  second_correct := by
    have composite : ¬ Nat.Prime (corner 10944 2) := by change ¬ Nat.Prime 479084545; exact PrattCertificates.composite_of_product (factor := 5) (cofactor := 95816909) rfl (by decide) (by decide)
    exact decide_eq_false composite
  third := true
  third_correct := by
    have prime : Nat.Prime (corner 10944 3) := by change Nat.Prime 479062657; exact ProjectEulerSolutions.P58PrimeNodes.p_479062657.prime
    exact decide_eq_true prime
theorem row_count_10944 : row 10944 = 1 := (row_10944.checked).trans rfl
theorem model_count_10944 : modelRow 10944 = 1 := by decide
theorem modelrow_agrees_10944 : modelRow 10944 = row 10944 := model_count_10944.trans row_count_10944.symm
def row_10945 : RowCertificate 10945 where
  first := false
  first_correct := by
    have composite : ¬ Nat.Prime (corner 10945 1) := by change ¬ Nat.Prime 479193991; exact PrattCertificates.composite_of_product (factor := 1051) (cofactor := 455941) rfl (by decide) (by decide)
    exact decide_eq_false composite
  second := false
  second_correct := by
    have composite : ¬ Nat.Prime (corner 10945 2) := by change ¬ Nat.Prime 479172101; exact PrattCertificates.composite_of_product (factor := 193) (cofactor := 2482757) rfl (by decide) (by decide)
    exact decide_eq_false composite
  third := false
  third_correct := by
    have composite : ¬ Nat.Prime (corner 10945 3) := by change ¬ Nat.Prime 479150211; exact PrattCertificates.composite_of_product (factor := 3) (cofactor := 159716737) rfl (by decide) (by decide)
    exact decide_eq_false composite
theorem row_count_10945 : row 10945 = 0 := (row_10945.checked).trans rfl
theorem model_count_10945 : modelRow 10945 = 0 := by decide
theorem modelrow_agrees_10945 : modelRow 10945 = row 10945 := model_count_10945.trans row_count_10945.symm
def row_10946 : RowCertificate 10946 where
  first := false
  first_correct := by
    have composite : ¬ Nat.Prime (corner 10946 1) := by change ¬ Nat.Prime 479281557; exact PrattCertificates.composite_of_product (factor := 3) (cofactor := 159760519) rfl (by decide) (by decide)
    exact decide_eq_false composite
  second := false
  second_correct := by
    have composite : ¬ Nat.Prime (corner 10946 2) := by change ¬ Nat.Prime 479259665; exact PrattCertificates.composite_of_product (factor := 5) (cofactor := 95851933) rfl (by decide) (by decide)
    exact decide_eq_false composite
  third := false
  third_correct := by
    have composite : ¬ Nat.Prime (corner 10946 3) := by change ¬ Nat.Prime 479237773; exact PrattCertificates.composite_of_product (factor := 7) (cofactor := 68462539) rfl (by decide) (by decide)
    exact decide_eq_false composite
theorem row_count_10946 : row 10946 = 0 := (row_10946.checked).trans rfl
theorem model_count_10946 : modelRow 10946 = 0 := by decide
theorem modelrow_agrees_10946 : modelRow 10946 = row 10946 := model_count_10946.trans row_count_10946.symm
def row_10947 : RowCertificate 10947 where
  first := true
  first_correct := by
    have prime : Nat.Prime (corner 10947 1) := by change Nat.Prime 479369131; exact ProjectEulerSolutions.P58PrimeNodes.p_479369131.prime
    exact decide_eq_true prime
  second := true
  second_correct := by
    have prime : Nat.Prime (corner 10947 2) := by change Nat.Prime 479347237; exact ProjectEulerSolutions.P58PrimeNodes.p_479347237.prime
    exact decide_eq_true prime
  third := false
  third_correct := by
    have composite : ¬ Nat.Prime (corner 10947 3) := by change ¬ Nat.Prime 479325343; exact PrattCertificates.composite_of_product (factor := 7) (cofactor := 68475049) rfl (by decide) (by decide)
    exact decide_eq_false composite
theorem row_count_10947 : row 10947 = 2 := (row_10947.checked).trans rfl
theorem model_count_10947 : modelRow 10947 = 2 := by decide
theorem modelrow_agrees_10947 : modelRow 10947 = row 10947 := model_count_10947.trans row_count_10947.symm
def row_10948 : RowCertificate 10948 where
  first := false
  first_correct := by
    have composite : ¬ Nat.Prime (corner 10948 1) := by change ¬ Nat.Prime 479456713; exact PrattCertificates.composite_of_product (factor := 15349) (cofactor := 31237) rfl (by decide) (by decide)
    exact decide_eq_false composite
  second := true
  second_correct := by
    have prime : Nat.Prime (corner 10948 2) := by change Nat.Prime 479434817; exact ProjectEulerSolutions.P58PrimeNodes.p_479434817.prime
    exact decide_eq_true prime
  third := false
  third_correct := by
    have composite : ¬ Nat.Prime (corner 10948 3) := by change ¬ Nat.Prime 479412921; exact PrattCertificates.composite_of_product (factor := 3) (cofactor := 159804307) rfl (by decide) (by decide)
    exact decide_eq_false composite
theorem row_count_10948 : row 10948 = 1 := (row_10948.checked).trans rfl
theorem model_count_10948 : modelRow 10948 = 1 := by decide
theorem modelrow_agrees_10948 : modelRow 10948 = row 10948 := model_count_10948.trans row_count_10948.symm
def row_10949 : RowCertificate 10949 where
  first := false
  first_correct := by
    have composite : ¬ Nat.Prime (corner 10949 1) := by change ¬ Nat.Prime 479544303; exact PrattCertificates.composite_of_product (factor := 3) (cofactor := 159848101) rfl (by decide) (by decide)
    exact decide_eq_false composite
  second := false
  second_correct := by
    have composite : ¬ Nat.Prime (corner 10949 2) := by change ¬ Nat.Prime 479522405; exact PrattCertificates.composite_of_product (factor := 5) (cofactor := 95904481) rfl (by decide) (by decide)
    exact decide_eq_false composite
  third := true
  third_correct := by
    have prime : Nat.Prime (corner 10949 3) := by change Nat.Prime 479500507; exact ProjectEulerSolutions.P58PrimeNodes.p_479500507.prime
    exact decide_eq_true prime
theorem row_count_10949 : row 10949 = 1 := (row_10949.checked).trans rfl
theorem model_count_10949 : modelRow 10949 = 1 := by decide
theorem modelrow_agrees_10949 : modelRow 10949 = row 10949 := model_count_10949.trans row_count_10949.symm
def row_10950 : RowCertificate 10950 where
  first := false
  first_correct := by
    have composite : ¬ Nat.Prime (corner 10950 1) := by change ¬ Nat.Prime 479631901; exact PrattCertificates.composite_of_product (factor := 7) (cofactor := 68518843) rfl (by decide) (by decide)
    exact decide_eq_false composite
  second := false
  second_correct := by
    have composite : ¬ Nat.Prime (corner 10950 2) := by change ¬ Nat.Prime 479610001; exact PrattCertificates.composite_of_product (factor := 13) (cofactor := 36893077) rfl (by decide) (by decide)
    exact decide_eq_false composite
  third := false
  third_correct := by
    have composite : ¬ Nat.Prime (corner 10950 3) := by change ¬ Nat.Prime 479588101; exact PrattCertificates.composite_of_product (factor := 19) (cofactor := 25241479) rfl (by decide) (by decide)
    exact decide_eq_false composite
theorem row_count_10950 : row 10950 = 0 := (row_10950.checked).trans rfl
theorem model_count_10950 : modelRow 10950 = 0 := by decide
theorem modelrow_agrees_10950 : modelRow 10950 = row 10950 := model_count_10950.trans row_count_10950.symm
def row_10951 : RowCertificate 10951 where
  first := true
  first_correct := by
    have prime : Nat.Prime (corner 10951 1) := by change Nat.Prime 479719507; exact ProjectEulerSolutions.P58PrimeNodes.p_479719507.prime
    exact decide_eq_true prime
  second := false
  second_correct := by
    have composite : ¬ Nat.Prime (corner 10951 2) := by change ¬ Nat.Prime 479697605; exact PrattCertificates.composite_of_product (factor := 5) (cofactor := 95939521) rfl (by decide) (by decide)
    exact decide_eq_false composite
  third := false
  third_correct := by
    have composite : ¬ Nat.Prime (corner 10951 3) := by change ¬ Nat.Prime 479675703; exact PrattCertificates.composite_of_product (factor := 3) (cofactor := 159891901) rfl (by decide) (by decide)
    exact decide_eq_false composite
theorem row_count_10951 : row 10951 = 1 := (row_10951.checked).trans rfl
theorem model_count_10951 : modelRow 10951 = 1 := by decide
theorem modelrow_agrees_10951 : modelRow 10951 = row 10951 := model_count_10951.trans row_count_10951.symm
def row_10952 : RowCertificate 10952 where
  first := false
  first_correct := by
    have composite : ¬ Nat.Prime (corner 10952 1) := by change ¬ Nat.Prime 479807121; exact PrattCertificates.composite_of_product (factor := 3) (cofactor := 159935707) rfl (by decide) (by decide)
    exact decide_eq_false composite
  second := false
  second_correct := by
    have composite : ¬ Nat.Prime (corner 10952 2) := by change ¬ Nat.Prime 479785217; exact PrattCertificates.composite_of_product (factor := 433) (cofactor := 1108049) rfl (by decide) (by decide)
    exact decide_eq_false composite
  third := false
  third_correct := by
    have composite : ¬ Nat.Prime (corner 10952 3) := by change ¬ Nat.Prime 479763313; exact PrattCertificates.composite_of_product (factor := 2689) (cofactor := 178417) rfl (by decide) (by decide)
    exact decide_eq_false composite
theorem row_count_10952 : row 10952 = 0 := (row_10952.checked).trans rfl
theorem model_count_10952 : modelRow 10952 = 0 := by decide
theorem modelrow_agrees_10952 : modelRow 10952 = row 10952 := model_count_10952.trans row_count_10952.symm
def row_10953 : RowCertificate 10953 where
  first := false
  first_correct := by
    have composite : ¬ Nat.Prime (corner 10953 1) := by change ¬ Nat.Prime 479894743; exact PrattCertificates.composite_of_product (factor := 79) (cofactor := 6074617) rfl (by decide) (by decide)
    exact decide_eq_false composite
  second := true
  second_correct := by
    have prime : Nat.Prime (corner 10953 2) := by change Nat.Prime 479872837; exact ProjectEulerSolutions.P58PrimeNodes.p_479872837.prime
    exact decide_eq_true prime
  third := false
  third_correct := by
    have composite : ¬ Nat.Prime (corner 10953 3) := by change ¬ Nat.Prime 479850931; exact PrattCertificates.composite_of_product (factor := 7) (cofactor := 68550133) rfl (by decide) (by decide)
    exact decide_eq_false composite
theorem row_count_10953 : row 10953 = 1 := (row_10953.checked).trans rfl
theorem model_count_10953 : modelRow 10953 = 1 := by decide
theorem modelrow_agrees_10953 : modelRow 10953 = row 10953 := model_count_10953.trans row_count_10953.symm
def row_10954 : RowCertificate 10954 where
  first := false
  first_correct := by
    have composite : ¬ Nat.Prime (corner 10954 1) := by change ¬ Nat.Prime 479982373; exact PrattCertificates.composite_of_product (factor := 13) (cofactor := 36921721) rfl (by decide) (by decide)
    exact decide_eq_false composite
  second := false
  second_correct := by
    have composite : ¬ Nat.Prime (corner 10954 2) := by change ¬ Nat.Prime 479960465; exact PrattCertificates.composite_of_product (factor := 5) (cofactor := 95992093) rfl (by decide) (by decide)
    exact decide_eq_false composite
  third := false
  third_correct := by
    have composite : ¬ Nat.Prime (corner 10954 3) := by change ¬ Nat.Prime 479938557; exact PrattCertificates.composite_of_product (factor := 3) (cofactor := 159979519) rfl (by decide) (by decide)
    exact decide_eq_false composite
theorem row_count_10954 : row 10954 = 0 := (row_10954.checked).trans rfl
theorem model_count_10954 : modelRow 10954 = 0 := by decide
theorem modelrow_agrees_10954 : modelRow 10954 = row 10954 := model_count_10954.trans row_count_10954.symm
def row_10955 : RowCertificate 10955 where
  first := false
  first_correct := by
    have composite : ¬ Nat.Prime (corner 10955 1) := by change ¬ Nat.Prime 480070011; exact PrattCertificates.composite_of_product (factor := 3) (cofactor := 160023337) rfl (by decide) (by decide)
    exact decide_eq_false composite
  second := false
  second_correct := by
    have composite : ¬ Nat.Prime (corner 10955 2) := by change ¬ Nat.Prime 480048101; exact PrattCertificates.composite_of_product (factor := 13) (cofactor := 36926777) rfl (by decide) (by decide)
    exact decide_eq_false composite
  third := false
  third_correct := by
    have composite : ¬ Nat.Prime (corner 10955 3) := by change ¬ Nat.Prime 480026191; exact PrattCertificates.composite_of_product (factor := 4483) (cofactor := 107077) rfl (by decide) (by decide)
    exact decide_eq_false composite
theorem row_count_10955 : row 10955 = 0 := (row_10955.checked).trans rfl
theorem model_count_10955 : modelRow 10955 = 0 := by decide
theorem modelrow_agrees_10955 : modelRow 10955 = row 10955 := model_count_10955.trans row_count_10955.symm
def row_10956 : RowCertificate 10956 where
  first := false
  first_correct := by
    have composite : ¬ Nat.Prime (corner 10956 1) := by change ¬ Nat.Prime 480157657; exact PrattCertificates.composite_of_product (factor := 7) (cofactor := 68593951) rfl (by decide) (by decide)
    exact decide_eq_false composite
  second := false
  second_correct := by
    have composite : ¬ Nat.Prime (corner 10956 2) := by change ¬ Nat.Prime 480135745; exact PrattCertificates.composite_of_product (factor := 5) (cofactor := 96027149) rfl (by decide) (by decide)
    exact decide_eq_false composite
  third := false
  third_correct := by
    have composite : ¬ Nat.Prime (corner 10956 3) := by change ¬ Nat.Prime 480113833; exact PrattCertificates.composite_of_product (factor := 31) (cofactor := 15487543) rfl (by decide) (by decide)
    exact decide_eq_false composite
theorem row_count_10956 : row 10956 = 0 := (row_10956.checked).trans rfl
theorem model_count_10956 : modelRow 10956 = 0 := by decide
theorem modelrow_agrees_10956 : modelRow 10956 = row 10956 := model_count_10956.trans row_count_10956.symm
def row_10957 : RowCertificate 10957 where
  first := false
  first_correct := by
    have composite : ¬ Nat.Prime (corner 10957 1) := by change ¬ Nat.Prime 480245311; exact PrattCertificates.composite_of_product (factor := 7) (cofactor := 68606473) rfl (by decide) (by decide)
    exact decide_eq_false composite
  second := true
  second_correct := by
    have prime : Nat.Prime (corner 10957 2) := by change Nat.Prime 480223397; exact ProjectEulerSolutions.P58PrimeNodes.p_480223397.prime
    exact decide_eq_true prime
  third := false
  third_correct := by
    have composite : ¬ Nat.Prime (corner 10957 3) := by change ¬ Nat.Prime 480201483; exact PrattCertificates.composite_of_product (factor := 3) (cofactor := 160067161) rfl (by decide) (by decide)
    exact decide_eq_false composite
theorem row_count_10957 : row 10957 = 1 := (row_10957.checked).trans rfl
theorem model_count_10957 : modelRow 10957 = 1 := by decide
theorem modelrow_agrees_10957 : modelRow 10957 = row 10957 := model_count_10957.trans row_count_10957.symm
def row_10958 : RowCertificate 10958 where
  first := false
  first_correct := by
    have composite : ¬ Nat.Prime (corner 10958 1) := by change ¬ Nat.Prime 480332973; exact PrattCertificates.composite_of_product (factor := 3) (cofactor := 160110991) rfl (by decide) (by decide)
    exact decide_eq_false composite
  second := true
  second_correct := by
    have prime : Nat.Prime (corner 10958 2) := by change Nat.Prime 480311057; exact ProjectEulerSolutions.P58PrimeNodes.p_480311057.prime
    exact decide_eq_true prime
  third := false
  third_correct := by
    have composite : ¬ Nat.Prime (corner 10958 3) := by change ¬ Nat.Prime 480289141; exact PrattCertificates.composite_of_product (factor := 241) (cofactor := 1992901) rfl (by decide) (by decide)
    exact decide_eq_false composite
theorem row_count_10958 : row 10958 = 1 := (row_10958.checked).trans rfl
theorem model_count_10958 : modelRow 10958 = 1 := by decide
theorem modelrow_agrees_10958 : modelRow 10958 = row 10958 := model_count_10958.trans row_count_10958.symm
def row_10959 : RowCertificate 10959 where
  first := false
  first_correct := by
    have composite : ¬ Nat.Prime (corner 10959 1) := by change ¬ Nat.Prime 480420643; exact PrattCertificates.composite_of_product (factor := 19) (cofactor := 25285297) rfl (by decide) (by decide)
    exact decide_eq_false composite
  second := false
  second_correct := by
    have composite : ¬ Nat.Prime (corner 10959 2) := by change ¬ Nat.Prime 480398725; exact PrattCertificates.composite_of_product (factor := 5) (cofactor := 96079745) rfl (by decide) (by decide)
    exact decide_eq_false composite
  third := false
  third_correct := by
    have composite : ¬ Nat.Prime (corner 10959 3) := by change ¬ Nat.Prime 480376807; exact PrattCertificates.composite_of_product (factor := 193) (cofactor := 2488999) rfl (by decide) (by decide)
    exact decide_eq_false composite
theorem row_count_10959 : row 10959 = 0 := (row_10959.checked).trans rfl
theorem model_count_10959 : modelRow 10959 = 0 := by decide
theorem modelrow_agrees_10959 : modelRow 10959 = row 10959 := model_count_10959.trans row_count_10959.symm
def row_10960 : RowCertificate 10960 where
  first := false
  first_correct := by
    have composite : ¬ Nat.Prime (corner 10960 1) := by change ¬ Nat.Prime 480508321; exact PrattCertificates.composite_of_product (factor := 547) (cofactor := 878443) rfl (by decide) (by decide)
    exact decide_eq_false composite
  second := false
  second_correct := by
    have composite : ¬ Nat.Prime (corner 10960 2) := by change ¬ Nat.Prime 480486401; exact PrattCertificates.composite_of_product (factor := 181) (cofactor := 2654621) rfl (by decide) (by decide)
    exact decide_eq_false composite
  third := false
  third_correct := by
    have composite : ¬ Nat.Prime (corner 10960 3) := by change ¬ Nat.Prime 480464481; exact PrattCertificates.composite_of_product (factor := 3) (cofactor := 160154827) rfl (by decide) (by decide)
    exact decide_eq_false composite
theorem row_count_10960 : row 10960 = 0 := (row_10960.checked).trans rfl
theorem model_count_10960 : modelRow 10960 = 0 := by decide
theorem modelrow_agrees_10960 : modelRow 10960 = row 10960 := model_count_10960.trans row_count_10960.symm
def row_10961 : RowCertificate 10961 where
  first := false
  first_correct := by
    have composite : ¬ Nat.Prime (corner 10961 1) := by change ¬ Nat.Prime 480596007; exact PrattCertificates.composite_of_product (factor := 3) (cofactor := 160198669) rfl (by decide) (by decide)
    exact decide_eq_false composite
  second := false
  second_correct := by
    have composite : ¬ Nat.Prime (corner 10961 2) := by change ¬ Nat.Prime 480574085; exact PrattCertificates.composite_of_product (factor := 5) (cofactor := 96114817) rfl (by decide) (by decide)
    exact decide_eq_false composite
  third := false
  third_correct := by
    have composite : ¬ Nat.Prime (corner 10961 3) := by change ¬ Nat.Prime 480552163; exact PrattCertificates.composite_of_product (factor := 7) (cofactor := 68650309) rfl (by decide) (by decide)
    exact decide_eq_false composite
theorem row_count_10961 : row 10961 = 0 := (row_10961.checked).trans rfl
theorem model_count_10961 : modelRow 10961 = 0 := by decide
theorem modelrow_agrees_10961 : modelRow 10961 = row 10961 := model_count_10961.trans row_count_10961.symm
def row_10962 : RowCertificate 10962 where
  first := true
  first_correct := by
    have prime : Nat.Prime (corner 10962 1) := by change Nat.Prime 480683701; exact ProjectEulerSolutions.P58PrimeNodes.p_480683701.prime
    exact decide_eq_true prime
  second := true
  second_correct := by
    have prime : Nat.Prime (corner 10962 2) := by change Nat.Prime 480661777; exact ProjectEulerSolutions.P58PrimeNodes.p_480661777.prime
    exact decide_eq_true prime
  third := false
  third_correct := by
    have composite : ¬ Nat.Prime (corner 10962 3) := by change ¬ Nat.Prime 480639853; exact PrattCertificates.composite_of_product (factor := 43) (cofactor := 11177671) rfl (by decide) (by decide)
    exact decide_eq_false composite
theorem row_count_10962 : row 10962 = 2 := (row_10962.checked).trans rfl
theorem model_count_10962 : modelRow 10962 = 2 := by decide
theorem modelrow_agrees_10962 : modelRow 10962 = row 10962 := model_count_10962.trans row_count_10962.symm
def row_10963 : RowCertificate 10963 where
  first := false
  first_correct := by
    have composite : ¬ Nat.Prime (corner 10963 1) := by change ¬ Nat.Prime 480771403; exact PrattCertificates.composite_of_product (factor := 7) (cofactor := 68681629) rfl (by decide) (by decide)
    exact decide_eq_false composite
  second := false
  second_correct := by
    have composite : ¬ Nat.Prime (corner 10963 2) := by change ¬ Nat.Prime 480749477; exact PrattCertificates.composite_of_product (factor := 13) (cofactor := 36980729) rfl (by decide) (by decide)
    exact decide_eq_false composite
  third := false
  third_correct := by
    have composite : ¬ Nat.Prime (corner 10963 3) := by change ¬ Nat.Prime 480727551; exact PrattCertificates.composite_of_product (factor := 3) (cofactor := 160242517) rfl (by decide) (by decide)
    exact decide_eq_false composite
theorem row_count_10963 : row 10963 = 0 := (row_10963.checked).trans rfl
theorem model_count_10963 : modelRow 10963 = 0 := by decide
theorem modelrow_agrees_10963 : modelRow 10963 = row 10963 := model_count_10963.trans row_count_10963.symm
def row_10964 : RowCertificate 10964 where
  first := false
  first_correct := by
    have composite : ¬ Nat.Prime (corner 10964 1) := by change ¬ Nat.Prime 480859113; exact PrattCertificates.composite_of_product (factor := 3) (cofactor := 160286371) rfl (by decide) (by decide)
    exact decide_eq_false composite
  second := false
  second_correct := by
    have composite : ¬ Nat.Prime (corner 10964 2) := by change ¬ Nat.Prime 480837185; exact PrattCertificates.composite_of_product (factor := 5) (cofactor := 96167437) rfl (by decide) (by decide)
    exact decide_eq_false composite
  third := false
  third_correct := by
    have composite : ¬ Nat.Prime (corner 10964 3) := by change ¬ Nat.Prime 480815257; exact PrattCertificates.composite_of_product (factor := 13) (cofactor := 36985789) rfl (by decide) (by decide)
    exact decide_eq_false composite
theorem row_count_10964 : row 10964 = 0 := (row_10964.checked).trans rfl
theorem model_count_10964 : modelRow 10964 = 0 := by decide
theorem modelrow_agrees_10964 : modelRow 10964 = row 10964 := model_count_10964.trans row_count_10964.symm
def row_10965 : RowCertificate 10965 where
  first := false
  first_correct := by
    have composite : ¬ Nat.Prime (corner 10965 1) := by change ¬ Nat.Prime 480946831; exact PrattCertificates.composite_of_product (factor := 37) (cofactor := 12998563) rfl (by decide) (by decide)
    exact decide_eq_false composite
  second := true
  second_correct := by
    have prime : Nat.Prime (corner 10965 2) := by change Nat.Prime 480924901; exact ProjectEulerSolutions.P58PrimeNodes.p_480924901.prime
    exact decide_eq_true prime
  third := false
  third_correct := by
    have composite : ¬ Nat.Prime (corner 10965 3) := by change ¬ Nat.Prime 480902971; exact PrattCertificates.composite_of_product (factor := 211) (cofactor := 2279161) rfl (by decide) (by decide)
    exact decide_eq_false composite
theorem row_count_10965 : row 10965 = 1 := (row_10965.checked).trans rfl
theorem model_count_10965 : modelRow 10965 = 1 := by decide
theorem modelrow_agrees_10965 : modelRow 10965 = row 10965 := model_count_10965.trans row_count_10965.symm
def row_10966 : RowCertificate 10966 where
  first := false
  first_correct := by
    have composite : ¬ Nat.Prime (corner 10966 1) := by change ¬ Nat.Prime 481034557; exact PrattCertificates.composite_of_product (factor := 4339) (cofactor := 110863) rfl (by decide) (by decide)
    exact decide_eq_false composite
  second := false
  second_correct := by
    have composite : ¬ Nat.Prime (corner 10966 2) := by change ¬ Nat.Prime 481012625; exact PrattCertificates.composite_of_product (factor := 5) (cofactor := 96202525) rfl (by decide) (by decide)
    exact decide_eq_false composite
  third := false
  third_correct := by
    have composite : ¬ Nat.Prime (corner 10966 3) := by change ¬ Nat.Prime 480990693; exact PrattCertificates.composite_of_product (factor := 3) (cofactor := 160330231) rfl (by decide) (by decide)
    exact decide_eq_false composite
theorem row_count_10966 : row 10966 = 0 := (row_10966.checked).trans rfl
theorem model_count_10966 : modelRow 10966 = 0 := by decide
theorem modelrow_agrees_10966 : modelRow 10966 = row 10966 := model_count_10966.trans row_count_10966.symm
def row_10967 : RowCertificate 10967 where
  first := false
  first_correct := by
    have composite : ¬ Nat.Prime (corner 10967 1) := by change ¬ Nat.Prime 481122291; exact PrattCertificates.composite_of_product (factor := 3) (cofactor := 160374097) rfl (by decide) (by decide)
    exact decide_eq_false composite
  second := false
  second_correct := by
    have composite : ¬ Nat.Prime (corner 10967 2) := by change ¬ Nat.Prime 481100357; exact PrattCertificates.composite_of_product (factor := 17) (cofactor := 28300021) rfl (by decide) (by decide)
    exact decide_eq_false composite
  third := false
  third_correct := by
    have composite : ¬ Nat.Prime (corner 10967 3) := by change ¬ Nat.Prime 481078423; exact PrattCertificates.composite_of_product (factor := 7) (cofactor := 68725489) rfl (by decide) (by decide)
    exact decide_eq_false composite
theorem row_count_10967 : row 10967 = 0 := (row_10967.checked).trans rfl
theorem model_count_10967 : modelRow 10967 = 0 := by decide
theorem modelrow_agrees_10967 : modelRow 10967 = row 10967 := model_count_10967.trans row_count_10967.symm
def row_10968 : RowCertificate 10968 where
  first := false
  first_correct := by
    have composite : ¬ Nat.Prime (corner 10968 1) := by change ¬ Nat.Prime 481210033; exact PrattCertificates.composite_of_product (factor := 43) (cofactor := 11190931) rfl (by decide) (by decide)
    exact decide_eq_false composite
  second := false
  second_correct := by
    have composite : ¬ Nat.Prime (corner 10968 2) := by change ¬ Nat.Prime 481188097; exact PrattCertificates.composite_of_product (factor := 13) (cofactor := 37014469) rfl (by decide) (by decide)
    exact decide_eq_false composite
  third := false
  third_correct := by
    have composite : ¬ Nat.Prime (corner 10968 3) := by change ¬ Nat.Prime 481166161; exact PrattCertificates.composite_of_product (factor := 7) (cofactor := 68738023) rfl (by decide) (by decide)
    exact decide_eq_false composite
theorem row_count_10968 : row 10968 = 0 := (row_10968.checked).trans rfl
theorem model_count_10968 : modelRow 10968 = 0 := by decide
theorem modelrow_agrees_10968 : modelRow 10968 = row 10968 := model_count_10968.trans row_count_10968.symm
def row_10969 : RowCertificate 10969 where
  first := false
  first_correct := by
    have composite : ¬ Nat.Prime (corner 10969 1) := by change ¬ Nat.Prime 481297783; exact PrattCertificates.composite_of_product (factor := 67) (cofactor := 7183549) rfl (by decide) (by decide)
    exact decide_eq_false composite
  second := false
  second_correct := by
    have composite : ¬ Nat.Prime (corner 10969 2) := by change ¬ Nat.Prime 481275845; exact PrattCertificates.composite_of_product (factor := 5) (cofactor := 96255169) rfl (by decide) (by decide)
    exact decide_eq_false composite
  third := false
  third_correct := by
    have composite : ¬ Nat.Prime (corner 10969 3) := by change ¬ Nat.Prime 481253907; exact PrattCertificates.composite_of_product (factor := 3) (cofactor := 160417969) rfl (by decide) (by decide)
    exact decide_eq_false composite
theorem row_count_10969 : row 10969 = 0 := (row_10969.checked).trans rfl
theorem model_count_10969 : modelRow 10969 = 0 := by decide
theorem modelrow_agrees_10969 : modelRow 10969 = row 10969 := model_count_10969.trans row_count_10969.symm
def row_10970 : RowCertificate 10970 where
  first := false
  first_correct := by
    have composite : ¬ Nat.Prime (corner 10970 1) := by change ¬ Nat.Prime 481385541; exact PrattCertificates.composite_of_product (factor := 3) (cofactor := 160461847) rfl (by decide) (by decide)
    exact decide_eq_false composite
  second := true
  second_correct := by
    have prime : Nat.Prime (corner 10970 2) := by change Nat.Prime 481363601; exact ProjectEulerSolutions.P58PrimeNodes.p_481363601.prime
    exact decide_eq_true prime
  third := true
  third_correct := by
    have prime : Nat.Prime (corner 10970 3) := by change Nat.Prime 481341661; exact ProjectEulerSolutions.P58PrimeNodes.p_481341661.prime
    exact decide_eq_true prime
theorem row_count_10970 : row 10970 = 2 := (row_10970.checked).trans rfl
theorem model_count_10970 : modelRow 10970 = 2 := by decide
theorem modelrow_agrees_10970 : modelRow 10970 = row 10970 := model_count_10970.trans row_count_10970.symm
def row_10971 : RowCertificate 10971 where
  first := false
  first_correct := by
    have composite : ¬ Nat.Prime (corner 10971 1) := by change ¬ Nat.Prime 481473307; exact PrattCertificates.composite_of_product (factor := 7) (cofactor := 68781901) rfl (by decide) (by decide)
    exact decide_eq_false composite
  second := false
  second_correct := by
    have composite : ¬ Nat.Prime (corner 10971 2) := by change ¬ Nat.Prime 481451365; exact PrattCertificates.composite_of_product (factor := 5) (cofactor := 96290273) rfl (by decide) (by decide)
    exact decide_eq_false composite
  third := false
  third_correct := by
    have composite : ¬ Nat.Prime (corner 10971 3) := by change ¬ Nat.Prime 481429423; exact PrattCertificates.composite_of_product (factor := 3001) (cofactor := 160423) rfl (by decide) (by decide)
    exact decide_eq_false composite
theorem row_count_10971 : row 10971 = 0 := (row_10971.checked).trans rfl
theorem model_count_10971 : modelRow 10971 = 0 := by decide
theorem modelrow_agrees_10971 : modelRow 10971 = row 10971 := model_count_10971.trans row_count_10971.symm
def row_10972 : RowCertificate 10972 where
  first := true
  first_correct := by
    have prime : Nat.Prime (corner 10972 1) := by change Nat.Prime 481561081; exact ProjectEulerSolutions.P58PrimeNodes.p_481561081.prime
    exact decide_eq_true prime
  second := false
  second_correct := by
    have composite : ¬ Nat.Prime (corner 10972 2) := by change ¬ Nat.Prime 481539137; exact PrattCertificates.composite_of_product (factor := 41) (cofactor := 11744857) rfl (by decide) (by decide)
    exact decide_eq_false composite
  third := false
  third_correct := by
    have composite : ¬ Nat.Prime (corner 10972 3) := by change ¬ Nat.Prime 481517193; exact PrattCertificates.composite_of_product (factor := 3) (cofactor := 160505731) rfl (by decide) (by decide)
    exact decide_eq_false composite
theorem row_count_10972 : row 10972 = 1 := (row_10972.checked).trans rfl
theorem model_count_10972 : modelRow 10972 = 1 := by decide
theorem modelrow_agrees_10972 : modelRow 10972 = row 10972 := model_count_10972.trans row_count_10972.symm
def row_10973 : RowCertificate 10973 where
  first := false
  first_correct := by
    have composite : ¬ Nat.Prime (corner 10973 1) := by change ¬ Nat.Prime 481648863; exact PrattCertificates.composite_of_product (factor := 3) (cofactor := 160549621) rfl (by decide) (by decide)
    exact decide_eq_false composite
  second := false
  second_correct := by
    have composite : ¬ Nat.Prime (corner 10973 2) := by change ¬ Nat.Prime 481626917; exact PrattCertificates.composite_of_product (factor := 73) (cofactor := 6597629) rfl (by decide) (by decide)
    exact decide_eq_false composite
  third := false
  third_correct := by
    have composite : ¬ Nat.Prime (corner 10973 3) := by change ¬ Nat.Prime 481604971; exact PrattCertificates.composite_of_product (factor := 1669) (cofactor := 288559) rfl (by decide) (by decide)
    exact decide_eq_false composite
theorem row_count_10973 : row 10973 = 0 := (row_10973.checked).trans rfl
theorem model_count_10973 : modelRow 10973 = 0 := by decide
theorem modelrow_agrees_10973 : modelRow 10973 = row 10973 := model_count_10973.trans row_count_10973.symm
def row_10974 : RowCertificate 10974 where
  first := true
  first_correct := by
    have prime : Nat.Prime (corner 10974 1) := by change Nat.Prime 481736653; exact ProjectEulerSolutions.P58PrimeNodes.p_481736653.prime
    exact decide_eq_true prime
  second := false
  second_correct := by
    have composite : ¬ Nat.Prime (corner 10974 2) := by change ¬ Nat.Prime 481714705; exact PrattCertificates.composite_of_product (factor := 5) (cofactor := 96342941) rfl (by decide) (by decide)
    exact decide_eq_false composite
  third := false
  third_correct := by
    have composite : ¬ Nat.Prime (corner 10974 3) := by change ¬ Nat.Prime 481692757; exact PrattCertificates.composite_of_product (factor := 7) (cofactor := 68813251) rfl (by decide) (by decide)
    exact decide_eq_false composite
theorem row_count_10974 : row 10974 = 1 := (row_10974.checked).trans rfl
theorem model_count_10974 : modelRow 10974 = 1 := by decide
theorem modelrow_agrees_10974 : modelRow 10974 = row 10974 := model_count_10974.trans row_count_10974.symm
def row_10975 : RowCertificate 10975 where
  first := true
  first_correct := by
    have prime : Nat.Prime (corner 10975 1) := by change Nat.Prime 481824451; exact ProjectEulerSolutions.P58PrimeNodes.p_481824451.prime
    exact decide_eq_true prime
  second := false
  second_correct := by
    have composite : ¬ Nat.Prime (corner 10975 2) := by change ¬ Nat.Prime 481802501; exact PrattCertificates.composite_of_product (factor := 577) (cofactor := 835013) rfl (by decide) (by decide)
    exact decide_eq_false composite
  third := false
  third_correct := by
    have composite : ¬ Nat.Prime (corner 10975 3) := by change ¬ Nat.Prime 481780551; exact PrattCertificates.composite_of_product (factor := 3) (cofactor := 160593517) rfl (by decide) (by decide)
    exact decide_eq_false composite
theorem row_count_10975 : row 10975 = 1 := (row_10975.checked).trans rfl
theorem model_count_10975 : modelRow 10975 = 1 := by decide
theorem modelrow_agrees_10975 : modelRow 10975 = row 10975 := model_count_10975.trans row_count_10975.symm
def row_10976 : RowCertificate 10976 where
  first := false
  first_correct := by
    have composite : ¬ Nat.Prime (corner 10976 1) := by change ¬ Nat.Prime 481912257; exact PrattCertificates.composite_of_product (factor := 3) (cofactor := 160637419) rfl (by decide) (by decide)
    exact decide_eq_false composite
  second := false
  second_correct := by
    have composite : ¬ Nat.Prime (corner 10976 2) := by change ¬ Nat.Prime 481890305; exact PrattCertificates.composite_of_product (factor := 5) (cofactor := 96378061) rfl (by decide) (by decide)
    exact decide_eq_false composite
  third := false
  third_correct := by
    have composite : ¬ Nat.Prime (corner 10976 3) := by change ¬ Nat.Prime 481868353; exact PrattCertificates.composite_of_product (factor := 37) (cofactor := 13023469) rfl (by decide) (by decide)
    exact decide_eq_false composite
theorem row_count_10976 : row 10976 = 0 := (row_10976.checked).trans rfl
theorem model_count_10976 : modelRow 10976 = 0 := by decide
theorem modelrow_agrees_10976 : modelRow 10976 = row 10976 := model_count_10976.trans row_count_10976.symm
def row_10977 : RowCertificate 10977 where
  first := false
  first_correct := by
    have composite : ¬ Nat.Prime (corner 10977 1) := by change ¬ Nat.Prime 482000071; exact PrattCertificates.composite_of_product (factor := 7) (cofactor := 68857153) rfl (by decide) (by decide)
    exact decide_eq_false composite
  second := true
  second_correct := by
    have prime : Nat.Prime (corner 10977 2) := by change Nat.Prime 481978117; exact ProjectEulerSolutions.P58PrimeNodes.p_481978117.prime
    exact decide_eq_true prime
  third := false
  third_correct := by
    have composite : ¬ Nat.Prime (corner 10977 3) := by change ¬ Nat.Prime 481956163; exact PrattCertificates.composite_of_product (factor := 13) (cofactor := 37073551) rfl (by decide) (by decide)
    exact decide_eq_false composite
theorem row_count_10977 : row 10977 = 1 := (row_10977.checked).trans rfl
theorem model_count_10977 : modelRow 10977 = 1 := by decide
theorem modelrow_agrees_10977 : modelRow 10977 = row 10977 := model_count_10977.trans row_count_10977.symm
def row_10978 : RowCertificate 10978 where
  first := false
  first_correct := by
    have composite : ¬ Nat.Prime (corner 10978 1) := by change ¬ Nat.Prime 482087893; exact PrattCertificates.composite_of_product (factor := 7) (cofactor := 68869699) rfl (by decide) (by decide)
    exact decide_eq_false composite
  second := true
  second_correct := by
    have prime : Nat.Prime (corner 10978 2) := by change Nat.Prime 482065937; exact ProjectEulerSolutions.P58PrimeNodes.p_482065937.prime
    exact decide_eq_true prime
  third := false
  third_correct := by
    have composite : ¬ Nat.Prime (corner 10978 3) := by change ¬ Nat.Prime 482043981; exact PrattCertificates.composite_of_product (factor := 3) (cofactor := 160681327) rfl (by decide) (by decide)
    exact decide_eq_false composite
theorem row_count_10978 : row 10978 = 1 := (row_10978.checked).trans rfl
theorem model_count_10978 : modelRow 10978 = 1 := by decide
theorem modelrow_agrees_10978 : modelRow 10978 = row 10978 := model_count_10978.trans row_count_10978.symm
def row_10979 : RowCertificate 10979 where
  first := false
  first_correct := by
    have composite : ¬ Nat.Prime (corner 10979 1) := by change ¬ Nat.Prime 482175723; exact PrattCertificates.composite_of_product (factor := 3) (cofactor := 160725241) rfl (by decide) (by decide)
    exact decide_eq_false composite
  second := false
  second_correct := by
    have composite : ¬ Nat.Prime (corner 10979 2) := by change ¬ Nat.Prime 482153765; exact PrattCertificates.composite_of_product (factor := 5) (cofactor := 96430753) rfl (by decide) (by decide)
    exact decide_eq_false composite
  third := false
  third_correct := by
    have composite : ¬ Nat.Prime (corner 10979 3) := by change ¬ Nat.Prime 482131807; exact PrattCertificates.composite_of_product (factor := 97) (cofactor := 4970431) rfl (by decide) (by decide)
    exact decide_eq_false composite
theorem row_count_10979 : row 10979 = 0 := (row_10979.checked).trans rfl
theorem model_count_10979 : modelRow 10979 = 0 := by decide
theorem modelrow_agrees_10979 : modelRow 10979 = row 10979 := model_count_10979.trans row_count_10979.symm
def row_10980 : RowCertificate 10980 where
  first := false
  first_correct := by
    have composite : ¬ Nat.Prime (corner 10980 1) := by change ¬ Nat.Prime 482263561; exact PrattCertificates.composite_of_product (factor := 13) (cofactor := 37097197) rfl (by decide) (by decide)
    exact decide_eq_false composite
  second := false
  second_correct := by
    have composite : ¬ Nat.Prime (corner 10980 2) := by change ¬ Nat.Prime 482241601; exact PrattCertificates.composite_of_product (factor := 17) (cofactor := 28367153) rfl (by decide) (by decide)
    exact decide_eq_false composite
  third := false
  third_correct := by
    have composite : ¬ Nat.Prime (corner 10980 3) := by change ¬ Nat.Prime 482219641; exact PrattCertificates.composite_of_product (factor := 1039) (cofactor := 464119) rfl (by decide) (by decide)
    exact decide_eq_false composite
theorem row_count_10980 : row 10980 = 0 := (row_10980.checked).trans rfl
theorem model_count_10980 : modelRow 10980 = 0 := by decide
theorem modelrow_agrees_10980 : modelRow 10980 = row 10980 := model_count_10980.trans row_count_10980.symm
def row_10981 : RowCertificate 10981 where
  first := true
  first_correct := by
    have prime : Nat.Prime (corner 10981 1) := by change Nat.Prime 482351407; exact ProjectEulerSolutions.P58PrimeNodes.p_482351407.prime
    exact decide_eq_true prime
  second := false
  second_correct := by
    have composite : ¬ Nat.Prime (corner 10981 2) := by change ¬ Nat.Prime 482329445; exact PrattCertificates.composite_of_product (factor := 5) (cofactor := 96465889) rfl (by decide) (by decide)
    exact decide_eq_false composite
  third := false
  third_correct := by
    have composite : ¬ Nat.Prime (corner 10981 3) := by change ¬ Nat.Prime 482307483; exact PrattCertificates.composite_of_product (factor := 3) (cofactor := 160769161) rfl (by decide) (by decide)
    exact decide_eq_false composite
theorem row_count_10981 : row 10981 = 1 := (row_10981.checked).trans rfl
theorem model_count_10981 : modelRow 10981 = 1 := by decide
theorem modelrow_agrees_10981 : modelRow 10981 = row 10981 := model_count_10981.trans row_count_10981.symm
def row_10982 : RowCertificate 10982 where
  first := false
  first_correct := by
    have composite : ¬ Nat.Prime (corner 10982 1) := by change ¬ Nat.Prime 482439261; exact PrattCertificates.composite_of_product (factor := 3) (cofactor := 160813087) rfl (by decide) (by decide)
    exact decide_eq_false composite
  second := true
  second_correct := by
    have prime : Nat.Prime (corner 10982 2) := by change Nat.Prime 482417297; exact ProjectEulerSolutions.P58PrimeNodes.p_482417297.prime
    exact decide_eq_true prime
  third := false
  third_correct := by
    have composite : ¬ Nat.Prime (corner 10982 3) := by change ¬ Nat.Prime 482395333; exact PrattCertificates.composite_of_product (factor := 7) (cofactor := 68913619) rfl (by decide) (by decide)
    exact decide_eq_false composite
theorem row_count_10982 : row 10982 = 1 := (row_10982.checked).trans rfl
theorem model_count_10982 : modelRow 10982 = 1 := by decide
theorem modelrow_agrees_10982 : modelRow 10982 = row 10982 := model_count_10982.trans row_count_10982.symm
def row_10983 : RowCertificate 10983 where
  first := false
  first_correct := by
    have composite : ¬ Nat.Prime (corner 10983 1) := by change ¬ Nat.Prime 482527123; exact PrattCertificates.composite_of_product (factor := 13) (cofactor := 37117471) rfl (by decide) (by decide)
    exact decide_eq_false composite
  second := false
  second_correct := by
    have composite : ¬ Nat.Prime (corner 10983 2) := by change ¬ Nat.Prime 482505157; exact PrattCertificates.composite_of_product (factor := 2677) (cofactor := 180241) rfl (by decide) (by decide)
    exact decide_eq_false composite
  third := false
  third_correct := by
    have composite : ¬ Nat.Prime (corner 10983 3) := by change ¬ Nat.Prime 482483191; exact PrattCertificates.composite_of_product (factor := 1237) (cofactor := 390043) rfl (by decide) (by decide)
    exact decide_eq_false composite
theorem row_count_10983 : row 10983 = 0 := (row_10983.checked).trans rfl
theorem model_count_10983 : modelRow 10983 = 0 := by decide
theorem modelrow_agrees_10983 : modelRow 10983 = row 10983 := model_count_10983.trans row_count_10983.symm
def row_10984 : RowCertificate 10984 where
  first := false
  first_correct := by
    have composite : ¬ Nat.Prime (corner 10984 1) := by change ¬ Nat.Prime 482614993; exact PrattCertificates.composite_of_product (factor := 7) (cofactor := 68944999) rfl (by decide) (by decide)
    exact decide_eq_false composite
  second := false
  second_correct := by
    have composite : ¬ Nat.Prime (corner 10984 2) := by change ¬ Nat.Prime 482593025; exact PrattCertificates.composite_of_product (factor := 5) (cofactor := 96518605) rfl (by decide) (by decide)
    exact decide_eq_false composite
  third := false
  third_correct := by
    have composite : ¬ Nat.Prime (corner 10984 3) := by change ¬ Nat.Prime 482571057; exact PrattCertificates.composite_of_product (factor := 3) (cofactor := 160857019) rfl (by decide) (by decide)
    exact decide_eq_false composite
theorem row_count_10984 : row 10984 = 0 := (row_10984.checked).trans rfl
theorem model_count_10984 : modelRow 10984 = 0 := by decide
theorem modelrow_agrees_10984 : modelRow 10984 = row 10984 := model_count_10984.trans row_count_10984.symm
def row_10985 : RowCertificate 10985 where
  first := false
  first_correct := by
    have composite : ¬ Nat.Prime (corner 10985 1) := by change ¬ Nat.Prime 482702871; exact PrattCertificates.composite_of_product (factor := 3) (cofactor := 160900957) rfl (by decide) (by decide)
    exact decide_eq_false composite
  second := false
  second_correct := by
    have composite : ¬ Nat.Prime (corner 10985 2) := by change ¬ Nat.Prime 482680901; exact PrattCertificates.composite_of_product (factor := 29) (cofactor := 16644169) rfl (by decide) (by decide)
    exact decide_eq_false composite
  third := true
  third_correct := by
    have prime : Nat.Prime (corner 10985 3) := by change Nat.Prime 482658931; exact ProjectEulerSolutions.P58PrimeNodes.p_482658931.prime
    exact decide_eq_true prime
theorem row_count_10985 : row 10985 = 1 := (row_10985.checked).trans rfl
theorem model_count_10985 : modelRow 10985 = 1 := by decide
theorem modelrow_agrees_10985 : modelRow 10985 = row 10985 := model_count_10985.trans row_count_10985.symm
def row_10986 : RowCertificate 10986 where
  first := false
  first_correct := by
    have composite : ¬ Nat.Prime (corner 10986 1) := by change ¬ Nat.Prime 482790757; exact PrattCertificates.composite_of_product (factor := 109) (cofactor := 4429273) rfl (by decide) (by decide)
    exact decide_eq_false composite
  second := false
  second_correct := by
    have composite : ¬ Nat.Prime (corner 10986 2) := by change ¬ Nat.Prime 482768785; exact PrattCertificates.composite_of_product (factor := 5) (cofactor := 96553757) rfl (by decide) (by decide)
    exact decide_eq_false composite
  third := false
  third_correct := by
    have composite : ¬ Nat.Prime (corner 10986 3) := by change ¬ Nat.Prime 482746813; exact PrattCertificates.composite_of_product (factor := 19) (cofactor := 25407727) rfl (by decide) (by decide)
    exact decide_eq_false composite
theorem row_count_10986 : row 10986 = 0 := (row_10986.checked).trans rfl
theorem model_count_10986 : modelRow 10986 = 0 := by decide
theorem modelrow_agrees_10986 : modelRow 10986 = row 10986 := model_count_10986.trans row_count_10986.symm
def row_10987 : RowCertificate 10987 where
  first := true
  first_correct := by
    have prime : Nat.Prime (corner 10987 1) := by change Nat.Prime 482878651; exact ProjectEulerSolutions.P58PrimeNodes.p_482878651.prime
    exact decide_eq_true prime
  second := false
  second_correct := by
    have composite : ¬ Nat.Prime (corner 10987 2) := by change ¬ Nat.Prime 482856677; exact PrattCertificates.composite_of_product (factor := 8089) (cofactor := 59693) rfl (by decide) (by decide)
    exact decide_eq_false composite
  third := false
  third_correct := by
    have composite : ¬ Nat.Prime (corner 10987 3) := by change ¬ Nat.Prime 482834703; exact PrattCertificates.composite_of_product (factor := 3) (cofactor := 160944901) rfl (by decide) (by decide)
    exact decide_eq_false composite
theorem row_count_10987 : row 10987 = 1 := (row_10987.checked).trans rfl
theorem model_count_10987 : modelRow 10987 = 1 := by decide
theorem modelrow_agrees_10987 : modelRow 10987 = row 10987 := model_count_10987.trans row_count_10987.symm
def row_10988 : RowCertificate 10988 where
  first := false
  first_correct := by
    have composite : ¬ Nat.Prime (corner 10988 1) := by change ¬ Nat.Prime 482966553; exact PrattCertificates.composite_of_product (factor := 3) (cofactor := 160988851) rfl (by decide) (by decide)
    exact decide_eq_false composite
  second := false
  second_correct := by
    have composite : ¬ Nat.Prime (corner 10988 2) := by change ¬ Nat.Prime 482944577; exact PrattCertificates.composite_of_product (factor := 269) (cofactor := 1795333) rfl (by decide) (by decide)
    exact decide_eq_false composite
  third := false
  third_correct := by
    have composite : ¬ Nat.Prime (corner 10988 3) := by change ¬ Nat.Prime 482922601; exact PrattCertificates.composite_of_product (factor := 7) (cofactor := 68988943) rfl (by decide) (by decide)
    exact decide_eq_false composite
theorem row_count_10988 : row 10988 = 0 := (row_10988.checked).trans rfl
theorem model_count_10988 : modelRow 10988 = 0 := by decide
theorem modelrow_agrees_10988 : modelRow 10988 = row 10988 := model_count_10988.trans row_count_10988.symm
def row_10989 : RowCertificate 10989 where
  first := false
  first_correct := by
    have composite : ¬ Nat.Prime (corner 10989 1) := by change ¬ Nat.Prime 483054463; exact PrattCertificates.composite_of_product (factor := 3853) (cofactor := 125371) rfl (by decide) (by decide)
    exact decide_eq_false composite
  second := false
  second_correct := by
    have composite : ¬ Nat.Prime (corner 10989 2) := by change ¬ Nat.Prime 483032485; exact PrattCertificates.composite_of_product (factor := 5) (cofactor := 96606497) rfl (by decide) (by decide)
    exact decide_eq_false composite
  third := false
  third_correct := by
    have composite : ¬ Nat.Prime (corner 10989 3) := by change ¬ Nat.Prime 483010507; exact PrattCertificates.composite_of_product (factor := 7) (cofactor := 69001501) rfl (by decide) (by decide)
    exact decide_eq_false composite
theorem row_count_10989 : row 10989 = 0 := (row_10989.checked).trans rfl
theorem model_count_10989 : modelRow 10989 = 0 := by decide
theorem modelrow_agrees_10989 : modelRow 10989 = row 10989 := model_count_10989.trans row_count_10989.symm
def row_10990 : RowCertificate 10990 where
  first := false
  first_correct := by
    have composite : ¬ Nat.Prime (corner 10990 1) := by change ¬ Nat.Prime 483142381; exact PrattCertificates.composite_of_product (factor := 757) (cofactor := 638233) rfl (by decide) (by decide)
    exact decide_eq_false composite
  second := true
  second_correct := by
    have prime : Nat.Prime (corner 10990 2) := by change Nat.Prime 483120401; exact ProjectEulerSolutions.P58PrimeNodes.p_483120401.prime
    exact decide_eq_true prime
  third := false
  third_correct := by
    have composite : ¬ Nat.Prime (corner 10990 3) := by change ¬ Nat.Prime 483098421; exact PrattCertificates.composite_of_product (factor := 3) (cofactor := 161032807) rfl (by decide) (by decide)
    exact decide_eq_false composite
theorem row_count_10990 : row 10990 = 1 := (row_10990.checked).trans rfl
theorem model_count_10990 : modelRow 10990 = 1 := by decide
theorem modelrow_agrees_10990 : modelRow 10990 = row 10990 := model_count_10990.trans row_count_10990.symm
def row_10991 : RowCertificate 10991 where
  first := false
  first_correct := by
    have composite : ¬ Nat.Prime (corner 10991 1) := by change ¬ Nat.Prime 483230307; exact PrattCertificates.composite_of_product (factor := 3) (cofactor := 161076769) rfl (by decide) (by decide)
    exact decide_eq_false composite
  second := false
  second_correct := by
    have composite : ¬ Nat.Prime (corner 10991 2) := by change ¬ Nat.Prime 483208325; exact PrattCertificates.composite_of_product (factor := 5) (cofactor := 96641665) rfl (by decide) (by decide)
    exact decide_eq_false composite
  third := false
  third_correct := by
    have composite : ¬ Nat.Prime (corner 10991 3) := by change ¬ Nat.Prime 483186343; exact PrattCertificates.composite_of_product (factor := 73) (cofactor := 6618991) rfl (by decide) (by decide)
    exact decide_eq_false composite
theorem row_count_10991 : row 10991 = 0 := (row_10991.checked).trans rfl
theorem model_count_10991 : modelRow 10991 = 0 := by decide
theorem modelrow_agrees_10991 : modelRow 10991 = row 10991 := model_count_10991.trans row_count_10991.symm
def row_10992 : RowCertificate 10992 where
  first := false
  first_correct := by
    have composite : ¬ Nat.Prime (corner 10992 1) := by change ¬ Nat.Prime 483318241; exact PrattCertificates.composite_of_product (factor := 7) (cofactor := 69045463) rfl (by decide) (by decide)
    exact decide_eq_false composite
  second := false
  second_correct := by
    have composite : ¬ Nat.Prime (corner 10992 2) := by change ¬ Nat.Prime 483296257; exact PrattCertificates.composite_of_product (factor := 37) (cofactor := 13062061) rfl (by decide) (by decide)
    exact decide_eq_false composite
  third := false
  third_correct := by
    have composite : ¬ Nat.Prime (corner 10992 3) := by change ¬ Nat.Prime 483274273; exact PrattCertificates.composite_of_product (factor := 97) (cofactor := 4982209) rfl (by decide) (by decide)
    exact decide_eq_false composite
theorem row_count_10992 : row 10992 = 0 := (row_10992.checked).trans rfl
theorem model_count_10992 : modelRow 10992 = 0 := by decide
theorem modelrow_agrees_10992 : modelRow 10992 = row 10992 := model_count_10992.trans row_count_10992.symm
def row_10993 : RowCertificate 10993 where
  first := false
  first_correct := by
    have composite : ¬ Nat.Prime (corner 10993 1) := by change ¬ Nat.Prime 483406183; exact PrattCertificates.composite_of_product (factor := 13) (cofactor := 37185091) rfl (by decide) (by decide)
    exact decide_eq_false composite
  second := true
  second_correct := by
    have prime : Nat.Prime (corner 10993 2) := by change Nat.Prime 483384197; exact ProjectEulerSolutions.P58PrimeNodes.p_483384197.prime
    exact decide_eq_true prime
  third := false
  third_correct := by
    have composite : ¬ Nat.Prime (corner 10993 3) := by change ¬ Nat.Prime 483362211; exact PrattCertificates.composite_of_product (factor := 3) (cofactor := 161120737) rfl (by decide) (by decide)
    exact decide_eq_false composite
theorem row_count_10993 : row 10993 = 1 := (row_10993.checked).trans rfl
theorem model_count_10993 : modelRow 10993 = 1 := by decide
theorem modelrow_agrees_10993 : modelRow 10993 = row 10993 := model_count_10993.trans row_count_10993.symm
def row_10994 : RowCertificate 10994 where
  first := false
  first_correct := by
    have composite : ¬ Nat.Prime (corner 10994 1) := by change ¬ Nat.Prime 483494133; exact PrattCertificates.composite_of_product (factor := 3) (cofactor := 161164711) rfl (by decide) (by decide)
    exact decide_eq_false composite
  second := false
  second_correct := by
    have composite : ¬ Nat.Prime (corner 10994 2) := by change ¬ Nat.Prime 483472145; exact PrattCertificates.composite_of_product (factor := 5) (cofactor := 96694429) rfl (by decide) (by decide)
    exact decide_eq_false composite
  third := false
  third_correct := by
    have composite : ¬ Nat.Prime (corner 10994 3) := by change ¬ Nat.Prime 483450157; exact PrattCertificates.composite_of_product (factor := 4177) (cofactor := 115741) rfl (by decide) (by decide)
    exact decide_eq_false composite
theorem row_count_10994 : row 10994 = 0 := (row_10994.checked).trans rfl
theorem model_count_10994 : modelRow 10994 = 0 := by decide
theorem modelrow_agrees_10994 : modelRow 10994 = row 10994 := model_count_10994.trans row_count_10994.symm
def row_10995 : RowCertificate 10995 where
  first := false
  first_correct := by
    have composite : ¬ Nat.Prime (corner 10995 1) := by change ¬ Nat.Prime 483582091; exact PrattCertificates.composite_of_product (factor := 19) (cofactor := 25451689) rfl (by decide) (by decide)
    exact decide_eq_false composite
  second := true
  second_correct := by
    have prime : Nat.Prime (corner 10995 2) := by change Nat.Prime 483560101; exact ProjectEulerSolutions.P58PrimeNodes.p_483560101.prime
    exact decide_eq_true prime
  third := false
  third_correct := by
    have composite : ¬ Nat.Prime (corner 10995 3) := by change ¬ Nat.Prime 483538111; exact PrattCertificates.composite_of_product (factor := 7) (cofactor := 69076873) rfl (by decide) (by decide)
    exact decide_eq_false composite
theorem row_count_10995 : row 10995 = 1 := (row_10995.checked).trans rfl
theorem model_count_10995 : modelRow 10995 = 1 := by decide
theorem modelrow_agrees_10995 : modelRow 10995 = row 10995 := model_count_10995.trans row_count_10995.symm
def row_10996 : RowCertificate 10996 where
  first := false
  first_correct := by
    have composite : ¬ Nat.Prime (corner 10996 1) := by change ¬ Nat.Prime 483670057; exact PrattCertificates.composite_of_product (factor := 13) (cofactor := 37205389) rfl (by decide) (by decide)
    exact decide_eq_false composite
  second := false
  second_correct := by
    have composite : ¬ Nat.Prime (corner 10996 2) := by change ¬ Nat.Prime 483648065; exact PrattCertificates.composite_of_product (factor := 5) (cofactor := 96729613) rfl (by decide) (by decide)
    exact decide_eq_false composite
  third := false
  third_correct := by
    have composite : ¬ Nat.Prime (corner 10996 3) := by change ¬ Nat.Prime 483626073; exact PrattCertificates.composite_of_product (factor := 3) (cofactor := 161208691) rfl (by decide) (by decide)
    exact decide_eq_false composite
theorem row_count_10996 : row 10996 = 0 := (row_10996.checked).trans rfl
theorem model_count_10996 : modelRow 10996 = 0 := by decide
theorem modelrow_agrees_10996 : modelRow 10996 = row 10996 := model_count_10996.trans row_count_10996.symm
def row_10997 : RowCertificate 10997 where
  first := false
  first_correct := by
    have composite : ¬ Nat.Prime (corner 10997 1) := by change ¬ Nat.Prime 483758031; exact PrattCertificates.composite_of_product (factor := 3) (cofactor := 161252677) rfl (by decide) (by decide)
    exact decide_eq_false composite
  second := false
  second_correct := by
    have composite : ¬ Nat.Prime (corner 10997 2) := by change ¬ Nat.Prime 483736037; exact PrattCertificates.composite_of_product (factor := 17) (cofactor := 28455061) rfl (by decide) (by decide)
    exact decide_eq_false composite
  third := false
  third_correct := by
    have composite : ¬ Nat.Prime (corner 10997 3) := by change ¬ Nat.Prime 483714043; exact PrattCertificates.composite_of_product (factor := 1399) (cofactor := 345757) rfl (by decide) (by decide)
    exact decide_eq_false composite
theorem row_count_10997 : row 10997 = 0 := (row_10997.checked).trans rfl
theorem model_count_10997 : modelRow 10997 = 0 := by decide
theorem modelrow_agrees_10997 : modelRow 10997 = row 10997 := model_count_10997.trans row_count_10997.symm
def row_10998 : RowCertificate 10998 where
  first := false
  first_correct := by
    have composite : ¬ Nat.Prime (corner 10998 1) := by change ¬ Nat.Prime 483846013; exact PrattCertificates.composite_of_product (factor := 7) (cofactor := 69120859) rfl (by decide) (by decide)
    exact decide_eq_false composite
  second := true
  second_correct := by
    have prime : Nat.Prime (corner 10998 2) := by change Nat.Prime 483824017; exact ProjectEulerSolutions.P58PrimeNodes.p_483824017.prime
    exact decide_eq_true prime
  third := false
  third_correct := by
    have composite : ¬ Nat.Prime (corner 10998 3) := by change ¬ Nat.Prime 483802021; exact PrattCertificates.composite_of_product (factor := 103) (cofactor := 4697107) rfl (by decide) (by decide)
    exact decide_eq_false composite
theorem row_count_10998 : row 10998 = 1 := (row_10998.checked).trans rfl
theorem model_count_10998 : modelRow 10998 = 1 := by decide
theorem modelrow_agrees_10998 : modelRow 10998 = row 10998 := model_count_10998.trans row_count_10998.symm
def row_10999 : RowCertificate 10999 where
  first := false
  first_correct := by
    have composite : ¬ Nat.Prime (corner 10999 1) := by change ¬ Nat.Prime 483934003; exact PrattCertificates.composite_of_product (factor := 7) (cofactor := 69133429) rfl (by decide) (by decide)
    exact decide_eq_false composite
  second := false
  second_correct := by
    have composite : ¬ Nat.Prime (corner 10999 2) := by change ¬ Nat.Prime 483912005; exact PrattCertificates.composite_of_product (factor := 5) (cofactor := 96782401) rfl (by decide) (by decide)
    exact decide_eq_false composite
  third := false
  third_correct := by
    have composite : ¬ Nat.Prime (corner 10999 3) := by change ¬ Nat.Prime 483890007; exact PrattCertificates.composite_of_product (factor := 3) (cofactor := 161296669) rfl (by decide) (by decide)
    exact decide_eq_false composite
theorem row_count_10999 : row 10999 = 0 := (row_10999.checked).trans rfl
theorem model_count_10999 : modelRow 10999 = 0 := by decide
theorem modelrow_agrees_10999 : modelRow 10999 = row 10999 := model_count_10999.trans row_count_10999.symm
def row_11000 : RowCertificate 11000 where
  first := false
  first_correct := by
    have composite : ¬ Nat.Prime (corner 11000 1) := by change ¬ Nat.Prime 484022001; exact PrattCertificates.composite_of_product (factor := 3) (cofactor := 161340667) rfl (by decide) (by decide)
    exact decide_eq_false composite
  second := false
  second_correct := by
    have composite : ¬ Nat.Prime (corner 11000 2) := by change ¬ Nat.Prime 484000001; exact PrattCertificates.composite_of_product (factor := 73) (cofactor := 6630137) rfl (by decide) (by decide)
    exact decide_eq_false composite
  third := false
  third_correct := by
    have composite : ¬ Nat.Prime (corner 11000 3) := by change ¬ Nat.Prime 483978001; exact PrattCertificates.composite_of_product (factor := 13) (cofactor := 37229077) rfl (by decide) (by decide)
    exact decide_eq_false composite
theorem row_count_11000 : row 11000 = 0 := (row_11000.checked).trans rfl
theorem model_count_11000 : modelRow 11000 = 0 := by decide
theorem modelrow_agrees_11000 : modelRow 11000 = row 11000 := model_count_11000.trans row_count_11000.symm
def row_11001 : RowCertificate 11001 where
  first := false
  first_correct := by
    have composite : ¬ Nat.Prime (corner 11001 1) := by change ¬ Nat.Prime 484110007; exact PrattCertificates.composite_of_product (factor := 613) (cofactor := 789739) rfl (by decide) (by decide)
    exact decide_eq_false composite
  second := false
  second_correct := by
    have composite : ¬ Nat.Prime (corner 11001 2) := by change ¬ Nat.Prime 484088005; exact PrattCertificates.composite_of_product (factor := 5) (cofactor := 96817601) rfl (by decide) (by decide)
    exact decide_eq_false composite
  third := false
  third_correct := by
    have composite : ¬ Nat.Prime (corner 11001 3) := by change ¬ Nat.Prime 484066003; exact PrattCertificates.composite_of_product (factor := 8887) (cofactor := 54469) rfl (by decide) (by decide)
    exact decide_eq_false composite
theorem row_count_11001 : row 11001 = 0 := (row_11001.checked).trans rfl
theorem model_count_11001 : modelRow 11001 = 0 := by decide
theorem modelrow_agrees_11001 : modelRow 11001 = row 11001 := model_count_11001.trans row_count_11001.symm
def row_11002 : RowCertificate 11002 where
  first := false
  first_correct := by
    have composite : ¬ Nat.Prime (corner 11002 1) := by change ¬ Nat.Prime 484198021; exact PrattCertificates.composite_of_product (factor := 31) (cofactor := 15619291) rfl (by decide) (by decide)
    exact decide_eq_false composite
  second := false
  second_correct := by
    have composite : ¬ Nat.Prime (corner 11002 2) := by change ¬ Nat.Prime 484176017; exact PrattCertificates.composite_of_product (factor := 13) (cofactor := 37244309) rfl (by decide) (by decide)
    exact decide_eq_false composite
  third := false
  third_correct := by
    have composite : ¬ Nat.Prime (corner 11002 3) := by change ¬ Nat.Prime 484154013; exact PrattCertificates.composite_of_product (factor := 3) (cofactor := 161384671) rfl (by decide) (by decide)
    exact decide_eq_false composite
theorem row_count_11002 : row 11002 = 0 := (row_11002.checked).trans rfl
theorem model_count_11002 : modelRow 11002 = 0 := by decide
theorem modelrow_agrees_11002 : modelRow 11002 = row 11002 := model_count_11002.trans row_count_11002.symm
def row_11003 : RowCertificate 11003 where
  first := false
  first_correct := by
    have composite : ¬ Nat.Prime (corner 11003 1) := by change ¬ Nat.Prime 484286043; exact PrattCertificates.composite_of_product (factor := 3) (cofactor := 161428681) rfl (by decide) (by decide)
    exact decide_eq_false composite
  second := false
  second_correct := by
    have composite : ¬ Nat.Prime (corner 11003 2) := by change ¬ Nat.Prime 484264037; exact PrattCertificates.composite_of_product (factor := 8689) (cofactor := 55733) rfl (by decide) (by decide)
    exact decide_eq_false composite
  third := false
  third_correct := by
    have composite : ¬ Nat.Prime (corner 11003 3) := by change ¬ Nat.Prime 484242031; exact PrattCertificates.composite_of_product (factor := 7) (cofactor := 69177433) rfl (by decide) (by decide)
    exact decide_eq_false composite
theorem row_count_11003 : row 11003 = 0 := (row_11003.checked).trans rfl
theorem model_count_11003 : modelRow 11003 = 0 := by decide
theorem modelrow_agrees_11003 : modelRow 11003 = row 11003 := model_count_11003.trans row_count_11003.symm
def row_11004 : RowCertificate 11004 where
  first := false
  first_correct := by
    have composite : ¬ Nat.Prime (corner 11004 1) := by change ¬ Nat.Prime 484374073; exact PrattCertificates.composite_of_product (factor := 313) (cofactor := 1547521) rfl (by decide) (by decide)
    exact decide_eq_false composite
  second := false
  second_correct := by
    have composite : ¬ Nat.Prime (corner 11004 2) := by change ¬ Nat.Prime 484352065; exact PrattCertificates.composite_of_product (factor := 5) (cofactor := 96870413) rfl (by decide) (by decide)
    exact decide_eq_false composite
  third := false
  third_correct := by
    have composite : ¬ Nat.Prime (corner 11004 3) := by change ¬ Nat.Prime 484330057; exact PrattCertificates.composite_of_product (factor := 61) (cofactor := 7939837) rfl (by decide) (by decide)
    exact decide_eq_false composite
theorem row_count_11004 : row 11004 = 0 := (row_11004.checked).trans rfl
theorem model_count_11004 : modelRow 11004 = 0 := by decide
theorem modelrow_agrees_11004 : modelRow 11004 = row 11004 := model_count_11004.trans row_count_11004.symm
def row_11005 : RowCertificate 11005 where
  first := false
  first_correct := by
    have composite : ¬ Nat.Prime (corner 11005 1) := by change ¬ Nat.Prime 484462111; exact PrattCertificates.composite_of_product (factor := 7) (cofactor := 69208873) rfl (by decide) (by decide)
    exact decide_eq_false composite
  second := false
  second_correct := by
    have composite : ¬ Nat.Prime (corner 11005 2) := by change ¬ Nat.Prime 484440101; exact PrattCertificates.composite_of_product (factor := 61) (cofactor := 7941641) rfl (by decide) (by decide)
    exact decide_eq_false composite
  third := false
  third_correct := by
    have composite : ¬ Nat.Prime (corner 11005 3) := by change ¬ Nat.Prime 484418091; exact PrattCertificates.composite_of_product (factor := 3) (cofactor := 161472697) rfl (by decide) (by decide)
    exact decide_eq_false composite
theorem row_count_11005 : row 11005 = 0 := (row_11005.checked).trans rfl
theorem model_count_11005 : modelRow 11005 = 0 := by decide
theorem modelrow_agrees_11005 : modelRow 11005 = row 11005 := model_count_11005.trans row_count_11005.symm
def row_11006 : RowCertificate 11006 where
  first := false
  first_correct := by
    have composite : ¬ Nat.Prime (corner 11006 1) := by change ¬ Nat.Prime 484550157; exact PrattCertificates.composite_of_product (factor := 3) (cofactor := 161516719) rfl (by decide) (by decide)
    exact decide_eq_false composite
  second := false
  second_correct := by
    have composite : ¬ Nat.Prime (corner 11006 2) := by change ¬ Nat.Prime 484528145; exact PrattCertificates.composite_of_product (factor := 5) (cofactor := 96905629) rfl (by decide) (by decide)
    exact decide_eq_false composite
  third := false
  third_correct := by
    have composite : ¬ Nat.Prime (corner 11006 3) := by change ¬ Nat.Prime 484506133; exact PrattCertificates.composite_of_product (factor := 1093) (cofactor := 443281) rfl (by decide) (by decide)
    exact decide_eq_false composite
theorem row_count_11006 : row 11006 = 0 := (row_11006.checked).trans rfl
theorem model_count_11006 : modelRow 11006 = 0 := by decide
theorem modelrow_agrees_11006 : modelRow 11006 = row 11006 := model_count_11006.trans row_count_11006.symm
def row_11007 : RowCertificate 11007 where
  first := true
  first_correct := by
    have prime : Nat.Prime (corner 11007 1) := by change Nat.Prime 484638211; exact ProjectEulerSolutions.P58PrimeNodes.p_484638211.prime
    exact decide_eq_true prime
  second := false
  second_correct := by
    have composite : ¬ Nat.Prime (corner 11007 2) := by change ¬ Nat.Prime 484616197; exact PrattCertificates.composite_of_product (factor := 13) (cofactor := 37278169) rfl (by decide) (by decide)
    exact decide_eq_false composite
  third := false
  third_correct := by
    have composite : ¬ Nat.Prime (corner 11007 3) := by change ¬ Nat.Prime 484594183; exact PrattCertificates.composite_of_product (factor := 19) (cofactor := 25504957) rfl (by decide) (by decide)
    exact decide_eq_false composite
theorem row_count_11007 : row 11007 = 1 := (row_11007.checked).trans rfl
theorem model_count_11007 : modelRow 11007 = 1 := by decide
theorem modelrow_agrees_11007 : modelRow 11007 = row 11007 := model_count_11007.trans row_count_11007.symm
def row_11008 : RowCertificate 11008 where
  first := true
  first_correct := by
    have prime : Nat.Prime (corner 11008 1) := by change Nat.Prime 484726273; exact ProjectEulerSolutions.P58PrimeNodes.p_484726273.prime
    exact decide_eq_true prime
  second := false
  second_correct := by
    have composite : ¬ Nat.Prime (corner 11008 2) := by change ¬ Nat.Prime 484704257; exact PrattCertificates.composite_of_product (factor := 3089) (cofactor := 156913) rfl (by decide) (by decide)
    exact decide_eq_false composite
  third := false
  third_correct := by
    have composite : ¬ Nat.Prime (corner 11008 3) := by change ¬ Nat.Prime 484682241; exact PrattCertificates.composite_of_product (factor := 3) (cofactor := 161560747) rfl (by decide) (by decide)
    exact decide_eq_false composite
theorem row_count_11008 : row 11008 = 1 := (row_11008.checked).trans rfl
theorem model_count_11008 : modelRow 11008 = 1 := by decide
theorem modelrow_agrees_11008 : modelRow 11008 = row 11008 := model_count_11008.trans row_count_11008.symm
private def atomic_10881 : CountSegment 1 10 13120 10880 10881 4457 4458 := CountSegment.single (first := 10880) (initial := 4457) row_count_10881 (by decide)
private theorem atomic_10881_agreement : RowAgreement 10880 10881 := RowAgreement.single (first := 10880) modelrow_agrees_10881
private def atomic_10882 : CountSegment 1 10 13120 10881 10882 4458 4458 := CountSegment.single (first := 10881) (initial := 4458) row_count_10882 (by decide)
private theorem atomic_10882_agreement : RowAgreement 10881 10882 := RowAgreement.single (first := 10881) modelrow_agrees_10882
private def atomic_10883 : CountSegment 1 10 13120 10882 10883 4458 4458 := CountSegment.single (first := 10882) (initial := 4458) row_count_10883 (by decide)
private theorem atomic_10883_agreement : RowAgreement 10882 10883 := RowAgreement.single (first := 10882) modelrow_agrees_10883
private def atomic_10884 : CountSegment 1 10 13120 10883 10884 4458 4458 := CountSegment.single (first := 10883) (initial := 4458) row_count_10884 (by decide)
private theorem atomic_10884_agreement : RowAgreement 10883 10884 := RowAgreement.single (first := 10883) modelrow_agrees_10884
private def atomic_10885 : CountSegment 1 10 13120 10884 10885 4458 4459 := CountSegment.single (first := 10884) (initial := 4458) row_count_10885 (by decide)
private theorem atomic_10885_agreement : RowAgreement 10884 10885 := RowAgreement.single (first := 10884) modelrow_agrees_10885
private def atomic_10886 : CountSegment 1 10 13120 10885 10886 4459 4459 := CountSegment.single (first := 10885) (initial := 4459) row_count_10886 (by decide)
private theorem atomic_10886_agreement : RowAgreement 10885 10886 := RowAgreement.single (first := 10885) modelrow_agrees_10886
private def atomic_10887 : CountSegment 1 10 13120 10886 10887 4459 4460 := CountSegment.single (first := 10886) (initial := 4459) row_count_10887 (by decide)
private theorem atomic_10887_agreement : RowAgreement 10886 10887 := RowAgreement.single (first := 10886) modelrow_agrees_10887
private def atomic_10888 : CountSegment 1 10 13120 10887 10888 4460 4461 := CountSegment.single (first := 10887) (initial := 4460) row_count_10888 (by decide)
private theorem atomic_10888_agreement : RowAgreement 10887 10888 := RowAgreement.single (first := 10887) modelrow_agrees_10888
private def atomic_10889 : CountSegment 1 10 13120 10888 10889 4461 4461 := CountSegment.single (first := 10888) (initial := 4461) row_count_10889 (by decide)
private theorem atomic_10889_agreement : RowAgreement 10888 10889 := RowAgreement.single (first := 10888) modelrow_agrees_10889
private def atomic_10890 : CountSegment 1 10 13120 10889 10890 4461 4461 := CountSegment.single (first := 10889) (initial := 4461) row_count_10890 (by decide)
private theorem atomic_10890_agreement : RowAgreement 10889 10890 := RowAgreement.single (first := 10889) modelrow_agrees_10890
private def atomic_10891 : CountSegment 1 10 13120 10890 10891 4461 4461 := CountSegment.single (first := 10890) (initial := 4461) row_count_10891 (by decide)
private theorem atomic_10891_agreement : RowAgreement 10890 10891 := RowAgreement.single (first := 10890) modelrow_agrees_10891
private def atomic_10892 : CountSegment 1 10 13120 10891 10892 4461 4462 := CountSegment.single (first := 10891) (initial := 4461) row_count_10892 (by decide)
private theorem atomic_10892_agreement : RowAgreement 10891 10892 := RowAgreement.single (first := 10891) modelrow_agrees_10892
private def atomic_10893 : CountSegment 1 10 13120 10892 10893 4462 4462 := CountSegment.single (first := 10892) (initial := 4462) row_count_10893 (by decide)
private theorem atomic_10893_agreement : RowAgreement 10892 10893 := RowAgreement.single (first := 10892) modelrow_agrees_10893
private def atomic_10894 : CountSegment 1 10 13120 10893 10894 4462 4462 := CountSegment.single (first := 10893) (initial := 4462) row_count_10894 (by decide)
private theorem atomic_10894_agreement : RowAgreement 10893 10894 := RowAgreement.single (first := 10893) modelrow_agrees_10894
private def atomic_10895 : CountSegment 1 10 13120 10894 10895 4462 4462 := CountSegment.single (first := 10894) (initial := 4462) row_count_10895 (by decide)
private theorem atomic_10895_agreement : RowAgreement 10894 10895 := RowAgreement.single (first := 10894) modelrow_agrees_10895
private def atomic_10896 : CountSegment 1 10 13120 10895 10896 4462 4463 := CountSegment.single (first := 10895) (initial := 4462) row_count_10896 (by decide)
private theorem atomic_10896_agreement : RowAgreement 10895 10896 := RowAgreement.single (first := 10895) modelrow_agrees_10896
private def atomic_10897 : CountSegment 1 10 13120 10896 10897 4463 4463 := CountSegment.single (first := 10896) (initial := 4463) row_count_10897 (by decide)
private theorem atomic_10897_agreement : RowAgreement 10896 10897 := RowAgreement.single (first := 10896) modelrow_agrees_10897
private def atomic_10898 : CountSegment 1 10 13120 10897 10898 4463 4463 := CountSegment.single (first := 10897) (initial := 4463) row_count_10898 (by decide)
private theorem atomic_10898_agreement : RowAgreement 10897 10898 := RowAgreement.single (first := 10897) modelrow_agrees_10898
private def atomic_10899 : CountSegment 1 10 13120 10898 10899 4463 4463 := CountSegment.single (first := 10898) (initial := 4463) row_count_10899 (by decide)
private theorem atomic_10899_agreement : RowAgreement 10898 10899 := RowAgreement.single (first := 10898) modelrow_agrees_10899
private def atomic_10900 : CountSegment 1 10 13120 10899 10900 4463 4463 := CountSegment.single (first := 10899) (initial := 4463) row_count_10900 (by decide)
private theorem atomic_10900_agreement : RowAgreement 10899 10900 := RowAgreement.single (first := 10899) modelrow_agrees_10900
private def atomic_10901 : CountSegment 1 10 13120 10900 10901 4463 4463 := CountSegment.single (first := 10900) (initial := 4463) row_count_10901 (by decide)
private theorem atomic_10901_agreement : RowAgreement 10900 10901 := RowAgreement.single (first := 10900) modelrow_agrees_10901
private def atomic_10902 : CountSegment 1 10 13120 10901 10902 4463 4464 := CountSegment.single (first := 10901) (initial := 4463) row_count_10902 (by decide)
private theorem atomic_10902_agreement : RowAgreement 10901 10902 := RowAgreement.single (first := 10901) modelrow_agrees_10902
private def atomic_10903 : CountSegment 1 10 13120 10902 10903 4464 4465 := CountSegment.single (first := 10902) (initial := 4464) row_count_10903 (by decide)
private theorem atomic_10903_agreement : RowAgreement 10902 10903 := RowAgreement.single (first := 10902) modelrow_agrees_10903
private def atomic_10904 : CountSegment 1 10 13120 10903 10904 4465 4465 := CountSegment.single (first := 10903) (initial := 4465) row_count_10904 (by decide)
private theorem atomic_10904_agreement : RowAgreement 10903 10904 := RowAgreement.single (first := 10903) modelrow_agrees_10904
private def atomic_10905 : CountSegment 1 10 13120 10904 10905 4465 4466 := CountSegment.single (first := 10904) (initial := 4465) row_count_10905 (by decide)
private theorem atomic_10905_agreement : RowAgreement 10904 10905 := RowAgreement.single (first := 10904) modelrow_agrees_10905
private def atomic_10906 : CountSegment 1 10 13120 10905 10906 4466 4466 := CountSegment.single (first := 10905) (initial := 4466) row_count_10906 (by decide)
private theorem atomic_10906_agreement : RowAgreement 10905 10906 := RowAgreement.single (first := 10905) modelrow_agrees_10906
private def atomic_10907 : CountSegment 1 10 13120 10906 10907 4466 4467 := CountSegment.single (first := 10906) (initial := 4466) row_count_10907 (by decide)
private theorem atomic_10907_agreement : RowAgreement 10906 10907 := RowAgreement.single (first := 10906) modelrow_agrees_10907
private def atomic_10908 : CountSegment 1 10 13120 10907 10908 4467 4468 := CountSegment.single (first := 10907) (initial := 4467) row_count_10908 (by decide)
private theorem atomic_10908_agreement : RowAgreement 10907 10908 := RowAgreement.single (first := 10907) modelrow_agrees_10908
private def atomic_10909 : CountSegment 1 10 13120 10908 10909 4468 4468 := CountSegment.single (first := 10908) (initial := 4468) row_count_10909 (by decide)
private theorem atomic_10909_agreement : RowAgreement 10908 10909 := RowAgreement.single (first := 10908) modelrow_agrees_10909
private def atomic_10910 : CountSegment 1 10 13120 10909 10910 4468 4468 := CountSegment.single (first := 10909) (initial := 4468) row_count_10910 (by decide)
private theorem atomic_10910_agreement : RowAgreement 10909 10910 := RowAgreement.single (first := 10909) modelrow_agrees_10910
private def atomic_10911 : CountSegment 1 10 13120 10910 10911 4468 4468 := CountSegment.single (first := 10910) (initial := 4468) row_count_10911 (by decide)
private theorem atomic_10911_agreement : RowAgreement 10910 10911 := RowAgreement.single (first := 10910) modelrow_agrees_10911
private def atomic_10912 : CountSegment 1 10 13120 10911 10912 4468 4468 := CountSegment.single (first := 10911) (initial := 4468) row_count_10912 (by decide)
private theorem atomic_10912_agreement : RowAgreement 10911 10912 := RowAgreement.single (first := 10911) modelrow_agrees_10912
private def atomic_10913 : CountSegment 1 10 13120 10912 10913 4468 4469 := CountSegment.single (first := 10912) (initial := 4468) row_count_10913 (by decide)
private theorem atomic_10913_agreement : RowAgreement 10912 10913 := RowAgreement.single (first := 10912) modelrow_agrees_10913
private def atomic_10914 : CountSegment 1 10 13120 10913 10914 4469 4469 := CountSegment.single (first := 10913) (initial := 4469) row_count_10914 (by decide)
private theorem atomic_10914_agreement : RowAgreement 10913 10914 := RowAgreement.single (first := 10913) modelrow_agrees_10914
private def atomic_10915 : CountSegment 1 10 13120 10914 10915 4469 4470 := CountSegment.single (first := 10914) (initial := 4469) row_count_10915 (by decide)
private theorem atomic_10915_agreement : RowAgreement 10914 10915 := RowAgreement.single (first := 10914) modelrow_agrees_10915
private def atomic_10916 : CountSegment 1 10 13120 10915 10916 4470 4471 := CountSegment.single (first := 10915) (initial := 4470) row_count_10916 (by decide)
private theorem atomic_10916_agreement : RowAgreement 10915 10916 := RowAgreement.single (first := 10915) modelrow_agrees_10916
private def atomic_10917 : CountSegment 1 10 13120 10916 10917 4471 4472 := CountSegment.single (first := 10916) (initial := 4471) row_count_10917 (by decide)
private theorem atomic_10917_agreement : RowAgreement 10916 10917 := RowAgreement.single (first := 10916) modelrow_agrees_10917
private def atomic_10918 : CountSegment 1 10 13120 10917 10918 4472 4472 := CountSegment.single (first := 10917) (initial := 4472) row_count_10918 (by decide)
private theorem atomic_10918_agreement : RowAgreement 10917 10918 := RowAgreement.single (first := 10917) modelrow_agrees_10918
private def atomic_10919 : CountSegment 1 10 13120 10918 10919 4472 4472 := CountSegment.single (first := 10918) (initial := 4472) row_count_10919 (by decide)
private theorem atomic_10919_agreement : RowAgreement 10918 10919 := RowAgreement.single (first := 10918) modelrow_agrees_10919
private def atomic_10920 : CountSegment 1 10 13120 10919 10920 4472 4473 := CountSegment.single (first := 10919) (initial := 4472) row_count_10920 (by decide)
private theorem atomic_10920_agreement : RowAgreement 10919 10920 := RowAgreement.single (first := 10919) modelrow_agrees_10920
private def atomic_10921 : CountSegment 1 10 13120 10920 10921 4473 4473 := CountSegment.single (first := 10920) (initial := 4473) row_count_10921 (by decide)
private theorem atomic_10921_agreement : RowAgreement 10920 10921 := RowAgreement.single (first := 10920) modelrow_agrees_10921
private def atomic_10922 : CountSegment 1 10 13120 10921 10922 4473 4473 := CountSegment.single (first := 10921) (initial := 4473) row_count_10922 (by decide)
private theorem atomic_10922_agreement : RowAgreement 10921 10922 := RowAgreement.single (first := 10921) modelrow_agrees_10922
private def atomic_10923 : CountSegment 1 10 13120 10922 10923 4473 4473 := CountSegment.single (first := 10922) (initial := 4473) row_count_10923 (by decide)
private theorem atomic_10923_agreement : RowAgreement 10922 10923 := RowAgreement.single (first := 10922) modelrow_agrees_10923
private def atomic_10924 : CountSegment 1 10 13120 10923 10924 4473 4473 := CountSegment.single (first := 10923) (initial := 4473) row_count_10924 (by decide)
private theorem atomic_10924_agreement : RowAgreement 10923 10924 := RowAgreement.single (first := 10923) modelrow_agrees_10924
private def atomic_10925 : CountSegment 1 10 13120 10924 10925 4473 4473 := CountSegment.single (first := 10924) (initial := 4473) row_count_10925 (by decide)
private theorem atomic_10925_agreement : RowAgreement 10924 10925 := RowAgreement.single (first := 10924) modelrow_agrees_10925
private def atomic_10926 : CountSegment 1 10 13120 10925 10926 4473 4473 := CountSegment.single (first := 10925) (initial := 4473) row_count_10926 (by decide)
private theorem atomic_10926_agreement : RowAgreement 10925 10926 := RowAgreement.single (first := 10925) modelrow_agrees_10926
private def atomic_10927 : CountSegment 1 10 13120 10926 10927 4473 4473 := CountSegment.single (first := 10926) (initial := 4473) row_count_10927 (by decide)
private theorem atomic_10927_agreement : RowAgreement 10926 10927 := RowAgreement.single (first := 10926) modelrow_agrees_10927
private def atomic_10928 : CountSegment 1 10 13120 10927 10928 4473 4475 := CountSegment.single (first := 10927) (initial := 4473) row_count_10928 (by decide)
private theorem atomic_10928_agreement : RowAgreement 10927 10928 := RowAgreement.single (first := 10927) modelrow_agrees_10928
private def atomic_10929 : CountSegment 1 10 13120 10928 10929 4475 4475 := CountSegment.single (first := 10928) (initial := 4475) row_count_10929 (by decide)
private theorem atomic_10929_agreement : RowAgreement 10928 10929 := RowAgreement.single (first := 10928) modelrow_agrees_10929
private def atomic_10930 : CountSegment 1 10 13120 10929 10930 4475 4475 := CountSegment.single (first := 10929) (initial := 4475) row_count_10930 (by decide)
private theorem atomic_10930_agreement : RowAgreement 10929 10930 := RowAgreement.single (first := 10929) modelrow_agrees_10930
private def atomic_10931 : CountSegment 1 10 13120 10930 10931 4475 4475 := CountSegment.single (first := 10930) (initial := 4475) row_count_10931 (by decide)
private theorem atomic_10931_agreement : RowAgreement 10930 10931 := RowAgreement.single (first := 10930) modelrow_agrees_10931
private def atomic_10932 : CountSegment 1 10 13120 10931 10932 4475 4475 := CountSegment.single (first := 10931) (initial := 4475) row_count_10932 (by decide)
private theorem atomic_10932_agreement : RowAgreement 10931 10932 := RowAgreement.single (first := 10931) modelrow_agrees_10932
private def atomic_10933 : CountSegment 1 10 13120 10932 10933 4475 4476 := CountSegment.single (first := 10932) (initial := 4475) row_count_10933 (by decide)
private theorem atomic_10933_agreement : RowAgreement 10932 10933 := RowAgreement.single (first := 10932) modelrow_agrees_10933
private def atomic_10934 : CountSegment 1 10 13120 10933 10934 4476 4476 := CountSegment.single (first := 10933) (initial := 4476) row_count_10934 (by decide)
private theorem atomic_10934_agreement : RowAgreement 10933 10934 := RowAgreement.single (first := 10933) modelrow_agrees_10934
private def atomic_10935 : CountSegment 1 10 13120 10934 10935 4476 4477 := CountSegment.single (first := 10934) (initial := 4476) row_count_10935 (by decide)
private theorem atomic_10935_agreement : RowAgreement 10934 10935 := RowAgreement.single (first := 10934) modelrow_agrees_10935
private def atomic_10936 : CountSegment 1 10 13120 10935 10936 4477 4477 := CountSegment.single (first := 10935) (initial := 4477) row_count_10936 (by decide)
private theorem atomic_10936_agreement : RowAgreement 10935 10936 := RowAgreement.single (first := 10935) modelrow_agrees_10936
private def atomic_10937 : CountSegment 1 10 13120 10936 10937 4477 4477 := CountSegment.single (first := 10936) (initial := 4477) row_count_10937 (by decide)
private theorem atomic_10937_agreement : RowAgreement 10936 10937 := RowAgreement.single (first := 10936) modelrow_agrees_10937
private def atomic_10938 : CountSegment 1 10 13120 10937 10938 4477 4477 := CountSegment.single (first := 10937) (initial := 4477) row_count_10938 (by decide)
private theorem atomic_10938_agreement : RowAgreement 10937 10938 := RowAgreement.single (first := 10937) modelrow_agrees_10938
private def atomic_10939 : CountSegment 1 10 13120 10938 10939 4477 4478 := CountSegment.single (first := 10938) (initial := 4477) row_count_10939 (by decide)
private theorem atomic_10939_agreement : RowAgreement 10938 10939 := RowAgreement.single (first := 10938) modelrow_agrees_10939
private def atomic_10940 : CountSegment 1 10 13120 10939 10940 4478 4479 := CountSegment.single (first := 10939) (initial := 4478) row_count_10940 (by decide)
private theorem atomic_10940_agreement : RowAgreement 10939 10940 := RowAgreement.single (first := 10939) modelrow_agrees_10940
private def atomic_10941 : CountSegment 1 10 13120 10940 10941 4479 4479 := CountSegment.single (first := 10940) (initial := 4479) row_count_10941 (by decide)
private theorem atomic_10941_agreement : RowAgreement 10940 10941 := RowAgreement.single (first := 10940) modelrow_agrees_10941
private def atomic_10942 : CountSegment 1 10 13120 10941 10942 4479 4479 := CountSegment.single (first := 10941) (initial := 4479) row_count_10942 (by decide)
private theorem atomic_10942_agreement : RowAgreement 10941 10942 := RowAgreement.single (first := 10941) modelrow_agrees_10942
private def atomic_10943 : CountSegment 1 10 13120 10942 10943 4479 4480 := CountSegment.single (first := 10942) (initial := 4479) row_count_10943 (by decide)
private theorem atomic_10943_agreement : RowAgreement 10942 10943 := RowAgreement.single (first := 10942) modelrow_agrees_10943
private def atomic_10944 : CountSegment 1 10 13120 10943 10944 4480 4481 := CountSegment.single (first := 10943) (initial := 4480) row_count_10944 (by decide)
private theorem atomic_10944_agreement : RowAgreement 10943 10944 := RowAgreement.single (first := 10943) modelrow_agrees_10944
private def atomic_10945 : CountSegment 1 10 13120 10944 10945 4481 4481 := CountSegment.single (first := 10944) (initial := 4481) row_count_10945 (by decide)
private theorem atomic_10945_agreement : RowAgreement 10944 10945 := RowAgreement.single (first := 10944) modelrow_agrees_10945
private def atomic_10946 : CountSegment 1 10 13120 10945 10946 4481 4481 := CountSegment.single (first := 10945) (initial := 4481) row_count_10946 (by decide)
private theorem atomic_10946_agreement : RowAgreement 10945 10946 := RowAgreement.single (first := 10945) modelrow_agrees_10946
private def atomic_10947 : CountSegment 1 10 13120 10946 10947 4481 4483 := CountSegment.single (first := 10946) (initial := 4481) row_count_10947 (by decide)
private theorem atomic_10947_agreement : RowAgreement 10946 10947 := RowAgreement.single (first := 10946) modelrow_agrees_10947
private def atomic_10948 : CountSegment 1 10 13120 10947 10948 4483 4484 := CountSegment.single (first := 10947) (initial := 4483) row_count_10948 (by decide)
private theorem atomic_10948_agreement : RowAgreement 10947 10948 := RowAgreement.single (first := 10947) modelrow_agrees_10948
private def atomic_10949 : CountSegment 1 10 13120 10948 10949 4484 4485 := CountSegment.single (first := 10948) (initial := 4484) row_count_10949 (by decide)
private theorem atomic_10949_agreement : RowAgreement 10948 10949 := RowAgreement.single (first := 10948) modelrow_agrees_10949
private def atomic_10950 : CountSegment 1 10 13120 10949 10950 4485 4485 := CountSegment.single (first := 10949) (initial := 4485) row_count_10950 (by decide)
private theorem atomic_10950_agreement : RowAgreement 10949 10950 := RowAgreement.single (first := 10949) modelrow_agrees_10950
private def atomic_10951 : CountSegment 1 10 13120 10950 10951 4485 4486 := CountSegment.single (first := 10950) (initial := 4485) row_count_10951 (by decide)
private theorem atomic_10951_agreement : RowAgreement 10950 10951 := RowAgreement.single (first := 10950) modelrow_agrees_10951
private def atomic_10952 : CountSegment 1 10 13120 10951 10952 4486 4486 := CountSegment.single (first := 10951) (initial := 4486) row_count_10952 (by decide)
private theorem atomic_10952_agreement : RowAgreement 10951 10952 := RowAgreement.single (first := 10951) modelrow_agrees_10952
private def atomic_10953 : CountSegment 1 10 13120 10952 10953 4486 4487 := CountSegment.single (first := 10952) (initial := 4486) row_count_10953 (by decide)
private theorem atomic_10953_agreement : RowAgreement 10952 10953 := RowAgreement.single (first := 10952) modelrow_agrees_10953
private def atomic_10954 : CountSegment 1 10 13120 10953 10954 4487 4487 := CountSegment.single (first := 10953) (initial := 4487) row_count_10954 (by decide)
private theorem atomic_10954_agreement : RowAgreement 10953 10954 := RowAgreement.single (first := 10953) modelrow_agrees_10954
private def atomic_10955 : CountSegment 1 10 13120 10954 10955 4487 4487 := CountSegment.single (first := 10954) (initial := 4487) row_count_10955 (by decide)
private theorem atomic_10955_agreement : RowAgreement 10954 10955 := RowAgreement.single (first := 10954) modelrow_agrees_10955
private def atomic_10956 : CountSegment 1 10 13120 10955 10956 4487 4487 := CountSegment.single (first := 10955) (initial := 4487) row_count_10956 (by decide)
private theorem atomic_10956_agreement : RowAgreement 10955 10956 := RowAgreement.single (first := 10955) modelrow_agrees_10956
private def atomic_10957 : CountSegment 1 10 13120 10956 10957 4487 4488 := CountSegment.single (first := 10956) (initial := 4487) row_count_10957 (by decide)
private theorem atomic_10957_agreement : RowAgreement 10956 10957 := RowAgreement.single (first := 10956) modelrow_agrees_10957
private def atomic_10958 : CountSegment 1 10 13120 10957 10958 4488 4489 := CountSegment.single (first := 10957) (initial := 4488) row_count_10958 (by decide)
private theorem atomic_10958_agreement : RowAgreement 10957 10958 := RowAgreement.single (first := 10957) modelrow_agrees_10958
private def atomic_10959 : CountSegment 1 10 13120 10958 10959 4489 4489 := CountSegment.single (first := 10958) (initial := 4489) row_count_10959 (by decide)
private theorem atomic_10959_agreement : RowAgreement 10958 10959 := RowAgreement.single (first := 10958) modelrow_agrees_10959
private def atomic_10960 : CountSegment 1 10 13120 10959 10960 4489 4489 := CountSegment.single (first := 10959) (initial := 4489) row_count_10960 (by decide)
private theorem atomic_10960_agreement : RowAgreement 10959 10960 := RowAgreement.single (first := 10959) modelrow_agrees_10960
private def atomic_10961 : CountSegment 1 10 13120 10960 10961 4489 4489 := CountSegment.single (first := 10960) (initial := 4489) row_count_10961 (by decide)
private theorem atomic_10961_agreement : RowAgreement 10960 10961 := RowAgreement.single (first := 10960) modelrow_agrees_10961
private def atomic_10962 : CountSegment 1 10 13120 10961 10962 4489 4491 := CountSegment.single (first := 10961) (initial := 4489) row_count_10962 (by decide)
private theorem atomic_10962_agreement : RowAgreement 10961 10962 := RowAgreement.single (first := 10961) modelrow_agrees_10962
private def atomic_10963 : CountSegment 1 10 13120 10962 10963 4491 4491 := CountSegment.single (first := 10962) (initial := 4491) row_count_10963 (by decide)
private theorem atomic_10963_agreement : RowAgreement 10962 10963 := RowAgreement.single (first := 10962) modelrow_agrees_10963
private def atomic_10964 : CountSegment 1 10 13120 10963 10964 4491 4491 := CountSegment.single (first := 10963) (initial := 4491) row_count_10964 (by decide)
private theorem atomic_10964_agreement : RowAgreement 10963 10964 := RowAgreement.single (first := 10963) modelrow_agrees_10964
private def atomic_10965 : CountSegment 1 10 13120 10964 10965 4491 4492 := CountSegment.single (first := 10964) (initial := 4491) row_count_10965 (by decide)
private theorem atomic_10965_agreement : RowAgreement 10964 10965 := RowAgreement.single (first := 10964) modelrow_agrees_10965
private def atomic_10966 : CountSegment 1 10 13120 10965 10966 4492 4492 := CountSegment.single (first := 10965) (initial := 4492) row_count_10966 (by decide)
private theorem atomic_10966_agreement : RowAgreement 10965 10966 := RowAgreement.single (first := 10965) modelrow_agrees_10966
private def atomic_10967 : CountSegment 1 10 13120 10966 10967 4492 4492 := CountSegment.single (first := 10966) (initial := 4492) row_count_10967 (by decide)
private theorem atomic_10967_agreement : RowAgreement 10966 10967 := RowAgreement.single (first := 10966) modelrow_agrees_10967
private def atomic_10968 : CountSegment 1 10 13120 10967 10968 4492 4492 := CountSegment.single (first := 10967) (initial := 4492) row_count_10968 (by decide)
private theorem atomic_10968_agreement : RowAgreement 10967 10968 := RowAgreement.single (first := 10967) modelrow_agrees_10968
private def atomic_10969 : CountSegment 1 10 13120 10968 10969 4492 4492 := CountSegment.single (first := 10968) (initial := 4492) row_count_10969 (by decide)
private theorem atomic_10969_agreement : RowAgreement 10968 10969 := RowAgreement.single (first := 10968) modelrow_agrees_10969
private def atomic_10970 : CountSegment 1 10 13120 10969 10970 4492 4494 := CountSegment.single (first := 10969) (initial := 4492) row_count_10970 (by decide)
private theorem atomic_10970_agreement : RowAgreement 10969 10970 := RowAgreement.single (first := 10969) modelrow_agrees_10970
private def atomic_10971 : CountSegment 1 10 13120 10970 10971 4494 4494 := CountSegment.single (first := 10970) (initial := 4494) row_count_10971 (by decide)
private theorem atomic_10971_agreement : RowAgreement 10970 10971 := RowAgreement.single (first := 10970) modelrow_agrees_10971
private def atomic_10972 : CountSegment 1 10 13120 10971 10972 4494 4495 := CountSegment.single (first := 10971) (initial := 4494) row_count_10972 (by decide)
private theorem atomic_10972_agreement : RowAgreement 10971 10972 := RowAgreement.single (first := 10971) modelrow_agrees_10972
private def atomic_10973 : CountSegment 1 10 13120 10972 10973 4495 4495 := CountSegment.single (first := 10972) (initial := 4495) row_count_10973 (by decide)
private theorem atomic_10973_agreement : RowAgreement 10972 10973 := RowAgreement.single (first := 10972) modelrow_agrees_10973
private def atomic_10974 : CountSegment 1 10 13120 10973 10974 4495 4496 := CountSegment.single (first := 10973) (initial := 4495) row_count_10974 (by decide)
private theorem atomic_10974_agreement : RowAgreement 10973 10974 := RowAgreement.single (first := 10973) modelrow_agrees_10974
private def atomic_10975 : CountSegment 1 10 13120 10974 10975 4496 4497 := CountSegment.single (first := 10974) (initial := 4496) row_count_10975 (by decide)
private theorem atomic_10975_agreement : RowAgreement 10974 10975 := RowAgreement.single (first := 10974) modelrow_agrees_10975
private def atomic_10976 : CountSegment 1 10 13120 10975 10976 4497 4497 := CountSegment.single (first := 10975) (initial := 4497) row_count_10976 (by decide)
private theorem atomic_10976_agreement : RowAgreement 10975 10976 := RowAgreement.single (first := 10975) modelrow_agrees_10976
private def atomic_10977 : CountSegment 1 10 13120 10976 10977 4497 4498 := CountSegment.single (first := 10976) (initial := 4497) row_count_10977 (by decide)
private theorem atomic_10977_agreement : RowAgreement 10976 10977 := RowAgreement.single (first := 10976) modelrow_agrees_10977
private def atomic_10978 : CountSegment 1 10 13120 10977 10978 4498 4499 := CountSegment.single (first := 10977) (initial := 4498) row_count_10978 (by decide)
private theorem atomic_10978_agreement : RowAgreement 10977 10978 := RowAgreement.single (first := 10977) modelrow_agrees_10978
private def atomic_10979 : CountSegment 1 10 13120 10978 10979 4499 4499 := CountSegment.single (first := 10978) (initial := 4499) row_count_10979 (by decide)
private theorem atomic_10979_agreement : RowAgreement 10978 10979 := RowAgreement.single (first := 10978) modelrow_agrees_10979
private def atomic_10980 : CountSegment 1 10 13120 10979 10980 4499 4499 := CountSegment.single (first := 10979) (initial := 4499) row_count_10980 (by decide)
private theorem atomic_10980_agreement : RowAgreement 10979 10980 := RowAgreement.single (first := 10979) modelrow_agrees_10980
private def atomic_10981 : CountSegment 1 10 13120 10980 10981 4499 4500 := CountSegment.single (first := 10980) (initial := 4499) row_count_10981 (by decide)
private theorem atomic_10981_agreement : RowAgreement 10980 10981 := RowAgreement.single (first := 10980) modelrow_agrees_10981
private def atomic_10982 : CountSegment 1 10 13120 10981 10982 4500 4501 := CountSegment.single (first := 10981) (initial := 4500) row_count_10982 (by decide)
private theorem atomic_10982_agreement : RowAgreement 10981 10982 := RowAgreement.single (first := 10981) modelrow_agrees_10982
private def atomic_10983 : CountSegment 1 10 13120 10982 10983 4501 4501 := CountSegment.single (first := 10982) (initial := 4501) row_count_10983 (by decide)
private theorem atomic_10983_agreement : RowAgreement 10982 10983 := RowAgreement.single (first := 10982) modelrow_agrees_10983
private def atomic_10984 : CountSegment 1 10 13120 10983 10984 4501 4501 := CountSegment.single (first := 10983) (initial := 4501) row_count_10984 (by decide)
private theorem atomic_10984_agreement : RowAgreement 10983 10984 := RowAgreement.single (first := 10983) modelrow_agrees_10984
private def atomic_10985 : CountSegment 1 10 13120 10984 10985 4501 4502 := CountSegment.single (first := 10984) (initial := 4501) row_count_10985 (by decide)
private theorem atomic_10985_agreement : RowAgreement 10984 10985 := RowAgreement.single (first := 10984) modelrow_agrees_10985
private def atomic_10986 : CountSegment 1 10 13120 10985 10986 4502 4502 := CountSegment.single (first := 10985) (initial := 4502) row_count_10986 (by decide)
private theorem atomic_10986_agreement : RowAgreement 10985 10986 := RowAgreement.single (first := 10985) modelrow_agrees_10986
private def atomic_10987 : CountSegment 1 10 13120 10986 10987 4502 4503 := CountSegment.single (first := 10986) (initial := 4502) row_count_10987 (by decide)
private theorem atomic_10987_agreement : RowAgreement 10986 10987 := RowAgreement.single (first := 10986) modelrow_agrees_10987
private def atomic_10988 : CountSegment 1 10 13120 10987 10988 4503 4503 := CountSegment.single (first := 10987) (initial := 4503) row_count_10988 (by decide)
private theorem atomic_10988_agreement : RowAgreement 10987 10988 := RowAgreement.single (first := 10987) modelrow_agrees_10988
private def atomic_10989 : CountSegment 1 10 13120 10988 10989 4503 4503 := CountSegment.single (first := 10988) (initial := 4503) row_count_10989 (by decide)
private theorem atomic_10989_agreement : RowAgreement 10988 10989 := RowAgreement.single (first := 10988) modelrow_agrees_10989
private def atomic_10990 : CountSegment 1 10 13120 10989 10990 4503 4504 := CountSegment.single (first := 10989) (initial := 4503) row_count_10990 (by decide)
private theorem atomic_10990_agreement : RowAgreement 10989 10990 := RowAgreement.single (first := 10989) modelrow_agrees_10990
private def atomic_10991 : CountSegment 1 10 13120 10990 10991 4504 4504 := CountSegment.single (first := 10990) (initial := 4504) row_count_10991 (by decide)
private theorem atomic_10991_agreement : RowAgreement 10990 10991 := RowAgreement.single (first := 10990) modelrow_agrees_10991
private def atomic_10992 : CountSegment 1 10 13120 10991 10992 4504 4504 := CountSegment.single (first := 10991) (initial := 4504) row_count_10992 (by decide)
private theorem atomic_10992_agreement : RowAgreement 10991 10992 := RowAgreement.single (first := 10991) modelrow_agrees_10992
private def atomic_10993 : CountSegment 1 10 13120 10992 10993 4504 4505 := CountSegment.single (first := 10992) (initial := 4504) row_count_10993 (by decide)
private theorem atomic_10993_agreement : RowAgreement 10992 10993 := RowAgreement.single (first := 10992) modelrow_agrees_10993
private def atomic_10994 : CountSegment 1 10 13120 10993 10994 4505 4505 := CountSegment.single (first := 10993) (initial := 4505) row_count_10994 (by decide)
private theorem atomic_10994_agreement : RowAgreement 10993 10994 := RowAgreement.single (first := 10993) modelrow_agrees_10994
private def atomic_10995 : CountSegment 1 10 13120 10994 10995 4505 4506 := CountSegment.single (first := 10994) (initial := 4505) row_count_10995 (by decide)
private theorem atomic_10995_agreement : RowAgreement 10994 10995 := RowAgreement.single (first := 10994) modelrow_agrees_10995
private def atomic_10996 : CountSegment 1 10 13120 10995 10996 4506 4506 := CountSegment.single (first := 10995) (initial := 4506) row_count_10996 (by decide)
private theorem atomic_10996_agreement : RowAgreement 10995 10996 := RowAgreement.single (first := 10995) modelrow_agrees_10996
private def atomic_10997 : CountSegment 1 10 13120 10996 10997 4506 4506 := CountSegment.single (first := 10996) (initial := 4506) row_count_10997 (by decide)
private theorem atomic_10997_agreement : RowAgreement 10996 10997 := RowAgreement.single (first := 10996) modelrow_agrees_10997
private def atomic_10998 : CountSegment 1 10 13120 10997 10998 4506 4507 := CountSegment.single (first := 10997) (initial := 4506) row_count_10998 (by decide)
private theorem atomic_10998_agreement : RowAgreement 10997 10998 := RowAgreement.single (first := 10997) modelrow_agrees_10998
private def atomic_10999 : CountSegment 1 10 13120 10998 10999 4507 4507 := CountSegment.single (first := 10998) (initial := 4507) row_count_10999 (by decide)
private theorem atomic_10999_agreement : RowAgreement 10998 10999 := RowAgreement.single (first := 10998) modelrow_agrees_10999
private def atomic_11000 : CountSegment 1 10 13120 10999 11000 4507 4507 := CountSegment.single (first := 10999) (initial := 4507) row_count_11000 (by decide)
private theorem atomic_11000_agreement : RowAgreement 10999 11000 := RowAgreement.single (first := 10999) modelrow_agrees_11000
private def atomic_11001 : CountSegment 1 10 13120 11000 11001 4507 4507 := CountSegment.single (first := 11000) (initial := 4507) row_count_11001 (by decide)
private theorem atomic_11001_agreement : RowAgreement 11000 11001 := RowAgreement.single (first := 11000) modelrow_agrees_11001
private def atomic_11002 : CountSegment 1 10 13120 11001 11002 4507 4507 := CountSegment.single (first := 11001) (initial := 4507) row_count_11002 (by decide)
private theorem atomic_11002_agreement : RowAgreement 11001 11002 := RowAgreement.single (first := 11001) modelrow_agrees_11002
private def atomic_11003 : CountSegment 1 10 13120 11002 11003 4507 4507 := CountSegment.single (first := 11002) (initial := 4507) row_count_11003 (by decide)
private theorem atomic_11003_agreement : RowAgreement 11002 11003 := RowAgreement.single (first := 11002) modelrow_agrees_11003
private def atomic_11004 : CountSegment 1 10 13120 11003 11004 4507 4507 := CountSegment.single (first := 11003) (initial := 4507) row_count_11004 (by decide)
private theorem atomic_11004_agreement : RowAgreement 11003 11004 := RowAgreement.single (first := 11003) modelrow_agrees_11004
private def atomic_11005 : CountSegment 1 10 13120 11004 11005 4507 4507 := CountSegment.single (first := 11004) (initial := 4507) row_count_11005 (by decide)
private theorem atomic_11005_agreement : RowAgreement 11004 11005 := RowAgreement.single (first := 11004) modelrow_agrees_11005
private def atomic_11006 : CountSegment 1 10 13120 11005 11006 4507 4507 := CountSegment.single (first := 11005) (initial := 4507) row_count_11006 (by decide)
private theorem atomic_11006_agreement : RowAgreement 11005 11006 := RowAgreement.single (first := 11005) modelrow_agrees_11006
private def atomic_11007 : CountSegment 1 10 13120 11006 11007 4507 4508 := CountSegment.single (first := 11006) (initial := 4507) row_count_11007 (by decide)
private theorem atomic_11007_agreement : RowAgreement 11006 11007 := RowAgreement.single (first := 11006) modelrow_agrees_11007
private def atomic_11008 : CountSegment 1 10 13120 11007 11008 4508 4509 := CountSegment.single (first := 11007) (initial := 4508) row_count_11008 (by decide)
private theorem atomic_11008_agreement : RowAgreement 11007 11008 := RowAgreement.single (first := 11007) modelrow_agrees_11008
private def within_085_000 : CountSegment 1 10 13120 10880 10882 4457 4458 := CountSegment.join atomic_10881 atomic_10882
private theorem within_085_000_agreement : RowAgreement 10880 10882 := RowAgreement.join atomic_10881_agreement atomic_10882_agreement
private def within_085_001 : CountSegment 1 10 13120 10882 10884 4458 4458 := CountSegment.join atomic_10883 atomic_10884
private theorem within_085_001_agreement : RowAgreement 10882 10884 := RowAgreement.join atomic_10883_agreement atomic_10884_agreement
private def within_085_002 : CountSegment 1 10 13120 10880 10884 4457 4458 := CountSegment.join within_085_000 within_085_001
private theorem within_085_002_agreement : RowAgreement 10880 10884 := RowAgreement.join within_085_000_agreement within_085_001_agreement
private def within_085_003 : CountSegment 1 10 13120 10884 10886 4458 4459 := CountSegment.join atomic_10885 atomic_10886
private theorem within_085_003_agreement : RowAgreement 10884 10886 := RowAgreement.join atomic_10885_agreement atomic_10886_agreement
private def within_085_004 : CountSegment 1 10 13120 10886 10888 4459 4461 := CountSegment.join atomic_10887 atomic_10888
private theorem within_085_004_agreement : RowAgreement 10886 10888 := RowAgreement.join atomic_10887_agreement atomic_10888_agreement
private def within_085_005 : CountSegment 1 10 13120 10884 10888 4458 4461 := CountSegment.join within_085_003 within_085_004
private theorem within_085_005_agreement : RowAgreement 10884 10888 := RowAgreement.join within_085_003_agreement within_085_004_agreement
private def within_085_006 : CountSegment 1 10 13120 10880 10888 4457 4461 := CountSegment.join within_085_002 within_085_005
private theorem within_085_006_agreement : RowAgreement 10880 10888 := RowAgreement.join within_085_002_agreement within_085_005_agreement
private def within_085_007 : CountSegment 1 10 13120 10888 10890 4461 4461 := CountSegment.join atomic_10889 atomic_10890
private theorem within_085_007_agreement : RowAgreement 10888 10890 := RowAgreement.join atomic_10889_agreement atomic_10890_agreement
private def within_085_008 : CountSegment 1 10 13120 10890 10892 4461 4462 := CountSegment.join atomic_10891 atomic_10892
private theorem within_085_008_agreement : RowAgreement 10890 10892 := RowAgreement.join atomic_10891_agreement atomic_10892_agreement
private def within_085_009 : CountSegment 1 10 13120 10888 10892 4461 4462 := CountSegment.join within_085_007 within_085_008
private theorem within_085_009_agreement : RowAgreement 10888 10892 := RowAgreement.join within_085_007_agreement within_085_008_agreement
private def within_085_010 : CountSegment 1 10 13120 10892 10894 4462 4462 := CountSegment.join atomic_10893 atomic_10894
private theorem within_085_010_agreement : RowAgreement 10892 10894 := RowAgreement.join atomic_10893_agreement atomic_10894_agreement
private def within_085_011 : CountSegment 1 10 13120 10894 10896 4462 4463 := CountSegment.join atomic_10895 atomic_10896
private theorem within_085_011_agreement : RowAgreement 10894 10896 := RowAgreement.join atomic_10895_agreement atomic_10896_agreement
private def within_085_012 : CountSegment 1 10 13120 10892 10896 4462 4463 := CountSegment.join within_085_010 within_085_011
private theorem within_085_012_agreement : RowAgreement 10892 10896 := RowAgreement.join within_085_010_agreement within_085_011_agreement
private def within_085_013 : CountSegment 1 10 13120 10888 10896 4461 4463 := CountSegment.join within_085_009 within_085_012
private theorem within_085_013_agreement : RowAgreement 10888 10896 := RowAgreement.join within_085_009_agreement within_085_012_agreement
private def within_085_014 : CountSegment 1 10 13120 10880 10896 4457 4463 := CountSegment.join within_085_006 within_085_013
private theorem within_085_014_agreement : RowAgreement 10880 10896 := RowAgreement.join within_085_006_agreement within_085_013_agreement
private def within_085_015 : CountSegment 1 10 13120 10896 10898 4463 4463 := CountSegment.join atomic_10897 atomic_10898
private theorem within_085_015_agreement : RowAgreement 10896 10898 := RowAgreement.join atomic_10897_agreement atomic_10898_agreement
private def within_085_016 : CountSegment 1 10 13120 10898 10900 4463 4463 := CountSegment.join atomic_10899 atomic_10900
private theorem within_085_016_agreement : RowAgreement 10898 10900 := RowAgreement.join atomic_10899_agreement atomic_10900_agreement
private def within_085_017 : CountSegment 1 10 13120 10896 10900 4463 4463 := CountSegment.join within_085_015 within_085_016
private theorem within_085_017_agreement : RowAgreement 10896 10900 := RowAgreement.join within_085_015_agreement within_085_016_agreement
private def within_085_018 : CountSegment 1 10 13120 10900 10902 4463 4464 := CountSegment.join atomic_10901 atomic_10902
private theorem within_085_018_agreement : RowAgreement 10900 10902 := RowAgreement.join atomic_10901_agreement atomic_10902_agreement
private def within_085_019 : CountSegment 1 10 13120 10902 10904 4464 4465 := CountSegment.join atomic_10903 atomic_10904
private theorem within_085_019_agreement : RowAgreement 10902 10904 := RowAgreement.join atomic_10903_agreement atomic_10904_agreement
private def within_085_020 : CountSegment 1 10 13120 10900 10904 4463 4465 := CountSegment.join within_085_018 within_085_019
private theorem within_085_020_agreement : RowAgreement 10900 10904 := RowAgreement.join within_085_018_agreement within_085_019_agreement
private def within_085_021 : CountSegment 1 10 13120 10896 10904 4463 4465 := CountSegment.join within_085_017 within_085_020
private theorem within_085_021_agreement : RowAgreement 10896 10904 := RowAgreement.join within_085_017_agreement within_085_020_agreement
private def within_085_022 : CountSegment 1 10 13120 10904 10906 4465 4466 := CountSegment.join atomic_10905 atomic_10906
private theorem within_085_022_agreement : RowAgreement 10904 10906 := RowAgreement.join atomic_10905_agreement atomic_10906_agreement
private def within_085_023 : CountSegment 1 10 13120 10906 10908 4466 4468 := CountSegment.join atomic_10907 atomic_10908
private theorem within_085_023_agreement : RowAgreement 10906 10908 := RowAgreement.join atomic_10907_agreement atomic_10908_agreement
private def within_085_024 : CountSegment 1 10 13120 10904 10908 4465 4468 := CountSegment.join within_085_022 within_085_023
private theorem within_085_024_agreement : RowAgreement 10904 10908 := RowAgreement.join within_085_022_agreement within_085_023_agreement
private def within_085_025 : CountSegment 1 10 13120 10908 10910 4468 4468 := CountSegment.join atomic_10909 atomic_10910
private theorem within_085_025_agreement : RowAgreement 10908 10910 := RowAgreement.join atomic_10909_agreement atomic_10910_agreement
private def within_085_026 : CountSegment 1 10 13120 10910 10912 4468 4468 := CountSegment.join atomic_10911 atomic_10912
private theorem within_085_026_agreement : RowAgreement 10910 10912 := RowAgreement.join atomic_10911_agreement atomic_10912_agreement
private def within_085_027 : CountSegment 1 10 13120 10908 10912 4468 4468 := CountSegment.join within_085_025 within_085_026
private theorem within_085_027_agreement : RowAgreement 10908 10912 := RowAgreement.join within_085_025_agreement within_085_026_agreement
private def within_085_028 : CountSegment 1 10 13120 10904 10912 4465 4468 := CountSegment.join within_085_024 within_085_027
private theorem within_085_028_agreement : RowAgreement 10904 10912 := RowAgreement.join within_085_024_agreement within_085_027_agreement
private def within_085_029 : CountSegment 1 10 13120 10896 10912 4463 4468 := CountSegment.join within_085_021 within_085_028
private theorem within_085_029_agreement : RowAgreement 10896 10912 := RowAgreement.join within_085_021_agreement within_085_028_agreement
private def within_085_030 : CountSegment 1 10 13120 10880 10912 4457 4468 := CountSegment.join within_085_014 within_085_029
private theorem within_085_030_agreement : RowAgreement 10880 10912 := RowAgreement.join within_085_014_agreement within_085_029_agreement
private def within_085_031 : CountSegment 1 10 13120 10912 10914 4468 4469 := CountSegment.join atomic_10913 atomic_10914
private theorem within_085_031_agreement : RowAgreement 10912 10914 := RowAgreement.join atomic_10913_agreement atomic_10914_agreement
private def within_085_032 : CountSegment 1 10 13120 10914 10916 4469 4471 := CountSegment.join atomic_10915 atomic_10916
private theorem within_085_032_agreement : RowAgreement 10914 10916 := RowAgreement.join atomic_10915_agreement atomic_10916_agreement
private def within_085_033 : CountSegment 1 10 13120 10912 10916 4468 4471 := CountSegment.join within_085_031 within_085_032
private theorem within_085_033_agreement : RowAgreement 10912 10916 := RowAgreement.join within_085_031_agreement within_085_032_agreement
private def within_085_034 : CountSegment 1 10 13120 10916 10918 4471 4472 := CountSegment.join atomic_10917 atomic_10918
private theorem within_085_034_agreement : RowAgreement 10916 10918 := RowAgreement.join atomic_10917_agreement atomic_10918_agreement
private def within_085_035 : CountSegment 1 10 13120 10918 10920 4472 4473 := CountSegment.join atomic_10919 atomic_10920
private theorem within_085_035_agreement : RowAgreement 10918 10920 := RowAgreement.join atomic_10919_agreement atomic_10920_agreement
private def within_085_036 : CountSegment 1 10 13120 10916 10920 4471 4473 := CountSegment.join within_085_034 within_085_035
private theorem within_085_036_agreement : RowAgreement 10916 10920 := RowAgreement.join within_085_034_agreement within_085_035_agreement
private def within_085_037 : CountSegment 1 10 13120 10912 10920 4468 4473 := CountSegment.join within_085_033 within_085_036
private theorem within_085_037_agreement : RowAgreement 10912 10920 := RowAgreement.join within_085_033_agreement within_085_036_agreement
private def within_085_038 : CountSegment 1 10 13120 10920 10922 4473 4473 := CountSegment.join atomic_10921 atomic_10922
private theorem within_085_038_agreement : RowAgreement 10920 10922 := RowAgreement.join atomic_10921_agreement atomic_10922_agreement
private def within_085_039 : CountSegment 1 10 13120 10922 10924 4473 4473 := CountSegment.join atomic_10923 atomic_10924
private theorem within_085_039_agreement : RowAgreement 10922 10924 := RowAgreement.join atomic_10923_agreement atomic_10924_agreement
private def within_085_040 : CountSegment 1 10 13120 10920 10924 4473 4473 := CountSegment.join within_085_038 within_085_039
private theorem within_085_040_agreement : RowAgreement 10920 10924 := RowAgreement.join within_085_038_agreement within_085_039_agreement
private def within_085_041 : CountSegment 1 10 13120 10924 10926 4473 4473 := CountSegment.join atomic_10925 atomic_10926
private theorem within_085_041_agreement : RowAgreement 10924 10926 := RowAgreement.join atomic_10925_agreement atomic_10926_agreement
private def within_085_042 : CountSegment 1 10 13120 10926 10928 4473 4475 := CountSegment.join atomic_10927 atomic_10928
private theorem within_085_042_agreement : RowAgreement 10926 10928 := RowAgreement.join atomic_10927_agreement atomic_10928_agreement
private def within_085_043 : CountSegment 1 10 13120 10924 10928 4473 4475 := CountSegment.join within_085_041 within_085_042
private theorem within_085_043_agreement : RowAgreement 10924 10928 := RowAgreement.join within_085_041_agreement within_085_042_agreement
private def within_085_044 : CountSegment 1 10 13120 10920 10928 4473 4475 := CountSegment.join within_085_040 within_085_043
private theorem within_085_044_agreement : RowAgreement 10920 10928 := RowAgreement.join within_085_040_agreement within_085_043_agreement
private def within_085_045 : CountSegment 1 10 13120 10912 10928 4468 4475 := CountSegment.join within_085_037 within_085_044
private theorem within_085_045_agreement : RowAgreement 10912 10928 := RowAgreement.join within_085_037_agreement within_085_044_agreement
private def within_085_046 : CountSegment 1 10 13120 10928 10930 4475 4475 := CountSegment.join atomic_10929 atomic_10930
private theorem within_085_046_agreement : RowAgreement 10928 10930 := RowAgreement.join atomic_10929_agreement atomic_10930_agreement
private def within_085_047 : CountSegment 1 10 13120 10930 10932 4475 4475 := CountSegment.join atomic_10931 atomic_10932
private theorem within_085_047_agreement : RowAgreement 10930 10932 := RowAgreement.join atomic_10931_agreement atomic_10932_agreement
private def within_085_048 : CountSegment 1 10 13120 10928 10932 4475 4475 := CountSegment.join within_085_046 within_085_047
private theorem within_085_048_agreement : RowAgreement 10928 10932 := RowAgreement.join within_085_046_agreement within_085_047_agreement
private def within_085_049 : CountSegment 1 10 13120 10932 10934 4475 4476 := CountSegment.join atomic_10933 atomic_10934
private theorem within_085_049_agreement : RowAgreement 10932 10934 := RowAgreement.join atomic_10933_agreement atomic_10934_agreement
private def within_085_050 : CountSegment 1 10 13120 10934 10936 4476 4477 := CountSegment.join atomic_10935 atomic_10936
private theorem within_085_050_agreement : RowAgreement 10934 10936 := RowAgreement.join atomic_10935_agreement atomic_10936_agreement
private def within_085_051 : CountSegment 1 10 13120 10932 10936 4475 4477 := CountSegment.join within_085_049 within_085_050
private theorem within_085_051_agreement : RowAgreement 10932 10936 := RowAgreement.join within_085_049_agreement within_085_050_agreement
private def within_085_052 : CountSegment 1 10 13120 10928 10936 4475 4477 := CountSegment.join within_085_048 within_085_051
private theorem within_085_052_agreement : RowAgreement 10928 10936 := RowAgreement.join within_085_048_agreement within_085_051_agreement
private def within_085_053 : CountSegment 1 10 13120 10936 10938 4477 4477 := CountSegment.join atomic_10937 atomic_10938
private theorem within_085_053_agreement : RowAgreement 10936 10938 := RowAgreement.join atomic_10937_agreement atomic_10938_agreement
private def within_085_054 : CountSegment 1 10 13120 10938 10940 4477 4479 := CountSegment.join atomic_10939 atomic_10940
private theorem within_085_054_agreement : RowAgreement 10938 10940 := RowAgreement.join atomic_10939_agreement atomic_10940_agreement
private def within_085_055 : CountSegment 1 10 13120 10936 10940 4477 4479 := CountSegment.join within_085_053 within_085_054
private theorem within_085_055_agreement : RowAgreement 10936 10940 := RowAgreement.join within_085_053_agreement within_085_054_agreement
private def within_085_056 : CountSegment 1 10 13120 10940 10942 4479 4479 := CountSegment.join atomic_10941 atomic_10942
private theorem within_085_056_agreement : RowAgreement 10940 10942 := RowAgreement.join atomic_10941_agreement atomic_10942_agreement
private def within_085_057 : CountSegment 1 10 13120 10942 10944 4479 4481 := CountSegment.join atomic_10943 atomic_10944
private theorem within_085_057_agreement : RowAgreement 10942 10944 := RowAgreement.join atomic_10943_agreement atomic_10944_agreement
private def within_085_058 : CountSegment 1 10 13120 10940 10944 4479 4481 := CountSegment.join within_085_056 within_085_057
private theorem within_085_058_agreement : RowAgreement 10940 10944 := RowAgreement.join within_085_056_agreement within_085_057_agreement
private def within_085_059 : CountSegment 1 10 13120 10936 10944 4477 4481 := CountSegment.join within_085_055 within_085_058
private theorem within_085_059_agreement : RowAgreement 10936 10944 := RowAgreement.join within_085_055_agreement within_085_058_agreement
private def within_085_060 : CountSegment 1 10 13120 10928 10944 4475 4481 := CountSegment.join within_085_052 within_085_059
private theorem within_085_060_agreement : RowAgreement 10928 10944 := RowAgreement.join within_085_052_agreement within_085_059_agreement
private def within_085_061 : CountSegment 1 10 13120 10912 10944 4468 4481 := CountSegment.join within_085_045 within_085_060
private theorem within_085_061_agreement : RowAgreement 10912 10944 := RowAgreement.join within_085_045_agreement within_085_060_agreement
private def within_085_062 : CountSegment 1 10 13120 10880 10944 4457 4481 := CountSegment.join within_085_030 within_085_061
private theorem within_085_062_agreement : RowAgreement 10880 10944 := RowAgreement.join within_085_030_agreement within_085_061_agreement
private def within_085_063 : CountSegment 1 10 13120 10944 10946 4481 4481 := CountSegment.join atomic_10945 atomic_10946
private theorem within_085_063_agreement : RowAgreement 10944 10946 := RowAgreement.join atomic_10945_agreement atomic_10946_agreement
private def within_085_064 : CountSegment 1 10 13120 10946 10948 4481 4484 := CountSegment.join atomic_10947 atomic_10948
private theorem within_085_064_agreement : RowAgreement 10946 10948 := RowAgreement.join atomic_10947_agreement atomic_10948_agreement
private def within_085_065 : CountSegment 1 10 13120 10944 10948 4481 4484 := CountSegment.join within_085_063 within_085_064
private theorem within_085_065_agreement : RowAgreement 10944 10948 := RowAgreement.join within_085_063_agreement within_085_064_agreement
private def within_085_066 : CountSegment 1 10 13120 10948 10950 4484 4485 := CountSegment.join atomic_10949 atomic_10950
private theorem within_085_066_agreement : RowAgreement 10948 10950 := RowAgreement.join atomic_10949_agreement atomic_10950_agreement
private def within_085_067 : CountSegment 1 10 13120 10950 10952 4485 4486 := CountSegment.join atomic_10951 atomic_10952
private theorem within_085_067_agreement : RowAgreement 10950 10952 := RowAgreement.join atomic_10951_agreement atomic_10952_agreement
private def within_085_068 : CountSegment 1 10 13120 10948 10952 4484 4486 := CountSegment.join within_085_066 within_085_067
private theorem within_085_068_agreement : RowAgreement 10948 10952 := RowAgreement.join within_085_066_agreement within_085_067_agreement
private def within_085_069 : CountSegment 1 10 13120 10944 10952 4481 4486 := CountSegment.join within_085_065 within_085_068
private theorem within_085_069_agreement : RowAgreement 10944 10952 := RowAgreement.join within_085_065_agreement within_085_068_agreement
private def within_085_070 : CountSegment 1 10 13120 10952 10954 4486 4487 := CountSegment.join atomic_10953 atomic_10954
private theorem within_085_070_agreement : RowAgreement 10952 10954 := RowAgreement.join atomic_10953_agreement atomic_10954_agreement
private def within_085_071 : CountSegment 1 10 13120 10954 10956 4487 4487 := CountSegment.join atomic_10955 atomic_10956
private theorem within_085_071_agreement : RowAgreement 10954 10956 := RowAgreement.join atomic_10955_agreement atomic_10956_agreement
private def within_085_072 : CountSegment 1 10 13120 10952 10956 4486 4487 := CountSegment.join within_085_070 within_085_071
private theorem within_085_072_agreement : RowAgreement 10952 10956 := RowAgreement.join within_085_070_agreement within_085_071_agreement
private def within_085_073 : CountSegment 1 10 13120 10956 10958 4487 4489 := CountSegment.join atomic_10957 atomic_10958
private theorem within_085_073_agreement : RowAgreement 10956 10958 := RowAgreement.join atomic_10957_agreement atomic_10958_agreement
private def within_085_074 : CountSegment 1 10 13120 10958 10960 4489 4489 := CountSegment.join atomic_10959 atomic_10960
private theorem within_085_074_agreement : RowAgreement 10958 10960 := RowAgreement.join atomic_10959_agreement atomic_10960_agreement
private def within_085_075 : CountSegment 1 10 13120 10956 10960 4487 4489 := CountSegment.join within_085_073 within_085_074
private theorem within_085_075_agreement : RowAgreement 10956 10960 := RowAgreement.join within_085_073_agreement within_085_074_agreement
private def within_085_076 : CountSegment 1 10 13120 10952 10960 4486 4489 := CountSegment.join within_085_072 within_085_075
private theorem within_085_076_agreement : RowAgreement 10952 10960 := RowAgreement.join within_085_072_agreement within_085_075_agreement
private def within_085_077 : CountSegment 1 10 13120 10944 10960 4481 4489 := CountSegment.join within_085_069 within_085_076
private theorem within_085_077_agreement : RowAgreement 10944 10960 := RowAgreement.join within_085_069_agreement within_085_076_agreement
private def within_085_078 : CountSegment 1 10 13120 10960 10962 4489 4491 := CountSegment.join atomic_10961 atomic_10962
private theorem within_085_078_agreement : RowAgreement 10960 10962 := RowAgreement.join atomic_10961_agreement atomic_10962_agreement
private def within_085_079 : CountSegment 1 10 13120 10962 10964 4491 4491 := CountSegment.join atomic_10963 atomic_10964
private theorem within_085_079_agreement : RowAgreement 10962 10964 := RowAgreement.join atomic_10963_agreement atomic_10964_agreement
private def within_085_080 : CountSegment 1 10 13120 10960 10964 4489 4491 := CountSegment.join within_085_078 within_085_079
private theorem within_085_080_agreement : RowAgreement 10960 10964 := RowAgreement.join within_085_078_agreement within_085_079_agreement
private def within_085_081 : CountSegment 1 10 13120 10964 10966 4491 4492 := CountSegment.join atomic_10965 atomic_10966
private theorem within_085_081_agreement : RowAgreement 10964 10966 := RowAgreement.join atomic_10965_agreement atomic_10966_agreement
private def within_085_082 : CountSegment 1 10 13120 10966 10968 4492 4492 := CountSegment.join atomic_10967 atomic_10968
private theorem within_085_082_agreement : RowAgreement 10966 10968 := RowAgreement.join atomic_10967_agreement atomic_10968_agreement
private def within_085_083 : CountSegment 1 10 13120 10964 10968 4491 4492 := CountSegment.join within_085_081 within_085_082
private theorem within_085_083_agreement : RowAgreement 10964 10968 := RowAgreement.join within_085_081_agreement within_085_082_agreement
private def within_085_084 : CountSegment 1 10 13120 10960 10968 4489 4492 := CountSegment.join within_085_080 within_085_083
private theorem within_085_084_agreement : RowAgreement 10960 10968 := RowAgreement.join within_085_080_agreement within_085_083_agreement
private def within_085_085 : CountSegment 1 10 13120 10968 10970 4492 4494 := CountSegment.join atomic_10969 atomic_10970
private theorem within_085_085_agreement : RowAgreement 10968 10970 := RowAgreement.join atomic_10969_agreement atomic_10970_agreement
private def within_085_086 : CountSegment 1 10 13120 10970 10972 4494 4495 := CountSegment.join atomic_10971 atomic_10972
private theorem within_085_086_agreement : RowAgreement 10970 10972 := RowAgreement.join atomic_10971_agreement atomic_10972_agreement
private def within_085_087 : CountSegment 1 10 13120 10968 10972 4492 4495 := CountSegment.join within_085_085 within_085_086
private theorem within_085_087_agreement : RowAgreement 10968 10972 := RowAgreement.join within_085_085_agreement within_085_086_agreement
private def within_085_088 : CountSegment 1 10 13120 10972 10974 4495 4496 := CountSegment.join atomic_10973 atomic_10974
private theorem within_085_088_agreement : RowAgreement 10972 10974 := RowAgreement.join atomic_10973_agreement atomic_10974_agreement
private def within_085_089 : CountSegment 1 10 13120 10974 10976 4496 4497 := CountSegment.join atomic_10975 atomic_10976
private theorem within_085_089_agreement : RowAgreement 10974 10976 := RowAgreement.join atomic_10975_agreement atomic_10976_agreement
private def within_085_090 : CountSegment 1 10 13120 10972 10976 4495 4497 := CountSegment.join within_085_088 within_085_089
private theorem within_085_090_agreement : RowAgreement 10972 10976 := RowAgreement.join within_085_088_agreement within_085_089_agreement
private def within_085_091 : CountSegment 1 10 13120 10968 10976 4492 4497 := CountSegment.join within_085_087 within_085_090
private theorem within_085_091_agreement : RowAgreement 10968 10976 := RowAgreement.join within_085_087_agreement within_085_090_agreement
private def within_085_092 : CountSegment 1 10 13120 10960 10976 4489 4497 := CountSegment.join within_085_084 within_085_091
private theorem within_085_092_agreement : RowAgreement 10960 10976 := RowAgreement.join within_085_084_agreement within_085_091_agreement
private def within_085_093 : CountSegment 1 10 13120 10944 10976 4481 4497 := CountSegment.join within_085_077 within_085_092
private theorem within_085_093_agreement : RowAgreement 10944 10976 := RowAgreement.join within_085_077_agreement within_085_092_agreement
private def within_085_094 : CountSegment 1 10 13120 10976 10978 4497 4499 := CountSegment.join atomic_10977 atomic_10978
private theorem within_085_094_agreement : RowAgreement 10976 10978 := RowAgreement.join atomic_10977_agreement atomic_10978_agreement
private def within_085_095 : CountSegment 1 10 13120 10978 10980 4499 4499 := CountSegment.join atomic_10979 atomic_10980
private theorem within_085_095_agreement : RowAgreement 10978 10980 := RowAgreement.join atomic_10979_agreement atomic_10980_agreement
private def within_085_096 : CountSegment 1 10 13120 10976 10980 4497 4499 := CountSegment.join within_085_094 within_085_095
private theorem within_085_096_agreement : RowAgreement 10976 10980 := RowAgreement.join within_085_094_agreement within_085_095_agreement
private def within_085_097 : CountSegment 1 10 13120 10980 10982 4499 4501 := CountSegment.join atomic_10981 atomic_10982
private theorem within_085_097_agreement : RowAgreement 10980 10982 := RowAgreement.join atomic_10981_agreement atomic_10982_agreement
private def within_085_098 : CountSegment 1 10 13120 10982 10984 4501 4501 := CountSegment.join atomic_10983 atomic_10984
private theorem within_085_098_agreement : RowAgreement 10982 10984 := RowAgreement.join atomic_10983_agreement atomic_10984_agreement
private def within_085_099 : CountSegment 1 10 13120 10980 10984 4499 4501 := CountSegment.join within_085_097 within_085_098
private theorem within_085_099_agreement : RowAgreement 10980 10984 := RowAgreement.join within_085_097_agreement within_085_098_agreement
private def within_085_100 : CountSegment 1 10 13120 10976 10984 4497 4501 := CountSegment.join within_085_096 within_085_099
private theorem within_085_100_agreement : RowAgreement 10976 10984 := RowAgreement.join within_085_096_agreement within_085_099_agreement
private def within_085_101 : CountSegment 1 10 13120 10984 10986 4501 4502 := CountSegment.join atomic_10985 atomic_10986
private theorem within_085_101_agreement : RowAgreement 10984 10986 := RowAgreement.join atomic_10985_agreement atomic_10986_agreement
private def within_085_102 : CountSegment 1 10 13120 10986 10988 4502 4503 := CountSegment.join atomic_10987 atomic_10988
private theorem within_085_102_agreement : RowAgreement 10986 10988 := RowAgreement.join atomic_10987_agreement atomic_10988_agreement
private def within_085_103 : CountSegment 1 10 13120 10984 10988 4501 4503 := CountSegment.join within_085_101 within_085_102
private theorem within_085_103_agreement : RowAgreement 10984 10988 := RowAgreement.join within_085_101_agreement within_085_102_agreement
private def within_085_104 : CountSegment 1 10 13120 10988 10990 4503 4504 := CountSegment.join atomic_10989 atomic_10990
private theorem within_085_104_agreement : RowAgreement 10988 10990 := RowAgreement.join atomic_10989_agreement atomic_10990_agreement
private def within_085_105 : CountSegment 1 10 13120 10990 10992 4504 4504 := CountSegment.join atomic_10991 atomic_10992
private theorem within_085_105_agreement : RowAgreement 10990 10992 := RowAgreement.join atomic_10991_agreement atomic_10992_agreement
private def within_085_106 : CountSegment 1 10 13120 10988 10992 4503 4504 := CountSegment.join within_085_104 within_085_105
private theorem within_085_106_agreement : RowAgreement 10988 10992 := RowAgreement.join within_085_104_agreement within_085_105_agreement
private def within_085_107 : CountSegment 1 10 13120 10984 10992 4501 4504 := CountSegment.join within_085_103 within_085_106
private theorem within_085_107_agreement : RowAgreement 10984 10992 := RowAgreement.join within_085_103_agreement within_085_106_agreement
private def within_085_108 : CountSegment 1 10 13120 10976 10992 4497 4504 := CountSegment.join within_085_100 within_085_107
private theorem within_085_108_agreement : RowAgreement 10976 10992 := RowAgreement.join within_085_100_agreement within_085_107_agreement
private def within_085_109 : CountSegment 1 10 13120 10992 10994 4504 4505 := CountSegment.join atomic_10993 atomic_10994
private theorem within_085_109_agreement : RowAgreement 10992 10994 := RowAgreement.join atomic_10993_agreement atomic_10994_agreement
private def within_085_110 : CountSegment 1 10 13120 10994 10996 4505 4506 := CountSegment.join atomic_10995 atomic_10996
private theorem within_085_110_agreement : RowAgreement 10994 10996 := RowAgreement.join atomic_10995_agreement atomic_10996_agreement
private def within_085_111 : CountSegment 1 10 13120 10992 10996 4504 4506 := CountSegment.join within_085_109 within_085_110
private theorem within_085_111_agreement : RowAgreement 10992 10996 := RowAgreement.join within_085_109_agreement within_085_110_agreement
private def within_085_112 : CountSegment 1 10 13120 10996 10998 4506 4507 := CountSegment.join atomic_10997 atomic_10998
private theorem within_085_112_agreement : RowAgreement 10996 10998 := RowAgreement.join atomic_10997_agreement atomic_10998_agreement
private def within_085_113 : CountSegment 1 10 13120 10998 11000 4507 4507 := CountSegment.join atomic_10999 atomic_11000
private theorem within_085_113_agreement : RowAgreement 10998 11000 := RowAgreement.join atomic_10999_agreement atomic_11000_agreement
private def within_085_114 : CountSegment 1 10 13120 10996 11000 4506 4507 := CountSegment.join within_085_112 within_085_113
private theorem within_085_114_agreement : RowAgreement 10996 11000 := RowAgreement.join within_085_112_agreement within_085_113_agreement
private def within_085_115 : CountSegment 1 10 13120 10992 11000 4504 4507 := CountSegment.join within_085_111 within_085_114
private theorem within_085_115_agreement : RowAgreement 10992 11000 := RowAgreement.join within_085_111_agreement within_085_114_agreement
private def within_085_116 : CountSegment 1 10 13120 11000 11002 4507 4507 := CountSegment.join atomic_11001 atomic_11002
private theorem within_085_116_agreement : RowAgreement 11000 11002 := RowAgreement.join atomic_11001_agreement atomic_11002_agreement
private def within_085_117 : CountSegment 1 10 13120 11002 11004 4507 4507 := CountSegment.join atomic_11003 atomic_11004
private theorem within_085_117_agreement : RowAgreement 11002 11004 := RowAgreement.join atomic_11003_agreement atomic_11004_agreement
private def within_085_118 : CountSegment 1 10 13120 11000 11004 4507 4507 := CountSegment.join within_085_116 within_085_117
private theorem within_085_118_agreement : RowAgreement 11000 11004 := RowAgreement.join within_085_116_agreement within_085_117_agreement
private def within_085_119 : CountSegment 1 10 13120 11004 11006 4507 4507 := CountSegment.join atomic_11005 atomic_11006
private theorem within_085_119_agreement : RowAgreement 11004 11006 := RowAgreement.join atomic_11005_agreement atomic_11006_agreement
private def within_085_120 : CountSegment 1 10 13120 11006 11008 4507 4509 := CountSegment.join atomic_11007 atomic_11008
private theorem within_085_120_agreement : RowAgreement 11006 11008 := RowAgreement.join atomic_11007_agreement atomic_11008_agreement
private def within_085_121 : CountSegment 1 10 13120 11004 11008 4507 4509 := CountSegment.join within_085_119 within_085_120
private theorem within_085_121_agreement : RowAgreement 11004 11008 := RowAgreement.join within_085_119_agreement within_085_120_agreement
private def within_085_122 : CountSegment 1 10 13120 11000 11008 4507 4509 := CountSegment.join within_085_118 within_085_121
private theorem within_085_122_agreement : RowAgreement 11000 11008 := RowAgreement.join within_085_118_agreement within_085_121_agreement
private def within_085_123 : CountSegment 1 10 13120 10992 11008 4504 4509 := CountSegment.join within_085_115 within_085_122
private theorem within_085_123_agreement : RowAgreement 10992 11008 := RowAgreement.join within_085_115_agreement within_085_122_agreement
private def within_085_124 : CountSegment 1 10 13120 10976 11008 4497 4509 := CountSegment.join within_085_108 within_085_123
private theorem within_085_124_agreement : RowAgreement 10976 11008 := RowAgreement.join within_085_108_agreement within_085_123_agreement
private def within_085_125 : CountSegment 1 10 13120 10944 11008 4481 4509 := CountSegment.join within_085_093 within_085_124
private theorem within_085_125_agreement : RowAgreement 10944 11008 := RowAgreement.join within_085_093_agreement within_085_124_agreement
private def within_085_126 : CountSegment 1 10 13120 10880 11008 4457 4509 := CountSegment.join within_085_062 within_085_125
private theorem within_085_126_agreement : RowAgreement 10880 11008 := RowAgreement.join within_085_062_agreement within_085_125_agreement
def segment_085 : CountSegment 1 10 13120 10880 11008 4457 4509 := within_085_126
theorem agreement_085 : RowAgreement 10880 11008 := within_085_126_agreement
end ProjectEulerSolutions.P58Numerical
