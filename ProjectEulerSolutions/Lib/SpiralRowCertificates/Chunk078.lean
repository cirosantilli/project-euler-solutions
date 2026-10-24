import ProjectEulerSolutions.Lib.SpiralPrimeSegments
import ProjectEulerSolutions.Lib.SpiralPrimeCertificates.All
import ProjectEulerSolutions.Lib.MillerRabin
import ProjectEulerSolutions.Lib.SpiralMillerAgreement
/-! Finite spiral row and counter certificates. Python execution is checked separately. -/
namespace ProjectEulerSolutions.P58Numerical
open SpiralPrimes
set_option Elab.async false
set_option maxRecDepth 4096
def row_9985 : RowCertificate 9985 where
  first := true
  first_correct := by
    have prime : Nat.Prime (corner 9985 1) := by change Nat.Prime 398820871; exact ProjectEulerSolutions.P58PrimeNodes.p_398820871.prime
    exact decide_eq_true prime
  second := false
  second_correct := by
    have composite : ¬ Nat.Prime (corner 9985 2) := by change ¬ Nat.Prime 398800901; exact PrattCertificates.composite_of_product (factor := 89) (cofactor := 4480909) rfl (by decide) (by decide)
    exact decide_eq_false composite
  third := false
  third_correct := by
    have composite : ¬ Nat.Prime (corner 9985 3) := by change ¬ Nat.Prime 398780931; exact PrattCertificates.composite_of_product (factor := 3) (cofactor := 132926977) rfl (by decide) (by decide)
    exact decide_eq_false composite
theorem row_count_9985 : row 9985 = 1 := (row_9985.checked).trans rfl
theorem model_count_9985 : modelRow 9985 = 1 := by decide
theorem modelrow_agrees_9985 : modelRow 9985 = row 9985 := model_count_9985.trans row_count_9985.symm
def row_9986 : RowCertificate 9986 where
  first := false
  first_correct := by
    have composite : ¬ Nat.Prime (corner 9986 1) := by change ¬ Nat.Prime 398900757; exact PrattCertificates.composite_of_product (factor := 3) (cofactor := 132966919) rfl (by decide) (by decide)
    exact decide_eq_false composite
  second := false
  second_correct := by
    have composite : ¬ Nat.Prime (corner 9986 2) := by change ¬ Nat.Prime 398880785; exact PrattCertificates.composite_of_product (factor := 5) (cofactor := 79776157) rfl (by decide) (by decide)
    exact decide_eq_false composite
  third := false
  third_correct := by
    have composite : ¬ Nat.Prime (corner 9986 3) := by change ¬ Nat.Prime 398860813; exact PrattCertificates.composite_of_product (factor := 13) (cofactor := 30681601) rfl (by decide) (by decide)
    exact decide_eq_false composite
theorem row_count_9986 : row 9986 = 0 := (row_9986.checked).trans rfl
theorem model_count_9986 : modelRow 9986 = 0 := by decide
theorem modelrow_agrees_9986 : modelRow 9986 = row 9986 := model_count_9986.trans row_count_9986.symm
def row_9987 : RowCertificate 9987 where
  first := false
  first_correct := by
    have composite : ¬ Nat.Prime (corner 9987 1) := by change ¬ Nat.Prime 398980651; exact PrattCertificates.composite_of_product (factor := 457) (cofactor := 873043) rfl (by decide) (by decide)
    exact decide_eq_false composite
  second := false
  second_correct := by
    have composite : ¬ Nat.Prime (corner 9987 2) := by change ¬ Nat.Prime 398960677; exact PrattCertificates.composite_of_product (factor := 37) (cofactor := 10782721) rfl (by decide) (by decide)
    exact decide_eq_false composite
  third := false
  third_correct := by
    have composite : ¬ Nat.Prime (corner 9987 3) := by change ¬ Nat.Prime 398940703; exact PrattCertificates.composite_of_product (factor := 7) (cofactor := 56991529) rfl (by decide) (by decide)
    exact decide_eq_false composite
theorem row_count_9987 : row 9987 = 0 := (row_9987.checked).trans rfl
theorem model_count_9987 : modelRow 9987 = 0 := by decide
theorem modelrow_agrees_9987 : modelRow 9987 = row 9987 := model_count_9987.trans row_count_9987.symm
def row_9988 : RowCertificate 9988 where
  first := false
  first_correct := by
    have composite : ¬ Nat.Prime (corner 9988 1) := by change ¬ Nat.Prime 399060553; exact PrattCertificates.composite_of_product (factor := 19) (cofactor := 21003187) rfl (by decide) (by decide)
    exact decide_eq_false composite
  second := false
  second_correct := by
    have composite : ¬ Nat.Prime (corner 9988 2) := by change ¬ Nat.Prime 399040577; exact PrattCertificates.composite_of_product (factor := 13) (cofactor := 30695429) rfl (by decide) (by decide)
    exact decide_eq_false composite
  third := false
  third_correct := by
    have composite : ¬ Nat.Prime (corner 9988 3) := by change ¬ Nat.Prime 399020601; exact PrattCertificates.composite_of_product (factor := 3) (cofactor := 133006867) rfl (by decide) (by decide)
    exact decide_eq_false composite
theorem row_count_9988 : row 9988 = 0 := (row_9988.checked).trans rfl
theorem model_count_9988 : modelRow 9988 = 0 := by decide
theorem modelrow_agrees_9988 : modelRow 9988 = row 9988 := model_count_9988.trans row_count_9988.symm
def row_9989 : RowCertificate 9989 where
  first := false
  first_correct := by
    have composite : ¬ Nat.Prime (corner 9989 1) := by change ¬ Nat.Prime 399140463; exact PrattCertificates.composite_of_product (factor := 3) (cofactor := 133046821) rfl (by decide) (by decide)
    exact decide_eq_false composite
  second := false
  second_correct := by
    have composite : ¬ Nat.Prime (corner 9989 2) := by change ¬ Nat.Prime 399120485; exact PrattCertificates.composite_of_product (factor := 5) (cofactor := 79824097) rfl (by decide) (by decide)
    exact decide_eq_false composite
  third := false
  third_correct := by
    have composite : ¬ Nat.Prime (corner 9989 3) := by change ¬ Nat.Prime 399100507; exact PrattCertificates.composite_of_product (factor := 13) (cofactor := 30700039) rfl (by decide) (by decide)
    exact decide_eq_false composite
theorem row_count_9989 : row 9989 = 0 := (row_9989.checked).trans rfl
theorem model_count_9989 : modelRow 9989 = 0 := by decide
theorem modelrow_agrees_9989 : modelRow 9989 = row 9989 := model_count_9989.trans row_count_9989.symm
def row_9990 : RowCertificate 9990 where
  first := false
  first_correct := by
    have composite : ¬ Nat.Prime (corner 9990 1) := by change ¬ Nat.Prime 399220381; exact PrattCertificates.composite_of_product (factor := 7) (cofactor := 57031483) rfl (by decide) (by decide)
    exact decide_eq_false composite
  second := false
  second_correct := by
    have composite : ¬ Nat.Prime (corner 9990 2) := by change ¬ Nat.Prime 399200401; exact PrattCertificates.composite_of_product (factor := 109) (cofactor := 3662389) rfl (by decide) (by decide)
    exact decide_eq_false composite
  third := true
  third_correct := by
    have prime : Nat.Prime (corner 9990 3) := by change Nat.Prime 399180421; exact ProjectEulerSolutions.P58PrimeNodes.p_399180421.prime
    exact decide_eq_true prime
theorem row_count_9990 : row 9990 = 1 := (row_9990.checked).trans rfl
theorem model_count_9990 : modelRow 9990 = 1 := by decide
theorem modelrow_agrees_9990 : modelRow 9990 = row 9990 := model_count_9990.trans row_count_9990.symm
def row_9991 : RowCertificate 9991 where
  first := false
  first_correct := by
    have composite : ¬ Nat.Prime (corner 9991 1) := by change ¬ Nat.Prime 399300307; exact PrattCertificates.composite_of_product (factor := 7) (cofactor := 57042901) rfl (by decide) (by decide)
    exact decide_eq_false composite
  second := false
  second_correct := by
    have composite : ¬ Nat.Prime (corner 9991 2) := by change ¬ Nat.Prime 399280325; exact PrattCertificates.composite_of_product (factor := 5) (cofactor := 79856065) rfl (by decide) (by decide)
    exact decide_eq_false composite
  third := false
  third_correct := by
    have composite : ¬ Nat.Prime (corner 9991 3) := by change ¬ Nat.Prime 399260343; exact PrattCertificates.composite_of_product (factor := 3) (cofactor := 133086781) rfl (by decide) (by decide)
    exact decide_eq_false composite
theorem row_count_9991 : row 9991 = 0 := (row_9991.checked).trans rfl
theorem model_count_9991 : modelRow 9991 = 0 := by decide
theorem modelrow_agrees_9991 : modelRow 9991 = row 9991 := model_count_9991.trans row_count_9991.symm
def row_9992 : RowCertificate 9992 where
  first := false
  first_correct := by
    have composite : ¬ Nat.Prime (corner 9992 1) := by change ¬ Nat.Prime 399380241; exact PrattCertificates.composite_of_product (factor := 3) (cofactor := 133126747) rfl (by decide) (by decide)
    exact decide_eq_false composite
  second := true
  second_correct := by
    have prime : Nat.Prime (corner 9992 2) := by change Nat.Prime 399360257; exact ProjectEulerSolutions.P58PrimeNodes.p_399360257.prime
    exact decide_eq_true prime
  third := true
  third_correct := by
    have prime : Nat.Prime (corner 9992 3) := by change Nat.Prime 399340273; exact ProjectEulerSolutions.P58PrimeNodes.p_399340273.prime
    exact decide_eq_true prime
theorem row_count_9992 : row 9992 = 2 := (row_9992.checked).trans rfl
theorem model_count_9992 : modelRow 9992 = 2 := by decide
theorem modelrow_agrees_9992 : modelRow 9992 = row 9992 := model_count_9992.trans row_count_9992.symm
def row_9993 : RowCertificate 9993 where
  first := true
  first_correct := by
    have prime : Nat.Prime (corner 9993 1) := by change Nat.Prime 399460183; exact ProjectEulerSolutions.P58PrimeNodes.p_399460183.prime
    exact decide_eq_true prime
  second := false
  second_correct := by
    have composite : ¬ Nat.Prime (corner 9993 2) := by change ¬ Nat.Prime 399440197; exact PrattCertificates.composite_of_product (factor := 13) (cofactor := 30726169) rfl (by decide) (by decide)
    exact decide_eq_false composite
  third := false
  third_correct := by
    have composite : ¬ Nat.Prime (corner 9993 3) := by change ¬ Nat.Prime 399420211; exact PrattCertificates.composite_of_product (factor := 1117) (cofactor := 357583) rfl (by decide) (by decide)
    exact decide_eq_false composite
theorem row_count_9993 : row 9993 = 1 := (row_9993.checked).trans rfl
theorem model_count_9993 : modelRow 9993 = 1 := by decide
theorem modelrow_agrees_9993 : modelRow 9993 = row 9993 := model_count_9993.trans row_count_9993.symm
def row_9994 : RowCertificate 9994 where
  first := false
  first_correct := by
    have composite : ¬ Nat.Prime (corner 9994 1) := by change ¬ Nat.Prime 399540133; exact PrattCertificates.composite_of_product (factor := 43) (cofactor := 9291631) rfl (by decide) (by decide)
    exact decide_eq_false composite
  second := false
  second_correct := by
    have composite : ¬ Nat.Prime (corner 9994 2) := by change ¬ Nat.Prime 399520145; exact PrattCertificates.composite_of_product (factor := 5) (cofactor := 79904029) rfl (by decide) (by decide)
    exact decide_eq_false composite
  third := false
  third_correct := by
    have composite : ¬ Nat.Prime (corner 9994 3) := by change ¬ Nat.Prime 399500157; exact PrattCertificates.composite_of_product (factor := 3) (cofactor := 133166719) rfl (by decide) (by decide)
    exact decide_eq_false composite
theorem row_count_9994 : row 9994 = 0 := (row_9994.checked).trans rfl
theorem model_count_9994 : modelRow 9994 = 0 := by decide
theorem modelrow_agrees_9994 : modelRow 9994 = row 9994 := model_count_9994.trans row_count_9994.symm
def row_9995 : RowCertificate 9995 where
  first := false
  first_correct := by
    have composite : ¬ Nat.Prime (corner 9995 1) := by change ¬ Nat.Prime 399620091; exact PrattCertificates.composite_of_product (factor := 3) (cofactor := 133206697) rfl (by decide) (by decide)
    exact decide_eq_false composite
  second := true
  second_correct := by
    have prime : Nat.Prime (corner 9995 2) := by change Nat.Prime 399600101; exact ProjectEulerSolutions.P58PrimeNodes.p_399600101.prime
    exact decide_eq_true prime
  third := false
  third_correct := by
    have composite : ¬ Nat.Prime (corner 9995 3) := by change ¬ Nat.Prime 399580111; exact PrattCertificates.composite_of_product (factor := 7) (cofactor := 57082873) rfl (by decide) (by decide)
    exact decide_eq_false composite
theorem row_count_9995 : row 9995 = 1 := (row_9995.checked).trans rfl
theorem model_count_9995 : modelRow 9995 = 1 := by decide
theorem modelrow_agrees_9995 : modelRow 9995 = row 9995 := model_count_9995.trans row_count_9995.symm
def row_9996 : RowCertificate 9996 where
  first := false
  first_correct := by
    have composite : ¬ Nat.Prime (corner 9996 1) := by change ¬ Nat.Prime 399700057; exact PrattCertificates.composite_of_product (factor := 109) (cofactor := 3666973) rfl (by decide) (by decide)
    exact decide_eq_false composite
  second := false
  second_correct := by
    have composite : ¬ Nat.Prime (corner 9996 2) := by change ¬ Nat.Prime 399680065; exact PrattCertificates.composite_of_product (factor := 5) (cofactor := 79936013) rfl (by decide) (by decide)
    exact decide_eq_false composite
  third := true
  third_correct := by
    have prime : Nat.Prime (corner 9996 3) := by change Nat.Prime 399660073; exact ProjectEulerSolutions.P58PrimeNodes.p_399660073.prime
    exact decide_eq_true prime
theorem row_count_9996 : row 9996 = 1 := (row_9996.checked).trans rfl
theorem model_count_9996 : modelRow 9996 = 1 := by decide
theorem modelrow_agrees_9996 : modelRow 9996 = row 9996 := model_count_9996.trans row_count_9996.symm
def row_9997 : RowCertificate 9997 where
  first := false
  first_correct := by
    have composite : ¬ Nat.Prime (corner 9997 1) := by change ¬ Nat.Prime 399780031; exact PrattCertificates.composite_of_product (factor := 7) (cofactor := 57111433) rfl (by decide) (by decide)
    exact decide_eq_false composite
  second := true
  second_correct := by
    have prime : Nat.Prime (corner 9997 2) := by change Nat.Prime 399760037; exact ProjectEulerSolutions.P58PrimeNodes.p_399760037.prime
    exact decide_eq_true prime
  third := false
  third_correct := by
    have composite : ¬ Nat.Prime (corner 9997 3) := by change ¬ Nat.Prime 399740043; exact PrattCertificates.composite_of_product (factor := 3) (cofactor := 133246681) rfl (by decide) (by decide)
    exact decide_eq_false composite
theorem row_count_9997 : row 9997 = 1 := (row_9997.checked).trans rfl
theorem model_count_9997 : modelRow 9997 = 1 := by decide
theorem modelrow_agrees_9997 : modelRow 9997 = row 9997 := model_count_9997.trans row_count_9997.symm
def row_9998 : RowCertificate 9998 where
  first := false
  first_correct := by
    have composite : ¬ Nat.Prime (corner 9998 1) := by change ¬ Nat.Prime 399860013; exact PrattCertificates.composite_of_product (factor := 3) (cofactor := 133286671) rfl (by decide) (by decide)
    exact decide_eq_false composite
  second := false
  second_correct := by
    have composite : ¬ Nat.Prime (corner 9998 2) := by change ¬ Nat.Prime 399840017; exact PrattCertificates.composite_of_product (factor := 17) (cofactor := 23520001) rfl (by decide) (by decide)
    exact decide_eq_false composite
  third := false
  third_correct := by
    have composite : ¬ Nat.Prime (corner 9998 3) := by change ¬ Nat.Prime 399820021; exact PrattCertificates.composite_of_product (factor := 19) (cofactor := 21043159) rfl (by decide) (by decide)
    exact decide_eq_false composite
theorem row_count_9998 : row 9998 = 0 := (row_9998.checked).trans rfl
theorem model_count_9998 : modelRow 9998 = 0 := by decide
theorem modelrow_agrees_9998 : modelRow 9998 = row 9998 := model_count_9998.trans row_count_9998.symm
def row_9999 : RowCertificate 9999 where
  first := true
  first_correct := by
    have prime : Nat.Prime (corner 9999 1) := by change Nat.Prime 399940003; exact ProjectEulerSolutions.P58PrimeNodes.p_399940003.prime
    exact decide_eq_true prime
  second := false
  second_correct := by
    have composite : ¬ Nat.Prime (corner 9999 2) := by change ¬ Nat.Prime 399920005; exact PrattCertificates.composite_of_product (factor := 5) (cofactor := 79984001) rfl (by decide) (by decide)
    exact decide_eq_false composite
  third := false
  third_correct := by
    have composite : ¬ Nat.Prime (corner 9999 3) := by change ¬ Nat.Prime 399900007; exact PrattCertificates.composite_of_product (factor := 13) (cofactor := 30761539) rfl (by decide) (by decide)
    exact decide_eq_false composite
theorem row_count_9999 : row 9999 = 1 := (row_9999.checked).trans rfl
theorem model_count_9999 : modelRow 9999 = 1 := by decide
theorem modelrow_agrees_9999 : modelRow 9999 = row 9999 := model_count_9999.trans row_count_9999.symm
def row_10000 : RowCertificate 10000 where
  first := false
  first_correct := by
    have composite : ¬ Nat.Prime (corner 10000 1) := by change ¬ Nat.Prime 400020001; exact PrattCertificates.composite_of_product (factor := 31) (cofactor := 12903871) rfl (by decide) (by decide)
    exact decide_eq_false composite
  second := false
  second_correct := by
    have composite : ¬ Nat.Prime (corner 10000 2) := by change ¬ Nat.Prime 400000001; exact PrattCertificates.composite_of_product (factor := 19801) (cofactor := 20201) rfl (by decide) (by decide)
    exact decide_eq_false composite
  third := false
  third_correct := by
    have composite : ¬ Nat.Prime (corner 10000 3) := by change ¬ Nat.Prime 399980001; exact PrattCertificates.composite_of_product (factor := 3) (cofactor := 133326667) rfl (by decide) (by decide)
    exact decide_eq_false composite
theorem row_count_10000 : row 10000 = 0 := (row_10000.checked).trans rfl
theorem model_count_10000 : modelRow 10000 = 0 := by decide
theorem modelrow_agrees_10000 : modelRow 10000 = row 10000 := model_count_10000.trans row_count_10000.symm
def row_10001 : RowCertificate 10001 where
  first := false
  first_correct := by
    have composite : ¬ Nat.Prime (corner 10001 1) := by change ¬ Nat.Prime 400100007; exact PrattCertificates.composite_of_product (factor := 3) (cofactor := 133366669) rfl (by decide) (by decide)
    exact decide_eq_false composite
  second := false
  second_correct := by
    have composite : ¬ Nat.Prime (corner 10001 2) := by change ¬ Nat.Prime 400080005; exact PrattCertificates.composite_of_product (factor := 5) (cofactor := 80016001) rfl (by decide) (by decide)
    exact decide_eq_false composite
  third := false
  third_correct := by
    have composite : ¬ Nat.Prime (corner 10001 3) := by change ¬ Nat.Prime 400060003; exact PrattCertificates.composite_of_product (factor := 7) (cofactor := 57151429) rfl (by decide) (by decide)
    exact decide_eq_false composite
theorem row_count_10001 : row 10001 = 0 := (row_10001.checked).trans rfl
theorem model_count_10001 : modelRow 10001 = 0 := by decide
theorem modelrow_agrees_10001 : modelRow 10001 = row 10001 := model_count_10001.trans row_count_10001.symm
def row_10002 : RowCertificate 10002 where
  first := false
  first_correct := by
    have composite : ¬ Nat.Prime (corner 10002 1) := by change ¬ Nat.Prime 400180021; exact PrattCertificates.composite_of_product (factor := 19501) (cofactor := 20521) rfl (by decide) (by decide)
    exact decide_eq_false composite
  second := false
  second_correct := by
    have composite : ¬ Nat.Prime (corner 10002 2) := by change ¬ Nat.Prime 400160017; exact PrattCertificates.composite_of_product (factor := 53) (cofactor := 7550189) rfl (by decide) (by decide)
    exact decide_eq_false composite
  third := false
  third_correct := by
    have composite : ¬ Nat.Prime (corner 10002 3) := by change ¬ Nat.Prime 400140013; exact PrattCertificates.composite_of_product (factor := 7) (cofactor := 57162859) rfl (by decide) (by decide)
    exact decide_eq_false composite
theorem row_count_10002 : row 10002 = 0 := (row_10002.checked).trans rfl
theorem model_count_10002 : modelRow 10002 = 0 := by decide
theorem modelrow_agrees_10002 : modelRow 10002 = row 10002 := model_count_10002.trans row_count_10002.symm
def row_10003 : RowCertificate 10003 where
  first := false
  first_correct := by
    have composite : ¬ Nat.Prime (corner 10003 1) := by change ¬ Nat.Prime 400260043; exact PrattCertificates.composite_of_product (factor := 37) (cofactor := 10817839) rfl (by decide) (by decide)
    exact decide_eq_false composite
  second := false
  second_correct := by
    have composite : ¬ Nat.Prime (corner 10003 2) := by change ¬ Nat.Prime 400240037; exact PrattCertificates.composite_of_product (factor := 13697) (cofactor := 29221) rfl (by decide) (by decide)
    exact decide_eq_false composite
  third := false
  third_correct := by
    have composite : ¬ Nat.Prime (corner 10003 3) := by change ¬ Nat.Prime 400220031; exact PrattCertificates.composite_of_product (factor := 3) (cofactor := 133406677) rfl (by decide) (by decide)
    exact decide_eq_false composite
theorem row_count_10003 : row 10003 = 0 := (row_10003.checked).trans rfl
theorem model_count_10003 : modelRow 10003 = 0 := by decide
theorem modelrow_agrees_10003 : modelRow 10003 = row 10003 := model_count_10003.trans row_count_10003.symm
def row_10004 : RowCertificate 10004 where
  first := false
  first_correct := by
    have composite : ¬ Nat.Prime (corner 10004 1) := by change ¬ Nat.Prime 400340073; exact PrattCertificates.composite_of_product (factor := 3) (cofactor := 133446691) rfl (by decide) (by decide)
    exact decide_eq_false composite
  second := false
  second_correct := by
    have composite : ¬ Nat.Prime (corner 10004 2) := by change ¬ Nat.Prime 400320065; exact PrattCertificates.composite_of_product (factor := 5) (cofactor := 80064013) rfl (by decide) (by decide)
    exact decide_eq_false composite
  third := true
  third_correct := by
    have prime : Nat.Prime (corner 10004 3) := by change Nat.Prime 400300057; exact ProjectEulerSolutions.P58PrimeNodes.p_400300057.prime
    exact decide_eq_true prime
theorem row_count_10004 : row 10004 = 1 := (row_10004.checked).trans rfl
theorem model_count_10004 : modelRow 10004 = 1 := by decide
theorem modelrow_agrees_10004 : modelRow 10004 = row 10004 := model_count_10004.trans row_count_10004.symm
def row_10005 : RowCertificate 10005 where
  first := false
  first_correct := by
    have composite : ¬ Nat.Prime (corner 10005 1) := by change ¬ Nat.Prime 400420111; exact PrattCertificates.composite_of_product (factor := 7) (cofactor := 57202873) rfl (by decide) (by decide)
    exact decide_eq_false composite
  second := false
  second_correct := by
    have composite : ¬ Nat.Prime (corner 10005 2) := by change ¬ Nat.Prime 400400101; exact PrattCertificates.composite_of_product (factor := 149) (cofactor := 2687249) rfl (by decide) (by decide)
    exact decide_eq_false composite
  third := false
  third_correct := by
    have composite : ¬ Nat.Prime (corner 10005 3) := by change ¬ Nat.Prime 400380091; exact PrattCertificates.composite_of_product (factor := 5323) (cofactor := 75217) rfl (by decide) (by decide)
    exact decide_eq_false composite
theorem row_count_10005 : row 10005 = 0 := (row_10005.checked).trans rfl
theorem model_count_10005 : modelRow 10005 = 0 := by decide
theorem modelrow_agrees_10005 : modelRow 10005 = row 10005 := model_count_10005.trans row_count_10005.symm
def row_10006 : RowCertificate 10006 where
  first := true
  first_correct := by
    have prime : Nat.Prime (corner 10006 1) := by change Nat.Prime 400500157; exact ProjectEulerSolutions.P58PrimeNodes.p_400500157.prime
    exact decide_eq_true prime
  second := false
  second_correct := by
    have composite : ¬ Nat.Prime (corner 10006 2) := by change ¬ Nat.Prime 400480145; exact PrattCertificates.composite_of_product (factor := 5) (cofactor := 80096029) rfl (by decide) (by decide)
    exact decide_eq_false composite
  third := false
  third_correct := by
    have composite : ¬ Nat.Prime (corner 10006 3) := by change ¬ Nat.Prime 400460133; exact PrattCertificates.composite_of_product (factor := 3) (cofactor := 133486711) rfl (by decide) (by decide)
    exact decide_eq_false composite
theorem row_count_10006 : row 10006 = 1 := (row_10006.checked).trans rfl
theorem model_count_10006 : modelRow 10006 = 1 := by decide
theorem modelrow_agrees_10006 : modelRow 10006 = row 10006 := model_count_10006.trans row_count_10006.symm
def row_10007 : RowCertificate 10007 where
  first := false
  first_correct := by
    have composite : ¬ Nat.Prime (corner 10007 1) := by change ¬ Nat.Prime 400580211; exact PrattCertificates.composite_of_product (factor := 3) (cofactor := 133526737) rfl (by decide) (by decide)
    exact decide_eq_false composite
  second := true
  second_correct := by
    have prime : Nat.Prime (corner 10007 2) := by change Nat.Prime 400560197; exact ProjectEulerSolutions.P58PrimeNodes.p_400560197.prime
    exact decide_eq_true prime
  third := true
  third_correct := by
    have prime : Nat.Prime (corner 10007 3) := by change Nat.Prime 400540183; exact ProjectEulerSolutions.P58PrimeNodes.p_400540183.prime
    exact decide_eq_true prime
theorem row_count_10007 : row 10007 = 2 := (row_10007.checked).trans rfl
theorem model_count_10007 : modelRow 10007 = 2 := by decide
theorem modelrow_agrees_10007 : modelRow 10007 = row 10007 := model_count_10007.trans row_count_10007.symm
def row_10008 : RowCertificate 10008 where
  first := false
  first_correct := by
    have composite : ¬ Nat.Prime (corner 10008 1) := by change ¬ Nat.Prime 400660273; exact PrattCertificates.composite_of_product (factor := 13) (cofactor := 30820021) rfl (by decide) (by decide)
    exact decide_eq_false composite
  second := false
  second_correct := by
    have composite : ¬ Nat.Prime (corner 10008 2) := by change ¬ Nat.Prime 400640257; exact PrattCertificates.composite_of_product (factor := 113) (cofactor := 3545489) rfl (by decide) (by decide)
    exact decide_eq_false composite
  third := false
  third_correct := by
    have composite : ¬ Nat.Prime (corner 10008 3) := by change ¬ Nat.Prime 400620241; exact PrattCertificates.composite_of_product (factor := 7) (cofactor := 57231463) rfl (by decide) (by decide)
    exact decide_eq_false composite
theorem row_count_10008 : row 10008 = 0 := (row_10008.checked).trans rfl
theorem model_count_10008 : modelRow 10008 = 0 := by decide
theorem modelrow_agrees_10008 : modelRow 10008 = row 10008 := model_count_10008.trans row_count_10008.symm
def row_10009 : RowCertificate 10009 where
  first := false
  first_correct := by
    have composite : ¬ Nat.Prime (corner 10009 1) := by change ¬ Nat.Prime 400740343; exact PrattCertificates.composite_of_product (factor := 19) (cofactor := 21091597) rfl (by decide) (by decide)
    exact decide_eq_false composite
  second := false
  second_correct := by
    have composite : ¬ Nat.Prime (corner 10009 2) := by change ¬ Nat.Prime 400720325; exact PrattCertificates.composite_of_product (factor := 5) (cofactor := 80144065) rfl (by decide) (by decide)
    exact decide_eq_false composite
  third := false
  third_correct := by
    have composite : ¬ Nat.Prime (corner 10009 3) := by change ¬ Nat.Prime 400700307; exact PrattCertificates.composite_of_product (factor := 3) (cofactor := 133566769) rfl (by decide) (by decide)
    exact decide_eq_false composite
theorem row_count_10009 : row 10009 = 0 := (row_10009.checked).trans rfl
theorem model_count_10009 : modelRow 10009 = 0 := by decide
theorem modelrow_agrees_10009 : modelRow 10009 = row 10009 := model_count_10009.trans row_count_10009.symm
def row_10010 : RowCertificate 10010 where
  first := false
  first_correct := by
    have composite : ¬ Nat.Prime (corner 10010 1) := by change ¬ Nat.Prime 400820421; exact PrattCertificates.composite_of_product (factor := 3) (cofactor := 133606807) rfl (by decide) (by decide)
    exact decide_eq_false composite
  second := false
  second_correct := by
    have composite : ¬ Nat.Prime (corner 10010 2) := by change ¬ Nat.Prime 400800401; exact PrattCertificates.composite_of_product (factor := 877) (cofactor := 457013) rfl (by decide) (by decide)
    exact decide_eq_false composite
  third := false
  third_correct := by
    have composite : ¬ Nat.Prime (corner 10010 3) := by change ¬ Nat.Prime 400780381; exact PrattCertificates.composite_of_product (factor := 3457) (cofactor := 115933) rfl (by decide) (by decide)
    exact decide_eq_false composite
theorem row_count_10010 : row 10010 = 0 := (row_10010.checked).trans rfl
theorem model_count_10010 : modelRow 10010 = 0 := by decide
theorem modelrow_agrees_10010 : modelRow 10010 = row 10010 := model_count_10010.trans row_count_10010.symm
def row_10011 : RowCertificate 10011 where
  first := false
  first_correct := by
    have composite : ¬ Nat.Prime (corner 10011 1) := by change ¬ Nat.Prime 400900507; exact PrattCertificates.composite_of_product (factor := 7) (cofactor := 57271501) rfl (by decide) (by decide)
    exact decide_eq_false composite
  second := false
  second_correct := by
    have composite : ¬ Nat.Prime (corner 10011 2) := by change ¬ Nat.Prime 400880485; exact PrattCertificates.composite_of_product (factor := 5) (cofactor := 80176097) rfl (by decide) (by decide)
    exact decide_eq_false composite
  third := false
  third_correct := by
    have composite : ¬ Nat.Prime (corner 10011 3) := by change ¬ Nat.Prime 400860463; exact PrattCertificates.composite_of_product (factor := 61) (cofactor := 6571483) rfl (by decide) (by decide)
    exact decide_eq_false composite
theorem row_count_10011 : row 10011 = 0 := (row_10011.checked).trans rfl
theorem model_count_10011 : modelRow 10011 = 0 := by decide
theorem modelrow_agrees_10011 : modelRow 10011 = row 10011 := model_count_10011.trans row_count_10011.symm
def row_10012 : RowCertificate 10012 where
  first := false
  first_correct := by
    have composite : ¬ Nat.Prime (corner 10012 1) := by change ¬ Nat.Prime 400980601; exact PrattCertificates.composite_of_product (factor := 7) (cofactor := 57282943) rfl (by decide) (by decide)
    exact decide_eq_false composite
  second := false
  second_correct := by
    have composite : ¬ Nat.Prime (corner 10012 2) := by change ¬ Nat.Prime 400960577; exact PrattCertificates.composite_of_product (factor := 7741) (cofactor := 51797) rfl (by decide) (by decide)
    exact decide_eq_false composite
  third := false
  third_correct := by
    have composite : ¬ Nat.Prime (corner 10012 3) := by change ¬ Nat.Prime 400940553; exact PrattCertificates.composite_of_product (factor := 3) (cofactor := 133646851) rfl (by decide) (by decide)
    exact decide_eq_false composite
theorem row_count_10012 : row 10012 = 0 := (row_10012.checked).trans rfl
theorem model_count_10012 : modelRow 10012 = 0 := by decide
theorem modelrow_agrees_10012 : modelRow 10012 = row 10012 := model_count_10012.trans row_count_10012.symm
def row_10013 : RowCertificate 10013 where
  first := false
  first_correct := by
    have composite : ¬ Nat.Prime (corner 10013 1) := by change ¬ Nat.Prime 401060703; exact PrattCertificates.composite_of_product (factor := 3) (cofactor := 133686901) rfl (by decide) (by decide)
    exact decide_eq_false composite
  second := true
  second_correct := by
    have prime : Nat.Prime (corner 10013 2) := by change Nat.Prime 401040677; exact ProjectEulerSolutions.P58PrimeNodes.p_401040677.prime
    exact decide_eq_true prime
  third := false
  third_correct := by
    have composite : ¬ Nat.Prime (corner 10013 3) := by change ¬ Nat.Prime 401020651; exact PrattCertificates.composite_of_product (factor := 271) (cofactor := 1479781) rfl (by decide) (by decide)
    exact decide_eq_false composite
theorem row_count_10013 : row 10013 = 1 := (row_10013.checked).trans rfl
theorem model_count_10013 : modelRow 10013 = 1 := by decide
theorem modelrow_agrees_10013 : modelRow 10013 = row 10013 := model_count_10013.trans row_count_10013.symm
def row_10014 : RowCertificate 10014 where
  first := false
  first_correct := by
    have composite : ¬ Nat.Prime (corner 10014 1) := by change ¬ Nat.Prime 401140813; exact PrattCertificates.composite_of_product (factor := 103) (cofactor := 3894571) rfl (by decide) (by decide)
    exact decide_eq_false composite
  second := false
  second_correct := by
    have composite : ¬ Nat.Prime (corner 10014 2) := by change ¬ Nat.Prime 401120785; exact PrattCertificates.composite_of_product (factor := 5) (cofactor := 80224157) rfl (by decide) (by decide)
    exact decide_eq_false composite
  third := false
  third_correct := by
    have composite : ¬ Nat.Prime (corner 10014 3) := by change ¬ Nat.Prime 401100757; exact PrattCertificates.composite_of_product (factor := 37) (cofactor := 10840561) rfl (by decide) (by decide)
    exact decide_eq_false composite
theorem row_count_10014 : row 10014 = 0 := (row_10014.checked).trans rfl
theorem model_count_10014 : modelRow 10014 = 0 := by decide
theorem modelrow_agrees_10014 : modelRow 10014 = row 10014 := model_count_10014.trans row_count_10014.symm
def row_10015 : RowCertificate 10015 where
  first := false
  first_correct := by
    have composite : ¬ Nat.Prime (corner 10015 1) := by change ¬ Nat.Prime 401220931; exact PrattCertificates.composite_of_product (factor := 211) (cofactor := 1901521) rfl (by decide) (by decide)
    exact decide_eq_false composite
  second := false
  second_correct := by
    have composite : ¬ Nat.Prime (corner 10015 2) := by change ¬ Nat.Prime 401200901; exact PrattCertificates.composite_of_product (factor := 17) (cofactor := 23600053) rfl (by decide) (by decide)
    exact decide_eq_false composite
  third := false
  third_correct := by
    have composite : ¬ Nat.Prime (corner 10015 3) := by change ¬ Nat.Prime 401180871; exact PrattCertificates.composite_of_product (factor := 3) (cofactor := 133726957) rfl (by decide) (by decide)
    exact decide_eq_false composite
theorem row_count_10015 : row 10015 = 0 := (row_10015.checked).trans rfl
theorem model_count_10015 : modelRow 10015 = 0 := by decide
theorem modelrow_agrees_10015 : modelRow 10015 = row 10015 := model_count_10015.trans row_count_10015.symm
def row_10016 : RowCertificate 10016 where
  first := false
  first_correct := by
    have composite : ¬ Nat.Prime (corner 10016 1) := by change ¬ Nat.Prime 401301057; exact PrattCertificates.composite_of_product (factor := 3) (cofactor := 133767019) rfl (by decide) (by decide)
    exact decide_eq_false composite
  second := false
  second_correct := by
    have composite : ¬ Nat.Prime (corner 10016 2) := by change ¬ Nat.Prime 401281025; exact PrattCertificates.composite_of_product (factor := 5) (cofactor := 80256205) rfl (by decide) (by decide)
    exact decide_eq_false composite
  third := false
  third_correct := by
    have composite : ¬ Nat.Prime (corner 10016 3) := by change ¬ Nat.Prime 401260993; exact PrattCertificates.composite_of_product (factor := 7) (cofactor := 57322999) rfl (by decide) (by decide)
    exact decide_eq_false composite
theorem row_count_10016 : row 10016 = 0 := (row_10016.checked).trans rfl
theorem model_count_10016 : modelRow 10016 = 0 := by decide
theorem modelrow_agrees_10016 : modelRow 10016 = row 10016 := model_count_10016.trans row_count_10016.symm
def row_10017 : RowCertificate 10017 where
  first := false
  first_correct := by
    have composite : ¬ Nat.Prime (corner 10017 1) := by change ¬ Nat.Prime 401381191; exact PrattCertificates.composite_of_product (factor := 2803) (cofactor := 143197) rfl (by decide) (by decide)
    exact decide_eq_false composite
  second := false
  second_correct := by
    have composite : ¬ Nat.Prime (corner 10017 2) := by change ¬ Nat.Prime 401361157; exact PrattCertificates.composite_of_product (factor := 3457) (cofactor := 116101) rfl (by decide) (by decide)
    exact decide_eq_false composite
  third := false
  third_correct := by
    have composite : ¬ Nat.Prime (corner 10017 3) := by change ¬ Nat.Prime 401341123; exact PrattCertificates.composite_of_product (factor := 19) (cofactor := 21123217) rfl (by decide) (by decide)
    exact decide_eq_false composite
theorem row_count_10017 : row 10017 = 0 := (row_10017.checked).trans rfl
theorem model_count_10017 : modelRow 10017 = 0 := by decide
theorem modelrow_agrees_10017 : modelRow 10017 = row 10017 := model_count_10017.trans row_count_10017.symm
def row_10018 : RowCertificate 10018 where
  first := false
  first_correct := by
    have composite : ¬ Nat.Prime (corner 10018 1) := by change ¬ Nat.Prime 401461333; exact PrattCertificates.composite_of_product (factor := 7) (cofactor := 57351619) rfl (by decide) (by decide)
    exact decide_eq_false composite
  second := true
  second_correct := by
    have prime : Nat.Prime (corner 10018 2) := by change Nat.Prime 401441297; exact ProjectEulerSolutions.P58PrimeNodes.p_401441297.prime
    exact decide_eq_true prime
  third := false
  third_correct := by
    have composite : ¬ Nat.Prime (corner 10018 3) := by change ¬ Nat.Prime 401421261; exact PrattCertificates.composite_of_product (factor := 3) (cofactor := 133807087) rfl (by decide) (by decide)
    exact decide_eq_false composite
theorem row_count_10018 : row 10018 = 1 := (row_10018.checked).trans rfl
theorem model_count_10018 : modelRow 10018 = 1 := by decide
theorem modelrow_agrees_10018 : modelRow 10018 = row 10018 := model_count_10018.trans row_count_10018.symm
def row_10019 : RowCertificate 10019 where
  first := false
  first_correct := by
    have composite : ¬ Nat.Prime (corner 10019 1) := by change ¬ Nat.Prime 401541483; exact PrattCertificates.composite_of_product (factor := 3) (cofactor := 133847161) rfl (by decide) (by decide)
    exact decide_eq_false composite
  second := false
  second_correct := by
    have composite : ¬ Nat.Prime (corner 10019 2) := by change ¬ Nat.Prime 401521445; exact PrattCertificates.composite_of_product (factor := 5) (cofactor := 80304289) rfl (by decide) (by decide)
    exact decide_eq_false composite
  third := false
  third_correct := by
    have composite : ¬ Nat.Prime (corner 10019 3) := by change ¬ Nat.Prime 401501407; exact PrattCertificates.composite_of_product (factor := 19) (cofactor := 21131653) rfl (by decide) (by decide)
    exact decide_eq_false composite
theorem row_count_10019 : row 10019 = 0 := (row_10019.checked).trans rfl
theorem model_count_10019 : modelRow 10019 = 0 := by decide
theorem modelrow_agrees_10019 : modelRow 10019 = row 10019 := model_count_10019.trans row_count_10019.symm
def row_10020 : RowCertificate 10020 where
  first := true
  first_correct := by
    have prime : Nat.Prime (corner 10020 1) := by change Nat.Prime 401621641; exact ProjectEulerSolutions.P58PrimeNodes.p_401621641.prime
    exact decide_eq_true prime
  second := false
  second_correct := by
    have composite : ¬ Nat.Prime (corner 10020 2) := by change ¬ Nat.Prime 401601601; exact PrattCertificates.composite_of_product (factor := 41) (cofactor := 9795161) rfl (by decide) (by decide)
    exact decide_eq_false composite
  third := true
  third_correct := by
    have prime : Nat.Prime (corner 10020 3) := by change Nat.Prime 401581561; exact ProjectEulerSolutions.P58PrimeNodes.p_401581561.prime
    exact decide_eq_true prime
theorem row_count_10020 : row 10020 = 2 := (row_10020.checked).trans rfl
theorem model_count_10020 : modelRow 10020 = 2 := by decide
theorem modelrow_agrees_10020 : modelRow 10020 = row 10020 := model_count_10020.trans row_count_10020.symm
def row_10021 : RowCertificate 10021 where
  first := false
  first_correct := by
    have composite : ¬ Nat.Prime (corner 10021 1) := by change ¬ Nat.Prime 401701807; exact PrattCertificates.composite_of_product (factor := 13) (cofactor := 30900139) rfl (by decide) (by decide)
    exact decide_eq_false composite
  second := false
  second_correct := by
    have composite : ¬ Nat.Prime (corner 10021 2) := by change ¬ Nat.Prime 401681765; exact PrattCertificates.composite_of_product (factor := 5) (cofactor := 80336353) rfl (by decide) (by decide)
    exact decide_eq_false composite
  third := false
  third_correct := by
    have composite : ¬ Nat.Prime (corner 10021 3) := by change ¬ Nat.Prime 401661723; exact PrattCertificates.composite_of_product (factor := 3) (cofactor := 133887241) rfl (by decide) (by decide)
    exact decide_eq_false composite
theorem row_count_10021 : row 10021 = 0 := (row_10021.checked).trans rfl
theorem model_count_10021 : modelRow 10021 = 0 := by decide
theorem modelrow_agrees_10021 : modelRow 10021 = row 10021 := model_count_10021.trans row_count_10021.symm
def row_10022 : RowCertificate 10022 where
  first := false
  first_correct := by
    have composite : ¬ Nat.Prime (corner 10022 1) := by change ¬ Nat.Prime 401781981; exact PrattCertificates.composite_of_product (factor := 3) (cofactor := 133927327) rfl (by decide) (by decide)
    exact decide_eq_false composite
  second := false
  second_correct := by
    have composite : ¬ Nat.Prime (corner 10022 2) := by change ¬ Nat.Prime 401761937; exact PrattCertificates.composite_of_product (factor := 14293) (cofactor := 28109) rfl (by decide) (by decide)
    exact decide_eq_false composite
  third := false
  third_correct := by
    have composite : ¬ Nat.Prime (corner 10022 3) := by change ¬ Nat.Prime 401741893; exact PrattCertificates.composite_of_product (factor := 7) (cofactor := 57391699) rfl (by decide) (by decide)
    exact decide_eq_false composite
theorem row_count_10022 : row 10022 = 0 := (row_10022.checked).trans rfl
theorem model_count_10022 : modelRow 10022 = 0 := by decide
theorem modelrow_agrees_10022 : modelRow 10022 = row 10022 := model_count_10022.trans row_count_10022.symm
def row_10023 : RowCertificate 10023 where
  first := false
  first_correct := by
    have composite : ¬ Nat.Prime (corner 10023 1) := by change ¬ Nat.Prime 401862163; exact PrattCertificates.composite_of_product (factor := 127) (cofactor := 3164269) rfl (by decide) (by decide)
    exact decide_eq_false composite
  second := false
  second_correct := by
    have composite : ¬ Nat.Prime (corner 10023 2) := by change ¬ Nat.Prime 401842117; exact PrattCertificates.composite_of_product (factor := 1021) (cofactor := 393577) rfl (by decide) (by decide)
    exact decide_eq_false composite
  third := false
  third_correct := by
    have composite : ¬ Nat.Prime (corner 10023 3) := by change ¬ Nat.Prime 401822071; exact PrattCertificates.composite_of_product (factor := 7) (cofactor := 57403153) rfl (by decide) (by decide)
    exact decide_eq_false composite
theorem row_count_10023 : row 10023 = 0 := (row_10023.checked).trans rfl
theorem model_count_10023 : modelRow 10023 = 0 := by decide
theorem modelrow_agrees_10023 : modelRow 10023 = row 10023 := model_count_10023.trans row_count_10023.symm
def row_10024 : RowCertificate 10024 where
  first := false
  first_correct := by
    have composite : ¬ Nat.Prime (corner 10024 1) := by change ¬ Nat.Prime 401942353; exact PrattCertificates.composite_of_product (factor := 283) (cofactor := 1420291) rfl (by decide) (by decide)
    exact decide_eq_false composite
  second := false
  second_correct := by
    have composite : ¬ Nat.Prime (corner 10024 2) := by change ¬ Nat.Prime 401922305; exact PrattCertificates.composite_of_product (factor := 5) (cofactor := 80384461) rfl (by decide) (by decide)
    exact decide_eq_false composite
  third := false
  third_correct := by
    have composite : ¬ Nat.Prime (corner 10024 3) := by change ¬ Nat.Prime 401902257; exact PrattCertificates.composite_of_product (factor := 3) (cofactor := 133967419) rfl (by decide) (by decide)
    exact decide_eq_false composite
theorem row_count_10024 : row 10024 = 0 := (row_10024.checked).trans rfl
theorem model_count_10024 : modelRow 10024 = 0 := by decide
theorem modelrow_agrees_10024 : modelRow 10024 = row 10024 := model_count_10024.trans row_count_10024.symm
def row_10025 : RowCertificate 10025 where
  first := false
  first_correct := by
    have composite : ¬ Nat.Prime (corner 10025 1) := by change ¬ Nat.Prime 402022551; exact PrattCertificates.composite_of_product (factor := 3) (cofactor := 134007517) rfl (by decide) (by decide)
    exact decide_eq_false composite
  second := false
  second_correct := by
    have composite : ¬ Nat.Prime (corner 10025 2) := by change ¬ Nat.Prime 402002501; exact PrattCertificates.composite_of_product (factor := 16421) (cofactor := 24481) rfl (by decide) (by decide)
    exact decide_eq_false composite
  third := false
  third_correct := by
    have composite : ¬ Nat.Prime (corner 10025 3) := by change ¬ Nat.Prime 401982451; exact PrattCertificates.composite_of_product (factor := 13) (cofactor := 30921727) rfl (by decide) (by decide)
    exact decide_eq_false composite
theorem row_count_10025 : row 10025 = 0 := (row_10025.checked).trans rfl
theorem model_count_10025 : modelRow 10025 = 0 := by decide
theorem modelrow_agrees_10025 : modelRow 10025 = row 10025 := model_count_10025.trans row_count_10025.symm
def row_10026 : RowCertificate 10026 where
  first := false
  first_correct := by
    have composite : ¬ Nat.Prime (corner 10026 1) := by change ¬ Nat.Prime 402102757; exact PrattCertificates.composite_of_product (factor := 7) (cofactor := 57443251) rfl (by decide) (by decide)
    exact decide_eq_false composite
  second := false
  second_correct := by
    have composite : ¬ Nat.Prime (corner 10026 2) := by change ¬ Nat.Prime 402082705; exact PrattCertificates.composite_of_product (factor := 5) (cofactor := 80416541) rfl (by decide) (by decide)
    exact decide_eq_false composite
  third := false
  third_correct := by
    have composite : ¬ Nat.Prime (corner 10026 3) := by change ¬ Nat.Prime 402062653; exact PrattCertificates.composite_of_product (factor := 31) (cofactor := 12969763) rfl (by decide) (by decide)
    exact decide_eq_false composite
theorem row_count_10026 : row 10026 = 0 := (row_10026.checked).trans rfl
theorem model_count_10026 : modelRow 10026 = 0 := by decide
theorem modelrow_agrees_10026 : modelRow 10026 = row 10026 := model_count_10026.trans row_count_10026.symm
def row_10027 : RowCertificate 10027 where
  first := false
  first_correct := by
    have composite : ¬ Nat.Prime (corner 10027 1) := by change ¬ Nat.Prime 402182971; exact PrattCertificates.composite_of_product (factor := 787) (cofactor := 511033) rfl (by decide) (by decide)
    exact decide_eq_false composite
  second := false
  second_correct := by
    have composite : ¬ Nat.Prime (corner 10027 2) := by change ¬ Nat.Prime 402162917; exact PrattCertificates.composite_of_product (factor := 13) (cofactor := 30935609) rfl (by decide) (by decide)
    exact decide_eq_false composite
  third := false
  third_correct := by
    have composite : ¬ Nat.Prime (corner 10027 3) := by change ¬ Nat.Prime 402142863; exact PrattCertificates.composite_of_product (factor := 3) (cofactor := 134047621) rfl (by decide) (by decide)
    exact decide_eq_false composite
theorem row_count_10027 : row 10027 = 0 := (row_10027.checked).trans rfl
theorem model_count_10027 : modelRow 10027 = 0 := by decide
theorem modelrow_agrees_10027 : modelRow 10027 = row 10027 := model_count_10027.trans row_count_10027.symm
def row_10028 : RowCertificate 10028 where
  first := false
  first_correct := by
    have composite : ¬ Nat.Prime (corner 10028 1) := by change ¬ Nat.Prime 402263193; exact PrattCertificates.composite_of_product (factor := 3) (cofactor := 134087731) rfl (by decide) (by decide)
    exact decide_eq_false composite
  second := false
  second_correct := by
    have composite : ¬ Nat.Prime (corner 10028 2) := by change ¬ Nat.Prime 402243137; exact PrattCertificates.composite_of_product (factor := 17) (cofactor := 23661361) rfl (by decide) (by decide)
    exact decide_eq_false composite
  third := false
  third_correct := by
    have composite : ¬ Nat.Prime (corner 10028 3) := by change ¬ Nat.Prime 402223081; exact PrattCertificates.composite_of_product (factor := 13) (cofactor := 30940237) rfl (by decide) (by decide)
    exact decide_eq_false composite
theorem row_count_10028 : row 10028 = 0 := (row_10028.checked).trans rfl
theorem model_count_10028 : modelRow 10028 = 0 := by decide
theorem modelrow_agrees_10028 : modelRow 10028 = row 10028 := model_count_10028.trans row_count_10028.symm
def row_10029 : RowCertificate 10029 where
  first := false
  first_correct := by
    have composite : ¬ Nat.Prime (corner 10029 1) := by change ¬ Nat.Prime 402343423; exact PrattCertificates.composite_of_product (factor := 139) (cofactor := 2894557) rfl (by decide) (by decide)
    exact decide_eq_false composite
  second := false
  second_correct := by
    have composite : ¬ Nat.Prime (corner 10029 2) := by change ¬ Nat.Prime 402323365; exact PrattCertificates.composite_of_product (factor := 5) (cofactor := 80464673) rfl (by decide) (by decide)
    exact decide_eq_false composite
  third := false
  third_correct := by
    have composite : ¬ Nat.Prime (corner 10029 3) := by change ¬ Nat.Prime 402303307; exact PrattCertificates.composite_of_product (factor := 7) (cofactor := 57471901) rfl (by decide) (by decide)
    exact decide_eq_false composite
theorem row_count_10029 : row 10029 = 0 := (row_10029.checked).trans rfl
theorem model_count_10029 : modelRow 10029 = 0 := by decide
theorem modelrow_agrees_10029 : modelRow 10029 = row 10029 := model_count_10029.trans row_count_10029.symm
def row_10030 : RowCertificate 10030 where
  first := false
  first_correct := by
    have composite : ¬ Nat.Prime (corner 10030 1) := by change ¬ Nat.Prime 402423661; exact PrattCertificates.composite_of_product (factor := 277) (cofactor := 1452793) rfl (by decide) (by decide)
    exact decide_eq_false composite
  second := false
  second_correct := by
    have composite : ¬ Nat.Prime (corner 10030 2) := by change ¬ Nat.Prime 402403601; exact PrattCertificates.composite_of_product (factor := 37) (cofactor := 10875773) rfl (by decide) (by decide)
    exact decide_eq_false composite
  third := false
  third_correct := by
    have composite : ¬ Nat.Prime (corner 10030 3) := by change ¬ Nat.Prime 402383541; exact PrattCertificates.composite_of_product (factor := 3) (cofactor := 134127847) rfl (by decide) (by decide)
    exact decide_eq_false composite
theorem row_count_10030 : row 10030 = 0 := (row_10030.checked).trans rfl
theorem model_count_10030 : modelRow 10030 = 0 := by decide
theorem modelrow_agrees_10030 : modelRow 10030 = row 10030 := model_count_10030.trans row_count_10030.symm
def row_10031 : RowCertificate 10031 where
  first := false
  first_correct := by
    have composite : ¬ Nat.Prime (corner 10031 1) := by change ¬ Nat.Prime 402503907; exact PrattCertificates.composite_of_product (factor := 3) (cofactor := 134167969) rfl (by decide) (by decide)
    exact decide_eq_false composite
  second := false
  second_correct := by
    have composite : ¬ Nat.Prime (corner 10031 2) := by change ¬ Nat.Prime 402483845; exact PrattCertificates.composite_of_product (factor := 5) (cofactor := 80496769) rfl (by decide) (by decide)
    exact decide_eq_false composite
  third := false
  third_correct := by
    have composite : ¬ Nat.Prime (corner 10031 3) := by change ¬ Nat.Prime 402463783; exact PrattCertificates.composite_of_product (factor := 6073) (cofactor := 66271) rfl (by decide) (by decide)
    exact decide_eq_false composite
theorem row_count_10031 : row 10031 = 0 := (row_10031.checked).trans rfl
theorem model_count_10031 : modelRow 10031 = 0 := by decide
theorem modelrow_agrees_10031 : modelRow 10031 = row 10031 := model_count_10031.trans row_count_10031.symm
def row_10032 : RowCertificate 10032 where
  first := false
  first_correct := by
    have composite : ¬ Nat.Prime (corner 10032 1) := by change ¬ Nat.Prime 402584161; exact PrattCertificates.composite_of_product (factor := 7) (cofactor := 57512023) rfl (by decide) (by decide)
    exact decide_eq_false composite
  second := false
  second_correct := by
    have composite : ¬ Nat.Prime (corner 10032 2) := by change ¬ Nat.Prime 402564097; exact PrattCertificates.composite_of_product (factor := 13) (cofactor := 30966469) rfl (by decide) (by decide)
    exact decide_eq_false composite
  third := false
  third_correct := by
    have composite : ¬ Nat.Prime (corner 10032 3) := by change ¬ Nat.Prime 402544033; exact PrattCertificates.composite_of_product (factor := 2089) (cofactor := 192697) rfl (by decide) (by decide)
    exact decide_eq_false composite
theorem row_count_10032 : row 10032 = 0 := (row_10032.checked).trans rfl
theorem model_count_10032 : modelRow 10032 = 0 := by decide
theorem modelrow_agrees_10032 : modelRow 10032 = row 10032 := model_count_10032.trans row_count_10032.symm
def row_10033 : RowCertificate 10033 where
  first := false
  first_correct := by
    have composite : ¬ Nat.Prime (corner 10033 1) := by change ¬ Nat.Prime 402664423; exact PrattCertificates.composite_of_product (factor := 7) (cofactor := 57523489) rfl (by decide) (by decide)
    exact decide_eq_false composite
  second := true
  second_correct := by
    have prime : Nat.Prime (corner 10033 2) := by change Nat.Prime 402644357; exact ProjectEulerSolutions.P58PrimeNodes.p_402644357.prime
    exact decide_eq_true prime
  third := false
  third_correct := by
    have composite : ¬ Nat.Prime (corner 10033 3) := by change ¬ Nat.Prime 402624291; exact PrattCertificates.composite_of_product (factor := 3) (cofactor := 134208097) rfl (by decide) (by decide)
    exact decide_eq_false composite
theorem row_count_10033 : row 10033 = 1 := (row_10033.checked).trans rfl
theorem model_count_10033 : modelRow 10033 = 1 := by decide
theorem modelrow_agrees_10033 : modelRow 10033 = row 10033 := model_count_10033.trans row_count_10033.symm
def row_10034 : RowCertificate 10034 where
  first := false
  first_correct := by
    have composite : ¬ Nat.Prime (corner 10034 1) := by change ¬ Nat.Prime 402744693; exact PrattCertificates.composite_of_product (factor := 3) (cofactor := 134248231) rfl (by decide) (by decide)
    exact decide_eq_false composite
  second := false
  second_correct := by
    have composite : ¬ Nat.Prime (corner 10034 2) := by change ¬ Nat.Prime 402724625; exact PrattCertificates.composite_of_product (factor := 5) (cofactor := 80544925) rfl (by decide) (by decide)
    exact decide_eq_false composite
  third := true
  third_correct := by
    have prime : Nat.Prime (corner 10034 3) := by change Nat.Prime 402704557; exact ProjectEulerSolutions.P58PrimeNodes.p_402704557.prime
    exact decide_eq_true prime
theorem row_count_10034 : row 10034 = 1 := (row_10034.checked).trans rfl
theorem model_count_10034 : modelRow 10034 = 1 := by decide
theorem modelrow_agrees_10034 : modelRow 10034 = row 10034 := model_count_10034.trans row_count_10034.symm
def row_10035 : RowCertificate 10035 where
  first := false
  first_correct := by
    have composite : ¬ Nat.Prime (corner 10035 1) := by change ¬ Nat.Prime 402824971; exact PrattCertificates.composite_of_product (factor := 67) (cofactor := 6012313) rfl (by decide) (by decide)
    exact decide_eq_false composite
  second := false
  second_correct := by
    have composite : ¬ Nat.Prime (corner 10035 2) := by change ¬ Nat.Prime 402804901; exact PrattCertificates.composite_of_product (factor := 401) (cofactor := 1004501) rfl (by decide) (by decide)
    exact decide_eq_false composite
  third := true
  third_correct := by
    have prime : Nat.Prime (corner 10035 3) := by change Nat.Prime 402784831; exact ProjectEulerSolutions.P58PrimeNodes.p_402784831.prime
    exact decide_eq_true prime
theorem row_count_10035 : row 10035 = 1 := (row_10035.checked).trans rfl
theorem model_count_10035 : modelRow 10035 = 1 := by decide
theorem modelrow_agrees_10035 : modelRow 10035 = row 10035 := model_count_10035.trans row_count_10035.symm
def row_10036 : RowCertificate 10036 where
  first := false
  first_correct := by
    have composite : ¬ Nat.Prime (corner 10036 1) := by change ¬ Nat.Prime 402905257; exact PrattCertificates.composite_of_product (factor := 3001) (cofactor := 134257) rfl (by decide) (by decide)
    exact decide_eq_false composite
  second := false
  second_correct := by
    have composite : ¬ Nat.Prime (corner 10036 2) := by change ¬ Nat.Prime 402885185; exact PrattCertificates.composite_of_product (factor := 5) (cofactor := 80577037) rfl (by decide) (by decide)
    exact decide_eq_false composite
  third := false
  third_correct := by
    have composite : ¬ Nat.Prime (corner 10036 3) := by change ¬ Nat.Prime 402865113; exact PrattCertificates.composite_of_product (factor := 3) (cofactor := 134288371) rfl (by decide) (by decide)
    exact decide_eq_false composite
theorem row_count_10036 : row 10036 = 0 := (row_10036.checked).trans rfl
theorem model_count_10036 : modelRow 10036 = 0 := by decide
theorem modelrow_agrees_10036 : modelRow 10036 = row 10036 := model_count_10036.trans row_count_10036.symm
def row_10037 : RowCertificate 10037 where
  first := false
  first_correct := by
    have composite : ¬ Nat.Prime (corner 10037 1) := by change ¬ Nat.Prime 402985551; exact PrattCertificates.composite_of_product (factor := 3) (cofactor := 134328517) rfl (by decide) (by decide)
    exact decide_eq_false composite
  second := false
  second_correct := by
    have composite : ¬ Nat.Prime (corner 10037 2) := by change ¬ Nat.Prime 402965477; exact PrattCertificates.composite_of_product (factor := 11933) (cofactor := 33769) rfl (by decide) (by decide)
    exact decide_eq_false composite
  third := false
  third_correct := by
    have composite : ¬ Nat.Prime (corner 10037 3) := by change ¬ Nat.Prime 402945403; exact PrattCertificates.composite_of_product (factor := 7) (cofactor := 57563629) rfl (by decide) (by decide)
    exact decide_eq_false composite
theorem row_count_10037 : row 10037 = 0 := (row_10037.checked).trans rfl
theorem model_count_10037 : modelRow 10037 = 0 := by decide
theorem modelrow_agrees_10037 : modelRow 10037 = row 10037 := model_count_10037.trans row_count_10037.symm
def row_10038 : RowCertificate 10038 where
  first := true
  first_correct := by
    have prime : Nat.Prime (corner 10038 1) := by change Nat.Prime 403065853; exact ProjectEulerSolutions.P58PrimeNodes.p_403065853.prime
    exact decide_eq_true prime
  second := true
  second_correct := by
    have prime : Nat.Prime (corner 10038 2) := by change Nat.Prime 403045777; exact ProjectEulerSolutions.P58PrimeNodes.p_403045777.prime
    exact decide_eq_true prime
  third := false
  third_correct := by
    have composite : ¬ Nat.Prime (corner 10038 3) := by change ¬ Nat.Prime 403025701; exact PrattCertificates.composite_of_product (factor := 13) (cofactor := 31001977) rfl (by decide) (by decide)
    exact decide_eq_false composite
theorem row_count_10038 : row 10038 = 2 := (row_10038.checked).trans rfl
theorem model_count_10038 : modelRow 10038 = 2 := by decide
theorem modelrow_agrees_10038 : modelRow 10038 = row 10038 := model_count_10038.trans row_count_10038.symm
def row_10039 : RowCertificate 10039 where
  first := false
  first_correct := by
    have composite : ¬ Nat.Prime (corner 10039 1) := by change ¬ Nat.Prime 403146163; exact PrattCertificates.composite_of_product (factor := 7) (cofactor := 57592309) rfl (by decide) (by decide)
    exact decide_eq_false composite
  second := false
  second_correct := by
    have composite : ¬ Nat.Prime (corner 10039 2) := by change ¬ Nat.Prime 403126085; exact PrattCertificates.composite_of_product (factor := 5) (cofactor := 80625217) rfl (by decide) (by decide)
    exact decide_eq_false composite
  third := false
  third_correct := by
    have composite : ¬ Nat.Prime (corner 10039 3) := by change ¬ Nat.Prime 403106007; exact PrattCertificates.composite_of_product (factor := 3) (cofactor := 134368669) rfl (by decide) (by decide)
    exact decide_eq_false composite
theorem row_count_10039 : row 10039 = 0 := (row_10039.checked).trans rfl
theorem model_count_10039 : modelRow 10039 = 0 := by decide
theorem modelrow_agrees_10039 : modelRow 10039 = row 10039 := model_count_10039.trans row_count_10039.symm
def row_10040 : RowCertificate 10040 where
  first := false
  first_correct := by
    have composite : ¬ Nat.Prime (corner 10040 1) := by change ¬ Nat.Prime 403226481; exact PrattCertificates.composite_of_product (factor := 3) (cofactor := 134408827) rfl (by decide) (by decide)
    exact decide_eq_false composite
  second := false
  second_correct := by
    have composite : ¬ Nat.Prime (corner 10040 2) := by change ¬ Nat.Prime 403206401; exact PrattCertificates.composite_of_product (factor := 13) (cofactor := 31015877) rfl (by decide) (by decide)
    exact decide_eq_false composite
  third := false
  third_correct := by
    have composite : ¬ Nat.Prime (corner 10040 3) := by change ¬ Nat.Prime 403186321; exact PrattCertificates.composite_of_product (factor := 709) (cofactor := 568669) rfl (by decide) (by decide)
    exact decide_eq_false composite
theorem row_count_10040 : row 10040 = 0 := (row_10040.checked).trans rfl
theorem model_count_10040 : modelRow 10040 = 0 := by decide
theorem modelrow_agrees_10040 : modelRow 10040 = row 10040 := model_count_10040.trans row_count_10040.symm
def row_10041 : RowCertificate 10041 where
  first := false
  first_correct := by
    have composite : ¬ Nat.Prime (corner 10041 1) := by change ¬ Nat.Prime 403306807; exact PrattCertificates.composite_of_product (factor := 31) (cofactor := 13009897) rfl (by decide) (by decide)
    exact decide_eq_false composite
  second := false
  second_correct := by
    have composite : ¬ Nat.Prime (corner 10041 2) := by change ¬ Nat.Prime 403286725; exact PrattCertificates.composite_of_product (factor := 5) (cofactor := 80657345) rfl (by decide) (by decide)
    exact decide_eq_false composite
  third := false
  third_correct := by
    have composite : ¬ Nat.Prime (corner 10041 3) := by change ¬ Nat.Prime 403266643; exact PrattCertificates.composite_of_product (factor := 13) (cofactor := 31020511) rfl (by decide) (by decide)
    exact decide_eq_false composite
theorem row_count_10041 : row 10041 = 0 := (row_10041.checked).trans rfl
theorem model_count_10041 : modelRow 10041 = 0 := by decide
theorem modelrow_agrees_10041 : modelRow 10041 = row 10041 := model_count_10041.trans row_count_10041.symm
def row_10042 : RowCertificate 10042 where
  first := false
  first_correct := by
    have composite : ¬ Nat.Prime (corner 10042 1) := by change ¬ Nat.Prime 403387141; exact PrattCertificates.composite_of_product (factor := 6577) (cofactor := 61333) rfl (by decide) (by decide)
    exact decide_eq_false composite
  second := false
  second_correct := by
    have composite : ¬ Nat.Prime (corner 10042 2) := by change ¬ Nat.Prime 403367057; exact PrattCertificates.composite_of_product (factor := 521) (cofactor := 774217) rfl (by decide) (by decide)
    exact decide_eq_false composite
  third := false
  third_correct := by
    have composite : ¬ Nat.Prime (corner 10042 3) := by change ¬ Nat.Prime 403346973; exact PrattCertificates.composite_of_product (factor := 3) (cofactor := 134448991) rfl (by decide) (by decide)
    exact decide_eq_false composite
theorem row_count_10042 : row 10042 = 0 := (row_10042.checked).trans rfl
theorem model_count_10042 : modelRow 10042 = 0 := by decide
theorem modelrow_agrees_10042 : modelRow 10042 = row 10042 := model_count_10042.trans row_count_10042.symm
def row_10043 : RowCertificate 10043 where
  first := false
  first_correct := by
    have composite : ¬ Nat.Prime (corner 10043 1) := by change ¬ Nat.Prime 403467483; exact PrattCertificates.composite_of_product (factor := 3) (cofactor := 134489161) rfl (by decide) (by decide)
    exact decide_eq_false composite
  second := true
  second_correct := by
    have prime : Nat.Prime (corner 10043 2) := by change Nat.Prime 403447397; exact ProjectEulerSolutions.P58PrimeNodes.p_403447397.prime
    exact decide_eq_true prime
  third := false
  third_correct := by
    have composite : ¬ Nat.Prime (corner 10043 3) := by change ¬ Nat.Prime 403427311; exact PrattCertificates.composite_of_product (factor := 7) (cofactor := 57632473) rfl (by decide) (by decide)
    exact decide_eq_false composite
theorem row_count_10043 : row 10043 = 1 := (row_10043.checked).trans rfl
theorem model_count_10043 : modelRow 10043 = 1 := by decide
theorem modelrow_agrees_10043 : modelRow 10043 = row 10043 := model_count_10043.trans row_count_10043.symm
def row_10044 : RowCertificate 10044 where
  first := false
  first_correct := by
    have composite : ¬ Nat.Prime (corner 10044 1) := by change ¬ Nat.Prime 403547833; exact PrattCertificates.composite_of_product (factor := 13) (cofactor := 31042141) rfl (by decide) (by decide)
    exact decide_eq_false composite
  second := false
  second_correct := by
    have composite : ¬ Nat.Prime (corner 10044 2) := by change ¬ Nat.Prime 403527745; exact PrattCertificates.composite_of_product (factor := 5) (cofactor := 80705549) rfl (by decide) (by decide)
    exact decide_eq_false composite
  third := false
  third_correct := by
    have composite : ¬ Nat.Prime (corner 10044 3) := by change ¬ Nat.Prime 403507657; exact PrattCertificates.composite_of_product (factor := 7) (cofactor := 57643951) rfl (by decide) (by decide)
    exact decide_eq_false composite
theorem row_count_10044 : row 10044 = 0 := (row_10044.checked).trans rfl
theorem model_count_10044 : modelRow 10044 = 0 := by decide
theorem modelrow_agrees_10044 : modelRow 10044 = row 10044 := model_count_10044.trans row_count_10044.symm
def row_10045 : RowCertificate 10045 where
  first := false
  first_correct := by
    have composite : ¬ Nat.Prime (corner 10045 1) := by change ¬ Nat.Prime 403628191; exact PrattCertificates.composite_of_product (factor := 19) (cofactor := 21243589) rfl (by decide) (by decide)
    exact decide_eq_false composite
  second := false
  second_correct := by
    have composite : ¬ Nat.Prime (corner 10045 2) := by change ¬ Nat.Prime 403608101; exact PrattCertificates.composite_of_product (factor := 13) (cofactor := 31046777) rfl (by decide) (by decide)
    exact decide_eq_false composite
  third := false
  third_correct := by
    have composite : ¬ Nat.Prime (corner 10045 3) := by change ¬ Nat.Prime 403588011; exact PrattCertificates.composite_of_product (factor := 3) (cofactor := 134529337) rfl (by decide) (by decide)
    exact decide_eq_false composite
theorem row_count_10045 : row 10045 = 0 := (row_10045.checked).trans rfl
theorem model_count_10045 : modelRow 10045 = 0 := by decide
theorem modelrow_agrees_10045 : modelRow 10045 = row 10045 := model_count_10045.trans row_count_10045.symm
def row_10046 : RowCertificate 10046 where
  first := false
  first_correct := by
    have composite : ¬ Nat.Prime (corner 10046 1) := by change ¬ Nat.Prime 403708557; exact PrattCertificates.composite_of_product (factor := 3) (cofactor := 134569519) rfl (by decide) (by decide)
    exact decide_eq_false composite
  second := false
  second_correct := by
    have composite : ¬ Nat.Prime (corner 10046 2) := by change ¬ Nat.Prime 403688465; exact PrattCertificates.composite_of_product (factor := 5) (cofactor := 80737693) rfl (by decide) (by decide)
    exact decide_eq_false composite
  third := false
  third_correct := by
    have composite : ¬ Nat.Prime (corner 10046 3) := by change ¬ Nat.Prime 403668373; exact PrattCertificates.composite_of_product (factor := 337) (cofactor := 1197829) rfl (by decide) (by decide)
    exact decide_eq_false composite
theorem row_count_10046 : row 10046 = 0 := (row_10046.checked).trans rfl
theorem model_count_10046 : modelRow 10046 = 0 := by decide
theorem modelrow_agrees_10046 : modelRow 10046 = row 10046 := model_count_10046.trans row_count_10046.symm
def row_10047 : RowCertificate 10047 where
  first := false
  first_correct := by
    have composite : ¬ Nat.Prime (corner 10047 1) := by change ¬ Nat.Prime 403788931; exact PrattCertificates.composite_of_product (factor := 7) (cofactor := 57684133) rfl (by decide) (by decide)
    exact decide_eq_false composite
  second := false
  second_correct := by
    have composite : ¬ Nat.Prime (corner 10047 2) := by change ¬ Nat.Prime 403768837; exact PrattCertificates.composite_of_product (factor := 653) (cofactor := 618329) rfl (by decide) (by decide)
    exact decide_eq_false composite
  third := false
  third_correct := by
    have composite : ¬ Nat.Prime (corner 10047 3) := by change ¬ Nat.Prime 403748743; exact PrattCertificates.composite_of_product (factor := 31) (cofactor := 13024153) rfl (by decide) (by decide)
    exact decide_eq_false composite
theorem row_count_10047 : row 10047 = 0 := (row_10047.checked).trans rfl
theorem model_count_10047 : modelRow 10047 = 0 := by decide
theorem modelrow_agrees_10047 : modelRow 10047 = row 10047 := model_count_10047.trans row_count_10047.symm
def row_10048 : RowCertificate 10048 where
  first := true
  first_correct := by
    have prime : Nat.Prime (corner 10048 1) := by change Nat.Prime 403869313; exact ProjectEulerSolutions.P58PrimeNodes.p_403869313.prime
    exact decide_eq_true prime
  second := true
  second_correct := by
    have prime : Nat.Prime (corner 10048 2) := by change Nat.Prime 403849217; exact ProjectEulerSolutions.P58PrimeNodes.p_403849217.prime
    exact decide_eq_true prime
  third := false
  third_correct := by
    have composite : ¬ Nat.Prime (corner 10048 3) := by change ¬ Nat.Prime 403829121; exact PrattCertificates.composite_of_product (factor := 3) (cofactor := 134609707) rfl (by decide) (by decide)
    exact decide_eq_false composite
theorem row_count_10048 : row 10048 = 2 := (row_10048.checked).trans rfl
theorem model_count_10048 : modelRow 10048 = 2 := by decide
theorem modelrow_agrees_10048 : modelRow 10048 = row 10048 := model_count_10048.trans row_count_10048.symm
def row_10049 : RowCertificate 10049 where
  first := false
  first_correct := by
    have composite : ¬ Nat.Prime (corner 10049 1) := by change ¬ Nat.Prime 403949703; exact PrattCertificates.composite_of_product (factor := 3) (cofactor := 134649901) rfl (by decide) (by decide)
    exact decide_eq_false composite
  second := false
  second_correct := by
    have composite : ¬ Nat.Prime (corner 10049 2) := by change ¬ Nat.Prime 403929605; exact PrattCertificates.composite_of_product (factor := 5) (cofactor := 80785921) rfl (by decide) (by decide)
    exact decide_eq_false composite
  third := true
  third_correct := by
    have prime : Nat.Prime (corner 10049 3) := by change Nat.Prime 403909507; exact ProjectEulerSolutions.P58PrimeNodes.p_403909507.prime
    exact decide_eq_true prime
theorem row_count_10049 : row 10049 = 1 := (row_10049.checked).trans rfl
theorem model_count_10049 : modelRow 10049 = 1 := by decide
theorem modelrow_agrees_10049 : modelRow 10049 = row 10049 := model_count_10049.trans row_count_10049.symm
def row_10050 : RowCertificate 10050 where
  first := false
  first_correct := by
    have composite : ¬ Nat.Prime (corner 10050 1) := by change ¬ Nat.Prime 404030101; exact PrattCertificates.composite_of_product (factor := 14251) (cofactor := 28351) rfl (by decide) (by decide)
    exact decide_eq_false composite
  second := true
  second_correct := by
    have prime : Nat.Prime (corner 10050 2) := by change Nat.Prime 404010001; exact ProjectEulerSolutions.P58PrimeNodes.p_404010001.prime
    exact decide_eq_true prime
  third := false
  third_correct := by
    have composite : ¬ Nat.Prime (corner 10050 3) := by change ¬ Nat.Prime 403989901; exact PrattCertificates.composite_of_product (factor := 7) (cofactor := 57712843) rfl (by decide) (by decide)
    exact decide_eq_false composite
theorem row_count_10050 : row 10050 = 1 := (row_10050.checked).trans rfl
theorem model_count_10050 : modelRow 10050 = 1 := by decide
theorem modelrow_agrees_10050 : modelRow 10050 = row 10050 := model_count_10050.trans row_count_10050.symm
def row_10051 : RowCertificate 10051 where
  first := true
  first_correct := by
    have prime : Nat.Prime (corner 10051 1) := by change Nat.Prime 404110507; exact ProjectEulerSolutions.P58PrimeNodes.p_404110507.prime
    exact decide_eq_true prime
  second := false
  second_correct := by
    have composite : ¬ Nat.Prime (corner 10051 2) := by change ¬ Nat.Prime 404090405; exact PrattCertificates.composite_of_product (factor := 5) (cofactor := 80818081) rfl (by decide) (by decide)
    exact decide_eq_false composite
  third := false
  third_correct := by
    have composite : ¬ Nat.Prime (corner 10051 3) := by change ¬ Nat.Prime 404070303; exact PrattCertificates.composite_of_product (factor := 3) (cofactor := 134690101) rfl (by decide) (by decide)
    exact decide_eq_false composite
theorem row_count_10051 : row 10051 = 1 := (row_10051.checked).trans rfl
theorem model_count_10051 : modelRow 10051 = 1 := by decide
theorem modelrow_agrees_10051 : modelRow 10051 = row 10051 := model_count_10051.trans row_count_10051.symm
def row_10052 : RowCertificate 10052 where
  first := false
  first_correct := by
    have composite : ¬ Nat.Prime (corner 10052 1) := by change ¬ Nat.Prime 404190921; exact PrattCertificates.composite_of_product (factor := 3) (cofactor := 134730307) rfl (by decide) (by decide)
    exact decide_eq_false composite
  second := true
  second_correct := by
    have prime : Nat.Prime (corner 10052 2) := by change Nat.Prime 404170817; exact ProjectEulerSolutions.P58PrimeNodes.p_404170817.prime
    exact decide_eq_true prime
  third := false
  third_correct := by
    have composite : ¬ Nat.Prime (corner 10052 3) := by change ¬ Nat.Prime 404150713; exact PrattCertificates.composite_of_product (factor := 1153) (cofactor := 350521) rfl (by decide) (by decide)
    exact decide_eq_false composite
theorem row_count_10052 : row 10052 = 1 := (row_10052.checked).trans rfl
theorem model_count_10052 : modelRow 10052 = 1 := by decide
theorem modelrow_agrees_10052 : modelRow 10052 = row 10052 := model_count_10052.trans row_count_10052.symm
def row_10053 : RowCertificate 10053 where
  first := false
  first_correct := by
    have composite : ¬ Nat.Prime (corner 10053 1) := by change ¬ Nat.Prime 404271343; exact PrattCertificates.composite_of_product (factor := 7) (cofactor := 57753049) rfl (by decide) (by decide)
    exact decide_eq_false composite
  second := false
  second_correct := by
    have composite : ¬ Nat.Prime (corner 10053 2) := by change ¬ Nat.Prime 404251237; exact PrattCertificates.composite_of_product (factor := 13) (cofactor := 31096249) rfl (by decide) (by decide)
    exact decide_eq_false composite
  third := true
  third_correct := by
    have prime : Nat.Prime (corner 10053 3) := by change Nat.Prime 404231131; exact ProjectEulerSolutions.P58PrimeNodes.p_404231131.prime
    exact decide_eq_true prime
theorem row_count_10053 : row 10053 = 1 := (row_10053.checked).trans rfl
theorem model_count_10053 : modelRow 10053 = 1 := by decide
theorem modelrow_agrees_10053 : modelRow 10053 = row 10053 := model_count_10053.trans row_count_10053.symm
def row_10054 : RowCertificate 10054 where
  first := false
  first_correct := by
    have composite : ¬ Nat.Prime (corner 10054 1) := by change ¬ Nat.Prime 404351773; exact PrattCertificates.composite_of_product (factor := 7) (cofactor := 57764539) rfl (by decide) (by decide)
    exact decide_eq_false composite
  second := false
  second_correct := by
    have composite : ¬ Nat.Prime (corner 10054 2) := by change ¬ Nat.Prime 404331665; exact PrattCertificates.composite_of_product (factor := 5) (cofactor := 80866333) rfl (by decide) (by decide)
    exact decide_eq_false composite
  third := false
  third_correct := by
    have composite : ¬ Nat.Prime (corner 10054 3) := by change ¬ Nat.Prime 404311557; exact PrattCertificates.composite_of_product (factor := 3) (cofactor := 134770519) rfl (by decide) (by decide)
    exact decide_eq_false composite
theorem row_count_10054 : row 10054 = 0 := (row_10054.checked).trans rfl
theorem model_count_10054 : modelRow 10054 = 0 := by decide
theorem modelrow_agrees_10054 : modelRow 10054 = row 10054 := model_count_10054.trans row_count_10054.symm
def row_10055 : RowCertificate 10055 where
  first := false
  first_correct := by
    have composite : ¬ Nat.Prime (corner 10055 1) := by change ¬ Nat.Prime 404432211; exact PrattCertificates.composite_of_product (factor := 3) (cofactor := 134810737) rfl (by decide) (by decide)
    exact decide_eq_false composite
  second := false
  second_correct := by
    have composite : ¬ Nat.Prime (corner 10055 2) := by change ¬ Nat.Prime 404412101; exact PrattCertificates.composite_of_product (factor := 53) (cofactor := 7630417) rfl (by decide) (by decide)
    exact decide_eq_false composite
  third := false
  third_correct := by
    have composite : ¬ Nat.Prime (corner 10055 3) := by change ¬ Nat.Prime 404391991; exact PrattCertificates.composite_of_product (factor := 19) (cofactor := 21283789) rfl (by decide) (by decide)
    exact decide_eq_false composite
theorem row_count_10055 : row 10055 = 0 := (row_10055.checked).trans rfl
theorem model_count_10055 : modelRow 10055 = 0 := by decide
theorem modelrow_agrees_10055 : modelRow 10055 = row 10055 := model_count_10055.trans row_count_10055.symm
def row_10056 : RowCertificate 10056 where
  first := false
  first_correct := by
    have composite : ¬ Nat.Prime (corner 10056 1) := by change ¬ Nat.Prime 404512657; exact PrattCertificates.composite_of_product (factor := 139) (cofactor := 2910163) rfl (by decide) (by decide)
    exact decide_eq_false composite
  second := false
  second_correct := by
    have composite : ¬ Nat.Prime (corner 10056 2) := by change ¬ Nat.Prime 404492545; exact PrattCertificates.composite_of_product (factor := 5) (cofactor := 80898509) rfl (by decide) (by decide)
    exact decide_eq_false composite
  third := true
  third_correct := by
    have prime : Nat.Prime (corner 10056 3) := by change Nat.Prime 404472433; exact ProjectEulerSolutions.P58PrimeNodes.p_404472433.prime
    exact decide_eq_true prime
theorem row_count_10056 : row 10056 = 1 := (row_10056.checked).trans rfl
theorem model_count_10056 : modelRow 10056 = 1 := by decide
theorem modelrow_agrees_10056 : modelRow 10056 = row 10056 := model_count_10056.trans row_count_10056.symm
def row_10057 : RowCertificate 10057 where
  first := false
  first_correct := by
    have composite : ¬ Nat.Prime (corner 10057 1) := by change ¬ Nat.Prime 404593111; exact PrattCertificates.composite_of_product (factor := 13) (cofactor := 31122547) rfl (by decide) (by decide)
    exact decide_eq_false composite
  second := false
  second_correct := by
    have composite : ¬ Nat.Prime (corner 10057 2) := by change ¬ Nat.Prime 404572997; exact PrattCertificates.composite_of_product (factor := 29) (cofactor := 13950793) rfl (by decide) (by decide)
    exact decide_eq_false composite
  third := false
  third_correct := by
    have composite : ¬ Nat.Prime (corner 10057 3) := by change ¬ Nat.Prime 404552883; exact PrattCertificates.composite_of_product (factor := 3) (cofactor := 134850961) rfl (by decide) (by decide)
    exact decide_eq_false composite
theorem row_count_10057 : row 10057 = 0 := (row_10057.checked).trans rfl
theorem model_count_10057 : modelRow 10057 = 0 := by decide
theorem modelrow_agrees_10057 : modelRow 10057 = row 10057 := model_count_10057.trans row_count_10057.symm
def row_10058 : RowCertificate 10058 where
  first := false
  first_correct := by
    have composite : ¬ Nat.Prime (corner 10058 1) := by change ¬ Nat.Prime 404673573; exact PrattCertificates.composite_of_product (factor := 3) (cofactor := 134891191) rfl (by decide) (by decide)
    exact decide_eq_false composite
  second := false
  second_correct := by
    have composite : ¬ Nat.Prime (corner 10058 2) := by change ¬ Nat.Prime 404653457; exact PrattCertificates.composite_of_product (factor := 13) (cofactor := 31127189) rfl (by decide) (by decide)
    exact decide_eq_false composite
  third := false
  third_correct := by
    have composite : ¬ Nat.Prime (corner 10058 3) := by change ¬ Nat.Prime 404633341; exact PrattCertificates.composite_of_product (factor := 7) (cofactor := 57804763) rfl (by decide) (by decide)
    exact decide_eq_false composite
theorem row_count_10058 : row 10058 = 0 := (row_10058.checked).trans rfl
theorem model_count_10058 : modelRow 10058 = 0 := by decide
theorem modelrow_agrees_10058 : modelRow 10058 = row 10058 := model_count_10058.trans row_count_10058.symm
def row_10059 : RowCertificate 10059 where
  first := true
  first_correct := by
    have prime : Nat.Prime (corner 10059 1) := by change Nat.Prime 404754043; exact ProjectEulerSolutions.P58PrimeNodes.p_404754043.prime
    exact decide_eq_true prime
  second := false
  second_correct := by
    have composite : ¬ Nat.Prime (corner 10059 2) := by change ¬ Nat.Prime 404733925; exact PrattCertificates.composite_of_product (factor := 5) (cofactor := 80946785) rfl (by decide) (by decide)
    exact decide_eq_false composite
  third := false
  third_correct := by
    have composite : ¬ Nat.Prime (corner 10059 3) := by change ¬ Nat.Prime 404713807; exact PrattCertificates.composite_of_product (factor := 37) (cofactor := 10938211) rfl (by decide) (by decide)
    exact decide_eq_false composite
theorem row_count_10059 : row 10059 = 1 := (row_10059.checked).trans rfl
theorem model_count_10059 : modelRow 10059 = 1 := by decide
theorem modelrow_agrees_10059 : modelRow 10059 = row 10059 := model_count_10059.trans row_count_10059.symm
def row_10060 : RowCertificate 10060 where
  first := false
  first_correct := by
    have composite : ¬ Nat.Prime (corner 10060 1) := by change ¬ Nat.Prime 404834521; exact PrattCertificates.composite_of_product (factor := 7) (cofactor := 57833503) rfl (by decide) (by decide)
    exact decide_eq_false composite
  second := true
  second_correct := by
    have prime : Nat.Prime (corner 10060 2) := by change Nat.Prime 404814401; exact ProjectEulerSolutions.P58PrimeNodes.p_404814401.prime
    exact decide_eq_true prime
  third := false
  third_correct := by
    have composite : ¬ Nat.Prime (corner 10060 3) := by change ¬ Nat.Prime 404794281; exact PrattCertificates.composite_of_product (factor := 3) (cofactor := 134931427) rfl (by decide) (by decide)
    exact decide_eq_false composite
theorem row_count_10060 : row 10060 = 1 := (row_10060.checked).trans rfl
theorem model_count_10060 : modelRow 10060 = 1 := by decide
theorem modelrow_agrees_10060 : modelRow 10060 = row 10060 := model_count_10060.trans row_count_10060.symm
def row_10061 : RowCertificate 10061 where
  first := false
  first_correct := by
    have composite : ¬ Nat.Prime (corner 10061 1) := by change ¬ Nat.Prime 404915007; exact PrattCertificates.composite_of_product (factor := 3) (cofactor := 134971669) rfl (by decide) (by decide)
    exact decide_eq_false composite
  second := false
  second_correct := by
    have composite : ¬ Nat.Prime (corner 10061 2) := by change ¬ Nat.Prime 404894885; exact PrattCertificates.composite_of_product (factor := 5) (cofactor := 80978977) rfl (by decide) (by decide)
    exact decide_eq_false composite
  third := false
  third_correct := by
    have composite : ¬ Nat.Prime (corner 10061 3) := by change ¬ Nat.Prime 404874763; exact PrattCertificates.composite_of_product (factor := 79) (cofactor := 5124997) rfl (by decide) (by decide)
    exact decide_eq_false composite
theorem row_count_10061 : row 10061 = 0 := (row_10061.checked).trans rfl
theorem model_count_10061 : modelRow 10061 = 0 := by decide
theorem modelrow_agrees_10061 : modelRow 10061 = row 10061 := model_count_10061.trans row_count_10061.symm
def row_10062 : RowCertificate 10062 where
  first := false
  first_correct := by
    have composite : ¬ Nat.Prime (corner 10062 1) := by change ¬ Nat.Prime 404995501; exact PrattCertificates.composite_of_product (factor := 31) (cofactor := 13064371) rfl (by decide) (by decide)
    exact decide_eq_false composite
  second := false
  second_correct := by
    have composite : ¬ Nat.Prime (corner 10062 2) := by change ¬ Nat.Prime 404975377; exact PrattCertificates.composite_of_product (factor := 17) (cofactor := 23822081) rfl (by decide) (by decide)
    exact decide_eq_false composite
  third := true
  third_correct := by
    have prime : Nat.Prime (corner 10062 3) := by change Nat.Prime 404955253; exact ProjectEulerSolutions.P58PrimeNodes.p_404955253.prime
    exact decide_eq_true prime
theorem row_count_10062 : row 10062 = 1 := (row_10062.checked).trans rfl
theorem model_count_10062 : modelRow 10062 = 1 := by decide
theorem modelrow_agrees_10062 : modelRow 10062 = row 10062 := model_count_10062.trans row_count_10062.symm
def row_10063 : RowCertificate 10063 where
  first := false
  first_correct := by
    have composite : ¬ Nat.Prime (corner 10063 1) := by change ¬ Nat.Prime 405076003; exact PrattCertificates.composite_of_product (factor := 661) (cofactor := 612823) rfl (by decide) (by decide)
    exact decide_eq_false composite
  second := false
  second_correct := by
    have composite : ¬ Nat.Prime (corner 10063 2) := by change ¬ Nat.Prime 405055877; exact PrattCertificates.composite_of_product (factor := 1213) (cofactor := 333929) rfl (by decide) (by decide)
    exact decide_eq_false composite
  third := false
  third_correct := by
    have composite : ¬ Nat.Prime (corner 10063 3) := by change ¬ Nat.Prime 405035751; exact PrattCertificates.composite_of_product (factor := 3) (cofactor := 135011917) rfl (by decide) (by decide)
    exact decide_eq_false composite
theorem row_count_10063 : row 10063 = 0 := (row_10063.checked).trans rfl
theorem model_count_10063 : modelRow 10063 = 0 := by decide
theorem modelrow_agrees_10063 : modelRow 10063 = row 10063 := model_count_10063.trans row_count_10063.symm
def row_10064 : RowCertificate 10064 where
  first := false
  first_correct := by
    have composite : ¬ Nat.Prime (corner 10064 1) := by change ¬ Nat.Prime 405156513; exact PrattCertificates.composite_of_product (factor := 3) (cofactor := 135052171) rfl (by decide) (by decide)
    exact decide_eq_false composite
  second := false
  second_correct := by
    have composite : ¬ Nat.Prime (corner 10064 2) := by change ¬ Nat.Prime 405136385; exact PrattCertificates.composite_of_product (factor := 5) (cofactor := 81027277) rfl (by decide) (by decide)
    exact decide_eq_false composite
  third := false
  third_correct := by
    have composite : ¬ Nat.Prime (corner 10064 3) := by change ¬ Nat.Prime 405116257; exact PrattCertificates.composite_of_product (factor := 7) (cofactor := 57873751) rfl (by decide) (by decide)
    exact decide_eq_false composite
theorem row_count_10064 : row 10064 = 0 := (row_10064.checked).trans rfl
theorem model_count_10064 : modelRow 10064 = 0 := by decide
theorem modelrow_agrees_10064 : modelRow 10064 = row 10064 := model_count_10064.trans row_count_10064.symm
def row_10065 : RowCertificate 10065 where
  first := false
  first_correct := by
    have composite : ¬ Nat.Prime (corner 10065 1) := by change ¬ Nat.Prime 405237031; exact PrattCertificates.composite_of_product (factor := 43) (cofactor := 9424117) rfl (by decide) (by decide)
    exact decide_eq_false composite
  second := false
  second_correct := by
    have composite : ¬ Nat.Prime (corner 10065 2) := by change ¬ Nat.Prime 405216901; exact PrattCertificates.composite_of_product (factor := 4993) (cofactor := 81157) rfl (by decide) (by decide)
    exact decide_eq_false composite
  third := false
  third_correct := by
    have composite : ¬ Nat.Prime (corner 10065 3) := by change ¬ Nat.Prime 405196771; exact PrattCertificates.composite_of_product (factor := 7) (cofactor := 57885253) rfl (by decide) (by decide)
    exact decide_eq_false composite
theorem row_count_10065 : row 10065 = 0 := (row_10065.checked).trans rfl
theorem model_count_10065 : modelRow 10065 = 0 := by decide
theorem modelrow_agrees_10065 : modelRow 10065 = row 10065 := model_count_10065.trans row_count_10065.symm
def row_10066 : RowCertificate 10066 where
  first := false
  first_correct := by
    have composite : ¬ Nat.Prime (corner 10066 1) := by change ¬ Nat.Prime 405317557; exact PrattCertificates.composite_of_product (factor := 19) (cofactor := 21332503) rfl (by decide) (by decide)
    exact decide_eq_false composite
  second := false
  second_correct := by
    have composite : ¬ Nat.Prime (corner 10066 2) := by change ¬ Nat.Prime 405297425; exact PrattCertificates.composite_of_product (factor := 5) (cofactor := 81059485) rfl (by decide) (by decide)
    exact decide_eq_false composite
  third := false
  third_correct := by
    have composite : ¬ Nat.Prime (corner 10066 3) := by change ¬ Nat.Prime 405277293; exact PrattCertificates.composite_of_product (factor := 3) (cofactor := 135092431) rfl (by decide) (by decide)
    exact decide_eq_false composite
theorem row_count_10066 : row 10066 = 0 := (row_10066.checked).trans rfl
theorem model_count_10066 : modelRow 10066 = 0 := by decide
theorem modelrow_agrees_10066 : modelRow 10066 = row 10066 := model_count_10066.trans row_count_10066.symm
def row_10067 : RowCertificate 10067 where
  first := false
  first_correct := by
    have composite : ¬ Nat.Prime (corner 10067 1) := by change ¬ Nat.Prime 405398091; exact PrattCertificates.composite_of_product (factor := 3) (cofactor := 135132697) rfl (by decide) (by decide)
    exact decide_eq_false composite
  second := false
  second_correct := by
    have composite : ¬ Nat.Prime (corner 10067 2) := by change ¬ Nat.Prime 405377957; exact PrattCertificates.composite_of_product (factor := 37) (cofactor := 10956161) rfl (by decide) (by decide)
    exact decide_eq_false composite
  third := false
  third_correct := by
    have composite : ¬ Nat.Prime (corner 10067 3) := by change ¬ Nat.Prime 405357823; exact PrattCertificates.composite_of_product (factor := 13) (cofactor := 31181371) rfl (by decide) (by decide)
    exact decide_eq_false composite
theorem row_count_10067 : row 10067 = 0 := (row_10067.checked).trans rfl
theorem model_count_10067 : modelRow 10067 = 0 := by decide
theorem modelrow_agrees_10067 : modelRow 10067 = row 10067 := model_count_10067.trans row_count_10067.symm
def row_10068 : RowCertificate 10068 where
  first := false
  first_correct := by
    have composite : ¬ Nat.Prime (corner 10068 1) := by change ¬ Nat.Prime 405478633; exact PrattCertificates.composite_of_product (factor := 7) (cofactor := 57925519) rfl (by decide) (by decide)
    exact decide_eq_false composite
  second := true
  second_correct := by
    have prime : Nat.Prime (corner 10068 2) := by change Nat.Prime 405458497; exact ProjectEulerSolutions.P58PrimeNodes.p_405458497.prime
    exact decide_eq_true prime
  third := true
  third_correct := by
    have prime : Nat.Prime (corner 10068 3) := by change Nat.Prime 405438361; exact ProjectEulerSolutions.P58PrimeNodes.p_405438361.prime
    exact decide_eq_true prime
theorem row_count_10068 : row 10068 = 2 := (row_10068.checked).trans rfl
theorem model_count_10068 : modelRow 10068 = 2 := by decide
theorem modelrow_agrees_10068 : modelRow 10068 = row 10068 := model_count_10068.trans row_count_10068.symm
def row_10069 : RowCertificate 10069 where
  first := false
  first_correct := by
    have composite : ¬ Nat.Prime (corner 10069 1) := by change ¬ Nat.Prime 405559183; exact PrattCertificates.composite_of_product (factor := 37) (cofactor := 10961059) rfl (by decide) (by decide)
    exact decide_eq_false composite
  second := false
  second_correct := by
    have composite : ¬ Nat.Prime (corner 10069 2) := by change ¬ Nat.Prime 405539045; exact PrattCertificates.composite_of_product (factor := 5) (cofactor := 81107809) rfl (by decide) (by decide)
    exact decide_eq_false composite
  third := false
  third_correct := by
    have composite : ¬ Nat.Prime (corner 10069 3) := by change ¬ Nat.Prime 405518907; exact PrattCertificates.composite_of_product (factor := 3) (cofactor := 135172969) rfl (by decide) (by decide)
    exact decide_eq_false composite
theorem row_count_10069 : row 10069 = 0 := (row_10069.checked).trans rfl
theorem model_count_10069 : modelRow 10069 = 0 := by decide
theorem modelrow_agrees_10069 : modelRow 10069 = row 10069 := model_count_10069.trans row_count_10069.symm
def row_10070 : RowCertificate 10070 where
  first := false
  first_correct := by
    have composite : ¬ Nat.Prime (corner 10070 1) := by change ¬ Nat.Prime 405639741; exact PrattCertificates.composite_of_product (factor := 3) (cofactor := 135213247) rfl (by decide) (by decide)
    exact decide_eq_false composite
  second := false
  second_correct := by
    have composite : ¬ Nat.Prime (corner 10070 2) := by change ¬ Nat.Prime 405619601; exact PrattCertificates.composite_of_product (factor := 41) (cofactor := 9893161) rfl (by decide) (by decide)
    exact decide_eq_false composite
  third := false
  third_correct := by
    have composite : ¬ Nat.Prime (corner 10070 3) := by change ¬ Nat.Prime 405599461; exact PrattCertificates.composite_of_product (factor := 73) (cofactor := 5556157) rfl (by decide) (by decide)
    exact decide_eq_false composite
theorem row_count_10070 : row 10070 = 0 := (row_10070.checked).trans rfl
theorem model_count_10070 : modelRow 10070 = 0 := by decide
theorem modelrow_agrees_10070 : modelRow 10070 = row 10070 := model_count_10070.trans row_count_10070.symm
def row_10071 : RowCertificate 10071 where
  first := true
  first_correct := by
    have prime : Nat.Prime (corner 10071 1) := by change Nat.Prime 405720307; exact ProjectEulerSolutions.P58PrimeNodes.p_405720307.prime
    exact decide_eq_true prime
  second := false
  second_correct := by
    have composite : ¬ Nat.Prime (corner 10071 2) := by change ¬ Nat.Prime 405700165; exact PrattCertificates.composite_of_product (factor := 5) (cofactor := 81140033) rfl (by decide) (by decide)
    exact decide_eq_false composite
  third := false
  third_correct := by
    have composite : ¬ Nat.Prime (corner 10071 3) := by change ¬ Nat.Prime 405680023; exact PrattCertificates.composite_of_product (factor := 7) (cofactor := 57954289) rfl (by decide) (by decide)
    exact decide_eq_false composite
theorem row_count_10071 : row 10071 = 1 := (row_10071.checked).trans rfl
theorem model_count_10071 : modelRow 10071 = 1 := by decide
theorem modelrow_agrees_10071 : modelRow 10071 = row 10071 := model_count_10071.trans row_count_10071.symm
def row_10072 : RowCertificate 10072 where
  first := false
  first_correct := by
    have composite : ¬ Nat.Prime (corner 10072 1) := by change ¬ Nat.Prime 405800881; exact PrattCertificates.composite_of_product (factor := 31) (cofactor := 13090351) rfl (by decide) (by decide)
    exact decide_eq_false composite
  second := false
  second_correct := by
    have composite : ¬ Nat.Prime (corner 10072 2) := by change ¬ Nat.Prime 405780737; exact PrattCertificates.composite_of_product (factor := 733) (cofactor := 553589) rfl (by decide) (by decide)
    exact decide_eq_false composite
  third := false
  third_correct := by
    have composite : ¬ Nat.Prime (corner 10072 3) := by change ¬ Nat.Prime 405760593; exact PrattCertificates.composite_of_product (factor := 3) (cofactor := 135253531) rfl (by decide) (by decide)
    exact decide_eq_false composite
theorem row_count_10072 : row 10072 = 0 := (row_10072.checked).trans rfl
theorem model_count_10072 : modelRow 10072 = 0 := by decide
theorem modelrow_agrees_10072 : modelRow 10072 = row 10072 := model_count_10072.trans row_count_10072.symm
def row_10073 : RowCertificate 10073 where
  first := false
  first_correct := by
    have composite : ¬ Nat.Prime (corner 10073 1) := by change ¬ Nat.Prime 405881463; exact PrattCertificates.composite_of_product (factor := 3) (cofactor := 135293821) rfl (by decide) (by decide)
    exact decide_eq_false composite
  second := true
  second_correct := by
    have prime : Nat.Prime (corner 10073 2) := by change Nat.Prime 405861317; exact ProjectEulerSolutions.P58PrimeNodes.p_405861317.prime
    exact decide_eq_true prime
  third := true
  third_correct := by
    have prime : Nat.Prime (corner 10073 3) := by change Nat.Prime 405841171; exact ProjectEulerSolutions.P58PrimeNodes.p_405841171.prime
    exact decide_eq_true prime
theorem row_count_10073 : row 10073 = 2 := (row_10073.checked).trans rfl
theorem model_count_10073 : modelRow 10073 = 2 := by decide
theorem modelrow_agrees_10073 : modelRow 10073 = row 10073 := model_count_10073.trans row_count_10073.symm
def row_10074 : RowCertificate 10074 where
  first := false
  first_correct := by
    have composite : ¬ Nat.Prime (corner 10074 1) := by change ¬ Nat.Prime 405962053; exact PrattCertificates.composite_of_product (factor := 7) (cofactor := 57994579) rfl (by decide) (by decide)
    exact decide_eq_false composite
  second := false
  second_correct := by
    have composite : ¬ Nat.Prime (corner 10074 2) := by change ¬ Nat.Prime 405941905; exact PrattCertificates.composite_of_product (factor := 5) (cofactor := 81188381) rfl (by decide) (by decide)
    exact decide_eq_false composite
  third := false
  third_correct := by
    have composite : ¬ Nat.Prime (corner 10074 3) := by change ¬ Nat.Prime 405921757; exact PrattCertificates.composite_of_product (factor := 19) (cofactor := 21364303) rfl (by decide) (by decide)
    exact decide_eq_false composite
theorem row_count_10074 : row 10074 = 0 := (row_10074.checked).trans rfl
theorem model_count_10074 : modelRow 10074 = 0 := by decide
theorem modelrow_agrees_10074 : modelRow 10074 = row 10074 := model_count_10074.trans row_count_10074.symm
def row_10075 : RowCertificate 10075 where
  first := false
  first_correct := by
    have composite : ¬ Nat.Prime (corner 10075 1) := by change ¬ Nat.Prime 406042651; exact PrattCertificates.composite_of_product (factor := 7) (cofactor := 58006093) rfl (by decide) (by decide)
    exact decide_eq_false composite
  second := false
  second_correct := by
    have composite : ¬ Nat.Prime (corner 10075 2) := by change ¬ Nat.Prime 406022501; exact PrattCertificates.composite_of_product (factor := 2377) (cofactor := 170813) rfl (by decide) (by decide)
    exact decide_eq_false composite
  third := false
  third_correct := by
    have composite : ¬ Nat.Prime (corner 10075 3) := by change ¬ Nat.Prime 406002351; exact PrattCertificates.composite_of_product (factor := 3) (cofactor := 135334117) rfl (by decide) (by decide)
    exact decide_eq_false composite
theorem row_count_10075 : row 10075 = 0 := (row_10075.checked).trans rfl
theorem model_count_10075 : modelRow 10075 = 0 := by decide
theorem modelrow_agrees_10075 : modelRow 10075 = row 10075 := model_count_10075.trans row_count_10075.symm
def row_10076 : RowCertificate 10076 where
  first := false
  first_correct := by
    have composite : ¬ Nat.Prime (corner 10076 1) := by change ¬ Nat.Prime 406123257; exact PrattCertificates.composite_of_product (factor := 3) (cofactor := 135374419) rfl (by decide) (by decide)
    exact decide_eq_false composite
  second := false
  second_correct := by
    have composite : ¬ Nat.Prime (corner 10076 2) := by change ¬ Nat.Prime 406103105; exact PrattCertificates.composite_of_product (factor := 5) (cofactor := 81220621) rfl (by decide) (by decide)
    exact decide_eq_false composite
  third := false
  third_correct := by
    have composite : ¬ Nat.Prime (corner 10076 3) := by change ¬ Nat.Prime 406082953; exact PrattCertificates.composite_of_product (factor := 19) (cofactor := 21372787) rfl (by decide) (by decide)
    exact decide_eq_false composite
theorem row_count_10076 : row 10076 = 0 := (row_10076.checked).trans rfl
theorem model_count_10076 : modelRow 10076 = 0 := by decide
theorem modelrow_agrees_10076 : modelRow 10076 = row 10076 := model_count_10076.trans row_count_10076.symm
def row_10077 : RowCertificate 10077 where
  first := false
  first_correct := by
    have composite : ¬ Nat.Prime (corner 10077 1) := by change ¬ Nat.Prime 406203871; exact PrattCertificates.composite_of_product (factor := 37) (cofactor := 10978483) rfl (by decide) (by decide)
    exact decide_eq_false composite
  second := false
  second_correct := by
    have composite : ¬ Nat.Prime (corner 10077 2) := by change ¬ Nat.Prime 406183717; exact PrattCertificates.composite_of_product (factor := 97) (cofactor := 4187461) rfl (by decide) (by decide)
    exact decide_eq_false composite
  third := false
  third_correct := by
    have composite : ¬ Nat.Prime (corner 10077 3) := by change ¬ Nat.Prime 406163563; exact PrattCertificates.composite_of_product (factor := 13) (cofactor := 31243351) rfl (by decide) (by decide)
    exact decide_eq_false composite
theorem row_count_10077 : row 10077 = 0 := (row_10077.checked).trans rfl
theorem model_count_10077 : modelRow 10077 = 0 := by decide
theorem modelrow_agrees_10077 : modelRow 10077 = row 10077 := model_count_10077.trans row_count_10077.symm
def row_10078 : RowCertificate 10078 where
  first := false
  first_correct := by
    have composite : ¬ Nat.Prime (corner 10078 1) := by change ¬ Nat.Prime 406284493; exact PrattCertificates.composite_of_product (factor := 73) (cofactor := 5565541) rfl (by decide) (by decide)
    exact decide_eq_false composite
  second := true
  second_correct := by
    have prime : Nat.Prime (corner 10078 2) := by change Nat.Prime 406264337; exact ProjectEulerSolutions.P58PrimeNodes.p_406264337.prime
    exact decide_eq_true prime
  third := false
  third_correct := by
    have composite : ¬ Nat.Prime (corner 10078 3) := by change ¬ Nat.Prime 406244181; exact PrattCertificates.composite_of_product (factor := 3) (cofactor := 135414727) rfl (by decide) (by decide)
    exact decide_eq_false composite
theorem row_count_10078 : row 10078 = 1 := (row_10078.checked).trans rfl
theorem model_count_10078 : modelRow 10078 = 1 := by decide
theorem modelrow_agrees_10078 : modelRow 10078 = row 10078 := model_count_10078.trans row_count_10078.symm
def row_10079 : RowCertificate 10079 where
  first := false
  first_correct := by
    have composite : ¬ Nat.Prime (corner 10079 1) := by change ¬ Nat.Prime 406365123; exact PrattCertificates.composite_of_product (factor := 3) (cofactor := 135455041) rfl (by decide) (by decide)
    exact decide_eq_false composite
  second := false
  second_correct := by
    have composite : ¬ Nat.Prime (corner 10079 2) := by change ¬ Nat.Prime 406344965; exact PrattCertificates.composite_of_product (factor := 5) (cofactor := 81268993) rfl (by decide) (by decide)
    exact decide_eq_false composite
  third := false
  third_correct := by
    have composite : ¬ Nat.Prime (corner 10079 3) := by change ¬ Nat.Prime 406324807; exact PrattCertificates.composite_of_product (factor := 7) (cofactor := 58046401) rfl (by decide) (by decide)
    exact decide_eq_false composite
theorem row_count_10079 : row 10079 = 0 := (row_10079.checked).trans rfl
theorem model_count_10079 : modelRow 10079 = 0 := by decide
theorem modelrow_agrees_10079 : modelRow 10079 = row 10079 := model_count_10079.trans row_count_10079.symm
def row_10080 : RowCertificate 10080 where
  first := false
  first_correct := by
    have composite : ¬ Nat.Prime (corner 10080 1) := by change ¬ Nat.Prime 406445761; exact PrattCertificates.composite_of_product (factor := 43) (cofactor := 9452227) rfl (by decide) (by decide)
    exact decide_eq_false composite
  second := true
  second_correct := by
    have prime : Nat.Prime (corner 10080 2) := by change Nat.Prime 406425601; exact ProjectEulerSolutions.P58PrimeNodes.p_406425601.prime
    exact decide_eq_true prime
  third := false
  third_correct := by
    have composite : ¬ Nat.Prime (corner 10080 3) := by change ¬ Nat.Prime 406405441; exact PrattCertificates.composite_of_product (factor := 13) (cofactor := 31261957) rfl (by decide) (by decide)
    exact decide_eq_false composite
theorem row_count_10080 : row 10080 = 1 := (row_10080.checked).trans rfl
theorem model_count_10080 : modelRow 10080 = 1 := by decide
theorem modelrow_agrees_10080 : modelRow 10080 = row 10080 := model_count_10080.trans row_count_10080.symm
def row_10081 : RowCertificate 10081 where
  first := false
  first_correct := by
    have composite : ¬ Nat.Prime (corner 10081 1) := by change ¬ Nat.Prime 406526407; exact PrattCertificates.composite_of_product (factor := 7) (cofactor := 58075201) rfl (by decide) (by decide)
    exact decide_eq_false composite
  second := false
  second_correct := by
    have composite : ¬ Nat.Prime (corner 10081 2) := by change ¬ Nat.Prime 406506245; exact PrattCertificates.composite_of_product (factor := 5) (cofactor := 81301249) rfl (by decide) (by decide)
    exact decide_eq_false composite
  third := false
  third_correct := by
    have composite : ¬ Nat.Prime (corner 10081 3) := by change ¬ Nat.Prime 406486083; exact PrattCertificates.composite_of_product (factor := 3) (cofactor := 135495361) rfl (by decide) (by decide)
    exact decide_eq_false composite
theorem row_count_10081 : row 10081 = 0 := (row_10081.checked).trans rfl
theorem model_count_10081 : modelRow 10081 = 0 := by decide
theorem modelrow_agrees_10081 : modelRow 10081 = row 10081 := model_count_10081.trans row_count_10081.symm
def row_10082 : RowCertificate 10082 where
  first := false
  first_correct := by
    have composite : ¬ Nat.Prime (corner 10082 1) := by change ¬ Nat.Prime 406607061; exact PrattCertificates.composite_of_product (factor := 3) (cofactor := 135535687) rfl (by decide) (by decide)
    exact decide_eq_false composite
  second := true
  second_correct := by
    have prime : Nat.Prime (corner 10082 2) := by change Nat.Prime 406586897; exact ProjectEulerSolutions.P58PrimeNodes.p_406586897.prime
    exact decide_eq_true prime
  third := false
  third_correct := by
    have composite : ¬ Nat.Prime (corner 10082 3) := by change ¬ Nat.Prime 406566733; exact PrattCertificates.composite_of_product (factor := 613) (cofactor := 663241) rfl (by decide) (by decide)
    exact decide_eq_false composite
theorem row_count_10082 : row 10082 = 1 := (row_10082.checked).trans rfl
theorem model_count_10082 : modelRow 10082 = 1 := by decide
theorem modelrow_agrees_10082 : modelRow 10082 = row 10082 := model_count_10082.trans row_count_10082.symm
def row_10083 : RowCertificate 10083 where
  first := false
  first_correct := by
    have composite : ¬ Nat.Prime (corner 10083 1) := by change ¬ Nat.Prime 406687723; exact PrattCertificates.composite_of_product (factor := 13) (cofactor := 31283671) rfl (by decide) (by decide)
    exact decide_eq_false composite
  second := false
  second_correct := by
    have composite : ¬ Nat.Prime (corner 10083 2) := by change ¬ Nat.Prime 406667557; exact PrattCertificates.composite_of_product (factor := 17) (cofactor := 23921621) rfl (by decide) (by decide)
    exact decide_eq_false composite
  third := false
  third_correct := by
    have composite : ¬ Nat.Prime (corner 10083 3) := by change ¬ Nat.Prime 406647391; exact PrattCertificates.composite_of_product (factor := 919) (cofactor := 442489) rfl (by decide) (by decide)
    exact decide_eq_false composite
theorem row_count_10083 : row 10083 = 0 := (row_10083.checked).trans rfl
theorem model_count_10083 : modelRow 10083 = 0 := by decide
theorem modelrow_agrees_10083 : modelRow 10083 = row 10083 := model_count_10083.trans row_count_10083.symm
def row_10084 : RowCertificate 10084 where
  first := false
  first_correct := by
    have composite : ¬ Nat.Prime (corner 10084 1) := by change ¬ Nat.Prime 406768393; exact PrattCertificates.composite_of_product (factor := 79) (cofactor := 5148967) rfl (by decide) (by decide)
    exact decide_eq_false composite
  second := false
  second_correct := by
    have composite : ¬ Nat.Prime (corner 10084 2) := by change ¬ Nat.Prime 406748225; exact PrattCertificates.composite_of_product (factor := 5) (cofactor := 81349645) rfl (by decide) (by decide)
    exact decide_eq_false composite
  third := false
  third_correct := by
    have composite : ¬ Nat.Prime (corner 10084 3) := by change ¬ Nat.Prime 406728057; exact PrattCertificates.composite_of_product (factor := 3) (cofactor := 135576019) rfl (by decide) (by decide)
    exact decide_eq_false composite
theorem row_count_10084 : row 10084 = 0 := (row_10084.checked).trans rfl
theorem model_count_10084 : modelRow 10084 = 0 := by decide
theorem modelrow_agrees_10084 : modelRow 10084 = row 10084 := model_count_10084.trans row_count_10084.symm
def row_10085 : RowCertificate 10085 where
  first := false
  first_correct := by
    have composite : ¬ Nat.Prime (corner 10085 1) := by change ¬ Nat.Prime 406849071; exact PrattCertificates.composite_of_product (factor := 3) (cofactor := 135616357) rfl (by decide) (by decide)
    exact decide_eq_false composite
  second := false
  second_correct := by
    have composite : ¬ Nat.Prime (corner 10085 2) := by change ¬ Nat.Prime 406828901; exact PrattCertificates.composite_of_product (factor := 53) (cofactor := 7676017) rfl (by decide) (by decide)
    exact decide_eq_false composite
  third := false
  third_correct := by
    have composite : ¬ Nat.Prime (corner 10085 3) := by change ¬ Nat.Prime 406808731; exact PrattCertificates.composite_of_product (factor := 7) (cofactor := 58115533) rfl (by decide) (by decide)
    exact decide_eq_false composite
theorem row_count_10085 : row 10085 = 0 := (row_10085.checked).trans rfl
theorem model_count_10085 : modelRow 10085 = 0 := by decide
theorem modelrow_agrees_10085 : modelRow 10085 = row 10085 := model_count_10085.trans row_count_10085.symm
def row_10086 : RowCertificate 10086 where
  first := false
  first_correct := by
    have composite : ¬ Nat.Prime (corner 10086 1) := by change ¬ Nat.Prime 406929757; exact PrattCertificates.composite_of_product (factor := 13) (cofactor := 31302289) rfl (by decide) (by decide)
    exact decide_eq_false composite
  second := false
  second_correct := by
    have composite : ¬ Nat.Prime (corner 10086 2) := by change ¬ Nat.Prime 406909585; exact PrattCertificates.composite_of_product (factor := 5) (cofactor := 81381917) rfl (by decide) (by decide)
    exact decide_eq_false composite
  third := false
  third_correct := by
    have composite : ¬ Nat.Prime (corner 10086 3) := by change ¬ Nat.Prime 406889413; exact PrattCertificates.composite_of_product (factor := 7) (cofactor := 58127059) rfl (by decide) (by decide)
    exact decide_eq_false composite
theorem row_count_10086 : row 10086 = 0 := (row_10086.checked).trans rfl
theorem model_count_10086 : modelRow 10086 = 0 := by decide
theorem modelrow_agrees_10086 : modelRow 10086 = row 10086 := model_count_10086.trans row_count_10086.symm
def row_10087 : RowCertificate 10087 where
  first := false
  first_correct := by
    have composite : ¬ Nat.Prime (corner 10087 1) := by change ¬ Nat.Prime 407010451; exact PrattCertificates.composite_of_product (factor := 619) (cofactor := 657529) rfl (by decide) (by decide)
    exact decide_eq_false composite
  second := false
  second_correct := by
    have composite : ¬ Nat.Prime (corner 10087 2) := by change ¬ Nat.Prime 406990277; exact PrattCertificates.composite_of_product (factor := 317) (cofactor := 1283881) rfl (by decide) (by decide)
    exact decide_eq_false composite
  third := false
  third_correct := by
    have composite : ¬ Nat.Prime (corner 10087 3) := by change ¬ Nat.Prime 406970103; exact PrattCertificates.composite_of_product (factor := 3) (cofactor := 135656701) rfl (by decide) (by decide)
    exact decide_eq_false composite
theorem row_count_10087 : row 10087 = 0 := (row_10087.checked).trans rfl
theorem model_count_10087 : modelRow 10087 = 0 := by decide
theorem modelrow_agrees_10087 : modelRow 10087 = row 10087 := model_count_10087.trans row_count_10087.symm
def row_10088 : RowCertificate 10088 where
  first := false
  first_correct := by
    have composite : ¬ Nat.Prime (corner 10088 1) := by change ¬ Nat.Prime 407091153; exact PrattCertificates.composite_of_product (factor := 3) (cofactor := 135697051) rfl (by decide) (by decide)
    exact decide_eq_false composite
  second := false
  second_correct := by
    have composite : ¬ Nat.Prime (corner 10088 2) := by change ¬ Nat.Prime 407070977; exact PrattCertificates.composite_of_product (factor := 137) (cofactor := 2971321) rfl (by decide) (by decide)
    exact decide_eq_false composite
  third := false
  third_correct := by
    have composite : ¬ Nat.Prime (corner 10088 3) := by change ¬ Nat.Prime 407050801; exact PrattCertificates.composite_of_product (factor := 31) (cofactor := 13130671) rfl (by decide) (by decide)
    exact decide_eq_false composite
theorem row_count_10088 : row 10088 = 0 := (row_10088.checked).trans rfl
theorem model_count_10088 : modelRow 10088 = 0 := by decide
theorem modelrow_agrees_10088 : modelRow 10088 = row 10088 := model_count_10088.trans row_count_10088.symm
def row_10089 : RowCertificate 10089 where
  first := false
  first_correct := by
    have composite : ¬ Nat.Prime (corner 10089 1) := by change ¬ Nat.Prime 407171863; exact PrattCertificates.composite_of_product (factor := 7) (cofactor := 58167409) rfl (by decide) (by decide)
    exact decide_eq_false composite
  second := false
  second_correct := by
    have composite : ¬ Nat.Prime (corner 10089 2) := by change ¬ Nat.Prime 407151685; exact PrattCertificates.composite_of_product (factor := 5) (cofactor := 81430337) rfl (by decide) (by decide)
    exact decide_eq_false composite
  third := false
  third_correct := by
    have composite : ¬ Nat.Prime (corner 10089 3) := by change ¬ Nat.Prime 407131507; exact PrattCertificates.composite_of_product (factor := 61) (cofactor := 6674287) rfl (by decide) (by decide)
    exact decide_eq_false composite
theorem row_count_10089 : row 10089 = 0 := (row_10089.checked).trans rfl
theorem model_count_10089 : modelRow 10089 = 0 := by decide
theorem modelrow_agrees_10089 : modelRow 10089 = row 10089 := model_count_10089.trans row_count_10089.symm
def row_10090 : RowCertificate 10090 where
  first := false
  first_correct := by
    have composite : ¬ Nat.Prime (corner 10090 1) := by change ¬ Nat.Prime 407252581; exact PrattCertificates.composite_of_product (factor := 193) (cofactor := 2110117) rfl (by decide) (by decide)
    exact decide_eq_false composite
  second := false
  second_correct := by
    have composite : ¬ Nat.Prime (corner 10090 2) := by change ¬ Nat.Prime 407232401; exact PrattCertificates.composite_of_product (factor := 61) (cofactor := 6675941) rfl (by decide) (by decide)
    exact decide_eq_false composite
  third := false
  third_correct := by
    have composite : ¬ Nat.Prime (corner 10090 3) := by change ¬ Nat.Prime 407212221; exact PrattCertificates.composite_of_product (factor := 3) (cofactor := 135737407) rfl (by decide) (by decide)
    exact decide_eq_false composite
theorem row_count_10090 : row 10090 = 0 := (row_10090.checked).trans rfl
theorem model_count_10090 : modelRow 10090 = 0 := by decide
theorem modelrow_agrees_10090 : modelRow 10090 = row 10090 := model_count_10090.trans row_count_10090.symm
def row_10091 : RowCertificate 10091 where
  first := false
  first_correct := by
    have composite : ¬ Nat.Prime (corner 10091 1) := by change ¬ Nat.Prime 407333307; exact PrattCertificates.composite_of_product (factor := 3) (cofactor := 135777769) rfl (by decide) (by decide)
    exact decide_eq_false composite
  second := false
  second_correct := by
    have composite : ¬ Nat.Prime (corner 10091 2) := by change ¬ Nat.Prime 407313125; exact PrattCertificates.composite_of_product (factor := 5) (cofactor := 81462625) rfl (by decide) (by decide)
    exact decide_eq_false composite
  third := true
  third_correct := by
    have prime : Nat.Prime (corner 10091 3) := by change Nat.Prime 407292943; exact ProjectEulerSolutions.P58PrimeNodes.p_407292943.prime
    exact decide_eq_true prime
theorem row_count_10091 : row 10091 = 1 := (row_10091.checked).trans rfl
theorem model_count_10091 : modelRow 10091 = 1 := by decide
theorem modelrow_agrees_10091 : modelRow 10091 = row 10091 := model_count_10091.trans row_count_10091.symm
def row_10092 : RowCertificate 10092 where
  first := true
  first_correct := by
    have prime : Nat.Prime (corner 10092 1) := by change Nat.Prime 407414041; exact ProjectEulerSolutions.P58PrimeNodes.p_407414041.prime
    exact decide_eq_true prime
  second := false
  second_correct := by
    have composite : ¬ Nat.Prime (corner 10092 2) := by change ¬ Nat.Prime 407393857; exact PrattCertificates.composite_of_product (factor := 13) (cofactor := 31337989) rfl (by decide) (by decide)
    exact decide_eq_false composite
  third := false
  third_correct := by
    have composite : ¬ Nat.Prime (corner 10092 3) := by change ¬ Nat.Prime 407373673; exact PrattCertificates.composite_of_product (factor := 7) (cofactor := 58196239) rfl (by decide) (by decide)
    exact decide_eq_false composite
theorem row_count_10092 : row 10092 = 1 := (row_10092.checked).trans rfl
theorem model_count_10092 : modelRow 10092 = 1 := by decide
theorem modelrow_agrees_10092 : modelRow 10092 = row 10092 := model_count_10092.trans row_count_10092.symm
def row_10093 : RowCertificate 10093 where
  first := false
  first_correct := by
    have composite : ¬ Nat.Prime (corner 10093 1) := by change ¬ Nat.Prime 407494783; exact PrattCertificates.composite_of_product (factor := 31) (cofactor := 13144993) rfl (by decide) (by decide)
    exact decide_eq_false composite
  second := true
  second_correct := by
    have prime : Nat.Prime (corner 10093 2) := by change Nat.Prime 407474597; exact ProjectEulerSolutions.P58PrimeNodes.p_407474597.prime
    exact decide_eq_true prime
  third := false
  third_correct := by
    have composite : ¬ Nat.Prime (corner 10093 3) := by change ¬ Nat.Prime 407454411; exact PrattCertificates.composite_of_product (factor := 3) (cofactor := 135818137) rfl (by decide) (by decide)
    exact decide_eq_false composite
theorem row_count_10093 : row 10093 = 1 := (row_10093.checked).trans rfl
theorem model_count_10093 : modelRow 10093 = 1 := by decide
theorem modelrow_agrees_10093 : modelRow 10093 = row 10093 := model_count_10093.trans row_count_10093.symm
def row_10094 : RowCertificate 10094 where
  first := false
  first_correct := by
    have composite : ¬ Nat.Prime (corner 10094 1) := by change ¬ Nat.Prime 407575533; exact PrattCertificates.composite_of_product (factor := 3) (cofactor := 135858511) rfl (by decide) (by decide)
    exact decide_eq_false composite
  second := false
  second_correct := by
    have composite : ¬ Nat.Prime (corner 10094 2) := by change ¬ Nat.Prime 407555345; exact PrattCertificates.composite_of_product (factor := 5) (cofactor := 81511069) rfl (by decide) (by decide)
    exact decide_eq_false composite
  third := false
  third_correct := by
    have composite : ¬ Nat.Prime (corner 10094 3) := by change ¬ Nat.Prime 407535157; exact PrattCertificates.composite_of_product (factor := 421) (cofactor := 968017) rfl (by decide) (by decide)
    exact decide_eq_false composite
theorem row_count_10094 : row 10094 = 0 := (row_10094.checked).trans rfl
theorem model_count_10094 : modelRow 10094 = 0 := by decide
theorem modelrow_agrees_10094 : modelRow 10094 = row 10094 := model_count_10094.trans row_count_10094.symm
def row_10095 : RowCertificate 10095 where
  first := false
  first_correct := by
    have composite : ¬ Nat.Prime (corner 10095 1) := by change ¬ Nat.Prime 407656291; exact PrattCertificates.composite_of_product (factor := 7) (cofactor := 58236613) rfl (by decide) (by decide)
    exact decide_eq_false composite
  second := false
  second_correct := by
    have composite : ¬ Nat.Prime (corner 10095 2) := by change ¬ Nat.Prime 407636101; exact PrattCertificates.composite_of_product (factor := 101) (cofactor := 4036001) rfl (by decide) (by decide)
    exact decide_eq_false composite
  third := false
  third_correct := by
    have composite : ¬ Nat.Prime (corner 10095 3) := by change ¬ Nat.Prime 407615911; exact PrattCertificates.composite_of_product (factor := 19) (cofactor := 21453469) rfl (by decide) (by decide)
    exact decide_eq_false composite
theorem row_count_10095 : row 10095 = 0 := (row_10095.checked).trans rfl
theorem model_count_10095 : modelRow 10095 = 0 := by decide
theorem modelrow_agrees_10095 : modelRow 10095 = row 10095 := model_count_10095.trans row_count_10095.symm
def row_10096 : RowCertificate 10096 where
  first := false
  first_correct := by
    have composite : ¬ Nat.Prime (corner 10096 1) := by change ¬ Nat.Prime 407737057; exact PrattCertificates.composite_of_product (factor := 7) (cofactor := 58248151) rfl (by decide) (by decide)
    exact decide_eq_false composite
  second := false
  second_correct := by
    have composite : ¬ Nat.Prime (corner 10096 2) := by change ¬ Nat.Prime 407716865; exact PrattCertificates.composite_of_product (factor := 5) (cofactor := 81543373) rfl (by decide) (by decide)
    exact decide_eq_false composite
  third := false
  third_correct := by
    have composite : ¬ Nat.Prime (corner 10096 3) := by change ¬ Nat.Prime 407696673; exact PrattCertificates.composite_of_product (factor := 3) (cofactor := 135898891) rfl (by decide) (by decide)
    exact decide_eq_false composite
theorem row_count_10096 : row 10096 = 0 := (row_10096.checked).trans rfl
theorem model_count_10096 : modelRow 10096 = 0 := by decide
theorem modelrow_agrees_10096 : modelRow 10096 = row 10096 := model_count_10096.trans row_count_10096.symm
def row_10097 : RowCertificate 10097 where
  first := false
  first_correct := by
    have composite : ¬ Nat.Prime (corner 10097 1) := by change ¬ Nat.Prime 407817831; exact PrattCertificates.composite_of_product (factor := 3) (cofactor := 135939277) rfl (by decide) (by decide)
    exact decide_eq_false composite
  second := false
  second_correct := by
    have composite : ¬ Nat.Prime (corner 10097 2) := by change ¬ Nat.Prime 407797637; exact PrattCertificates.composite_of_product (factor := 13) (cofactor := 31369049) rfl (by decide) (by decide)
    exact decide_eq_false composite
  third := true
  third_correct := by
    have prime : Nat.Prime (corner 10097 3) := by change Nat.Prime 407777443; exact ProjectEulerSolutions.P58PrimeNodes.p_407777443.prime
    exact decide_eq_true prime
theorem row_count_10097 : row 10097 = 1 := (row_10097.checked).trans rfl
theorem model_count_10097 : modelRow 10097 = 1 := by decide
theorem modelrow_agrees_10097 : modelRow 10097 = row 10097 := model_count_10097.trans row_count_10097.symm
def row_10098 : RowCertificate 10098 where
  first := false
  first_correct := by
    have composite : ¬ Nat.Prime (corner 10098 1) := by change ¬ Nat.Prime 407898613; exact PrattCertificates.composite_of_product (factor := 67) (cofactor := 6088039) rfl (by decide) (by decide)
    exact decide_eq_false composite
  second := false
  second_correct := by
    have composite : ¬ Nat.Prime (corner 10098 2) := by change ¬ Nat.Prime 407878417; exact PrattCertificates.composite_of_product (factor := 29) (cofactor := 14064773) rfl (by decide) (by decide)
    exact decide_eq_false composite
  third := true
  third_correct := by
    have prime : Nat.Prime (corner 10098 3) := by change Nat.Prime 407858221; exact ProjectEulerSolutions.P58PrimeNodes.p_407858221.prime
    exact decide_eq_true prime
theorem row_count_10098 : row 10098 = 1 := (row_10098.checked).trans rfl
theorem model_count_10098 : modelRow 10098 = 1 := by decide
theorem modelrow_agrees_10098 : modelRow 10098 = row 10098 := model_count_10098.trans row_count_10098.symm
def row_10099 : RowCertificate 10099 where
  first := false
  first_correct := by
    have composite : ¬ Nat.Prime (corner 10099 1) := by change ¬ Nat.Prime 407979403; exact PrattCertificates.composite_of_product (factor := 13) (cofactor := 31383031) rfl (by decide) (by decide)
    exact decide_eq_false composite
  second := false
  second_correct := by
    have composite : ¬ Nat.Prime (corner 10099 2) := by change ¬ Nat.Prime 407959205; exact PrattCertificates.composite_of_product (factor := 5) (cofactor := 81591841) rfl (by decide) (by decide)
    exact decide_eq_false composite
  third := false
  third_correct := by
    have composite : ¬ Nat.Prime (corner 10099 3) := by change ¬ Nat.Prime 407939007; exact PrattCertificates.composite_of_product (factor := 3) (cofactor := 135979669) rfl (by decide) (by decide)
    exact decide_eq_false composite
theorem row_count_10099 : row 10099 = 0 := (row_10099.checked).trans rfl
theorem model_count_10099 : modelRow 10099 = 0 := by decide
theorem modelrow_agrees_10099 : modelRow 10099 = row 10099 := model_count_10099.trans row_count_10099.symm
def row_10100 : RowCertificate 10100 where
  first := false
  first_correct := by
    have composite : ¬ Nat.Prime (corner 10100 1) := by change ¬ Nat.Prime 408060201; exact PrattCertificates.composite_of_product (factor := 3) (cofactor := 136020067) rfl (by decide) (by decide)
    exact decide_eq_false composite
  second := false
  second_correct := by
    have composite : ¬ Nat.Prime (corner 10100 2) := by change ¬ Nat.Prime 408040001; exact PrattCertificates.composite_of_product (factor := 17) (cofactor := 24002353) rfl (by decide) (by decide)
    exact decide_eq_false composite
  third := false
  third_correct := by
    have composite : ¬ Nat.Prime (corner 10100 3) := by change ¬ Nat.Prime 408019801; exact PrattCertificates.composite_of_product (factor := 7) (cofactor := 58288543) rfl (by decide) (by decide)
    exact decide_eq_false composite
theorem row_count_10100 : row 10100 = 0 := (row_10100.checked).trans rfl
theorem model_count_10100 : modelRow 10100 = 0 := by decide
theorem modelrow_agrees_10100 : modelRow 10100 = row 10100 := model_count_10100.trans row_count_10100.symm
def row_10101 : RowCertificate 10101 where
  first := false
  first_correct := by
    have composite : ¬ Nat.Prime (corner 10101 1) := by change ¬ Nat.Prime 408141007; exact PrattCertificates.composite_of_product (factor := 2713) (cofactor := 150439) rfl (by decide) (by decide)
    exact decide_eq_false composite
  second := false
  second_correct := by
    have composite : ¬ Nat.Prime (corner 10101 2) := by change ¬ Nat.Prime 408120805; exact PrattCertificates.composite_of_product (factor := 5) (cofactor := 81624161) rfl (by decide) (by decide)
    exact decide_eq_false composite
  third := false
  third_correct := by
    have composite : ¬ Nat.Prime (corner 10101 3) := by change ¬ Nat.Prime 408100603; exact PrattCertificates.composite_of_product (factor := 151) (cofactor := 2702653) rfl (by decide) (by decide)
    exact decide_eq_false composite
theorem row_count_10101 : row 10101 = 0 := (row_10101.checked).trans rfl
theorem model_count_10101 : modelRow 10101 = 0 := by decide
theorem modelrow_agrees_10101 : modelRow 10101 = row 10101 := model_count_10101.trans row_count_10101.symm
def row_10102 : RowCertificate 10102 where
  first := false
  first_correct := by
    have composite : ¬ Nat.Prime (corner 10102 1) := by change ¬ Nat.Prime 408221821; exact PrattCertificates.composite_of_product (factor := 7) (cofactor := 58317403) rfl (by decide) (by decide)
    exact decide_eq_false composite
  second := false
  second_correct := by
    have composite : ¬ Nat.Prime (corner 10102 2) := by change ¬ Nat.Prime 408201617; exact PrattCertificates.composite_of_product (factor := 41) (cofactor := 9956137) rfl (by decide) (by decide)
    exact decide_eq_false composite
  third := false
  third_correct := by
    have composite : ¬ Nat.Prime (corner 10102 3) := by change ¬ Nat.Prime 408181413; exact PrattCertificates.composite_of_product (factor := 3) (cofactor := 136060471) rfl (by decide) (by decide)
    exact decide_eq_false composite
theorem row_count_10102 : row 10102 = 0 := (row_10102.checked).trans rfl
theorem model_count_10102 : modelRow 10102 = 0 := by decide
theorem modelrow_agrees_10102 : modelRow 10102 = row 10102 := model_count_10102.trans row_count_10102.symm
def row_10103 : RowCertificate 10103 where
  first := false
  first_correct := by
    have composite : ¬ Nat.Prime (corner 10103 1) := by change ¬ Nat.Prime 408302643; exact PrattCertificates.composite_of_product (factor := 3) (cofactor := 136100881) rfl (by decide) (by decide)
    exact decide_eq_false composite
  second := true
  second_correct := by
    have prime : Nat.Prime (corner 10103 2) := by change Nat.Prime 408282437; exact ProjectEulerSolutions.P58PrimeNodes.p_408282437.prime
    exact decide_eq_true prime
  third := false
  third_correct := by
    have composite : ¬ Nat.Prime (corner 10103 3) := by change ¬ Nat.Prime 408262231; exact PrattCertificates.composite_of_product (factor := 13) (cofactor := 31404787) rfl (by decide) (by decide)
    exact decide_eq_false composite
theorem row_count_10103 : row 10103 = 1 := (row_10103.checked).trans rfl
theorem model_count_10103 : modelRow 10103 = 1 := by decide
theorem modelrow_agrees_10103 : modelRow 10103 = row 10103 := model_count_10103.trans row_count_10103.symm
def row_10104 : RowCertificate 10104 where
  first := false
  first_correct := by
    have composite : ¬ Nat.Prime (corner 10104 1) := by change ¬ Nat.Prime 408383473; exact PrattCertificates.composite_of_product (factor := 19) (cofactor := 21493867) rfl (by decide) (by decide)
    exact decide_eq_false composite
  second := false
  second_correct := by
    have composite : ¬ Nat.Prime (corner 10104 2) := by change ¬ Nat.Prime 408363265; exact PrattCertificates.composite_of_product (factor := 5) (cofactor := 81672653) rfl (by decide) (by decide)
    exact decide_eq_false composite
  third := true
  third_correct := by
    have prime : Nat.Prime (corner 10104 3) := by change Nat.Prime 408343057; exact ProjectEulerSolutions.P58PrimeNodes.p_408343057.prime
    exact decide_eq_true prime
theorem row_count_10104 : row 10104 = 1 := (row_10104.checked).trans rfl
theorem model_count_10104 : modelRow 10104 = 1 := by decide
theorem modelrow_agrees_10104 : modelRow 10104 = row 10104 := model_count_10104.trans row_count_10104.symm
def row_10105 : RowCertificate 10105 where
  first := false
  first_correct := by
    have composite : ¬ Nat.Prime (corner 10105 1) := by change ¬ Nat.Prime 408464311; exact PrattCertificates.composite_of_product (factor := 109) (cofactor := 3747379) rfl (by decide) (by decide)
    exact decide_eq_false composite
  second := false
  second_correct := by
    have composite : ¬ Nat.Prime (corner 10105 2) := by change ¬ Nat.Prime 408444101; exact PrattCertificates.composite_of_product (factor := 13) (cofactor := 31418777) rfl (by decide) (by decide)
    exact decide_eq_false composite
  third := false
  third_correct := by
    have composite : ¬ Nat.Prime (corner 10105 3) := by change ¬ Nat.Prime 408423891; exact PrattCertificates.composite_of_product (factor := 3) (cofactor := 136141297) rfl (by decide) (by decide)
    exact decide_eq_false composite
theorem row_count_10105 : row 10105 = 0 := (row_10105.checked).trans rfl
theorem model_count_10105 : modelRow 10105 = 0 := by decide
theorem modelrow_agrees_10105 : modelRow 10105 = row 10105 := model_count_10105.trans row_count_10105.symm
def row_10106 : RowCertificate 10106 where
  first := false
  first_correct := by
    have composite : ¬ Nat.Prime (corner 10106 1) := by change ¬ Nat.Prime 408545157; exact PrattCertificates.composite_of_product (factor := 3) (cofactor := 136181719) rfl (by decide) (by decide)
    exact decide_eq_false composite
  second := false
  second_correct := by
    have composite : ¬ Nat.Prime (corner 10106 2) := by change ¬ Nat.Prime 408524945; exact PrattCertificates.composite_of_product (factor := 5) (cofactor := 81704989) rfl (by decide) (by decide)
    exact decide_eq_false composite
  third := false
  third_correct := by
    have composite : ¬ Nat.Prime (corner 10106 3) := by change ¬ Nat.Prime 408504733; exact PrattCertificates.composite_of_product (factor := 7) (cofactor := 58357819) rfl (by decide) (by decide)
    exact decide_eq_false composite
theorem row_count_10106 : row 10106 = 0 := (row_10106.checked).trans rfl
theorem model_count_10106 : modelRow 10106 = 0 := by decide
theorem modelrow_agrees_10106 : modelRow 10106 = row 10106 := model_count_10106.trans row_count_10106.symm
def row_10107 : RowCertificate 10107 where
  first := true
  first_correct := by
    have prime : Nat.Prime (corner 10107 1) := by change Nat.Prime 408626011; exact ProjectEulerSolutions.P58PrimeNodes.p_408626011.prime
    exact decide_eq_true prime
  second := true
  second_correct := by
    have prime : Nat.Prime (corner 10107 2) := by change Nat.Prime 408605797; exact ProjectEulerSolutions.P58PrimeNodes.p_408605797.prime
    exact decide_eq_true prime
  third := false
  third_correct := by
    have composite : ¬ Nat.Prime (corner 10107 3) := by change ¬ Nat.Prime 408585583; exact PrattCertificates.composite_of_product (factor := 7) (cofactor := 58369369) rfl (by decide) (by decide)
    exact decide_eq_false composite
theorem row_count_10107 : row 10107 = 2 := (row_10107.checked).trans rfl
theorem model_count_10107 : modelRow 10107 = 2 := by decide
theorem modelrow_agrees_10107 : modelRow 10107 = row 10107 := model_count_10107.trans row_count_10107.symm
def row_10108 : RowCertificate 10108 where
  first := false
  first_correct := by
    have composite : ¬ Nat.Prime (corner 10108 1) := by change ¬ Nat.Prime 408706873; exact PrattCertificates.composite_of_product (factor := 43) (cofactor := 9504811) rfl (by decide) (by decide)
    exact decide_eq_false composite
  second := false
  second_correct := by
    have composite : ¬ Nat.Prime (corner 10108 2) := by change ¬ Nat.Prime 408686657; exact PrattCertificates.composite_of_product (factor := 53) (cofactor := 7711069) rfl (by decide) (by decide)
    exact decide_eq_false composite
  third := false
  third_correct := by
    have composite : ¬ Nat.Prime (corner 10108 3) := by change ¬ Nat.Prime 408666441; exact PrattCertificates.composite_of_product (factor := 3) (cofactor := 136222147) rfl (by decide) (by decide)
    exact decide_eq_false composite
theorem row_count_10108 : row 10108 = 0 := (row_10108.checked).trans rfl
theorem model_count_10108 : modelRow 10108 = 0 := by decide
theorem modelrow_agrees_10108 : modelRow 10108 = row 10108 := model_count_10108.trans row_count_10108.symm
def row_10109 : RowCertificate 10109 where
  first := false
  first_correct := by
    have composite : ¬ Nat.Prime (corner 10109 1) := by change ¬ Nat.Prime 408787743; exact PrattCertificates.composite_of_product (factor := 3) (cofactor := 136262581) rfl (by decide) (by decide)
    exact decide_eq_false composite
  second := false
  second_correct := by
    have composite : ¬ Nat.Prime (corner 10109 2) := by change ¬ Nat.Prime 408767525; exact PrattCertificates.composite_of_product (factor := 5) (cofactor := 81753505) rfl (by decide) (by decide)
    exact decide_eq_false composite
  third := false
  third_correct := by
    have composite : ¬ Nat.Prime (corner 10109 3) := by change ¬ Nat.Prime 408747307; exact PrattCertificates.composite_of_product (factor := 31) (cofactor := 13185397) rfl (by decide) (by decide)
    exact decide_eq_false composite
theorem row_count_10109 : row 10109 = 0 := (row_10109.checked).trans rfl
theorem model_count_10109 : modelRow 10109 = 0 := by decide
theorem modelrow_agrees_10109 : modelRow 10109 = row 10109 := model_count_10109.trans row_count_10109.symm
def row_10110 : RowCertificate 10110 where
  first := false
  first_correct := by
    have composite : ¬ Nat.Prime (corner 10110 1) := by change ¬ Nat.Prime 408868621; exact PrattCertificates.composite_of_product (factor := 7) (cofactor := 58409803) rfl (by decide) (by decide)
    exact decide_eq_false composite
  second := false
  second_correct := by
    have composite : ¬ Nat.Prime (corner 10110 2) := by change ¬ Nat.Prime 408848401; exact PrattCertificates.composite_of_product (factor := 13) (cofactor := 31449877) rfl (by decide) (by decide)
    exact decide_eq_false composite
  third := true
  third_correct := by
    have prime : Nat.Prime (corner 10110 3) := by change Nat.Prime 408828181; exact ProjectEulerSolutions.P58PrimeNodes.p_408828181.prime
    exact decide_eq_true prime
theorem row_count_10110 : row 10110 = 1 := (row_10110.checked).trans rfl
theorem model_count_10110 : modelRow 10110 = 1 := by decide
theorem modelrow_agrees_10110 : modelRow 10110 = row 10110 := model_count_10110.trans row_count_10110.symm
def row_10111 : RowCertificate 10111 where
  first := false
  first_correct := by
    have composite : ¬ Nat.Prime (corner 10111 1) := by change ¬ Nat.Prime 408949507; exact PrattCertificates.composite_of_product (factor := 331) (cofactor := 1235497) rfl (by decide) (by decide)
    exact decide_eq_false composite
  second := false
  second_correct := by
    have composite : ¬ Nat.Prime (corner 10111 2) := by change ¬ Nat.Prime 408929285; exact PrattCertificates.composite_of_product (factor := 5) (cofactor := 81785857) rfl (by decide) (by decide)
    exact decide_eq_false composite
  third := false
  third_correct := by
    have composite : ¬ Nat.Prime (corner 10111 3) := by change ¬ Nat.Prime 408909063; exact PrattCertificates.composite_of_product (factor := 3) (cofactor := 136303021) rfl (by decide) (by decide)
    exact decide_eq_false composite
theorem row_count_10111 : row 10111 = 0 := (row_10111.checked).trans rfl
theorem model_count_10111 : modelRow 10111 = 0 := by decide
theorem modelrow_agrees_10111 : modelRow 10111 = row 10111 := model_count_10111.trans row_count_10111.symm
def row_10112 : RowCertificate 10112 where
  first := false
  first_correct := by
    have composite : ¬ Nat.Prime (corner 10112 1) := by change ¬ Nat.Prime 409030401; exact PrattCertificates.composite_of_product (factor := 3) (cofactor := 136343467) rfl (by decide) (by decide)
    exact decide_eq_false composite
  second := false
  second_correct := by
    have composite : ¬ Nat.Prime (corner 10112 2) := by change ¬ Nat.Prime 409010177; exact PrattCertificates.composite_of_product (factor := 1693) (cofactor := 241589) rfl (by decide) (by decide)
    exact decide_eq_false composite
  third := false
  third_correct := by
    have composite : ¬ Nat.Prime (corner 10112 3) := by change ¬ Nat.Prime 408989953; exact PrattCertificates.composite_of_product (factor := 19) (cofactor := 21525787) rfl (by decide) (by decide)
    exact decide_eq_false composite
theorem row_count_10112 : row 10112 = 0 := (row_10112.checked).trans rfl
theorem model_count_10112 : modelRow 10112 = 0 := by decide
theorem modelrow_agrees_10112 : modelRow 10112 = row 10112 := model_count_10112.trans row_count_10112.symm
private def atomic_9985 : CountSegment 1 10 13120 9984 9985 4129 4130 := CountSegment.single (first := 9984) (initial := 4129) row_count_9985 (by decide)
private theorem atomic_9985_agreement : RowAgreement 9984 9985 := RowAgreement.single (first := 9984) modelrow_agrees_9985
private def atomic_9986 : CountSegment 1 10 13120 9985 9986 4130 4130 := CountSegment.single (first := 9985) (initial := 4130) row_count_9986 (by decide)
private theorem atomic_9986_agreement : RowAgreement 9985 9986 := RowAgreement.single (first := 9985) modelrow_agrees_9986
private def atomic_9987 : CountSegment 1 10 13120 9986 9987 4130 4130 := CountSegment.single (first := 9986) (initial := 4130) row_count_9987 (by decide)
private theorem atomic_9987_agreement : RowAgreement 9986 9987 := RowAgreement.single (first := 9986) modelrow_agrees_9987
private def atomic_9988 : CountSegment 1 10 13120 9987 9988 4130 4130 := CountSegment.single (first := 9987) (initial := 4130) row_count_9988 (by decide)
private theorem atomic_9988_agreement : RowAgreement 9987 9988 := RowAgreement.single (first := 9987) modelrow_agrees_9988
private def atomic_9989 : CountSegment 1 10 13120 9988 9989 4130 4130 := CountSegment.single (first := 9988) (initial := 4130) row_count_9989 (by decide)
private theorem atomic_9989_agreement : RowAgreement 9988 9989 := RowAgreement.single (first := 9988) modelrow_agrees_9989
private def atomic_9990 : CountSegment 1 10 13120 9989 9990 4130 4131 := CountSegment.single (first := 9989) (initial := 4130) row_count_9990 (by decide)
private theorem atomic_9990_agreement : RowAgreement 9989 9990 := RowAgreement.single (first := 9989) modelrow_agrees_9990
private def atomic_9991 : CountSegment 1 10 13120 9990 9991 4131 4131 := CountSegment.single (first := 9990) (initial := 4131) row_count_9991 (by decide)
private theorem atomic_9991_agreement : RowAgreement 9990 9991 := RowAgreement.single (first := 9990) modelrow_agrees_9991
private def atomic_9992 : CountSegment 1 10 13120 9991 9992 4131 4133 := CountSegment.single (first := 9991) (initial := 4131) row_count_9992 (by decide)
private theorem atomic_9992_agreement : RowAgreement 9991 9992 := RowAgreement.single (first := 9991) modelrow_agrees_9992
private def atomic_9993 : CountSegment 1 10 13120 9992 9993 4133 4134 := CountSegment.single (first := 9992) (initial := 4133) row_count_9993 (by decide)
private theorem atomic_9993_agreement : RowAgreement 9992 9993 := RowAgreement.single (first := 9992) modelrow_agrees_9993
private def atomic_9994 : CountSegment 1 10 13120 9993 9994 4134 4134 := CountSegment.single (first := 9993) (initial := 4134) row_count_9994 (by decide)
private theorem atomic_9994_agreement : RowAgreement 9993 9994 := RowAgreement.single (first := 9993) modelrow_agrees_9994
private def atomic_9995 : CountSegment 1 10 13120 9994 9995 4134 4135 := CountSegment.single (first := 9994) (initial := 4134) row_count_9995 (by decide)
private theorem atomic_9995_agreement : RowAgreement 9994 9995 := RowAgreement.single (first := 9994) modelrow_agrees_9995
private def atomic_9996 : CountSegment 1 10 13120 9995 9996 4135 4136 := CountSegment.single (first := 9995) (initial := 4135) row_count_9996 (by decide)
private theorem atomic_9996_agreement : RowAgreement 9995 9996 := RowAgreement.single (first := 9995) modelrow_agrees_9996
private def atomic_9997 : CountSegment 1 10 13120 9996 9997 4136 4137 := CountSegment.single (first := 9996) (initial := 4136) row_count_9997 (by decide)
private theorem atomic_9997_agreement : RowAgreement 9996 9997 := RowAgreement.single (first := 9996) modelrow_agrees_9997
private def atomic_9998 : CountSegment 1 10 13120 9997 9998 4137 4137 := CountSegment.single (first := 9997) (initial := 4137) row_count_9998 (by decide)
private theorem atomic_9998_agreement : RowAgreement 9997 9998 := RowAgreement.single (first := 9997) modelrow_agrees_9998
private def atomic_9999 : CountSegment 1 10 13120 9998 9999 4137 4138 := CountSegment.single (first := 9998) (initial := 4137) row_count_9999 (by decide)
private theorem atomic_9999_agreement : RowAgreement 9998 9999 := RowAgreement.single (first := 9998) modelrow_agrees_9999
private def atomic_10000 : CountSegment 1 10 13120 9999 10000 4138 4138 := CountSegment.single (first := 9999) (initial := 4138) row_count_10000 (by decide)
private theorem atomic_10000_agreement : RowAgreement 9999 10000 := RowAgreement.single (first := 9999) modelrow_agrees_10000
private def atomic_10001 : CountSegment 1 10 13120 10000 10001 4138 4138 := CountSegment.single (first := 10000) (initial := 4138) row_count_10001 (by decide)
private theorem atomic_10001_agreement : RowAgreement 10000 10001 := RowAgreement.single (first := 10000) modelrow_agrees_10001
private def atomic_10002 : CountSegment 1 10 13120 10001 10002 4138 4138 := CountSegment.single (first := 10001) (initial := 4138) row_count_10002 (by decide)
private theorem atomic_10002_agreement : RowAgreement 10001 10002 := RowAgreement.single (first := 10001) modelrow_agrees_10002
private def atomic_10003 : CountSegment 1 10 13120 10002 10003 4138 4138 := CountSegment.single (first := 10002) (initial := 4138) row_count_10003 (by decide)
private theorem atomic_10003_agreement : RowAgreement 10002 10003 := RowAgreement.single (first := 10002) modelrow_agrees_10003
private def atomic_10004 : CountSegment 1 10 13120 10003 10004 4138 4139 := CountSegment.single (first := 10003) (initial := 4138) row_count_10004 (by decide)
private theorem atomic_10004_agreement : RowAgreement 10003 10004 := RowAgreement.single (first := 10003) modelrow_agrees_10004
private def atomic_10005 : CountSegment 1 10 13120 10004 10005 4139 4139 := CountSegment.single (first := 10004) (initial := 4139) row_count_10005 (by decide)
private theorem atomic_10005_agreement : RowAgreement 10004 10005 := RowAgreement.single (first := 10004) modelrow_agrees_10005
private def atomic_10006 : CountSegment 1 10 13120 10005 10006 4139 4140 := CountSegment.single (first := 10005) (initial := 4139) row_count_10006 (by decide)
private theorem atomic_10006_agreement : RowAgreement 10005 10006 := RowAgreement.single (first := 10005) modelrow_agrees_10006
private def atomic_10007 : CountSegment 1 10 13120 10006 10007 4140 4142 := CountSegment.single (first := 10006) (initial := 4140) row_count_10007 (by decide)
private theorem atomic_10007_agreement : RowAgreement 10006 10007 := RowAgreement.single (first := 10006) modelrow_agrees_10007
private def atomic_10008 : CountSegment 1 10 13120 10007 10008 4142 4142 := CountSegment.single (first := 10007) (initial := 4142) row_count_10008 (by decide)
private theorem atomic_10008_agreement : RowAgreement 10007 10008 := RowAgreement.single (first := 10007) modelrow_agrees_10008
private def atomic_10009 : CountSegment 1 10 13120 10008 10009 4142 4142 := CountSegment.single (first := 10008) (initial := 4142) row_count_10009 (by decide)
private theorem atomic_10009_agreement : RowAgreement 10008 10009 := RowAgreement.single (first := 10008) modelrow_agrees_10009
private def atomic_10010 : CountSegment 1 10 13120 10009 10010 4142 4142 := CountSegment.single (first := 10009) (initial := 4142) row_count_10010 (by decide)
private theorem atomic_10010_agreement : RowAgreement 10009 10010 := RowAgreement.single (first := 10009) modelrow_agrees_10010
private def atomic_10011 : CountSegment 1 10 13120 10010 10011 4142 4142 := CountSegment.single (first := 10010) (initial := 4142) row_count_10011 (by decide)
private theorem atomic_10011_agreement : RowAgreement 10010 10011 := RowAgreement.single (first := 10010) modelrow_agrees_10011
private def atomic_10012 : CountSegment 1 10 13120 10011 10012 4142 4142 := CountSegment.single (first := 10011) (initial := 4142) row_count_10012 (by decide)
private theorem atomic_10012_agreement : RowAgreement 10011 10012 := RowAgreement.single (first := 10011) modelrow_agrees_10012
private def atomic_10013 : CountSegment 1 10 13120 10012 10013 4142 4143 := CountSegment.single (first := 10012) (initial := 4142) row_count_10013 (by decide)
private theorem atomic_10013_agreement : RowAgreement 10012 10013 := RowAgreement.single (first := 10012) modelrow_agrees_10013
private def atomic_10014 : CountSegment 1 10 13120 10013 10014 4143 4143 := CountSegment.single (first := 10013) (initial := 4143) row_count_10014 (by decide)
private theorem atomic_10014_agreement : RowAgreement 10013 10014 := RowAgreement.single (first := 10013) modelrow_agrees_10014
private def atomic_10015 : CountSegment 1 10 13120 10014 10015 4143 4143 := CountSegment.single (first := 10014) (initial := 4143) row_count_10015 (by decide)
private theorem atomic_10015_agreement : RowAgreement 10014 10015 := RowAgreement.single (first := 10014) modelrow_agrees_10015
private def atomic_10016 : CountSegment 1 10 13120 10015 10016 4143 4143 := CountSegment.single (first := 10015) (initial := 4143) row_count_10016 (by decide)
private theorem atomic_10016_agreement : RowAgreement 10015 10016 := RowAgreement.single (first := 10015) modelrow_agrees_10016
private def atomic_10017 : CountSegment 1 10 13120 10016 10017 4143 4143 := CountSegment.single (first := 10016) (initial := 4143) row_count_10017 (by decide)
private theorem atomic_10017_agreement : RowAgreement 10016 10017 := RowAgreement.single (first := 10016) modelrow_agrees_10017
private def atomic_10018 : CountSegment 1 10 13120 10017 10018 4143 4144 := CountSegment.single (first := 10017) (initial := 4143) row_count_10018 (by decide)
private theorem atomic_10018_agreement : RowAgreement 10017 10018 := RowAgreement.single (first := 10017) modelrow_agrees_10018
private def atomic_10019 : CountSegment 1 10 13120 10018 10019 4144 4144 := CountSegment.single (first := 10018) (initial := 4144) row_count_10019 (by decide)
private theorem atomic_10019_agreement : RowAgreement 10018 10019 := RowAgreement.single (first := 10018) modelrow_agrees_10019
private def atomic_10020 : CountSegment 1 10 13120 10019 10020 4144 4146 := CountSegment.single (first := 10019) (initial := 4144) row_count_10020 (by decide)
private theorem atomic_10020_agreement : RowAgreement 10019 10020 := RowAgreement.single (first := 10019) modelrow_agrees_10020
private def atomic_10021 : CountSegment 1 10 13120 10020 10021 4146 4146 := CountSegment.single (first := 10020) (initial := 4146) row_count_10021 (by decide)
private theorem atomic_10021_agreement : RowAgreement 10020 10021 := RowAgreement.single (first := 10020) modelrow_agrees_10021
private def atomic_10022 : CountSegment 1 10 13120 10021 10022 4146 4146 := CountSegment.single (first := 10021) (initial := 4146) row_count_10022 (by decide)
private theorem atomic_10022_agreement : RowAgreement 10021 10022 := RowAgreement.single (first := 10021) modelrow_agrees_10022
private def atomic_10023 : CountSegment 1 10 13120 10022 10023 4146 4146 := CountSegment.single (first := 10022) (initial := 4146) row_count_10023 (by decide)
private theorem atomic_10023_agreement : RowAgreement 10022 10023 := RowAgreement.single (first := 10022) modelrow_agrees_10023
private def atomic_10024 : CountSegment 1 10 13120 10023 10024 4146 4146 := CountSegment.single (first := 10023) (initial := 4146) row_count_10024 (by decide)
private theorem atomic_10024_agreement : RowAgreement 10023 10024 := RowAgreement.single (first := 10023) modelrow_agrees_10024
private def atomic_10025 : CountSegment 1 10 13120 10024 10025 4146 4146 := CountSegment.single (first := 10024) (initial := 4146) row_count_10025 (by decide)
private theorem atomic_10025_agreement : RowAgreement 10024 10025 := RowAgreement.single (first := 10024) modelrow_agrees_10025
private def atomic_10026 : CountSegment 1 10 13120 10025 10026 4146 4146 := CountSegment.single (first := 10025) (initial := 4146) row_count_10026 (by decide)
private theorem atomic_10026_agreement : RowAgreement 10025 10026 := RowAgreement.single (first := 10025) modelrow_agrees_10026
private def atomic_10027 : CountSegment 1 10 13120 10026 10027 4146 4146 := CountSegment.single (first := 10026) (initial := 4146) row_count_10027 (by decide)
private theorem atomic_10027_agreement : RowAgreement 10026 10027 := RowAgreement.single (first := 10026) modelrow_agrees_10027
private def atomic_10028 : CountSegment 1 10 13120 10027 10028 4146 4146 := CountSegment.single (first := 10027) (initial := 4146) row_count_10028 (by decide)
private theorem atomic_10028_agreement : RowAgreement 10027 10028 := RowAgreement.single (first := 10027) modelrow_agrees_10028
private def atomic_10029 : CountSegment 1 10 13120 10028 10029 4146 4146 := CountSegment.single (first := 10028) (initial := 4146) row_count_10029 (by decide)
private theorem atomic_10029_agreement : RowAgreement 10028 10029 := RowAgreement.single (first := 10028) modelrow_agrees_10029
private def atomic_10030 : CountSegment 1 10 13120 10029 10030 4146 4146 := CountSegment.single (first := 10029) (initial := 4146) row_count_10030 (by decide)
private theorem atomic_10030_agreement : RowAgreement 10029 10030 := RowAgreement.single (first := 10029) modelrow_agrees_10030
private def atomic_10031 : CountSegment 1 10 13120 10030 10031 4146 4146 := CountSegment.single (first := 10030) (initial := 4146) row_count_10031 (by decide)
private theorem atomic_10031_agreement : RowAgreement 10030 10031 := RowAgreement.single (first := 10030) modelrow_agrees_10031
private def atomic_10032 : CountSegment 1 10 13120 10031 10032 4146 4146 := CountSegment.single (first := 10031) (initial := 4146) row_count_10032 (by decide)
private theorem atomic_10032_agreement : RowAgreement 10031 10032 := RowAgreement.single (first := 10031) modelrow_agrees_10032
private def atomic_10033 : CountSegment 1 10 13120 10032 10033 4146 4147 := CountSegment.single (first := 10032) (initial := 4146) row_count_10033 (by decide)
private theorem atomic_10033_agreement : RowAgreement 10032 10033 := RowAgreement.single (first := 10032) modelrow_agrees_10033
private def atomic_10034 : CountSegment 1 10 13120 10033 10034 4147 4148 := CountSegment.single (first := 10033) (initial := 4147) row_count_10034 (by decide)
private theorem atomic_10034_agreement : RowAgreement 10033 10034 := RowAgreement.single (first := 10033) modelrow_agrees_10034
private def atomic_10035 : CountSegment 1 10 13120 10034 10035 4148 4149 := CountSegment.single (first := 10034) (initial := 4148) row_count_10035 (by decide)
private theorem atomic_10035_agreement : RowAgreement 10034 10035 := RowAgreement.single (first := 10034) modelrow_agrees_10035
private def atomic_10036 : CountSegment 1 10 13120 10035 10036 4149 4149 := CountSegment.single (first := 10035) (initial := 4149) row_count_10036 (by decide)
private theorem atomic_10036_agreement : RowAgreement 10035 10036 := RowAgreement.single (first := 10035) modelrow_agrees_10036
private def atomic_10037 : CountSegment 1 10 13120 10036 10037 4149 4149 := CountSegment.single (first := 10036) (initial := 4149) row_count_10037 (by decide)
private theorem atomic_10037_agreement : RowAgreement 10036 10037 := RowAgreement.single (first := 10036) modelrow_agrees_10037
private def atomic_10038 : CountSegment 1 10 13120 10037 10038 4149 4151 := CountSegment.single (first := 10037) (initial := 4149) row_count_10038 (by decide)
private theorem atomic_10038_agreement : RowAgreement 10037 10038 := RowAgreement.single (first := 10037) modelrow_agrees_10038
private def atomic_10039 : CountSegment 1 10 13120 10038 10039 4151 4151 := CountSegment.single (first := 10038) (initial := 4151) row_count_10039 (by decide)
private theorem atomic_10039_agreement : RowAgreement 10038 10039 := RowAgreement.single (first := 10038) modelrow_agrees_10039
private def atomic_10040 : CountSegment 1 10 13120 10039 10040 4151 4151 := CountSegment.single (first := 10039) (initial := 4151) row_count_10040 (by decide)
private theorem atomic_10040_agreement : RowAgreement 10039 10040 := RowAgreement.single (first := 10039) modelrow_agrees_10040
private def atomic_10041 : CountSegment 1 10 13120 10040 10041 4151 4151 := CountSegment.single (first := 10040) (initial := 4151) row_count_10041 (by decide)
private theorem atomic_10041_agreement : RowAgreement 10040 10041 := RowAgreement.single (first := 10040) modelrow_agrees_10041
private def atomic_10042 : CountSegment 1 10 13120 10041 10042 4151 4151 := CountSegment.single (first := 10041) (initial := 4151) row_count_10042 (by decide)
private theorem atomic_10042_agreement : RowAgreement 10041 10042 := RowAgreement.single (first := 10041) modelrow_agrees_10042
private def atomic_10043 : CountSegment 1 10 13120 10042 10043 4151 4152 := CountSegment.single (first := 10042) (initial := 4151) row_count_10043 (by decide)
private theorem atomic_10043_agreement : RowAgreement 10042 10043 := RowAgreement.single (first := 10042) modelrow_agrees_10043
private def atomic_10044 : CountSegment 1 10 13120 10043 10044 4152 4152 := CountSegment.single (first := 10043) (initial := 4152) row_count_10044 (by decide)
private theorem atomic_10044_agreement : RowAgreement 10043 10044 := RowAgreement.single (first := 10043) modelrow_agrees_10044
private def atomic_10045 : CountSegment 1 10 13120 10044 10045 4152 4152 := CountSegment.single (first := 10044) (initial := 4152) row_count_10045 (by decide)
private theorem atomic_10045_agreement : RowAgreement 10044 10045 := RowAgreement.single (first := 10044) modelrow_agrees_10045
private def atomic_10046 : CountSegment 1 10 13120 10045 10046 4152 4152 := CountSegment.single (first := 10045) (initial := 4152) row_count_10046 (by decide)
private theorem atomic_10046_agreement : RowAgreement 10045 10046 := RowAgreement.single (first := 10045) modelrow_agrees_10046
private def atomic_10047 : CountSegment 1 10 13120 10046 10047 4152 4152 := CountSegment.single (first := 10046) (initial := 4152) row_count_10047 (by decide)
private theorem atomic_10047_agreement : RowAgreement 10046 10047 := RowAgreement.single (first := 10046) modelrow_agrees_10047
private def atomic_10048 : CountSegment 1 10 13120 10047 10048 4152 4154 := CountSegment.single (first := 10047) (initial := 4152) row_count_10048 (by decide)
private theorem atomic_10048_agreement : RowAgreement 10047 10048 := RowAgreement.single (first := 10047) modelrow_agrees_10048
private def atomic_10049 : CountSegment 1 10 13120 10048 10049 4154 4155 := CountSegment.single (first := 10048) (initial := 4154) row_count_10049 (by decide)
private theorem atomic_10049_agreement : RowAgreement 10048 10049 := RowAgreement.single (first := 10048) modelrow_agrees_10049
private def atomic_10050 : CountSegment 1 10 13120 10049 10050 4155 4156 := CountSegment.single (first := 10049) (initial := 4155) row_count_10050 (by decide)
private theorem atomic_10050_agreement : RowAgreement 10049 10050 := RowAgreement.single (first := 10049) modelrow_agrees_10050
private def atomic_10051 : CountSegment 1 10 13120 10050 10051 4156 4157 := CountSegment.single (first := 10050) (initial := 4156) row_count_10051 (by decide)
private theorem atomic_10051_agreement : RowAgreement 10050 10051 := RowAgreement.single (first := 10050) modelrow_agrees_10051
private def atomic_10052 : CountSegment 1 10 13120 10051 10052 4157 4158 := CountSegment.single (first := 10051) (initial := 4157) row_count_10052 (by decide)
private theorem atomic_10052_agreement : RowAgreement 10051 10052 := RowAgreement.single (first := 10051) modelrow_agrees_10052
private def atomic_10053 : CountSegment 1 10 13120 10052 10053 4158 4159 := CountSegment.single (first := 10052) (initial := 4158) row_count_10053 (by decide)
private theorem atomic_10053_agreement : RowAgreement 10052 10053 := RowAgreement.single (first := 10052) modelrow_agrees_10053
private def atomic_10054 : CountSegment 1 10 13120 10053 10054 4159 4159 := CountSegment.single (first := 10053) (initial := 4159) row_count_10054 (by decide)
private theorem atomic_10054_agreement : RowAgreement 10053 10054 := RowAgreement.single (first := 10053) modelrow_agrees_10054
private def atomic_10055 : CountSegment 1 10 13120 10054 10055 4159 4159 := CountSegment.single (first := 10054) (initial := 4159) row_count_10055 (by decide)
private theorem atomic_10055_agreement : RowAgreement 10054 10055 := RowAgreement.single (first := 10054) modelrow_agrees_10055
private def atomic_10056 : CountSegment 1 10 13120 10055 10056 4159 4160 := CountSegment.single (first := 10055) (initial := 4159) row_count_10056 (by decide)
private theorem atomic_10056_agreement : RowAgreement 10055 10056 := RowAgreement.single (first := 10055) modelrow_agrees_10056
private def atomic_10057 : CountSegment 1 10 13120 10056 10057 4160 4160 := CountSegment.single (first := 10056) (initial := 4160) row_count_10057 (by decide)
private theorem atomic_10057_agreement : RowAgreement 10056 10057 := RowAgreement.single (first := 10056) modelrow_agrees_10057
private def atomic_10058 : CountSegment 1 10 13120 10057 10058 4160 4160 := CountSegment.single (first := 10057) (initial := 4160) row_count_10058 (by decide)
private theorem atomic_10058_agreement : RowAgreement 10057 10058 := RowAgreement.single (first := 10057) modelrow_agrees_10058
private def atomic_10059 : CountSegment 1 10 13120 10058 10059 4160 4161 := CountSegment.single (first := 10058) (initial := 4160) row_count_10059 (by decide)
private theorem atomic_10059_agreement : RowAgreement 10058 10059 := RowAgreement.single (first := 10058) modelrow_agrees_10059
private def atomic_10060 : CountSegment 1 10 13120 10059 10060 4161 4162 := CountSegment.single (first := 10059) (initial := 4161) row_count_10060 (by decide)
private theorem atomic_10060_agreement : RowAgreement 10059 10060 := RowAgreement.single (first := 10059) modelrow_agrees_10060
private def atomic_10061 : CountSegment 1 10 13120 10060 10061 4162 4162 := CountSegment.single (first := 10060) (initial := 4162) row_count_10061 (by decide)
private theorem atomic_10061_agreement : RowAgreement 10060 10061 := RowAgreement.single (first := 10060) modelrow_agrees_10061
private def atomic_10062 : CountSegment 1 10 13120 10061 10062 4162 4163 := CountSegment.single (first := 10061) (initial := 4162) row_count_10062 (by decide)
private theorem atomic_10062_agreement : RowAgreement 10061 10062 := RowAgreement.single (first := 10061) modelrow_agrees_10062
private def atomic_10063 : CountSegment 1 10 13120 10062 10063 4163 4163 := CountSegment.single (first := 10062) (initial := 4163) row_count_10063 (by decide)
private theorem atomic_10063_agreement : RowAgreement 10062 10063 := RowAgreement.single (first := 10062) modelrow_agrees_10063
private def atomic_10064 : CountSegment 1 10 13120 10063 10064 4163 4163 := CountSegment.single (first := 10063) (initial := 4163) row_count_10064 (by decide)
private theorem atomic_10064_agreement : RowAgreement 10063 10064 := RowAgreement.single (first := 10063) modelrow_agrees_10064
private def atomic_10065 : CountSegment 1 10 13120 10064 10065 4163 4163 := CountSegment.single (first := 10064) (initial := 4163) row_count_10065 (by decide)
private theorem atomic_10065_agreement : RowAgreement 10064 10065 := RowAgreement.single (first := 10064) modelrow_agrees_10065
private def atomic_10066 : CountSegment 1 10 13120 10065 10066 4163 4163 := CountSegment.single (first := 10065) (initial := 4163) row_count_10066 (by decide)
private theorem atomic_10066_agreement : RowAgreement 10065 10066 := RowAgreement.single (first := 10065) modelrow_agrees_10066
private def atomic_10067 : CountSegment 1 10 13120 10066 10067 4163 4163 := CountSegment.single (first := 10066) (initial := 4163) row_count_10067 (by decide)
private theorem atomic_10067_agreement : RowAgreement 10066 10067 := RowAgreement.single (first := 10066) modelrow_agrees_10067
private def atomic_10068 : CountSegment 1 10 13120 10067 10068 4163 4165 := CountSegment.single (first := 10067) (initial := 4163) row_count_10068 (by decide)
private theorem atomic_10068_agreement : RowAgreement 10067 10068 := RowAgreement.single (first := 10067) modelrow_agrees_10068
private def atomic_10069 : CountSegment 1 10 13120 10068 10069 4165 4165 := CountSegment.single (first := 10068) (initial := 4165) row_count_10069 (by decide)
private theorem atomic_10069_agreement : RowAgreement 10068 10069 := RowAgreement.single (first := 10068) modelrow_agrees_10069
private def atomic_10070 : CountSegment 1 10 13120 10069 10070 4165 4165 := CountSegment.single (first := 10069) (initial := 4165) row_count_10070 (by decide)
private theorem atomic_10070_agreement : RowAgreement 10069 10070 := RowAgreement.single (first := 10069) modelrow_agrees_10070
private def atomic_10071 : CountSegment 1 10 13120 10070 10071 4165 4166 := CountSegment.single (first := 10070) (initial := 4165) row_count_10071 (by decide)
private theorem atomic_10071_agreement : RowAgreement 10070 10071 := RowAgreement.single (first := 10070) modelrow_agrees_10071
private def atomic_10072 : CountSegment 1 10 13120 10071 10072 4166 4166 := CountSegment.single (first := 10071) (initial := 4166) row_count_10072 (by decide)
private theorem atomic_10072_agreement : RowAgreement 10071 10072 := RowAgreement.single (first := 10071) modelrow_agrees_10072
private def atomic_10073 : CountSegment 1 10 13120 10072 10073 4166 4168 := CountSegment.single (first := 10072) (initial := 4166) row_count_10073 (by decide)
private theorem atomic_10073_agreement : RowAgreement 10072 10073 := RowAgreement.single (first := 10072) modelrow_agrees_10073
private def atomic_10074 : CountSegment 1 10 13120 10073 10074 4168 4168 := CountSegment.single (first := 10073) (initial := 4168) row_count_10074 (by decide)
private theorem atomic_10074_agreement : RowAgreement 10073 10074 := RowAgreement.single (first := 10073) modelrow_agrees_10074
private def atomic_10075 : CountSegment 1 10 13120 10074 10075 4168 4168 := CountSegment.single (first := 10074) (initial := 4168) row_count_10075 (by decide)
private theorem atomic_10075_agreement : RowAgreement 10074 10075 := RowAgreement.single (first := 10074) modelrow_agrees_10075
private def atomic_10076 : CountSegment 1 10 13120 10075 10076 4168 4168 := CountSegment.single (first := 10075) (initial := 4168) row_count_10076 (by decide)
private theorem atomic_10076_agreement : RowAgreement 10075 10076 := RowAgreement.single (first := 10075) modelrow_agrees_10076
private def atomic_10077 : CountSegment 1 10 13120 10076 10077 4168 4168 := CountSegment.single (first := 10076) (initial := 4168) row_count_10077 (by decide)
private theorem atomic_10077_agreement : RowAgreement 10076 10077 := RowAgreement.single (first := 10076) modelrow_agrees_10077
private def atomic_10078 : CountSegment 1 10 13120 10077 10078 4168 4169 := CountSegment.single (first := 10077) (initial := 4168) row_count_10078 (by decide)
private theorem atomic_10078_agreement : RowAgreement 10077 10078 := RowAgreement.single (first := 10077) modelrow_agrees_10078
private def atomic_10079 : CountSegment 1 10 13120 10078 10079 4169 4169 := CountSegment.single (first := 10078) (initial := 4169) row_count_10079 (by decide)
private theorem atomic_10079_agreement : RowAgreement 10078 10079 := RowAgreement.single (first := 10078) modelrow_agrees_10079
private def atomic_10080 : CountSegment 1 10 13120 10079 10080 4169 4170 := CountSegment.single (first := 10079) (initial := 4169) row_count_10080 (by decide)
private theorem atomic_10080_agreement : RowAgreement 10079 10080 := RowAgreement.single (first := 10079) modelrow_agrees_10080
private def atomic_10081 : CountSegment 1 10 13120 10080 10081 4170 4170 := CountSegment.single (first := 10080) (initial := 4170) row_count_10081 (by decide)
private theorem atomic_10081_agreement : RowAgreement 10080 10081 := RowAgreement.single (first := 10080) modelrow_agrees_10081
private def atomic_10082 : CountSegment 1 10 13120 10081 10082 4170 4171 := CountSegment.single (first := 10081) (initial := 4170) row_count_10082 (by decide)
private theorem atomic_10082_agreement : RowAgreement 10081 10082 := RowAgreement.single (first := 10081) modelrow_agrees_10082
private def atomic_10083 : CountSegment 1 10 13120 10082 10083 4171 4171 := CountSegment.single (first := 10082) (initial := 4171) row_count_10083 (by decide)
private theorem atomic_10083_agreement : RowAgreement 10082 10083 := RowAgreement.single (first := 10082) modelrow_agrees_10083
private def atomic_10084 : CountSegment 1 10 13120 10083 10084 4171 4171 := CountSegment.single (first := 10083) (initial := 4171) row_count_10084 (by decide)
private theorem atomic_10084_agreement : RowAgreement 10083 10084 := RowAgreement.single (first := 10083) modelrow_agrees_10084
private def atomic_10085 : CountSegment 1 10 13120 10084 10085 4171 4171 := CountSegment.single (first := 10084) (initial := 4171) row_count_10085 (by decide)
private theorem atomic_10085_agreement : RowAgreement 10084 10085 := RowAgreement.single (first := 10084) modelrow_agrees_10085
private def atomic_10086 : CountSegment 1 10 13120 10085 10086 4171 4171 := CountSegment.single (first := 10085) (initial := 4171) row_count_10086 (by decide)
private theorem atomic_10086_agreement : RowAgreement 10085 10086 := RowAgreement.single (first := 10085) modelrow_agrees_10086
private def atomic_10087 : CountSegment 1 10 13120 10086 10087 4171 4171 := CountSegment.single (first := 10086) (initial := 4171) row_count_10087 (by decide)
private theorem atomic_10087_agreement : RowAgreement 10086 10087 := RowAgreement.single (first := 10086) modelrow_agrees_10087
private def atomic_10088 : CountSegment 1 10 13120 10087 10088 4171 4171 := CountSegment.single (first := 10087) (initial := 4171) row_count_10088 (by decide)
private theorem atomic_10088_agreement : RowAgreement 10087 10088 := RowAgreement.single (first := 10087) modelrow_agrees_10088
private def atomic_10089 : CountSegment 1 10 13120 10088 10089 4171 4171 := CountSegment.single (first := 10088) (initial := 4171) row_count_10089 (by decide)
private theorem atomic_10089_agreement : RowAgreement 10088 10089 := RowAgreement.single (first := 10088) modelrow_agrees_10089
private def atomic_10090 : CountSegment 1 10 13120 10089 10090 4171 4171 := CountSegment.single (first := 10089) (initial := 4171) row_count_10090 (by decide)
private theorem atomic_10090_agreement : RowAgreement 10089 10090 := RowAgreement.single (first := 10089) modelrow_agrees_10090
private def atomic_10091 : CountSegment 1 10 13120 10090 10091 4171 4172 := CountSegment.single (first := 10090) (initial := 4171) row_count_10091 (by decide)
private theorem atomic_10091_agreement : RowAgreement 10090 10091 := RowAgreement.single (first := 10090) modelrow_agrees_10091
private def atomic_10092 : CountSegment 1 10 13120 10091 10092 4172 4173 := CountSegment.single (first := 10091) (initial := 4172) row_count_10092 (by decide)
private theorem atomic_10092_agreement : RowAgreement 10091 10092 := RowAgreement.single (first := 10091) modelrow_agrees_10092
private def atomic_10093 : CountSegment 1 10 13120 10092 10093 4173 4174 := CountSegment.single (first := 10092) (initial := 4173) row_count_10093 (by decide)
private theorem atomic_10093_agreement : RowAgreement 10092 10093 := RowAgreement.single (first := 10092) modelrow_agrees_10093
private def atomic_10094 : CountSegment 1 10 13120 10093 10094 4174 4174 := CountSegment.single (first := 10093) (initial := 4174) row_count_10094 (by decide)
private theorem atomic_10094_agreement : RowAgreement 10093 10094 := RowAgreement.single (first := 10093) modelrow_agrees_10094
private def atomic_10095 : CountSegment 1 10 13120 10094 10095 4174 4174 := CountSegment.single (first := 10094) (initial := 4174) row_count_10095 (by decide)
private theorem atomic_10095_agreement : RowAgreement 10094 10095 := RowAgreement.single (first := 10094) modelrow_agrees_10095
private def atomic_10096 : CountSegment 1 10 13120 10095 10096 4174 4174 := CountSegment.single (first := 10095) (initial := 4174) row_count_10096 (by decide)
private theorem atomic_10096_agreement : RowAgreement 10095 10096 := RowAgreement.single (first := 10095) modelrow_agrees_10096
private def atomic_10097 : CountSegment 1 10 13120 10096 10097 4174 4175 := CountSegment.single (first := 10096) (initial := 4174) row_count_10097 (by decide)
private theorem atomic_10097_agreement : RowAgreement 10096 10097 := RowAgreement.single (first := 10096) modelrow_agrees_10097
private def atomic_10098 : CountSegment 1 10 13120 10097 10098 4175 4176 := CountSegment.single (first := 10097) (initial := 4175) row_count_10098 (by decide)
private theorem atomic_10098_agreement : RowAgreement 10097 10098 := RowAgreement.single (first := 10097) modelrow_agrees_10098
private def atomic_10099 : CountSegment 1 10 13120 10098 10099 4176 4176 := CountSegment.single (first := 10098) (initial := 4176) row_count_10099 (by decide)
private theorem atomic_10099_agreement : RowAgreement 10098 10099 := RowAgreement.single (first := 10098) modelrow_agrees_10099
private def atomic_10100 : CountSegment 1 10 13120 10099 10100 4176 4176 := CountSegment.single (first := 10099) (initial := 4176) row_count_10100 (by decide)
private theorem atomic_10100_agreement : RowAgreement 10099 10100 := RowAgreement.single (first := 10099) modelrow_agrees_10100
private def atomic_10101 : CountSegment 1 10 13120 10100 10101 4176 4176 := CountSegment.single (first := 10100) (initial := 4176) row_count_10101 (by decide)
private theorem atomic_10101_agreement : RowAgreement 10100 10101 := RowAgreement.single (first := 10100) modelrow_agrees_10101
private def atomic_10102 : CountSegment 1 10 13120 10101 10102 4176 4176 := CountSegment.single (first := 10101) (initial := 4176) row_count_10102 (by decide)
private theorem atomic_10102_agreement : RowAgreement 10101 10102 := RowAgreement.single (first := 10101) modelrow_agrees_10102
private def atomic_10103 : CountSegment 1 10 13120 10102 10103 4176 4177 := CountSegment.single (first := 10102) (initial := 4176) row_count_10103 (by decide)
private theorem atomic_10103_agreement : RowAgreement 10102 10103 := RowAgreement.single (first := 10102) modelrow_agrees_10103
private def atomic_10104 : CountSegment 1 10 13120 10103 10104 4177 4178 := CountSegment.single (first := 10103) (initial := 4177) row_count_10104 (by decide)
private theorem atomic_10104_agreement : RowAgreement 10103 10104 := RowAgreement.single (first := 10103) modelrow_agrees_10104
private def atomic_10105 : CountSegment 1 10 13120 10104 10105 4178 4178 := CountSegment.single (first := 10104) (initial := 4178) row_count_10105 (by decide)
private theorem atomic_10105_agreement : RowAgreement 10104 10105 := RowAgreement.single (first := 10104) modelrow_agrees_10105
private def atomic_10106 : CountSegment 1 10 13120 10105 10106 4178 4178 := CountSegment.single (first := 10105) (initial := 4178) row_count_10106 (by decide)
private theorem atomic_10106_agreement : RowAgreement 10105 10106 := RowAgreement.single (first := 10105) modelrow_agrees_10106
private def atomic_10107 : CountSegment 1 10 13120 10106 10107 4178 4180 := CountSegment.single (first := 10106) (initial := 4178) row_count_10107 (by decide)
private theorem atomic_10107_agreement : RowAgreement 10106 10107 := RowAgreement.single (first := 10106) modelrow_agrees_10107
private def atomic_10108 : CountSegment 1 10 13120 10107 10108 4180 4180 := CountSegment.single (first := 10107) (initial := 4180) row_count_10108 (by decide)
private theorem atomic_10108_agreement : RowAgreement 10107 10108 := RowAgreement.single (first := 10107) modelrow_agrees_10108
private def atomic_10109 : CountSegment 1 10 13120 10108 10109 4180 4180 := CountSegment.single (first := 10108) (initial := 4180) row_count_10109 (by decide)
private theorem atomic_10109_agreement : RowAgreement 10108 10109 := RowAgreement.single (first := 10108) modelrow_agrees_10109
private def atomic_10110 : CountSegment 1 10 13120 10109 10110 4180 4181 := CountSegment.single (first := 10109) (initial := 4180) row_count_10110 (by decide)
private theorem atomic_10110_agreement : RowAgreement 10109 10110 := RowAgreement.single (first := 10109) modelrow_agrees_10110
private def atomic_10111 : CountSegment 1 10 13120 10110 10111 4181 4181 := CountSegment.single (first := 10110) (initial := 4181) row_count_10111 (by decide)
private theorem atomic_10111_agreement : RowAgreement 10110 10111 := RowAgreement.single (first := 10110) modelrow_agrees_10111
private def atomic_10112 : CountSegment 1 10 13120 10111 10112 4181 4181 := CountSegment.single (first := 10111) (initial := 4181) row_count_10112 (by decide)
private theorem atomic_10112_agreement : RowAgreement 10111 10112 := RowAgreement.single (first := 10111) modelrow_agrees_10112
private def within_078_000 : CountSegment 1 10 13120 9984 9986 4129 4130 := CountSegment.join atomic_9985 atomic_9986
private theorem within_078_000_agreement : RowAgreement 9984 9986 := RowAgreement.join atomic_9985_agreement atomic_9986_agreement
private def within_078_001 : CountSegment 1 10 13120 9986 9988 4130 4130 := CountSegment.join atomic_9987 atomic_9988
private theorem within_078_001_agreement : RowAgreement 9986 9988 := RowAgreement.join atomic_9987_agreement atomic_9988_agreement
private def within_078_002 : CountSegment 1 10 13120 9984 9988 4129 4130 := CountSegment.join within_078_000 within_078_001
private theorem within_078_002_agreement : RowAgreement 9984 9988 := RowAgreement.join within_078_000_agreement within_078_001_agreement
private def within_078_003 : CountSegment 1 10 13120 9988 9990 4130 4131 := CountSegment.join atomic_9989 atomic_9990
private theorem within_078_003_agreement : RowAgreement 9988 9990 := RowAgreement.join atomic_9989_agreement atomic_9990_agreement
private def within_078_004 : CountSegment 1 10 13120 9990 9992 4131 4133 := CountSegment.join atomic_9991 atomic_9992
private theorem within_078_004_agreement : RowAgreement 9990 9992 := RowAgreement.join atomic_9991_agreement atomic_9992_agreement
private def within_078_005 : CountSegment 1 10 13120 9988 9992 4130 4133 := CountSegment.join within_078_003 within_078_004
private theorem within_078_005_agreement : RowAgreement 9988 9992 := RowAgreement.join within_078_003_agreement within_078_004_agreement
private def within_078_006 : CountSegment 1 10 13120 9984 9992 4129 4133 := CountSegment.join within_078_002 within_078_005
private theorem within_078_006_agreement : RowAgreement 9984 9992 := RowAgreement.join within_078_002_agreement within_078_005_agreement
private def within_078_007 : CountSegment 1 10 13120 9992 9994 4133 4134 := CountSegment.join atomic_9993 atomic_9994
private theorem within_078_007_agreement : RowAgreement 9992 9994 := RowAgreement.join atomic_9993_agreement atomic_9994_agreement
private def within_078_008 : CountSegment 1 10 13120 9994 9996 4134 4136 := CountSegment.join atomic_9995 atomic_9996
private theorem within_078_008_agreement : RowAgreement 9994 9996 := RowAgreement.join atomic_9995_agreement atomic_9996_agreement
private def within_078_009 : CountSegment 1 10 13120 9992 9996 4133 4136 := CountSegment.join within_078_007 within_078_008
private theorem within_078_009_agreement : RowAgreement 9992 9996 := RowAgreement.join within_078_007_agreement within_078_008_agreement
private def within_078_010 : CountSegment 1 10 13120 9996 9998 4136 4137 := CountSegment.join atomic_9997 atomic_9998
private theorem within_078_010_agreement : RowAgreement 9996 9998 := RowAgreement.join atomic_9997_agreement atomic_9998_agreement
private def within_078_011 : CountSegment 1 10 13120 9998 10000 4137 4138 := CountSegment.join atomic_9999 atomic_10000
private theorem within_078_011_agreement : RowAgreement 9998 10000 := RowAgreement.join atomic_9999_agreement atomic_10000_agreement
private def within_078_012 : CountSegment 1 10 13120 9996 10000 4136 4138 := CountSegment.join within_078_010 within_078_011
private theorem within_078_012_agreement : RowAgreement 9996 10000 := RowAgreement.join within_078_010_agreement within_078_011_agreement
private def within_078_013 : CountSegment 1 10 13120 9992 10000 4133 4138 := CountSegment.join within_078_009 within_078_012
private theorem within_078_013_agreement : RowAgreement 9992 10000 := RowAgreement.join within_078_009_agreement within_078_012_agreement
private def within_078_014 : CountSegment 1 10 13120 9984 10000 4129 4138 := CountSegment.join within_078_006 within_078_013
private theorem within_078_014_agreement : RowAgreement 9984 10000 := RowAgreement.join within_078_006_agreement within_078_013_agreement
private def within_078_015 : CountSegment 1 10 13120 10000 10002 4138 4138 := CountSegment.join atomic_10001 atomic_10002
private theorem within_078_015_agreement : RowAgreement 10000 10002 := RowAgreement.join atomic_10001_agreement atomic_10002_agreement
private def within_078_016 : CountSegment 1 10 13120 10002 10004 4138 4139 := CountSegment.join atomic_10003 atomic_10004
private theorem within_078_016_agreement : RowAgreement 10002 10004 := RowAgreement.join atomic_10003_agreement atomic_10004_agreement
private def within_078_017 : CountSegment 1 10 13120 10000 10004 4138 4139 := CountSegment.join within_078_015 within_078_016
private theorem within_078_017_agreement : RowAgreement 10000 10004 := RowAgreement.join within_078_015_agreement within_078_016_agreement
private def within_078_018 : CountSegment 1 10 13120 10004 10006 4139 4140 := CountSegment.join atomic_10005 atomic_10006
private theorem within_078_018_agreement : RowAgreement 10004 10006 := RowAgreement.join atomic_10005_agreement atomic_10006_agreement
private def within_078_019 : CountSegment 1 10 13120 10006 10008 4140 4142 := CountSegment.join atomic_10007 atomic_10008
private theorem within_078_019_agreement : RowAgreement 10006 10008 := RowAgreement.join atomic_10007_agreement atomic_10008_agreement
private def within_078_020 : CountSegment 1 10 13120 10004 10008 4139 4142 := CountSegment.join within_078_018 within_078_019
private theorem within_078_020_agreement : RowAgreement 10004 10008 := RowAgreement.join within_078_018_agreement within_078_019_agreement
private def within_078_021 : CountSegment 1 10 13120 10000 10008 4138 4142 := CountSegment.join within_078_017 within_078_020
private theorem within_078_021_agreement : RowAgreement 10000 10008 := RowAgreement.join within_078_017_agreement within_078_020_agreement
private def within_078_022 : CountSegment 1 10 13120 10008 10010 4142 4142 := CountSegment.join atomic_10009 atomic_10010
private theorem within_078_022_agreement : RowAgreement 10008 10010 := RowAgreement.join atomic_10009_agreement atomic_10010_agreement
private def within_078_023 : CountSegment 1 10 13120 10010 10012 4142 4142 := CountSegment.join atomic_10011 atomic_10012
private theorem within_078_023_agreement : RowAgreement 10010 10012 := RowAgreement.join atomic_10011_agreement atomic_10012_agreement
private def within_078_024 : CountSegment 1 10 13120 10008 10012 4142 4142 := CountSegment.join within_078_022 within_078_023
private theorem within_078_024_agreement : RowAgreement 10008 10012 := RowAgreement.join within_078_022_agreement within_078_023_agreement
private def within_078_025 : CountSegment 1 10 13120 10012 10014 4142 4143 := CountSegment.join atomic_10013 atomic_10014
private theorem within_078_025_agreement : RowAgreement 10012 10014 := RowAgreement.join atomic_10013_agreement atomic_10014_agreement
private def within_078_026 : CountSegment 1 10 13120 10014 10016 4143 4143 := CountSegment.join atomic_10015 atomic_10016
private theorem within_078_026_agreement : RowAgreement 10014 10016 := RowAgreement.join atomic_10015_agreement atomic_10016_agreement
private def within_078_027 : CountSegment 1 10 13120 10012 10016 4142 4143 := CountSegment.join within_078_025 within_078_026
private theorem within_078_027_agreement : RowAgreement 10012 10016 := RowAgreement.join within_078_025_agreement within_078_026_agreement
private def within_078_028 : CountSegment 1 10 13120 10008 10016 4142 4143 := CountSegment.join within_078_024 within_078_027
private theorem within_078_028_agreement : RowAgreement 10008 10016 := RowAgreement.join within_078_024_agreement within_078_027_agreement
private def within_078_029 : CountSegment 1 10 13120 10000 10016 4138 4143 := CountSegment.join within_078_021 within_078_028
private theorem within_078_029_agreement : RowAgreement 10000 10016 := RowAgreement.join within_078_021_agreement within_078_028_agreement
private def within_078_030 : CountSegment 1 10 13120 9984 10016 4129 4143 := CountSegment.join within_078_014 within_078_029
private theorem within_078_030_agreement : RowAgreement 9984 10016 := RowAgreement.join within_078_014_agreement within_078_029_agreement
private def within_078_031 : CountSegment 1 10 13120 10016 10018 4143 4144 := CountSegment.join atomic_10017 atomic_10018
private theorem within_078_031_agreement : RowAgreement 10016 10018 := RowAgreement.join atomic_10017_agreement atomic_10018_agreement
private def within_078_032 : CountSegment 1 10 13120 10018 10020 4144 4146 := CountSegment.join atomic_10019 atomic_10020
private theorem within_078_032_agreement : RowAgreement 10018 10020 := RowAgreement.join atomic_10019_agreement atomic_10020_agreement
private def within_078_033 : CountSegment 1 10 13120 10016 10020 4143 4146 := CountSegment.join within_078_031 within_078_032
private theorem within_078_033_agreement : RowAgreement 10016 10020 := RowAgreement.join within_078_031_agreement within_078_032_agreement
private def within_078_034 : CountSegment 1 10 13120 10020 10022 4146 4146 := CountSegment.join atomic_10021 atomic_10022
private theorem within_078_034_agreement : RowAgreement 10020 10022 := RowAgreement.join atomic_10021_agreement atomic_10022_agreement
private def within_078_035 : CountSegment 1 10 13120 10022 10024 4146 4146 := CountSegment.join atomic_10023 atomic_10024
private theorem within_078_035_agreement : RowAgreement 10022 10024 := RowAgreement.join atomic_10023_agreement atomic_10024_agreement
private def within_078_036 : CountSegment 1 10 13120 10020 10024 4146 4146 := CountSegment.join within_078_034 within_078_035
private theorem within_078_036_agreement : RowAgreement 10020 10024 := RowAgreement.join within_078_034_agreement within_078_035_agreement
private def within_078_037 : CountSegment 1 10 13120 10016 10024 4143 4146 := CountSegment.join within_078_033 within_078_036
private theorem within_078_037_agreement : RowAgreement 10016 10024 := RowAgreement.join within_078_033_agreement within_078_036_agreement
private def within_078_038 : CountSegment 1 10 13120 10024 10026 4146 4146 := CountSegment.join atomic_10025 atomic_10026
private theorem within_078_038_agreement : RowAgreement 10024 10026 := RowAgreement.join atomic_10025_agreement atomic_10026_agreement
private def within_078_039 : CountSegment 1 10 13120 10026 10028 4146 4146 := CountSegment.join atomic_10027 atomic_10028
private theorem within_078_039_agreement : RowAgreement 10026 10028 := RowAgreement.join atomic_10027_agreement atomic_10028_agreement
private def within_078_040 : CountSegment 1 10 13120 10024 10028 4146 4146 := CountSegment.join within_078_038 within_078_039
private theorem within_078_040_agreement : RowAgreement 10024 10028 := RowAgreement.join within_078_038_agreement within_078_039_agreement
private def within_078_041 : CountSegment 1 10 13120 10028 10030 4146 4146 := CountSegment.join atomic_10029 atomic_10030
private theorem within_078_041_agreement : RowAgreement 10028 10030 := RowAgreement.join atomic_10029_agreement atomic_10030_agreement
private def within_078_042 : CountSegment 1 10 13120 10030 10032 4146 4146 := CountSegment.join atomic_10031 atomic_10032
private theorem within_078_042_agreement : RowAgreement 10030 10032 := RowAgreement.join atomic_10031_agreement atomic_10032_agreement
private def within_078_043 : CountSegment 1 10 13120 10028 10032 4146 4146 := CountSegment.join within_078_041 within_078_042
private theorem within_078_043_agreement : RowAgreement 10028 10032 := RowAgreement.join within_078_041_agreement within_078_042_agreement
private def within_078_044 : CountSegment 1 10 13120 10024 10032 4146 4146 := CountSegment.join within_078_040 within_078_043
private theorem within_078_044_agreement : RowAgreement 10024 10032 := RowAgreement.join within_078_040_agreement within_078_043_agreement
private def within_078_045 : CountSegment 1 10 13120 10016 10032 4143 4146 := CountSegment.join within_078_037 within_078_044
private theorem within_078_045_agreement : RowAgreement 10016 10032 := RowAgreement.join within_078_037_agreement within_078_044_agreement
private def within_078_046 : CountSegment 1 10 13120 10032 10034 4146 4148 := CountSegment.join atomic_10033 atomic_10034
private theorem within_078_046_agreement : RowAgreement 10032 10034 := RowAgreement.join atomic_10033_agreement atomic_10034_agreement
private def within_078_047 : CountSegment 1 10 13120 10034 10036 4148 4149 := CountSegment.join atomic_10035 atomic_10036
private theorem within_078_047_agreement : RowAgreement 10034 10036 := RowAgreement.join atomic_10035_agreement atomic_10036_agreement
private def within_078_048 : CountSegment 1 10 13120 10032 10036 4146 4149 := CountSegment.join within_078_046 within_078_047
private theorem within_078_048_agreement : RowAgreement 10032 10036 := RowAgreement.join within_078_046_agreement within_078_047_agreement
private def within_078_049 : CountSegment 1 10 13120 10036 10038 4149 4151 := CountSegment.join atomic_10037 atomic_10038
private theorem within_078_049_agreement : RowAgreement 10036 10038 := RowAgreement.join atomic_10037_agreement atomic_10038_agreement
private def within_078_050 : CountSegment 1 10 13120 10038 10040 4151 4151 := CountSegment.join atomic_10039 atomic_10040
private theorem within_078_050_agreement : RowAgreement 10038 10040 := RowAgreement.join atomic_10039_agreement atomic_10040_agreement
private def within_078_051 : CountSegment 1 10 13120 10036 10040 4149 4151 := CountSegment.join within_078_049 within_078_050
private theorem within_078_051_agreement : RowAgreement 10036 10040 := RowAgreement.join within_078_049_agreement within_078_050_agreement
private def within_078_052 : CountSegment 1 10 13120 10032 10040 4146 4151 := CountSegment.join within_078_048 within_078_051
private theorem within_078_052_agreement : RowAgreement 10032 10040 := RowAgreement.join within_078_048_agreement within_078_051_agreement
private def within_078_053 : CountSegment 1 10 13120 10040 10042 4151 4151 := CountSegment.join atomic_10041 atomic_10042
private theorem within_078_053_agreement : RowAgreement 10040 10042 := RowAgreement.join atomic_10041_agreement atomic_10042_agreement
private def within_078_054 : CountSegment 1 10 13120 10042 10044 4151 4152 := CountSegment.join atomic_10043 atomic_10044
private theorem within_078_054_agreement : RowAgreement 10042 10044 := RowAgreement.join atomic_10043_agreement atomic_10044_agreement
private def within_078_055 : CountSegment 1 10 13120 10040 10044 4151 4152 := CountSegment.join within_078_053 within_078_054
private theorem within_078_055_agreement : RowAgreement 10040 10044 := RowAgreement.join within_078_053_agreement within_078_054_agreement
private def within_078_056 : CountSegment 1 10 13120 10044 10046 4152 4152 := CountSegment.join atomic_10045 atomic_10046
private theorem within_078_056_agreement : RowAgreement 10044 10046 := RowAgreement.join atomic_10045_agreement atomic_10046_agreement
private def within_078_057 : CountSegment 1 10 13120 10046 10048 4152 4154 := CountSegment.join atomic_10047 atomic_10048
private theorem within_078_057_agreement : RowAgreement 10046 10048 := RowAgreement.join atomic_10047_agreement atomic_10048_agreement
private def within_078_058 : CountSegment 1 10 13120 10044 10048 4152 4154 := CountSegment.join within_078_056 within_078_057
private theorem within_078_058_agreement : RowAgreement 10044 10048 := RowAgreement.join within_078_056_agreement within_078_057_agreement
private def within_078_059 : CountSegment 1 10 13120 10040 10048 4151 4154 := CountSegment.join within_078_055 within_078_058
private theorem within_078_059_agreement : RowAgreement 10040 10048 := RowAgreement.join within_078_055_agreement within_078_058_agreement
private def within_078_060 : CountSegment 1 10 13120 10032 10048 4146 4154 := CountSegment.join within_078_052 within_078_059
private theorem within_078_060_agreement : RowAgreement 10032 10048 := RowAgreement.join within_078_052_agreement within_078_059_agreement
private def within_078_061 : CountSegment 1 10 13120 10016 10048 4143 4154 := CountSegment.join within_078_045 within_078_060
private theorem within_078_061_agreement : RowAgreement 10016 10048 := RowAgreement.join within_078_045_agreement within_078_060_agreement
private def within_078_062 : CountSegment 1 10 13120 9984 10048 4129 4154 := CountSegment.join within_078_030 within_078_061
private theorem within_078_062_agreement : RowAgreement 9984 10048 := RowAgreement.join within_078_030_agreement within_078_061_agreement
private def within_078_063 : CountSegment 1 10 13120 10048 10050 4154 4156 := CountSegment.join atomic_10049 atomic_10050
private theorem within_078_063_agreement : RowAgreement 10048 10050 := RowAgreement.join atomic_10049_agreement atomic_10050_agreement
private def within_078_064 : CountSegment 1 10 13120 10050 10052 4156 4158 := CountSegment.join atomic_10051 atomic_10052
private theorem within_078_064_agreement : RowAgreement 10050 10052 := RowAgreement.join atomic_10051_agreement atomic_10052_agreement
private def within_078_065 : CountSegment 1 10 13120 10048 10052 4154 4158 := CountSegment.join within_078_063 within_078_064
private theorem within_078_065_agreement : RowAgreement 10048 10052 := RowAgreement.join within_078_063_agreement within_078_064_agreement
private def within_078_066 : CountSegment 1 10 13120 10052 10054 4158 4159 := CountSegment.join atomic_10053 atomic_10054
private theorem within_078_066_agreement : RowAgreement 10052 10054 := RowAgreement.join atomic_10053_agreement atomic_10054_agreement
private def within_078_067 : CountSegment 1 10 13120 10054 10056 4159 4160 := CountSegment.join atomic_10055 atomic_10056
private theorem within_078_067_agreement : RowAgreement 10054 10056 := RowAgreement.join atomic_10055_agreement atomic_10056_agreement
private def within_078_068 : CountSegment 1 10 13120 10052 10056 4158 4160 := CountSegment.join within_078_066 within_078_067
private theorem within_078_068_agreement : RowAgreement 10052 10056 := RowAgreement.join within_078_066_agreement within_078_067_agreement
private def within_078_069 : CountSegment 1 10 13120 10048 10056 4154 4160 := CountSegment.join within_078_065 within_078_068
private theorem within_078_069_agreement : RowAgreement 10048 10056 := RowAgreement.join within_078_065_agreement within_078_068_agreement
private def within_078_070 : CountSegment 1 10 13120 10056 10058 4160 4160 := CountSegment.join atomic_10057 atomic_10058
private theorem within_078_070_agreement : RowAgreement 10056 10058 := RowAgreement.join atomic_10057_agreement atomic_10058_agreement
private def within_078_071 : CountSegment 1 10 13120 10058 10060 4160 4162 := CountSegment.join atomic_10059 atomic_10060
private theorem within_078_071_agreement : RowAgreement 10058 10060 := RowAgreement.join atomic_10059_agreement atomic_10060_agreement
private def within_078_072 : CountSegment 1 10 13120 10056 10060 4160 4162 := CountSegment.join within_078_070 within_078_071
private theorem within_078_072_agreement : RowAgreement 10056 10060 := RowAgreement.join within_078_070_agreement within_078_071_agreement
private def within_078_073 : CountSegment 1 10 13120 10060 10062 4162 4163 := CountSegment.join atomic_10061 atomic_10062
private theorem within_078_073_agreement : RowAgreement 10060 10062 := RowAgreement.join atomic_10061_agreement atomic_10062_agreement
private def within_078_074 : CountSegment 1 10 13120 10062 10064 4163 4163 := CountSegment.join atomic_10063 atomic_10064
private theorem within_078_074_agreement : RowAgreement 10062 10064 := RowAgreement.join atomic_10063_agreement atomic_10064_agreement
private def within_078_075 : CountSegment 1 10 13120 10060 10064 4162 4163 := CountSegment.join within_078_073 within_078_074
private theorem within_078_075_agreement : RowAgreement 10060 10064 := RowAgreement.join within_078_073_agreement within_078_074_agreement
private def within_078_076 : CountSegment 1 10 13120 10056 10064 4160 4163 := CountSegment.join within_078_072 within_078_075
private theorem within_078_076_agreement : RowAgreement 10056 10064 := RowAgreement.join within_078_072_agreement within_078_075_agreement
private def within_078_077 : CountSegment 1 10 13120 10048 10064 4154 4163 := CountSegment.join within_078_069 within_078_076
private theorem within_078_077_agreement : RowAgreement 10048 10064 := RowAgreement.join within_078_069_agreement within_078_076_agreement
private def within_078_078 : CountSegment 1 10 13120 10064 10066 4163 4163 := CountSegment.join atomic_10065 atomic_10066
private theorem within_078_078_agreement : RowAgreement 10064 10066 := RowAgreement.join atomic_10065_agreement atomic_10066_agreement
private def within_078_079 : CountSegment 1 10 13120 10066 10068 4163 4165 := CountSegment.join atomic_10067 atomic_10068
private theorem within_078_079_agreement : RowAgreement 10066 10068 := RowAgreement.join atomic_10067_agreement atomic_10068_agreement
private def within_078_080 : CountSegment 1 10 13120 10064 10068 4163 4165 := CountSegment.join within_078_078 within_078_079
private theorem within_078_080_agreement : RowAgreement 10064 10068 := RowAgreement.join within_078_078_agreement within_078_079_agreement
private def within_078_081 : CountSegment 1 10 13120 10068 10070 4165 4165 := CountSegment.join atomic_10069 atomic_10070
private theorem within_078_081_agreement : RowAgreement 10068 10070 := RowAgreement.join atomic_10069_agreement atomic_10070_agreement
private def within_078_082 : CountSegment 1 10 13120 10070 10072 4165 4166 := CountSegment.join atomic_10071 atomic_10072
private theorem within_078_082_agreement : RowAgreement 10070 10072 := RowAgreement.join atomic_10071_agreement atomic_10072_agreement
private def within_078_083 : CountSegment 1 10 13120 10068 10072 4165 4166 := CountSegment.join within_078_081 within_078_082
private theorem within_078_083_agreement : RowAgreement 10068 10072 := RowAgreement.join within_078_081_agreement within_078_082_agreement
private def within_078_084 : CountSegment 1 10 13120 10064 10072 4163 4166 := CountSegment.join within_078_080 within_078_083
private theorem within_078_084_agreement : RowAgreement 10064 10072 := RowAgreement.join within_078_080_agreement within_078_083_agreement
private def within_078_085 : CountSegment 1 10 13120 10072 10074 4166 4168 := CountSegment.join atomic_10073 atomic_10074
private theorem within_078_085_agreement : RowAgreement 10072 10074 := RowAgreement.join atomic_10073_agreement atomic_10074_agreement
private def within_078_086 : CountSegment 1 10 13120 10074 10076 4168 4168 := CountSegment.join atomic_10075 atomic_10076
private theorem within_078_086_agreement : RowAgreement 10074 10076 := RowAgreement.join atomic_10075_agreement atomic_10076_agreement
private def within_078_087 : CountSegment 1 10 13120 10072 10076 4166 4168 := CountSegment.join within_078_085 within_078_086
private theorem within_078_087_agreement : RowAgreement 10072 10076 := RowAgreement.join within_078_085_agreement within_078_086_agreement
private def within_078_088 : CountSegment 1 10 13120 10076 10078 4168 4169 := CountSegment.join atomic_10077 atomic_10078
private theorem within_078_088_agreement : RowAgreement 10076 10078 := RowAgreement.join atomic_10077_agreement atomic_10078_agreement
private def within_078_089 : CountSegment 1 10 13120 10078 10080 4169 4170 := CountSegment.join atomic_10079 atomic_10080
private theorem within_078_089_agreement : RowAgreement 10078 10080 := RowAgreement.join atomic_10079_agreement atomic_10080_agreement
private def within_078_090 : CountSegment 1 10 13120 10076 10080 4168 4170 := CountSegment.join within_078_088 within_078_089
private theorem within_078_090_agreement : RowAgreement 10076 10080 := RowAgreement.join within_078_088_agreement within_078_089_agreement
private def within_078_091 : CountSegment 1 10 13120 10072 10080 4166 4170 := CountSegment.join within_078_087 within_078_090
private theorem within_078_091_agreement : RowAgreement 10072 10080 := RowAgreement.join within_078_087_agreement within_078_090_agreement
private def within_078_092 : CountSegment 1 10 13120 10064 10080 4163 4170 := CountSegment.join within_078_084 within_078_091
private theorem within_078_092_agreement : RowAgreement 10064 10080 := RowAgreement.join within_078_084_agreement within_078_091_agreement
private def within_078_093 : CountSegment 1 10 13120 10048 10080 4154 4170 := CountSegment.join within_078_077 within_078_092
private theorem within_078_093_agreement : RowAgreement 10048 10080 := RowAgreement.join within_078_077_agreement within_078_092_agreement
private def within_078_094 : CountSegment 1 10 13120 10080 10082 4170 4171 := CountSegment.join atomic_10081 atomic_10082
private theorem within_078_094_agreement : RowAgreement 10080 10082 := RowAgreement.join atomic_10081_agreement atomic_10082_agreement
private def within_078_095 : CountSegment 1 10 13120 10082 10084 4171 4171 := CountSegment.join atomic_10083 atomic_10084
private theorem within_078_095_agreement : RowAgreement 10082 10084 := RowAgreement.join atomic_10083_agreement atomic_10084_agreement
private def within_078_096 : CountSegment 1 10 13120 10080 10084 4170 4171 := CountSegment.join within_078_094 within_078_095
private theorem within_078_096_agreement : RowAgreement 10080 10084 := RowAgreement.join within_078_094_agreement within_078_095_agreement
private def within_078_097 : CountSegment 1 10 13120 10084 10086 4171 4171 := CountSegment.join atomic_10085 atomic_10086
private theorem within_078_097_agreement : RowAgreement 10084 10086 := RowAgreement.join atomic_10085_agreement atomic_10086_agreement
private def within_078_098 : CountSegment 1 10 13120 10086 10088 4171 4171 := CountSegment.join atomic_10087 atomic_10088
private theorem within_078_098_agreement : RowAgreement 10086 10088 := RowAgreement.join atomic_10087_agreement atomic_10088_agreement
private def within_078_099 : CountSegment 1 10 13120 10084 10088 4171 4171 := CountSegment.join within_078_097 within_078_098
private theorem within_078_099_agreement : RowAgreement 10084 10088 := RowAgreement.join within_078_097_agreement within_078_098_agreement
private def within_078_100 : CountSegment 1 10 13120 10080 10088 4170 4171 := CountSegment.join within_078_096 within_078_099
private theorem within_078_100_agreement : RowAgreement 10080 10088 := RowAgreement.join within_078_096_agreement within_078_099_agreement
private def within_078_101 : CountSegment 1 10 13120 10088 10090 4171 4171 := CountSegment.join atomic_10089 atomic_10090
private theorem within_078_101_agreement : RowAgreement 10088 10090 := RowAgreement.join atomic_10089_agreement atomic_10090_agreement
private def within_078_102 : CountSegment 1 10 13120 10090 10092 4171 4173 := CountSegment.join atomic_10091 atomic_10092
private theorem within_078_102_agreement : RowAgreement 10090 10092 := RowAgreement.join atomic_10091_agreement atomic_10092_agreement
private def within_078_103 : CountSegment 1 10 13120 10088 10092 4171 4173 := CountSegment.join within_078_101 within_078_102
private theorem within_078_103_agreement : RowAgreement 10088 10092 := RowAgreement.join within_078_101_agreement within_078_102_agreement
private def within_078_104 : CountSegment 1 10 13120 10092 10094 4173 4174 := CountSegment.join atomic_10093 atomic_10094
private theorem within_078_104_agreement : RowAgreement 10092 10094 := RowAgreement.join atomic_10093_agreement atomic_10094_agreement
private def within_078_105 : CountSegment 1 10 13120 10094 10096 4174 4174 := CountSegment.join atomic_10095 atomic_10096
private theorem within_078_105_agreement : RowAgreement 10094 10096 := RowAgreement.join atomic_10095_agreement atomic_10096_agreement
private def within_078_106 : CountSegment 1 10 13120 10092 10096 4173 4174 := CountSegment.join within_078_104 within_078_105
private theorem within_078_106_agreement : RowAgreement 10092 10096 := RowAgreement.join within_078_104_agreement within_078_105_agreement
private def within_078_107 : CountSegment 1 10 13120 10088 10096 4171 4174 := CountSegment.join within_078_103 within_078_106
private theorem within_078_107_agreement : RowAgreement 10088 10096 := RowAgreement.join within_078_103_agreement within_078_106_agreement
private def within_078_108 : CountSegment 1 10 13120 10080 10096 4170 4174 := CountSegment.join within_078_100 within_078_107
private theorem within_078_108_agreement : RowAgreement 10080 10096 := RowAgreement.join within_078_100_agreement within_078_107_agreement
private def within_078_109 : CountSegment 1 10 13120 10096 10098 4174 4176 := CountSegment.join atomic_10097 atomic_10098
private theorem within_078_109_agreement : RowAgreement 10096 10098 := RowAgreement.join atomic_10097_agreement atomic_10098_agreement
private def within_078_110 : CountSegment 1 10 13120 10098 10100 4176 4176 := CountSegment.join atomic_10099 atomic_10100
private theorem within_078_110_agreement : RowAgreement 10098 10100 := RowAgreement.join atomic_10099_agreement atomic_10100_agreement
private def within_078_111 : CountSegment 1 10 13120 10096 10100 4174 4176 := CountSegment.join within_078_109 within_078_110
private theorem within_078_111_agreement : RowAgreement 10096 10100 := RowAgreement.join within_078_109_agreement within_078_110_agreement
private def within_078_112 : CountSegment 1 10 13120 10100 10102 4176 4176 := CountSegment.join atomic_10101 atomic_10102
private theorem within_078_112_agreement : RowAgreement 10100 10102 := RowAgreement.join atomic_10101_agreement atomic_10102_agreement
private def within_078_113 : CountSegment 1 10 13120 10102 10104 4176 4178 := CountSegment.join atomic_10103 atomic_10104
private theorem within_078_113_agreement : RowAgreement 10102 10104 := RowAgreement.join atomic_10103_agreement atomic_10104_agreement
private def within_078_114 : CountSegment 1 10 13120 10100 10104 4176 4178 := CountSegment.join within_078_112 within_078_113
private theorem within_078_114_agreement : RowAgreement 10100 10104 := RowAgreement.join within_078_112_agreement within_078_113_agreement
private def within_078_115 : CountSegment 1 10 13120 10096 10104 4174 4178 := CountSegment.join within_078_111 within_078_114
private theorem within_078_115_agreement : RowAgreement 10096 10104 := RowAgreement.join within_078_111_agreement within_078_114_agreement
private def within_078_116 : CountSegment 1 10 13120 10104 10106 4178 4178 := CountSegment.join atomic_10105 atomic_10106
private theorem within_078_116_agreement : RowAgreement 10104 10106 := RowAgreement.join atomic_10105_agreement atomic_10106_agreement
private def within_078_117 : CountSegment 1 10 13120 10106 10108 4178 4180 := CountSegment.join atomic_10107 atomic_10108
private theorem within_078_117_agreement : RowAgreement 10106 10108 := RowAgreement.join atomic_10107_agreement atomic_10108_agreement
private def within_078_118 : CountSegment 1 10 13120 10104 10108 4178 4180 := CountSegment.join within_078_116 within_078_117
private theorem within_078_118_agreement : RowAgreement 10104 10108 := RowAgreement.join within_078_116_agreement within_078_117_agreement
private def within_078_119 : CountSegment 1 10 13120 10108 10110 4180 4181 := CountSegment.join atomic_10109 atomic_10110
private theorem within_078_119_agreement : RowAgreement 10108 10110 := RowAgreement.join atomic_10109_agreement atomic_10110_agreement
private def within_078_120 : CountSegment 1 10 13120 10110 10112 4181 4181 := CountSegment.join atomic_10111 atomic_10112
private theorem within_078_120_agreement : RowAgreement 10110 10112 := RowAgreement.join atomic_10111_agreement atomic_10112_agreement
private def within_078_121 : CountSegment 1 10 13120 10108 10112 4180 4181 := CountSegment.join within_078_119 within_078_120
private theorem within_078_121_agreement : RowAgreement 10108 10112 := RowAgreement.join within_078_119_agreement within_078_120_agreement
private def within_078_122 : CountSegment 1 10 13120 10104 10112 4178 4181 := CountSegment.join within_078_118 within_078_121
private theorem within_078_122_agreement : RowAgreement 10104 10112 := RowAgreement.join within_078_118_agreement within_078_121_agreement
private def within_078_123 : CountSegment 1 10 13120 10096 10112 4174 4181 := CountSegment.join within_078_115 within_078_122
private theorem within_078_123_agreement : RowAgreement 10096 10112 := RowAgreement.join within_078_115_agreement within_078_122_agreement
private def within_078_124 : CountSegment 1 10 13120 10080 10112 4170 4181 := CountSegment.join within_078_108 within_078_123
private theorem within_078_124_agreement : RowAgreement 10080 10112 := RowAgreement.join within_078_108_agreement within_078_123_agreement
private def within_078_125 : CountSegment 1 10 13120 10048 10112 4154 4181 := CountSegment.join within_078_093 within_078_124
private theorem within_078_125_agreement : RowAgreement 10048 10112 := RowAgreement.join within_078_093_agreement within_078_124_agreement
private def within_078_126 : CountSegment 1 10 13120 9984 10112 4129 4181 := CountSegment.join within_078_062 within_078_125
private theorem within_078_126_agreement : RowAgreement 9984 10112 := RowAgreement.join within_078_062_agreement within_078_125_agreement
def segment_078 : CountSegment 1 10 13120 9984 10112 4129 4181 := within_078_126
theorem agreement_078 : RowAgreement 9984 10112 := within_078_126_agreement
end ProjectEulerSolutions.P58Numerical
