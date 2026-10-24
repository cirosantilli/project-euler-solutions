import ProjectEulerSolutions.Lib.SpiralPrimeSegments
import ProjectEulerSolutions.Lib.SpiralPrimeCertificates.All
import ProjectEulerSolutions.Lib.MillerRabin
import ProjectEulerSolutions.Lib.SpiralMillerAgreement
/-! Finite spiral row and counter certificates. Python execution is checked separately. -/
namespace ProjectEulerSolutions.P58Numerical
open SpiralPrimes
set_option Elab.async false
set_option maxRecDepth 4096
def row_9729 : RowCertificate 9729 where
  first := true
  first_correct := by
    have prime : Nat.Prime (corner 9729 1) := by change Nat.Prime 378633223; exact ProjectEulerSolutions.P58PrimeNodes.p_378633223.prime
    exact decide_eq_true prime
  second := false
  second_correct := by
    have composite : ¬ Nat.Prime (corner 9729 2) := by change ¬ Nat.Prime 378613765; exact PrattCertificates.composite_of_product (factor := 5) (cofactor := 75722753) rfl (by decide) (by decide)
    exact decide_eq_false composite
  third := false
  third_correct := by
    have composite : ¬ Nat.Prime (corner 9729 3) := by change ¬ Nat.Prime 378594307; exact PrattCertificates.composite_of_product (factor := 7) (cofactor := 54084901) rfl (by decide) (by decide)
    exact decide_eq_false composite
theorem row_count_9729 : row 9729 = 1 := (row_9729.checked).trans rfl
theorem model_count_9729 : modelRow 9729 = 1 := by decide
theorem modelrow_agrees_9729 : modelRow 9729 = row 9729 := model_count_9729.trans row_count_9729.symm
def row_9730 : RowCertificate 9730 where
  first := true
  first_correct := by
    have prime : Nat.Prime (corner 9730 1) := by change Nat.Prime 378711061; exact ProjectEulerSolutions.P58PrimeNodes.p_378711061.prime
    exact decide_eq_true prime
  second := false
  second_correct := by
    have composite : ¬ Nat.Prime (corner 9730 2) := by change ¬ Nat.Prime 378691601; exact PrattCertificates.composite_of_product (factor := 9049) (cofactor := 41849) rfl (by decide) (by decide)
    exact decide_eq_false composite
  third := false
  third_correct := by
    have composite : ¬ Nat.Prime (corner 9730 3) := by change ¬ Nat.Prime 378672141; exact PrattCertificates.composite_of_product (factor := 3) (cofactor := 126224047) rfl (by decide) (by decide)
    exact decide_eq_false composite
theorem row_count_9730 : row 9730 = 1 := (row_9730.checked).trans rfl
theorem model_count_9730 : modelRow 9730 = 1 := by decide
theorem modelrow_agrees_9730 : modelRow 9730 = row 9730 := model_count_9730.trans row_count_9730.symm
def row_9731 : RowCertificate 9731 where
  first := false
  first_correct := by
    have composite : ¬ Nat.Prime (corner 9731 1) := by change ¬ Nat.Prime 378788907; exact PrattCertificates.composite_of_product (factor := 3) (cofactor := 126262969) rfl (by decide) (by decide)
    exact decide_eq_false composite
  second := false
  second_correct := by
    have composite : ¬ Nat.Prime (corner 9731 2) := by change ¬ Nat.Prime 378769445; exact PrattCertificates.composite_of_product (factor := 5) (cofactor := 75753889) rfl (by decide) (by decide)
    exact decide_eq_false composite
  third := false
  third_correct := by
    have composite : ¬ Nat.Prime (corner 9731 3) := by change ¬ Nat.Prime 378749983; exact PrattCertificates.composite_of_product (factor := 97) (cofactor := 3904639) rfl (by decide) (by decide)
    exact decide_eq_false composite
theorem row_count_9731 : row 9731 = 0 := (row_9731.checked).trans rfl
theorem model_count_9731 : modelRow 9731 = 0 := by decide
theorem modelrow_agrees_9731 : modelRow 9731 = row 9731 := model_count_9731.trans row_count_9731.symm
def row_9732 : RowCertificate 9732 where
  first := false
  first_correct := by
    have composite : ¬ Nat.Prime (corner 9732 1) := by change ¬ Nat.Prime 378866761; exact PrattCertificates.composite_of_product (factor := 7) (cofactor := 54123823) rfl (by decide) (by decide)
    exact decide_eq_false composite
  second := false
  second_correct := by
    have composite : ¬ Nat.Prime (corner 9732 2) := by change ¬ Nat.Prime 378847297; exact PrattCertificates.composite_of_product (factor := 73) (cofactor := 5189689) rfl (by decide) (by decide)
    exact decide_eq_false composite
  third := false
  third_correct := by
    have composite : ¬ Nat.Prime (corner 9732 3) := by change ¬ Nat.Prime 378827833; exact PrattCertificates.composite_of_product (factor := 19) (cofactor := 19938307) rfl (by decide) (by decide)
    exact decide_eq_false composite
theorem row_count_9732 : row 9732 = 0 := (row_9732.checked).trans rfl
theorem model_count_9732 : modelRow 9732 = 0 := by decide
theorem modelrow_agrees_9732 : modelRow 9732 = row 9732 := model_count_9732.trans row_count_9732.symm
def row_9733 : RowCertificate 9733 where
  first := true
  first_correct := by
    have prime : Nat.Prime (corner 9733 1) := by change Nat.Prime 378944623; exact ProjectEulerSolutions.P58PrimeNodes.p_378944623.prime
    exact decide_eq_true prime
  second := false
  second_correct := by
    have composite : ¬ Nat.Prime (corner 9733 2) := by change ¬ Nat.Prime 378925157; exact PrattCertificates.composite_of_product (factor := 13) (cofactor := 29148089) rfl (by decide) (by decide)
    exact decide_eq_false composite
  third := false
  third_correct := by
    have composite : ¬ Nat.Prime (corner 9733 3) := by change ¬ Nat.Prime 378905691; exact PrattCertificates.composite_of_product (factor := 3) (cofactor := 126301897) rfl (by decide) (by decide)
    exact decide_eq_false composite
theorem row_count_9733 : row 9733 = 1 := (row_9733.checked).trans rfl
theorem model_count_9733 : modelRow 9733 = 1 := by decide
theorem modelrow_agrees_9733 : modelRow 9733 = row 9733 := model_count_9733.trans row_count_9733.symm
def row_9734 : RowCertificate 9734 where
  first := false
  first_correct := by
    have composite : ¬ Nat.Prime (corner 9734 1) := by change ¬ Nat.Prime 379022493; exact PrattCertificates.composite_of_product (factor := 3) (cofactor := 126340831) rfl (by decide) (by decide)
    exact decide_eq_false composite
  second := false
  second_correct := by
    have composite : ¬ Nat.Prime (corner 9734 2) := by change ¬ Nat.Prime 379003025; exact PrattCertificates.composite_of_product (factor := 5) (cofactor := 75800605) rfl (by decide) (by decide)
    exact decide_eq_false composite
  third := false
  third_correct := by
    have composite : ¬ Nat.Prime (corner 9734 3) := by change ¬ Nat.Prime 378983557; exact PrattCertificates.composite_of_product (factor := 19) (cofactor := 19946503) rfl (by decide) (by decide)
    exact decide_eq_false composite
theorem row_count_9734 : row 9734 = 0 := (row_9734.checked).trans rfl
theorem model_count_9734 : modelRow 9734 = 0 := by decide
theorem modelrow_agrees_9734 : modelRow 9734 = row 9734 := model_count_9734.trans row_count_9734.symm
def row_9735 : RowCertificate 9735 where
  first := false
  first_correct := by
    have composite : ¬ Nat.Prime (corner 9735 1) := by change ¬ Nat.Prime 379100371; exact PrattCertificates.composite_of_product (factor := 13) (cofactor := 29161567) rfl (by decide) (by decide)
    exact decide_eq_false composite
  second := false
  second_correct := by
    have composite : ¬ Nat.Prime (corner 9735 2) := by change ¬ Nat.Prime 379080901; exact PrattCertificates.composite_of_product (factor := 61) (cofactor := 6214441) rfl (by decide) (by decide)
    exact decide_eq_false composite
  third := false
  third_correct := by
    have composite : ¬ Nat.Prime (corner 9735 3) := by change ¬ Nat.Prime 379061431; exact PrattCertificates.composite_of_product (factor := 7) (cofactor := 54151633) rfl (by decide) (by decide)
    exact decide_eq_false composite
theorem row_count_9735 : row 9735 = 0 := (row_9735.checked).trans rfl
theorem model_count_9735 : modelRow 9735 = 0 := by decide
theorem modelrow_agrees_9735 : modelRow 9735 = row 9735 := model_count_9735.trans row_count_9735.symm
def row_9736 : RowCertificate 9736 where
  first := false
  first_correct := by
    have composite : ¬ Nat.Prime (corner 9736 1) := by change ¬ Nat.Prime 379178257; exact PrattCertificates.composite_of_product (factor := 37) (cofactor := 10248061) rfl (by decide) (by decide)
    exact decide_eq_false composite
  second := false
  second_correct := by
    have composite : ¬ Nat.Prime (corner 9736 2) := by change ¬ Nat.Prime 379158785; exact PrattCertificates.composite_of_product (factor := 5) (cofactor := 75831757) rfl (by decide) (by decide)
    exact decide_eq_false composite
  third := false
  third_correct := by
    have composite : ¬ Nat.Prime (corner 9736 3) := by change ¬ Nat.Prime 379139313; exact PrattCertificates.composite_of_product (factor := 3) (cofactor := 126379771) rfl (by decide) (by decide)
    exact decide_eq_false composite
theorem row_count_9736 : row 9736 = 0 := (row_9736.checked).trans rfl
theorem model_count_9736 : modelRow 9736 = 0 := by decide
theorem modelrow_agrees_9736 : modelRow 9736 = row 9736 := model_count_9736.trans row_count_9736.symm
def row_9737 : RowCertificate 9737 where
  first := false
  first_correct := by
    have composite : ¬ Nat.Prime (corner 9737 1) := by change ¬ Nat.Prime 379256151; exact PrattCertificates.composite_of_product (factor := 3) (cofactor := 126418717) rfl (by decide) (by decide)
    exact decide_eq_false composite
  second := false
  second_correct := by
    have composite : ¬ Nat.Prime (corner 9737 2) := by change ¬ Nat.Prime 379236677; exact PrattCertificates.composite_of_product (factor := 53) (cofactor := 7155409) rfl (by decide) (by decide)
    exact decide_eq_false composite
  third := false
  third_correct := by
    have composite : ¬ Nat.Prime (corner 9737 3) := by change ¬ Nat.Prime 379217203; exact PrattCertificates.composite_of_product (factor := 31) (cofactor := 12232813) rfl (by decide) (by decide)
    exact decide_eq_false composite
theorem row_count_9737 : row 9737 = 0 := (row_9737.checked).trans rfl
theorem model_count_9737 : modelRow 9737 = 0 := by decide
theorem modelrow_agrees_9737 : modelRow 9737 = row 9737 := model_count_9737.trans row_count_9737.symm
def row_9738 : RowCertificate 9738 where
  first := false
  first_correct := by
    have composite : ¬ Nat.Prime (corner 9738 1) := by change ¬ Nat.Prime 379334053; exact PrattCertificates.composite_of_product (factor := 7) (cofactor := 54190579) rfl (by decide) (by decide)
    exact decide_eq_false composite
  second := false
  second_correct := by
    have composite : ¬ Nat.Prime (corner 9738 2) := by change ¬ Nat.Prime 379314577; exact PrattCertificates.composite_of_product (factor := 29) (cofactor := 13079813) rfl (by decide) (by decide)
    exact decide_eq_false composite
  third := true
  third_correct := by
    have prime : Nat.Prime (corner 9738 3) := by change Nat.Prime 379295101; exact ProjectEulerSolutions.P58PrimeNodes.p_379295101.prime
    exact decide_eq_true prime
theorem row_count_9738 : row 9738 = 1 := (row_9738.checked).trans rfl
theorem model_count_9738 : modelRow 9738 = 1 := by decide
theorem modelrow_agrees_9738 : modelRow 9738 = row 9738 := model_count_9738.trans row_count_9738.symm
def row_9739 : RowCertificate 9739 where
  first := false
  first_correct := by
    have composite : ¬ Nat.Prime (corner 9739 1) := by change ¬ Nat.Prime 379411963; exact PrattCertificates.composite_of_product (factor := 7) (cofactor := 54201709) rfl (by decide) (by decide)
    exact decide_eq_false composite
  second := false
  second_correct := by
    have composite : ¬ Nat.Prime (corner 9739 2) := by change ¬ Nat.Prime 379392485; exact PrattCertificates.composite_of_product (factor := 5) (cofactor := 75878497) rfl (by decide) (by decide)
    exact decide_eq_false composite
  third := false
  third_correct := by
    have composite : ¬ Nat.Prime (corner 9739 3) := by change ¬ Nat.Prime 379373007; exact PrattCertificates.composite_of_product (factor := 3) (cofactor := 126457669) rfl (by decide) (by decide)
    exact decide_eq_false composite
theorem row_count_9739 : row 9739 = 0 := (row_9739.checked).trans rfl
theorem model_count_9739 : modelRow 9739 = 0 := by decide
theorem modelrow_agrees_9739 : modelRow 9739 = row 9739 := model_count_9739.trans row_count_9739.symm
def row_9740 : RowCertificate 9740 where
  first := false
  first_correct := by
    have composite : ¬ Nat.Prime (corner 9740 1) := by change ¬ Nat.Prime 379489881; exact PrattCertificates.composite_of_product (factor := 3) (cofactor := 126496627) rfl (by decide) (by decide)
    exact decide_eq_false composite
  second := true
  second_correct := by
    have prime : Nat.Prime (corner 9740 2) := by change Nat.Prime 379470401; exact ProjectEulerSolutions.P58PrimeNodes.p_379470401.prime
    exact decide_eq_true prime
  third := true
  third_correct := by
    have prime : Nat.Prime (corner 9740 3) := by change Nat.Prime 379450921; exact ProjectEulerSolutions.P58PrimeNodes.p_379450921.prime
    exact decide_eq_true prime
theorem row_count_9740 : row 9740 = 2 := (row_9740.checked).trans rfl
theorem model_count_9740 : modelRow 9740 = 2 := by decide
theorem modelrow_agrees_9740 : modelRow 9740 = row 9740 := model_count_9740.trans row_count_9740.symm
def row_9741 : RowCertificate 9741 where
  first := false
  first_correct := by
    have composite : ¬ Nat.Prime (corner 9741 1) := by change ¬ Nat.Prime 379567807; exact PrattCertificates.composite_of_product (factor := 19) (cofactor := 19977253) rfl (by decide) (by decide)
    exact decide_eq_false composite
  second := false
  second_correct := by
    have composite : ¬ Nat.Prime (corner 9741 2) := by change ¬ Nat.Prime 379548325; exact PrattCertificates.composite_of_product (factor := 5) (cofactor := 75909665) rfl (by decide) (by decide)
    exact decide_eq_false composite
  third := true
  third_correct := by
    have prime : Nat.Prime (corner 9741 3) := by change Nat.Prime 379528843; exact ProjectEulerSolutions.P58PrimeNodes.p_379528843.prime
    exact decide_eq_true prime
theorem row_count_9741 : row 9741 = 1 := (row_9741.checked).trans rfl
theorem model_count_9741 : modelRow 9741 = 1 := by decide
theorem modelrow_agrees_9741 : modelRow 9741 = row 9741 := model_count_9741.trans row_count_9741.symm
def row_9742 : RowCertificate 9742 where
  first := false
  first_correct := by
    have composite : ¬ Nat.Prime (corner 9742 1) := by change ¬ Nat.Prime 379645741; exact PrattCertificates.composite_of_product (factor := 373) (cofactor := 1017817) rfl (by decide) (by decide)
    exact decide_eq_false composite
  second := false
  second_correct := by
    have composite : ¬ Nat.Prime (corner 9742 2) := by change ¬ Nat.Prime 379626257; exact PrattCertificates.composite_of_product (factor := 41) (cofactor := 9259177) rfl (by decide) (by decide)
    exact decide_eq_false composite
  third := false
  third_correct := by
    have composite : ¬ Nat.Prime (corner 9742 3) := by change ¬ Nat.Prime 379606773; exact PrattCertificates.composite_of_product (factor := 3) (cofactor := 126535591) rfl (by decide) (by decide)
    exact decide_eq_false composite
theorem row_count_9742 : row 9742 = 0 := (row_9742.checked).trans rfl
theorem model_count_9742 : modelRow 9742 = 0 := by decide
theorem modelrow_agrees_9742 : modelRow 9742 = row 9742 := model_count_9742.trans row_count_9742.symm
def row_9743 : RowCertificate 9743 where
  first := false
  first_correct := by
    have composite : ¬ Nat.Prime (corner 9743 1) := by change ¬ Nat.Prime 379723683; exact PrattCertificates.composite_of_product (factor := 3) (cofactor := 126574561) rfl (by decide) (by decide)
    exact decide_eq_false composite
  second := false
  second_correct := by
    have composite : ¬ Nat.Prime (corner 9743 2) := by change ¬ Nat.Prime 379704197; exact PrattCertificates.composite_of_product (factor := 17) (cofactor := 22335541) rfl (by decide) (by decide)
    exact decide_eq_false composite
  third := false
  third_correct := by
    have composite : ¬ Nat.Prime (corner 9743 3) := by change ¬ Nat.Prime 379684711; exact PrattCertificates.composite_of_product (factor := 7) (cofactor := 54240673) rfl (by decide) (by decide)
    exact decide_eq_false composite
theorem row_count_9743 : row 9743 = 0 := (row_9743.checked).trans rfl
theorem model_count_9743 : modelRow 9743 = 0 := by decide
theorem modelrow_agrees_9743 : modelRow 9743 = row 9743 := model_count_9743.trans row_count_9743.symm
def row_9744 : RowCertificate 9744 where
  first := false
  first_correct := by
    have composite : ¬ Nat.Prime (corner 9744 1) := by change ¬ Nat.Prime 379801633; exact PrattCertificates.composite_of_product (factor := 37) (cofactor := 10264909) rfl (by decide) (by decide)
    exact decide_eq_false composite
  second := false
  second_correct := by
    have composite : ¬ Nat.Prime (corner 9744 2) := by change ¬ Nat.Prime 379782145; exact PrattCertificates.composite_of_product (factor := 5) (cofactor := 75956429) rfl (by decide) (by decide)
    exact decide_eq_false composite
  third := false
  third_correct := by
    have composite : ¬ Nat.Prime (corner 9744 3) := by change ¬ Nat.Prime 379762657; exact PrattCertificates.composite_of_product (factor := 397) (cofactor := 956581) rfl (by decide) (by decide)
    exact decide_eq_false composite
theorem row_count_9744 : row 9744 = 0 := (row_9744.checked).trans rfl
theorem model_count_9744 : modelRow 9744 = 0 := by decide
theorem modelrow_agrees_9744 : modelRow 9744 = row 9744 := model_count_9744.trans row_count_9744.symm
def row_9745 : RowCertificate 9745 where
  first := false
  first_correct := by
    have composite : ¬ Nat.Prime (corner 9745 1) := by change ¬ Nat.Prime 379879591; exact PrattCertificates.composite_of_product (factor := 7) (cofactor := 54268513) rfl (by decide) (by decide)
    exact decide_eq_false composite
  second := true
  second_correct := by
    have prime : Nat.Prime (corner 9745 2) := by change Nat.Prime 379860101; exact ProjectEulerSolutions.P58PrimeNodes.p_379860101.prime
    exact decide_eq_true prime
  third := false
  third_correct := by
    have composite : ¬ Nat.Prime (corner 9745 3) := by change ¬ Nat.Prime 379840611; exact PrattCertificates.composite_of_product (factor := 3) (cofactor := 126613537) rfl (by decide) (by decide)
    exact decide_eq_false composite
theorem row_count_9745 : row 9745 = 1 := (row_9745.checked).trans rfl
theorem model_count_9745 : modelRow 9745 = 1 := by decide
theorem modelrow_agrees_9745 : modelRow 9745 = row 9745 := model_count_9745.trans row_count_9745.symm
def row_9746 : RowCertificate 9746 where
  first := false
  first_correct := by
    have composite : ¬ Nat.Prime (corner 9746 1) := by change ¬ Nat.Prime 379957557; exact PrattCertificates.composite_of_product (factor := 3) (cofactor := 126652519) rfl (by decide) (by decide)
    exact decide_eq_false composite
  second := false
  second_correct := by
    have composite : ¬ Nat.Prime (corner 9746 2) := by change ¬ Nat.Prime 379938065; exact PrattCertificates.composite_of_product (factor := 5) (cofactor := 75987613) rfl (by decide) (by decide)
    exact decide_eq_false composite
  third := false
  third_correct := by
    have composite : ¬ Nat.Prime (corner 9746 3) := by change ¬ Nat.Prime 379918573; exact PrattCertificates.composite_of_product (factor := 1237) (cofactor := 307129) rfl (by decide) (by decide)
    exact decide_eq_false composite
theorem row_count_9746 : row 9746 = 0 := (row_9746.checked).trans rfl
theorem model_count_9746 : modelRow 9746 = 0 := by decide
theorem modelrow_agrees_9746 : modelRow 9746 = row 9746 := model_count_9746.trans row_count_9746.symm
def row_9747 : RowCertificate 9747 where
  first := true
  first_correct := by
    have prime : Nat.Prime (corner 9747 1) := by change Nat.Prime 380035531; exact ProjectEulerSolutions.P58PrimeNodes.p_380035531.prime
    exact decide_eq_true prime
  second := true
  second_correct := by
    have prime : Nat.Prime (corner 9747 2) := by change Nat.Prime 380016037; exact ProjectEulerSolutions.P58PrimeNodes.p_380016037.prime
    exact decide_eq_true prime
  third := false
  third_correct := by
    have composite : ¬ Nat.Prime (corner 9747 3) := by change ¬ Nat.Prime 379996543; exact PrattCertificates.composite_of_product (factor := 31) (cofactor := 12257953) rfl (by decide) (by decide)
    exact decide_eq_false composite
theorem row_count_9747 : row 9747 = 2 := (row_9747.checked).trans rfl
theorem model_count_9747 : modelRow 9747 = 2 := by decide
theorem modelrow_agrees_9747 : modelRow 9747 = row 9747 := model_count_9747.trans row_count_9747.symm
def row_9748 : RowCertificate 9748 where
  first := false
  first_correct := by
    have composite : ¬ Nat.Prime (corner 9748 1) := by change ¬ Nat.Prime 380113513; exact PrattCertificates.composite_of_product (factor := 13) (cofactor := 29239501) rfl (by decide) (by decide)
    exact decide_eq_false composite
  second := false
  second_correct := by
    have composite : ¬ Nat.Prime (corner 9748 2) := by change ¬ Nat.Prime 380094017; exact PrattCertificates.composite_of_product (factor := 157) (cofactor := 2420981) rfl (by decide) (by decide)
    exact decide_eq_false composite
  third := false
  third_correct := by
    have composite : ¬ Nat.Prime (corner 9748 3) := by change ¬ Nat.Prime 380074521; exact PrattCertificates.composite_of_product (factor := 3) (cofactor := 126691507) rfl (by decide) (by decide)
    exact decide_eq_false composite
theorem row_count_9748 : row 9748 = 0 := (row_9748.checked).trans rfl
theorem model_count_9748 : modelRow 9748 = 0 := by decide
theorem modelrow_agrees_9748 : modelRow 9748 = row 9748 := model_count_9748.trans row_count_9748.symm
def row_9749 : RowCertificate 9749 where
  first := false
  first_correct := by
    have composite : ¬ Nat.Prime (corner 9749 1) := by change ¬ Nat.Prime 380191503; exact PrattCertificates.composite_of_product (factor := 3) (cofactor := 126730501) rfl (by decide) (by decide)
    exact decide_eq_false composite
  second := false
  second_correct := by
    have composite : ¬ Nat.Prime (corner 9749 2) := by change ¬ Nat.Prime 380172005; exact PrattCertificates.composite_of_product (factor := 5) (cofactor := 76034401) rfl (by decide) (by decide)
    exact decide_eq_false composite
  third := false
  third_correct := by
    have composite : ¬ Nat.Prime (corner 9749 3) := by change ¬ Nat.Prime 380152507; exact PrattCertificates.composite_of_product (factor := 7) (cofactor := 54307501) rfl (by decide) (by decide)
    exact decide_eq_false composite
theorem row_count_9749 : row 9749 = 0 := (row_9749.checked).trans rfl
theorem model_count_9749 : modelRow 9749 = 0 := by decide
theorem modelrow_agrees_9749 : modelRow 9749 = row 9749 := model_count_9749.trans row_count_9749.symm
def row_9750 : RowCertificate 9750 where
  first := true
  first_correct := by
    have prime : Nat.Prime (corner 9750 1) := by change Nat.Prime 380269501; exact ProjectEulerSolutions.P58PrimeNodes.p_380269501.prime
    exact decide_eq_true prime
  second := false
  second_correct := by
    have composite : ¬ Nat.Prime (corner 9750 2) := by change ¬ Nat.Prime 380250001; exact PrattCertificates.composite_of_product (factor := 29) (cofactor := 13112069) rfl (by decide) (by decide)
    exact decide_eq_false composite
  third := false
  third_correct := by
    have composite : ¬ Nat.Prime (corner 9750 3) := by change ¬ Nat.Prime 380230501; exact PrattCertificates.composite_of_product (factor := 7) (cofactor := 54318643) rfl (by decide) (by decide)
    exact decide_eq_false composite
theorem row_count_9750 : row 9750 = 1 := (row_9750.checked).trans rfl
theorem model_count_9750 : modelRow 9750 = 1 := by decide
theorem modelrow_agrees_9750 : modelRow 9750 = row 9750 := model_count_9750.trans row_count_9750.symm
def row_9751 : RowCertificate 9751 where
  first := false
  first_correct := by
    have composite : ¬ Nat.Prime (corner 9751 1) := by change ¬ Nat.Prime 380347507; exact PrattCertificates.composite_of_product (factor := 139) (cofactor := 2736313) rfl (by decide) (by decide)
    exact decide_eq_false composite
  second := false
  second_correct := by
    have composite : ¬ Nat.Prime (corner 9751 2) := by change ¬ Nat.Prime 380328005; exact PrattCertificates.composite_of_product (factor := 5) (cofactor := 76065601) rfl (by decide) (by decide)
    exact decide_eq_false composite
  third := false
  third_correct := by
    have composite : ¬ Nat.Prime (corner 9751 3) := by change ¬ Nat.Prime 380308503; exact PrattCertificates.composite_of_product (factor := 3) (cofactor := 126769501) rfl (by decide) (by decide)
    exact decide_eq_false composite
theorem row_count_9751 : row 9751 = 0 := (row_9751.checked).trans rfl
theorem model_count_9751 : modelRow 9751 = 0 := by decide
theorem modelrow_agrees_9751 : modelRow 9751 = row 9751 := model_count_9751.trans row_count_9751.symm
def row_9752 : RowCertificate 9752 where
  first := false
  first_correct := by
    have composite : ¬ Nat.Prime (corner 9752 1) := by change ¬ Nat.Prime 380425521; exact PrattCertificates.composite_of_product (factor := 3) (cofactor := 126808507) rfl (by decide) (by decide)
    exact decide_eq_false composite
  second := false
  second_correct := by
    have composite : ¬ Nat.Prime (corner 9752 2) := by change ¬ Nat.Prime 380406017; exact PrattCertificates.composite_of_product (factor := 857) (cofactor := 443881) rfl (by decide) (by decide)
    exact decide_eq_false composite
  third := false
  third_correct := by
    have composite : ¬ Nat.Prime (corner 9752 3) := by change ¬ Nat.Prime 380386513; exact PrattCertificates.composite_of_product (factor := 13) (cofactor := 29260501) rfl (by decide) (by decide)
    exact decide_eq_false composite
theorem row_count_9752 : row 9752 = 0 := (row_9752.checked).trans rfl
theorem model_count_9752 : modelRow 9752 = 0 := by decide
theorem modelrow_agrees_9752 : modelRow 9752 = row 9752 := model_count_9752.trans row_count_9752.symm
def row_9753 : RowCertificate 9753 where
  first := false
  first_correct := by
    have composite : ¬ Nat.Prime (corner 9753 1) := by change ¬ Nat.Prime 380503543; exact PrattCertificates.composite_of_product (factor := 7) (cofactor := 54357649) rfl (by decide) (by decide)
    exact decide_eq_false composite
  second := true
  second_correct := by
    have prime : Nat.Prime (corner 9753 2) := by change Nat.Prime 380484037; exact ProjectEulerSolutions.P58PrimeNodes.p_380484037.prime
    exact decide_eq_true prime
  third := false
  third_correct := by
    have composite : ¬ Nat.Prime (corner 9753 3) := by change ¬ Nat.Prime 380464531; exact PrattCertificates.composite_of_product (factor := 19) (cofactor := 20024449) rfl (by decide) (by decide)
    exact decide_eq_false composite
theorem row_count_9753 : row 9753 = 1 := (row_9753.checked).trans rfl
theorem model_count_9753 : modelRow 9753 = 1 := by decide
theorem modelrow_agrees_9753 : modelRow 9753 = row 9753 := model_count_9753.trans row_count_9753.symm
def row_9754 : RowCertificate 9754 where
  first := true
  first_correct := by
    have prime : Nat.Prime (corner 9754 1) := by change Nat.Prime 380581573; exact ProjectEulerSolutions.P58PrimeNodes.p_380581573.prime
    exact decide_eq_true prime
  second := false
  second_correct := by
    have composite : ¬ Nat.Prime (corner 9754 2) := by change ¬ Nat.Prime 380562065; exact PrattCertificates.composite_of_product (factor := 5) (cofactor := 76112413) rfl (by decide) (by decide)
    exact decide_eq_false composite
  third := false
  third_correct := by
    have composite : ¬ Nat.Prime (corner 9754 3) := by change ¬ Nat.Prime 380542557; exact PrattCertificates.composite_of_product (factor := 3) (cofactor := 126847519) rfl (by decide) (by decide)
    exact decide_eq_false composite
theorem row_count_9754 : row 9754 = 1 := (row_9754.checked).trans rfl
theorem model_count_9754 : modelRow 9754 = 1 := by decide
theorem modelrow_agrees_9754 : modelRow 9754 = row 9754 := model_count_9754.trans row_count_9754.symm
def row_9755 : RowCertificate 9755 where
  first := false
  first_correct := by
    have composite : ¬ Nat.Prime (corner 9755 1) := by change ¬ Nat.Prime 380659611; exact PrattCertificates.composite_of_product (factor := 3) (cofactor := 126886537) rfl (by decide) (by decide)
    exact decide_eq_false composite
  second := true
  second_correct := by
    have prime : Nat.Prime (corner 9755 2) := by change Nat.Prime 380640101; exact ProjectEulerSolutions.P58PrimeNodes.p_380640101.prime
    exact decide_eq_true prime
  third := false
  third_correct := by
    have composite : ¬ Nat.Prime (corner 9755 3) := by change ¬ Nat.Prime 380620591; exact PrattCertificates.composite_of_product (factor := 13) (cofactor := 29278507) rfl (by decide) (by decide)
    exact decide_eq_false composite
theorem row_count_9755 : row 9755 = 1 := (row_9755.checked).trans rfl
theorem model_count_9755 : modelRow 9755 = 1 := by decide
theorem modelrow_agrees_9755 : modelRow 9755 = row 9755 := model_count_9755.trans row_count_9755.symm
def row_9756 : RowCertificate 9756 where
  first := true
  first_correct := by
    have prime : Nat.Prime (corner 9756 1) := by change Nat.Prime 380737657; exact ProjectEulerSolutions.P58PrimeNodes.p_380737657.prime
    exact decide_eq_true prime
  second := false
  second_correct := by
    have composite : ¬ Nat.Prime (corner 9756 2) := by change ¬ Nat.Prime 380718145; exact PrattCertificates.composite_of_product (factor := 5) (cofactor := 76143629) rfl (by decide) (by decide)
    exact decide_eq_false composite
  third := false
  third_correct := by
    have composite : ¬ Nat.Prime (corner 9756 3) := by change ¬ Nat.Prime 380698633; exact PrattCertificates.composite_of_product (factor := 7) (cofactor := 54385519) rfl (by decide) (by decide)
    exact decide_eq_false composite
theorem row_count_9756 : row 9756 = 1 := (row_9756.checked).trans rfl
theorem model_count_9756 : modelRow 9756 = 1 := by decide
theorem modelrow_agrees_9756 : modelRow 9756 = row 9756 := model_count_9756.trans row_count_9756.symm
def row_9757 : RowCertificate 9757 where
  first := false
  first_correct := by
    have composite : ¬ Nat.Prime (corner 9757 1) := by change ¬ Nat.Prime 380815711; exact PrattCertificates.composite_of_product (factor := 3163) (cofactor := 120397) rfl (by decide) (by decide)
    exact decide_eq_false composite
  second := false
  second_correct := by
    have composite : ¬ Nat.Prime (corner 9757 2) := by change ¬ Nat.Prime 380796197; exact PrattCertificates.composite_of_product (factor := 7417) (cofactor := 51341) rfl (by decide) (by decide)
    exact decide_eq_false composite
  third := false
  third_correct := by
    have composite : ¬ Nat.Prime (corner 9757 3) := by change ¬ Nat.Prime 380776683; exact PrattCertificates.composite_of_product (factor := 3) (cofactor := 126925561) rfl (by decide) (by decide)
    exact decide_eq_false composite
theorem row_count_9757 : row 9757 = 0 := (row_9757.checked).trans rfl
theorem model_count_9757 : modelRow 9757 = 0 := by decide
theorem modelrow_agrees_9757 : modelRow 9757 = row 9757 := model_count_9757.trans row_count_9757.symm
def row_9758 : RowCertificate 9758 where
  first := false
  first_correct := by
    have composite : ¬ Nat.Prime (corner 9758 1) := by change ¬ Nat.Prime 380893773; exact PrattCertificates.composite_of_product (factor := 3) (cofactor := 126964591) rfl (by decide) (by decide)
    exact decide_eq_false composite
  second := false
  second_correct := by
    have composite : ¬ Nat.Prime (corner 9758 2) := by change ¬ Nat.Prime 380874257; exact PrattCertificates.composite_of_product (factor := 257) (cofactor := 1482001) rfl (by decide) (by decide)
    exact decide_eq_false composite
  third := false
  third_correct := by
    have composite : ¬ Nat.Prime (corner 9758 3) := by change ¬ Nat.Prime 380854741; exact PrattCertificates.composite_of_product (factor := 43) (cofactor := 8857087) rfl (by decide) (by decide)
    exact decide_eq_false composite
theorem row_count_9758 : row 9758 = 0 := (row_9758.checked).trans rfl
theorem model_count_9758 : modelRow 9758 = 0 := by decide
theorem modelrow_agrees_9758 : modelRow 9758 = row 9758 := model_count_9758.trans row_count_9758.symm
def row_9759 : RowCertificate 9759 where
  first := false
  first_correct := by
    have composite : ¬ Nat.Prime (corner 9759 1) := by change ¬ Nat.Prime 380971843; exact PrattCertificates.composite_of_product (factor := 7) (cofactor := 54424549) rfl (by decide) (by decide)
    exact decide_eq_false composite
  second := false
  second_correct := by
    have composite : ¬ Nat.Prime (corner 9759 2) := by change ¬ Nat.Prime 380952325; exact PrattCertificates.composite_of_product (factor := 5) (cofactor := 76190465) rfl (by decide) (by decide)
    exact decide_eq_false composite
  third := false
  third_correct := by
    have composite : ¬ Nat.Prime (corner 9759 3) := by change ¬ Nat.Prime 380932807; exact PrattCertificates.composite_of_product (factor := 457) (cofactor := 833551) rfl (by decide) (by decide)
    exact decide_eq_false composite
theorem row_count_9759 : row 9759 = 0 := (row_9759.checked).trans rfl
theorem model_count_9759 : modelRow 9759 = 0 := by decide
theorem modelrow_agrees_9759 : modelRow 9759 = row 9759 := model_count_9759.trans row_count_9759.symm
def row_9760 : RowCertificate 9760 where
  first := false
  first_correct := by
    have composite : ¬ Nat.Prime (corner 9760 1) := by change ¬ Nat.Prime 381049921; exact PrattCertificates.composite_of_product (factor := 7) (cofactor := 54435703) rfl (by decide) (by decide)
    exact decide_eq_false composite
  second := false
  second_correct := by
    have composite : ¬ Nat.Prime (corner 9760 2) := by change ¬ Nat.Prime 381030401; exact PrattCertificates.composite_of_product (factor := 17) (cofactor := 22413553) rfl (by decide) (by decide)
    exact decide_eq_false composite
  third := false
  third_correct := by
    have composite : ¬ Nat.Prime (corner 9760 3) := by change ¬ Nat.Prime 381010881; exact PrattCertificates.composite_of_product (factor := 3) (cofactor := 127003627) rfl (by decide) (by decide)
    exact decide_eq_false composite
theorem row_count_9760 : row 9760 = 0 := (row_9760.checked).trans rfl
theorem model_count_9760 : modelRow 9760 = 0 := by decide
theorem modelrow_agrees_9760 : modelRow 9760 = row 9760 := model_count_9760.trans row_count_9760.symm
def row_9761 : RowCertificate 9761 where
  first := false
  first_correct := by
    have composite : ¬ Nat.Prime (corner 9761 1) := by change ¬ Nat.Prime 381128007; exact PrattCertificates.composite_of_product (factor := 3) (cofactor := 127042669) rfl (by decide) (by decide)
    exact decide_eq_false composite
  second := false
  second_correct := by
    have composite : ¬ Nat.Prime (corner 9761 2) := by change ¬ Nat.Prime 381108485; exact PrattCertificates.composite_of_product (factor := 5) (cofactor := 76221697) rfl (by decide) (by decide)
    exact decide_eq_false composite
  third := true
  third_correct := by
    have prime : Nat.Prime (corner 9761 3) := by change Nat.Prime 381088963; exact ProjectEulerSolutions.P58PrimeNodes.p_381088963.prime
    exact decide_eq_true prime
theorem row_count_9761 : row 9761 = 1 := (row_9761.checked).trans rfl
theorem model_count_9761 : modelRow 9761 = 1 := by decide
theorem modelrow_agrees_9761 : modelRow 9761 = row 9761 := model_count_9761.trans row_count_9761.symm
def row_9762 : RowCertificate 9762 where
  first := false
  first_correct := by
    have composite : ¬ Nat.Prime (corner 9762 1) := by change ¬ Nat.Prime 381206101; exact PrattCertificates.composite_of_product (factor := 19) (cofactor := 20063479) rfl (by decide) (by decide)
    exact decide_eq_false composite
  second := true
  second_correct := by
    have prime : Nat.Prime (corner 9762 2) := by change Nat.Prime 381186577; exact ProjectEulerSolutions.P58PrimeNodes.p_381186577.prime
    exact decide_eq_true prime
  third := false
  third_correct := by
    have composite : ¬ Nat.Prime (corner 9762 3) := by change ¬ Nat.Prime 381167053; exact PrattCertificates.composite_of_product (factor := 103) (cofactor := 3700651) rfl (by decide) (by decide)
    exact decide_eq_false composite
theorem row_count_9762 : row 9762 = 1 := (row_9762.checked).trans rfl
theorem model_count_9762 : modelRow 9762 = 1 := by decide
theorem modelrow_agrees_9762 : modelRow 9762 = row 9762 := model_count_9762.trans row_count_9762.symm
def row_9763 : RowCertificate 9763 where
  first := false
  first_correct := by
    have composite : ¬ Nat.Prime (corner 9763 1) := by change ¬ Nat.Prime 381284203; exact PrattCertificates.composite_of_product (factor := 67) (cofactor := 5690809) rfl (by decide) (by decide)
    exact decide_eq_false composite
  second := false
  second_correct := by
    have composite : ¬ Nat.Prime (corner 9763 2) := by change ¬ Nat.Prime 381264677; exact PrattCertificates.composite_of_product (factor := 3833) (cofactor := 99469) rfl (by decide) (by decide)
    exact decide_eq_false composite
  third := false
  third_correct := by
    have composite : ¬ Nat.Prime (corner 9763 3) := by change ¬ Nat.Prime 381245151; exact PrattCertificates.composite_of_product (factor := 3) (cofactor := 127081717) rfl (by decide) (by decide)
    exact decide_eq_false composite
theorem row_count_9763 : row 9763 = 0 := (row_9763.checked).trans rfl
theorem model_count_9763 : modelRow 9763 = 0 := by decide
theorem modelrow_agrees_9763 : modelRow 9763 = row 9763 := model_count_9763.trans row_count_9763.symm
def row_9764 : RowCertificate 9764 where
  first := false
  first_correct := by
    have composite : ¬ Nat.Prime (corner 9764 1) := by change ¬ Nat.Prime 381362313; exact PrattCertificates.composite_of_product (factor := 3) (cofactor := 127120771) rfl (by decide) (by decide)
    exact decide_eq_false composite
  second := false
  second_correct := by
    have composite : ¬ Nat.Prime (corner 9764 2) := by change ¬ Nat.Prime 381342785; exact PrattCertificates.composite_of_product (factor := 5) (cofactor := 76268557) rfl (by decide) (by decide)
    exact decide_eq_false composite
  third := false
  third_correct := by
    have composite : ¬ Nat.Prime (corner 9764 3) := by change ¬ Nat.Prime 381323257; exact PrattCertificates.composite_of_product (factor := 7) (cofactor := 54474751) rfl (by decide) (by decide)
    exact decide_eq_false composite
theorem row_count_9764 : row 9764 = 0 := (row_9764.checked).trans rfl
theorem model_count_9764 : modelRow 9764 = 0 := by decide
theorem modelrow_agrees_9764 : modelRow 9764 = row 9764 := model_count_9764.trans row_count_9764.symm
def row_9765 : RowCertificate 9765 where
  first := true
  first_correct := by
    have prime : Nat.Prime (corner 9765 1) := by change Nat.Prime 381440431; exact ProjectEulerSolutions.P58PrimeNodes.p_381440431.prime
    exact decide_eq_true prime
  second := false
  second_correct := by
    have composite : ¬ Nat.Prime (corner 9765 2) := by change ¬ Nat.Prime 381420901; exact PrattCertificates.composite_of_product (factor := 37) (cofactor := 10308673) rfl (by decide) (by decide)
    exact decide_eq_false composite
  third := false
  third_correct := by
    have composite : ¬ Nat.Prime (corner 9765 3) := by change ¬ Nat.Prime 381401371; exact PrattCertificates.composite_of_product (factor := 13) (cofactor := 29338567) rfl (by decide) (by decide)
    exact decide_eq_false composite
theorem row_count_9765 : row 9765 = 1 := (row_9765.checked).trans rfl
theorem model_count_9765 : modelRow 9765 = 1 := by decide
theorem modelrow_agrees_9765 : modelRow 9765 = row 9765 := model_count_9765.trans row_count_9765.symm
def row_9766 : RowCertificate 9766 where
  first := false
  first_correct := by
    have composite : ¬ Nat.Prime (corner 9766 1) := by change ¬ Nat.Prime 381518557; exact PrattCertificates.composite_of_product (factor := 7) (cofactor := 54502651) rfl (by decide) (by decide)
    exact decide_eq_false composite
  second := false
  second_correct := by
    have composite : ¬ Nat.Prime (corner 9766 2) := by change ¬ Nat.Prime 381499025; exact PrattCertificates.composite_of_product (factor := 5) (cofactor := 76299805) rfl (by decide) (by decide)
    exact decide_eq_false composite
  third := false
  third_correct := by
    have composite : ¬ Nat.Prime (corner 9766 3) := by change ¬ Nat.Prime 381479493; exact PrattCertificates.composite_of_product (factor := 3) (cofactor := 127159831) rfl (by decide) (by decide)
    exact decide_eq_false composite
theorem row_count_9766 : row 9766 = 0 := (row_9766.checked).trans rfl
theorem model_count_9766 : modelRow 9766 = 0 := by decide
theorem modelrow_agrees_9766 : modelRow 9766 = row 9766 := model_count_9766.trans row_count_9766.symm
def row_9767 : RowCertificate 9767 where
  first := false
  first_correct := by
    have composite : ¬ Nat.Prime (corner 9767 1) := by change ¬ Nat.Prime 381596691; exact PrattCertificates.composite_of_product (factor := 3) (cofactor := 127198897) rfl (by decide) (by decide)
    exact decide_eq_false composite
  second := false
  second_correct := by
    have composite : ¬ Nat.Prime (corner 9767 2) := by change ¬ Nat.Prime 381577157; exact PrattCertificates.composite_of_product (factor := 13) (cofactor := 29352089) rfl (by decide) (by decide)
    exact decide_eq_false composite
  third := false
  third_correct := by
    have composite : ¬ Nat.Prime (corner 9767 3) := by change ¬ Nat.Prime 381557623; exact PrattCertificates.composite_of_product (factor := 61) (cofactor := 6255043) rfl (by decide) (by decide)
    exact decide_eq_false composite
theorem row_count_9767 : row 9767 = 0 := (row_9767.checked).trans rfl
theorem model_count_9767 : modelRow 9767 = 0 := by decide
theorem modelrow_agrees_9767 : modelRow 9767 = row 9767 := model_count_9767.trans row_count_9767.symm
def row_9768 : RowCertificate 9768 where
  first := false
  first_correct := by
    have composite : ¬ Nat.Prime (corner 9768 1) := by change ¬ Nat.Prime 381674833; exact PrattCertificates.composite_of_product (factor := 79) (cofactor := 4831327) rfl (by decide) (by decide)
    exact decide_eq_false composite
  second := true
  second_correct := by
    have prime : Nat.Prime (corner 9768 2) := by change Nat.Prime 381655297; exact ProjectEulerSolutions.P58PrimeNodes.p_381655297.prime
    exact decide_eq_true prime
  third := false
  third_correct := by
    have composite : ¬ Nat.Prime (corner 9768 3) := by change ¬ Nat.Prime 381635761; exact PrattCertificates.composite_of_product (factor := 13) (cofactor := 29356597) rfl (by decide) (by decide)
    exact decide_eq_false composite
theorem row_count_9768 : row 9768 = 1 := (row_9768.checked).trans rfl
theorem model_count_9768 : modelRow 9768 = 1 := by decide
theorem modelrow_agrees_9768 : modelRow 9768 = row 9768 := model_count_9768.trans row_count_9768.symm
def row_9769 : RowCertificate 9769 where
  first := false
  first_correct := by
    have composite : ¬ Nat.Prime (corner 9769 1) := by change ¬ Nat.Prime 381752983; exact PrattCertificates.composite_of_product (factor := 127) (cofactor := 3005929) rfl (by decide) (by decide)
    exact decide_eq_false composite
  second := false
  second_correct := by
    have composite : ¬ Nat.Prime (corner 9769 2) := by change ¬ Nat.Prime 381733445; exact PrattCertificates.composite_of_product (factor := 5) (cofactor := 76346689) rfl (by decide) (by decide)
    exact decide_eq_false composite
  third := false
  third_correct := by
    have composite : ¬ Nat.Prime (corner 9769 3) := by change ¬ Nat.Prime 381713907; exact PrattCertificates.composite_of_product (factor := 3) (cofactor := 127237969) rfl (by decide) (by decide)
    exact decide_eq_false composite
theorem row_count_9769 : row 9769 = 0 := (row_9769.checked).trans rfl
theorem model_count_9769 : modelRow 9769 = 0 := by decide
theorem modelrow_agrees_9769 : modelRow 9769 = row 9769 := model_count_9769.trans row_count_9769.symm
def row_9770 : RowCertificate 9770 where
  first := false
  first_correct := by
    have composite : ¬ Nat.Prime (corner 9770 1) := by change ¬ Nat.Prime 381831141; exact PrattCertificates.composite_of_product (factor := 3) (cofactor := 127277047) rfl (by decide) (by decide)
    exact decide_eq_false composite
  second := false
  second_correct := by
    have composite : ¬ Nat.Prime (corner 9770 2) := by change ¬ Nat.Prime 381811601; exact PrattCertificates.composite_of_product (factor := 317) (cofactor := 1204453) rfl (by decide) (by decide)
    exact decide_eq_false composite
  third := false
  third_correct := by
    have composite : ¬ Nat.Prime (corner 9770 3) := by change ¬ Nat.Prime 381792061; exact PrattCertificates.composite_of_product (factor := 7) (cofactor := 54541723) rfl (by decide) (by decide)
    exact decide_eq_false composite
theorem row_count_9770 : row 9770 = 0 := (row_9770.checked).trans rfl
theorem model_count_9770 : modelRow 9770 = 0 := by decide
theorem modelrow_agrees_9770 : modelRow 9770 = row 9770 := model_count_9770.trans row_count_9770.symm
def row_9771 : RowCertificate 9771 where
  first := false
  first_correct := by
    have composite : ¬ Nat.Prime (corner 9771 1) := by change ¬ Nat.Prime 381909307; exact PrattCertificates.composite_of_product (factor := 13) (cofactor := 29377639) rfl (by decide) (by decide)
    exact decide_eq_false composite
  second := false
  second_correct := by
    have composite : ¬ Nat.Prime (corner 9771 2) := by change ¬ Nat.Prime 381889765; exact PrattCertificates.composite_of_product (factor := 5) (cofactor := 76377953) rfl (by decide) (by decide)
    exact decide_eq_false composite
  third := false
  third_correct := by
    have composite : ¬ Nat.Prime (corner 9771 3) := by change ¬ Nat.Prime 381870223; exact PrattCertificates.composite_of_product (factor := 7) (cofactor := 54552889) rfl (by decide) (by decide)
    exact decide_eq_false composite
theorem row_count_9771 : row 9771 = 0 := (row_9771.checked).trans rfl
theorem model_count_9771 : modelRow 9771 = 0 := by decide
theorem modelrow_agrees_9771 : modelRow 9771 = row 9771 := model_count_9771.trans row_count_9771.symm
def row_9772 : RowCertificate 9772 where
  first := true
  first_correct := by
    have prime : Nat.Prime (corner 9772 1) := by change Nat.Prime 381987481; exact ProjectEulerSolutions.P58PrimeNodes.p_381987481.prime
    exact decide_eq_true prime
  second := false
  second_correct := by
    have composite : ¬ Nat.Prime (corner 9772 2) := by change ¬ Nat.Prime 381967937; exact PrattCertificates.composite_of_product (factor := 13) (cofactor := 29382149) rfl (by decide) (by decide)
    exact decide_eq_false composite
  third := false
  third_correct := by
    have composite : ¬ Nat.Prime (corner 9772 3) := by change ¬ Nat.Prime 381948393; exact PrattCertificates.composite_of_product (factor := 3) (cofactor := 127316131) rfl (by decide) (by decide)
    exact decide_eq_false composite
theorem row_count_9772 : row 9772 = 1 := (row_9772.checked).trans rfl
theorem model_count_9772 : modelRow 9772 = 1 := by decide
theorem modelrow_agrees_9772 : modelRow 9772 = row 9772 := model_count_9772.trans row_count_9772.symm
def row_9773 : RowCertificate 9773 where
  first := false
  first_correct := by
    have composite : ¬ Nat.Prime (corner 9773 1) := by change ¬ Nat.Prime 382065663; exact PrattCertificates.composite_of_product (factor := 3) (cofactor := 127355221) rfl (by decide) (by decide)
    exact decide_eq_false composite
  second := false
  second_correct := by
    have composite : ¬ Nat.Prime (corner 9773 2) := by change ¬ Nat.Prime 382046117; exact PrattCertificates.composite_of_product (factor := 17) (cofactor := 22473301) rfl (by decide) (by decide)
    exact decide_eq_false composite
  third := false
  third_correct := by
    have composite : ¬ Nat.Prime (corner 9773 3) := by change ¬ Nat.Prime 382026571; exact PrattCertificates.composite_of_product (factor := 1009) (cofactor := 378619) rfl (by decide) (by decide)
    exact decide_eq_false composite
theorem row_count_9773 : row 9773 = 0 := (row_9773.checked).trans rfl
theorem model_count_9773 : modelRow 9773 = 0 := by decide
theorem modelrow_agrees_9773 : modelRow 9773 = row 9773 := model_count_9773.trans row_count_9773.symm
def row_9774 : RowCertificate 9774 where
  first := false
  first_correct := by
    have composite : ¬ Nat.Prime (corner 9774 1) := by change ¬ Nat.Prime 382143853; exact PrattCertificates.composite_of_product (factor := 7) (cofactor := 54591979) rfl (by decide) (by decide)
    exact decide_eq_false composite
  second := false
  second_correct := by
    have composite : ¬ Nat.Prime (corner 9774 2) := by change ¬ Nat.Prime 382124305; exact PrattCertificates.composite_of_product (factor := 5) (cofactor := 76424861) rfl (by decide) (by decide)
    exact decide_eq_false composite
  third := false
  third_correct := by
    have composite : ¬ Nat.Prime (corner 9774 3) := by change ¬ Nat.Prime 382104757; exact PrattCertificates.composite_of_product (factor := 7369) (cofactor := 51853) rfl (by decide) (by decide)
    exact decide_eq_false composite
theorem row_count_9774 : row 9774 = 0 := (row_9774.checked).trans rfl
theorem model_count_9774 : modelRow 9774 = 0 := by decide
theorem modelrow_agrees_9774 : modelRow 9774 = row 9774 := model_count_9774.trans row_count_9774.symm
def row_9775 : RowCertificate 9775 where
  first := false
  first_correct := by
    have composite : ¬ Nat.Prime (corner 9775 1) := by change ¬ Nat.Prime 382222051; exact PrattCertificates.composite_of_product (factor := 277) (cofactor := 1379863) rfl (by decide) (by decide)
    exact decide_eq_false composite
  second := false
  second_correct := by
    have composite : ¬ Nat.Prime (corner 9775 2) := by change ¬ Nat.Prime 382202501; exact PrattCertificates.composite_of_product (factor := 509) (cofactor := 750889) rfl (by decide) (by decide)
    exact decide_eq_false composite
  third := false
  third_correct := by
    have composite : ¬ Nat.Prime (corner 9775 3) := by change ¬ Nat.Prime 382182951; exact PrattCertificates.composite_of_product (factor := 3) (cofactor := 127394317) rfl (by decide) (by decide)
    exact decide_eq_false composite
theorem row_count_9775 : row 9775 = 0 := (row_9775.checked).trans rfl
theorem model_count_9775 : modelRow 9775 = 0 := by decide
theorem modelrow_agrees_9775 : modelRow 9775 = row 9775 := model_count_9775.trans row_count_9775.symm
def row_9776 : RowCertificate 9776 where
  first := false
  first_correct := by
    have composite : ¬ Nat.Prime (corner 9776 1) := by change ¬ Nat.Prime 382300257; exact PrattCertificates.composite_of_product (factor := 3) (cofactor := 127433419) rfl (by decide) (by decide)
    exact decide_eq_false composite
  second := false
  second_correct := by
    have composite : ¬ Nat.Prime (corner 9776 2) := by change ¬ Nat.Prime 382280705; exact PrattCertificates.composite_of_product (factor := 5) (cofactor := 76456141) rfl (by decide) (by decide)
    exact decide_eq_false composite
  third := true
  third_correct := by
    have prime : Nat.Prime (corner 9776 3) := by change Nat.Prime 382261153; exact ProjectEulerSolutions.P58PrimeNodes.p_382261153.prime
    exact decide_eq_true prime
theorem row_count_9776 : row 9776 = 1 := (row_9776.checked).trans rfl
theorem model_count_9776 : modelRow 9776 = 1 := by decide
theorem modelrow_agrees_9776 : modelRow 9776 = row 9776 := model_count_9776.trans row_count_9776.symm
def row_9777 : RowCertificate 9777 where
  first := true
  first_correct := by
    have prime : Nat.Prime (corner 9777 1) := by change Nat.Prime 382378471; exact ProjectEulerSolutions.P58PrimeNodes.p_382378471.prime
    exact decide_eq_true prime
  second := false
  second_correct := by
    have composite : ¬ Nat.Prime (corner 9777 2) := by change ¬ Nat.Prime 382358917; exact PrattCertificates.composite_of_product (factor := 17) (cofactor := 22491701) rfl (by decide) (by decide)
    exact decide_eq_false composite
  third := false
  third_correct := by
    have composite : ¬ Nat.Prime (corner 9777 3) := by change ¬ Nat.Prime 382339363; exact PrattCertificates.composite_of_product (factor := 7) (cofactor := 54619909) rfl (by decide) (by decide)
    exact decide_eq_false composite
theorem row_count_9777 : row 9777 = 1 := (row_9777.checked).trans rfl
theorem model_count_9777 : modelRow 9777 = 1 := by decide
theorem modelrow_agrees_9777 : modelRow 9777 = row 9777 := model_count_9777.trans row_count_9777.symm
def row_9778 : RowCertificate 9778 where
  first := false
  first_correct := by
    have composite : ¬ Nat.Prime (corner 9778 1) := by change ¬ Nat.Prime 382456693; exact PrattCertificates.composite_of_product (factor := 109) (cofactor := 3508777) rfl (by decide) (by decide)
    exact decide_eq_false composite
  second := true
  second_correct := by
    have prime : Nat.Prime (corner 9778 2) := by change Nat.Prime 382437137; exact ProjectEulerSolutions.P58PrimeNodes.p_382437137.prime
    exact decide_eq_true prime
  third := false
  third_correct := by
    have composite : ¬ Nat.Prime (corner 9778 3) := by change ¬ Nat.Prime 382417581; exact PrattCertificates.composite_of_product (factor := 3) (cofactor := 127472527) rfl (by decide) (by decide)
    exact decide_eq_false composite
theorem row_count_9778 : row 9778 = 1 := (row_9778.checked).trans rfl
theorem model_count_9778 : modelRow 9778 = 1 := by decide
theorem modelrow_agrees_9778 : modelRow 9778 = row 9778 := model_count_9778.trans row_count_9778.symm
def row_9779 : RowCertificate 9779 where
  first := false
  first_correct := by
    have composite : ¬ Nat.Prime (corner 9779 1) := by change ¬ Nat.Prime 382534923; exact PrattCertificates.composite_of_product (factor := 3) (cofactor := 127511641) rfl (by decide) (by decide)
    exact decide_eq_false composite
  second := false
  second_correct := by
    have composite : ¬ Nat.Prime (corner 9779 2) := by change ¬ Nat.Prime 382515365; exact PrattCertificates.composite_of_product (factor := 5) (cofactor := 76503073) rfl (by decide) (by decide)
    exact decide_eq_false composite
  third := false
  third_correct := by
    have composite : ¬ Nat.Prime (corner 9779 3) := by change ¬ Nat.Prime 382495807; exact PrattCertificates.composite_of_product (factor := 1327) (cofactor := 288241) rfl (by decide) (by decide)
    exact decide_eq_false composite
theorem row_count_9779 : row 9779 = 0 := (row_9779.checked).trans rfl
theorem model_count_9779 : modelRow 9779 = 0 := by decide
theorem modelrow_agrees_9779 : modelRow 9779 = row 9779 := model_count_9779.trans row_count_9779.symm
def row_9780 : RowCertificate 9780 where
  first := false
  first_correct := by
    have composite : ¬ Nat.Prime (corner 9780 1) := by change ¬ Nat.Prime 382613161; exact PrattCertificates.composite_of_product (factor := 7) (cofactor := 54659023) rfl (by decide) (by decide)
    exact decide_eq_false composite
  second := false
  second_correct := by
    have composite : ¬ Nat.Prime (corner 9780 2) := by change ¬ Nat.Prime 382593601; exact PrattCertificates.composite_of_product (factor := 13) (cofactor := 29430277) rfl (by decide) (by decide)
    exact decide_eq_false composite
  third := false
  third_correct := by
    have composite : ¬ Nat.Prime (corner 9780 3) := by change ¬ Nat.Prime 382574041; exact PrattCertificates.composite_of_product (factor := 229) (cofactor := 1670629) rfl (by decide) (by decide)
    exact decide_eq_false composite
theorem row_count_9780 : row 9780 = 0 := (row_9780.checked).trans rfl
theorem model_count_9780 : modelRow 9780 = 0 := by decide
theorem modelrow_agrees_9780 : modelRow 9780 = row 9780 := model_count_9780.trans row_count_9780.symm
def row_9781 : RowCertificate 9781 where
  first := false
  first_correct := by
    have composite : ¬ Nat.Prime (corner 9781 1) := by change ¬ Nat.Prime 382691407; exact PrattCertificates.composite_of_product (factor := 7) (cofactor := 54670201) rfl (by decide) (by decide)
    exact decide_eq_false composite
  second := false
  second_correct := by
    have composite : ¬ Nat.Prime (corner 9781 2) := by change ¬ Nat.Prime 382671845; exact PrattCertificates.composite_of_product (factor := 5) (cofactor := 76534369) rfl (by decide) (by decide)
    exact decide_eq_false composite
  third := false
  third_correct := by
    have composite : ¬ Nat.Prime (corner 9781 3) := by change ¬ Nat.Prime 382652283; exact PrattCertificates.composite_of_product (factor := 3) (cofactor := 127550761) rfl (by decide) (by decide)
    exact decide_eq_false composite
theorem row_count_9781 : row 9781 = 0 := (row_9781.checked).trans rfl
theorem model_count_9781 : modelRow 9781 = 0 := by decide
theorem modelrow_agrees_9781 : modelRow 9781 = row 9781 := model_count_9781.trans row_count_9781.symm
def row_9782 : RowCertificate 9782 where
  first := false
  first_correct := by
    have composite : ¬ Nat.Prime (corner 9782 1) := by change ¬ Nat.Prime 382769661; exact PrattCertificates.composite_of_product (factor := 3) (cofactor := 127589887) rfl (by decide) (by decide)
    exact decide_eq_false composite
  second := false
  second_correct := by
    have composite : ¬ Nat.Prime (corner 9782 2) := by change ¬ Nat.Prime 382750097; exact PrattCertificates.composite_of_product (factor := 113) (cofactor := 3387169) rfl (by decide) (by decide)
    exact decide_eq_false composite
  third := false
  third_correct := by
    have composite : ¬ Nat.Prime (corner 9782 3) := by change ¬ Nat.Prime 382730533; exact PrattCertificates.composite_of_product (factor := 2731) (cofactor := 140143) rfl (by decide) (by decide)
    exact decide_eq_false composite
theorem row_count_9782 : row 9782 = 0 := (row_9782.checked).trans rfl
theorem model_count_9782 : modelRow 9782 = 0 := by decide
theorem modelrow_agrees_9782 : modelRow 9782 = row 9782 := model_count_9782.trans row_count_9782.symm
def row_9783 : RowCertificate 9783 where
  first := false
  first_correct := by
    have composite : ¬ Nat.Prime (corner 9783 1) := by change ¬ Nat.Prime 382847923; exact PrattCertificates.composite_of_product (factor := 31) (cofactor := 12349933) rfl (by decide) (by decide)
    exact decide_eq_false composite
  second := false
  second_correct := by
    have composite : ¬ Nat.Prime (corner 9783 2) := by change ¬ Nat.Prime 382828357; exact PrattCertificates.composite_of_product (factor := 41) (cofactor := 9337277) rfl (by decide) (by decide)
    exact decide_eq_false composite
  third := false
  third_correct := by
    have composite : ¬ Nat.Prime (corner 9783 3) := by change ¬ Nat.Prime 382808791; exact PrattCertificates.composite_of_product (factor := 6451) (cofactor := 59341) rfl (by decide) (by decide)
    exact decide_eq_false composite
theorem row_count_9783 : row 9783 = 0 := (row_9783.checked).trans rfl
theorem model_count_9783 : modelRow 9783 = 0 := by decide
theorem modelrow_agrees_9783 : modelRow 9783 = row 9783 := model_count_9783.trans row_count_9783.symm
def row_9784 : RowCertificate 9784 where
  first := false
  first_correct := by
    have composite : ¬ Nat.Prime (corner 9784 1) := by change ¬ Nat.Prime 382926193; exact PrattCertificates.composite_of_product (factor := 13) (cofactor := 29455861) rfl (by decide) (by decide)
    exact decide_eq_false composite
  second := false
  second_correct := by
    have composite : ¬ Nat.Prime (corner 9784 2) := by change ¬ Nat.Prime 382906625; exact PrattCertificates.composite_of_product (factor := 5) (cofactor := 76581325) rfl (by decide) (by decide)
    exact decide_eq_false composite
  third := false
  third_correct := by
    have composite : ¬ Nat.Prime (corner 9784 3) := by change ¬ Nat.Prime 382887057; exact PrattCertificates.composite_of_product (factor := 3) (cofactor := 127629019) rfl (by decide) (by decide)
    exact decide_eq_false composite
theorem row_count_9784 : row 9784 = 0 := (row_9784.checked).trans rfl
theorem model_count_9784 : modelRow 9784 = 0 := by decide
theorem modelrow_agrees_9784 : modelRow 9784 = row 9784 := model_count_9784.trans row_count_9784.symm
def row_9785 : RowCertificate 9785 where
  first := false
  first_correct := by
    have composite : ¬ Nat.Prime (corner 9785 1) := by change ¬ Nat.Prime 383004471; exact PrattCertificates.composite_of_product (factor := 3) (cofactor := 127668157) rfl (by decide) (by decide)
    exact decide_eq_false composite
  second := false
  second_correct := by
    have composite : ¬ Nat.Prime (corner 9785 2) := by change ¬ Nat.Prime 382984901; exact PrattCertificates.composite_of_product (factor := 13) (cofactor := 29460377) rfl (by decide) (by decide)
    exact decide_eq_false composite
  third := false
  third_correct := by
    have composite : ¬ Nat.Prime (corner 9785 3) := by change ¬ Nat.Prime 382965331; exact PrattCertificates.composite_of_product (factor := 7) (cofactor := 54709333) rfl (by decide) (by decide)
    exact decide_eq_false composite
theorem row_count_9785 : row 9785 = 0 := (row_9785.checked).trans rfl
theorem model_count_9785 : modelRow 9785 = 0 := by decide
theorem modelrow_agrees_9785 : modelRow 9785 = row 9785 := model_count_9785.trans row_count_9785.symm
def row_9786 : RowCertificate 9786 where
  first := false
  first_correct := by
    have composite : ¬ Nat.Prime (corner 9786 1) := by change ¬ Nat.Prime 383082757; exact PrattCertificates.composite_of_product (factor := 73) (cofactor := 5247709) rfl (by decide) (by decide)
    exact decide_eq_false composite
  second := false
  second_correct := by
    have composite : ¬ Nat.Prime (corner 9786 2) := by change ¬ Nat.Prime 383063185; exact PrattCertificates.composite_of_product (factor := 5) (cofactor := 76612637) rfl (by decide) (by decide)
    exact decide_eq_false composite
  third := false
  third_correct := by
    have composite : ¬ Nat.Prime (corner 9786 3) := by change ¬ Nat.Prime 383043613; exact PrattCertificates.composite_of_product (factor := 43) (cofactor := 8907991) rfl (by decide) (by decide)
    exact decide_eq_false composite
theorem row_count_9786 : row 9786 = 0 := (row_9786.checked).trans rfl
theorem model_count_9786 : modelRow 9786 = 0 := by decide
theorem modelrow_agrees_9786 : modelRow 9786 = row 9786 := model_count_9786.trans row_count_9786.symm
def row_9787 : RowCertificate 9787 where
  first := false
  first_correct := by
    have composite : ¬ Nat.Prime (corner 9787 1) := by change ¬ Nat.Prime 383161051; exact PrattCertificates.composite_of_product (factor := 7) (cofactor := 54737293) rfl (by decide) (by decide)
    exact decide_eq_false composite
  second := false
  second_correct := by
    have composite : ¬ Nat.Prime (corner 9787 2) := by change ¬ Nat.Prime 383141477; exact PrattCertificates.composite_of_product (factor := 193) (cofactor := 1985189) rfl (by decide) (by decide)
    exact decide_eq_false composite
  third := false
  third_correct := by
    have composite : ¬ Nat.Prime (corner 9787 3) := by change ¬ Nat.Prime 383121903; exact PrattCertificates.composite_of_product (factor := 3) (cofactor := 127707301) rfl (by decide) (by decide)
    exact decide_eq_false composite
theorem row_count_9787 : row 9787 = 0 := (row_9787.checked).trans rfl
theorem model_count_9787 : modelRow 9787 = 0 := by decide
theorem modelrow_agrees_9787 : modelRow 9787 = row 9787 := model_count_9787.trans row_count_9787.symm
def row_9788 : RowCertificate 9788 where
  first := false
  first_correct := by
    have composite : ¬ Nat.Prime (corner 9788 1) := by change ¬ Nat.Prime 383239353; exact PrattCertificates.composite_of_product (factor := 3) (cofactor := 127746451) rfl (by decide) (by decide)
    exact decide_eq_false composite
  second := true
  second_correct := by
    have prime : Nat.Prime (corner 9788 2) := by change Nat.Prime 383219777; exact ProjectEulerSolutions.P58PrimeNodes.p_383219777.prime
    exact decide_eq_true prime
  third := true
  third_correct := by
    have prime : Nat.Prime (corner 9788 3) := by change Nat.Prime 383200201; exact ProjectEulerSolutions.P58PrimeNodes.p_383200201.prime
    exact decide_eq_true prime
theorem row_count_9788 : row 9788 = 2 := (row_9788.checked).trans rfl
theorem model_count_9788 : modelRow 9788 = 2 := by decide
theorem modelrow_agrees_9788 : modelRow 9788 = row 9788 := model_count_9788.trans row_count_9788.symm
def row_9789 : RowCertificate 9789 where
  first := false
  first_correct := by
    have composite : ¬ Nat.Prime (corner 9789 1) := by change ¬ Nat.Prime 383317663; exact PrattCertificates.composite_of_product (factor := 409) (cofactor := 937207) rfl (by decide) (by decide)
    exact decide_eq_false composite
  second := false
  second_correct := by
    have composite : ¬ Nat.Prime (corner 9789 2) := by change ¬ Nat.Prime 383298085; exact PrattCertificates.composite_of_product (factor := 5) (cofactor := 76659617) rfl (by decide) (by decide)
    exact decide_eq_false composite
  third := false
  third_correct := by
    have composite : ¬ Nat.Prime (corner 9789 3) := by change ¬ Nat.Prime 383278507; exact PrattCertificates.composite_of_product (factor := 19) (cofactor := 20172553) rfl (by decide) (by decide)
    exact decide_eq_false composite
theorem row_count_9789 : row 9789 = 0 := (row_9789.checked).trans rfl
theorem model_count_9789 : modelRow 9789 = 0 := by decide
theorem modelrow_agrees_9789 : modelRow 9789 = row 9789 := model_count_9789.trans row_count_9789.symm
def row_9790 : RowCertificate 9790 where
  first := false
  first_correct := by
    have composite : ¬ Nat.Prime (corner 9790 1) := by change ¬ Nat.Prime 383395981; exact PrattCertificates.composite_of_product (factor := 1129) (cofactor := 339589) rfl (by decide) (by decide)
    exact decide_eq_false composite
  second := false
  second_correct := by
    have composite : ¬ Nat.Prime (corner 9790 2) := by change ¬ Nat.Prime 383376401; exact PrattCertificates.composite_of_product (factor := 17) (cofactor := 22551553) rfl (by decide) (by decide)
    exact decide_eq_false composite
  third := false
  third_correct := by
    have composite : ¬ Nat.Prime (corner 9790 3) := by change ¬ Nat.Prime 383356821; exact PrattCertificates.composite_of_product (factor := 3) (cofactor := 127785607) rfl (by decide) (by decide)
    exact decide_eq_false composite
theorem row_count_9790 : row 9790 = 0 := (row_9790.checked).trans rfl
theorem model_count_9790 : modelRow 9790 = 0 := by decide
theorem modelrow_agrees_9790 : modelRow 9790 = row 9790 := model_count_9790.trans row_count_9790.symm
def row_9791 : RowCertificate 9791 where
  first := false
  first_correct := by
    have composite : ¬ Nat.Prime (corner 9791 1) := by change ¬ Nat.Prime 383474307; exact PrattCertificates.composite_of_product (factor := 3) (cofactor := 127824769) rfl (by decide) (by decide)
    exact decide_eq_false composite
  second := false
  second_correct := by
    have composite : ¬ Nat.Prime (corner 9791 2) := by change ¬ Nat.Prime 383454725; exact PrattCertificates.composite_of_product (factor := 5) (cofactor := 76690945) rfl (by decide) (by decide)
    exact decide_eq_false composite
  third := false
  third_correct := by
    have composite : ¬ Nat.Prime (corner 9791 3) := by change ¬ Nat.Prime 383435143; exact PrattCertificates.composite_of_product (factor := 7) (cofactor := 54776449) rfl (by decide) (by decide)
    exact decide_eq_false composite
theorem row_count_9791 : row 9791 = 0 := (row_9791.checked).trans rfl
theorem model_count_9791 : modelRow 9791 = 0 := by decide
theorem modelrow_agrees_9791 : modelRow 9791 = row 9791 := model_count_9791.trans row_count_9791.symm
def row_9792 : RowCertificate 9792 where
  first := false
  first_correct := by
    have composite : ¬ Nat.Prime (corner 9792 1) := by change ¬ Nat.Prime 383552641; exact PrattCertificates.composite_of_product (factor := 223) (cofactor := 1719967) rfl (by decide) (by decide)
    exact decide_eq_false composite
  second := false
  second_correct := by
    have composite : ¬ Nat.Prime (corner 9792 2) := by change ¬ Nat.Prime 383533057; exact PrattCertificates.composite_of_product (factor := 101) (cofactor := 3797357) rfl (by decide) (by decide)
    exact decide_eq_false composite
  third := false
  third_correct := by
    have composite : ¬ Nat.Prime (corner 9792 3) := by change ¬ Nat.Prime 383513473; exact PrattCertificates.composite_of_product (factor := 7) (cofactor := 54787639) rfl (by decide) (by decide)
    exact decide_eq_false composite
theorem row_count_9792 : row 9792 = 0 := (row_9792.checked).trans rfl
theorem model_count_9792 : modelRow 9792 = 0 := by decide
theorem modelrow_agrees_9792 : modelRow 9792 = row 9792 := model_count_9792.trans row_count_9792.symm
def row_9793 : RowCertificate 9793 where
  first := false
  first_correct := by
    have composite : ¬ Nat.Prime (corner 9793 1) := by change ¬ Nat.Prime 383630983; exact PrattCertificates.composite_of_product (factor := 31) (cofactor := 12375193) rfl (by decide) (by decide)
    exact decide_eq_false composite
  second := false
  second_correct := by
    have composite : ¬ Nat.Prime (corner 9793 2) := by change ¬ Nat.Prime 383611397; exact PrattCertificates.composite_of_product (factor := 13) (cofactor := 29508569) rfl (by decide) (by decide)
    exact decide_eq_false composite
  third := false
  third_correct := by
    have composite : ¬ Nat.Prime (corner 9793 3) := by change ¬ Nat.Prime 383591811; exact PrattCertificates.composite_of_product (factor := 3) (cofactor := 127863937) rfl (by decide) (by decide)
    exact decide_eq_false composite
theorem row_count_9793 : row 9793 = 0 := (row_9793.checked).trans rfl
theorem model_count_9793 : modelRow 9793 = 0 := by decide
theorem modelrow_agrees_9793 : modelRow 9793 = row 9793 := model_count_9793.trans row_count_9793.symm
def row_9794 : RowCertificate 9794 where
  first := false
  first_correct := by
    have composite : ¬ Nat.Prime (corner 9794 1) := by change ¬ Nat.Prime 383709333; exact PrattCertificates.composite_of_product (factor := 3) (cofactor := 127903111) rfl (by decide) (by decide)
    exact decide_eq_false composite
  second := false
  second_correct := by
    have composite : ¬ Nat.Prime (corner 9794 2) := by change ¬ Nat.Prime 383689745; exact PrattCertificates.composite_of_product (factor := 5) (cofactor := 76737949) rfl (by decide) (by decide)
    exact decide_eq_false composite
  third := false
  third_correct := by
    have composite : ¬ Nat.Prime (corner 9794 3) := by change ¬ Nat.Prime 383670157; exact PrattCertificates.composite_of_product (factor := 13) (cofactor := 29513089) rfl (by decide) (by decide)
    exact decide_eq_false composite
theorem row_count_9794 : row 9794 = 0 := (row_9794.checked).trans rfl
theorem model_count_9794 : modelRow 9794 = 0 := by decide
theorem modelrow_agrees_9794 : modelRow 9794 = row 9794 := model_count_9794.trans row_count_9794.symm
def row_9795 : RowCertificate 9795 where
  first := false
  first_correct := by
    have composite : ¬ Nat.Prime (corner 9795 1) := by change ¬ Nat.Prime 383787691; exact PrattCertificates.composite_of_product (factor := 7) (cofactor := 54826813) rfl (by decide) (by decide)
    exact decide_eq_false composite
  second := false
  second_correct := by
    have composite : ¬ Nat.Prime (corner 9795 2) := by change ¬ Nat.Prime 383768101; exact PrattCertificates.composite_of_product (factor := 2089) (cofactor := 183709) rfl (by decide) (by decide)
    exact decide_eq_false composite
  third := true
  third_correct := by
    have prime : Nat.Prime (corner 9795 3) := by change Nat.Prime 383748511; exact ProjectEulerSolutions.P58PrimeNodes.p_383748511.prime
    exact decide_eq_true prime
theorem row_count_9795 : row 9795 = 1 := (row_9795.checked).trans rfl
theorem model_count_9795 : modelRow 9795 = 1 := by decide
theorem modelrow_agrees_9795 : modelRow 9795 = row 9795 := model_count_9795.trans row_count_9795.symm
def row_9796 : RowCertificate 9796 where
  first := true
  first_correct := by
    have prime : Nat.Prime (corner 9796 1) := by change Nat.Prime 383866057; exact ProjectEulerSolutions.P58PrimeNodes.p_383866057.prime
    exact decide_eq_true prime
  second := false
  second_correct := by
    have composite : ¬ Nat.Prime (corner 9796 2) := by change ¬ Nat.Prime 383846465; exact PrattCertificates.composite_of_product (factor := 5) (cofactor := 76769293) rfl (by decide) (by decide)
    exact decide_eq_false composite
  third := false
  third_correct := by
    have composite : ¬ Nat.Prime (corner 9796 3) := by change ¬ Nat.Prime 383826873; exact PrattCertificates.composite_of_product (factor := 3) (cofactor := 127942291) rfl (by decide) (by decide)
    exact decide_eq_false composite
theorem row_count_9796 : row 9796 = 1 := (row_9796.checked).trans rfl
theorem model_count_9796 : modelRow 9796 = 1 := by decide
theorem modelrow_agrees_9796 : modelRow 9796 = row 9796 := model_count_9796.trans row_count_9796.symm
def row_9797 : RowCertificate 9797 where
  first := false
  first_correct := by
    have composite : ¬ Nat.Prime (corner 9797 1) := by change ¬ Nat.Prime 383944431; exact PrattCertificates.composite_of_product (factor := 3) (cofactor := 127981477) rfl (by decide) (by decide)
    exact decide_eq_false composite
  second := false
  second_correct := by
    have composite : ¬ Nat.Prime (corner 9797 2) := by change ¬ Nat.Prime 383924837; exact PrattCertificates.composite_of_product (factor := 577) (cofactor := 665381) rfl (by decide) (by decide)
    exact decide_eq_false composite
  third := false
  third_correct := by
    have composite : ¬ Nat.Prime (corner 9797 3) := by change ¬ Nat.Prime 383905243; exact PrattCertificates.composite_of_product (factor := 67) (cofactor := 5729929) rfl (by decide) (by decide)
    exact decide_eq_false composite
theorem row_count_9797 : row 9797 = 0 := (row_9797.checked).trans rfl
theorem model_count_9797 : modelRow 9797 = 0 := by decide
theorem modelrow_agrees_9797 : modelRow 9797 = row 9797 := model_count_9797.trans row_count_9797.symm
def row_9798 : RowCertificate 9798 where
  first := false
  first_correct := by
    have composite : ¬ Nat.Prime (corner 9798 1) := by change ¬ Nat.Prime 384022813; exact PrattCertificates.composite_of_product (factor := 19) (cofactor := 20211727) rfl (by decide) (by decide)
    exact decide_eq_false composite
  second := false
  second_correct := by
    have composite : ¬ Nat.Prime (corner 9798 2) := by change ¬ Nat.Prime 384003217; exact PrattCertificates.composite_of_product (factor := 13) (cofactor := 29538709) rfl (by decide) (by decide)
    exact decide_eq_false composite
  third := false
  third_correct := by
    have composite : ¬ Nat.Prime (corner 9798 3) := by change ¬ Nat.Prime 383983621; exact PrattCertificates.composite_of_product (factor := 7) (cofactor := 54854803) rfl (by decide) (by decide)
    exact decide_eq_false composite
theorem row_count_9798 : row 9798 = 0 := (row_9798.checked).trans rfl
theorem model_count_9798 : modelRow 9798 = 0 := by decide
theorem modelrow_agrees_9798 : modelRow 9798 = row 9798 := model_count_9798.trans row_count_9798.symm
def row_9799 : RowCertificate 9799 where
  first := false
  first_correct := by
    have composite : ¬ Nat.Prime (corner 9799 1) := by change ¬ Nat.Prime 384101203; exact PrattCertificates.composite_of_product (factor := 751) (cofactor := 511453) rfl (by decide) (by decide)
    exact decide_eq_false composite
  second := false
  second_correct := by
    have composite : ¬ Nat.Prime (corner 9799 2) := by change ¬ Nat.Prime 384081605; exact PrattCertificates.composite_of_product (factor := 5) (cofactor := 76816321) rfl (by decide) (by decide)
    exact decide_eq_false composite
  third := false
  third_correct := by
    have composite : ¬ Nat.Prime (corner 9799 3) := by change ¬ Nat.Prime 384062007; exact PrattCertificates.composite_of_product (factor := 3) (cofactor := 128020669) rfl (by decide) (by decide)
    exact decide_eq_false composite
theorem row_count_9799 : row 9799 = 0 := (row_9799.checked).trans rfl
theorem model_count_9799 : modelRow 9799 = 0 := by decide
theorem modelrow_agrees_9799 : modelRow 9799 = row 9799 := model_count_9799.trans row_count_9799.symm
def row_9800 : RowCertificate 9800 where
  first := false
  first_correct := by
    have composite : ¬ Nat.Prime (corner 9800 1) := by change ¬ Nat.Prime 384179601; exact PrattCertificates.composite_of_product (factor := 3) (cofactor := 128059867) rfl (by decide) (by decide)
    exact decide_eq_false composite
  second := true
  second_correct := by
    have prime : Nat.Prime (corner 9800 2) := by change Nat.Prime 384160001; exact ProjectEulerSolutions.P58PrimeNodes.p_384160001.prime
    exact decide_eq_true prime
  third := false
  third_correct := by
    have composite : ¬ Nat.Prime (corner 9800 3) := by change ¬ Nat.Prime 384140401; exact PrattCertificates.composite_of_product (factor := 37) (cofactor := 10382173) rfl (by decide) (by decide)
    exact decide_eq_false composite
theorem row_count_9800 : row 9800 = 1 := (row_9800.checked).trans rfl
theorem model_count_9800 : modelRow 9800 = 1 := by decide
theorem modelrow_agrees_9800 : modelRow 9800 = row 9800 := model_count_9800.trans row_count_9800.symm
def row_9801 : RowCertificate 9801 where
  first := false
  first_correct := by
    have composite : ¬ Nat.Prime (corner 9801 1) := by change ¬ Nat.Prime 384258007; exact PrattCertificates.composite_of_product (factor := 7) (cofactor := 54894001) rfl (by decide) (by decide)
    exact decide_eq_false composite
  second := false
  second_correct := by
    have composite : ¬ Nat.Prime (corner 9801 2) := by change ¬ Nat.Prime 384238405; exact PrattCertificates.composite_of_product (factor := 5) (cofactor := 76847681) rfl (by decide) (by decide)
    exact decide_eq_false composite
  third := false
  third_correct := by
    have composite : ¬ Nat.Prime (corner 9801 3) := by change ¬ Nat.Prime 384218803; exact PrattCertificates.composite_of_product (factor := 43) (cofactor := 8935321) rfl (by decide) (by decide)
    exact decide_eq_false composite
theorem row_count_9801 : row 9801 = 0 := (row_9801.checked).trans rfl
theorem model_count_9801 : modelRow 9801 = 0 := by decide
theorem modelrow_agrees_9801 : modelRow 9801 = row 9801 := model_count_9801.trans row_count_9801.symm
def row_9802 : RowCertificate 9802 where
  first := false
  first_correct := by
    have composite : ¬ Nat.Prime (corner 9802 1) := by change ¬ Nat.Prime 384336421; exact PrattCertificates.composite_of_product (factor := 7) (cofactor := 54905203) rfl (by decide) (by decide)
    exact decide_eq_false composite
  second := false
  second_correct := by
    have composite : ¬ Nat.Prime (corner 9802 2) := by change ¬ Nat.Prime 384316817; exact PrattCertificates.composite_of_product (factor := 37) (cofactor := 10386941) rfl (by decide) (by decide)
    exact decide_eq_false composite
  third := false
  third_correct := by
    have composite : ¬ Nat.Prime (corner 9802 3) := by change ¬ Nat.Prime 384297213; exact PrattCertificates.composite_of_product (factor := 3) (cofactor := 128099071) rfl (by decide) (by decide)
    exact decide_eq_false composite
theorem row_count_9802 : row 9802 = 0 := (row_9802.checked).trans rfl
theorem model_count_9802 : modelRow 9802 = 0 := by decide
theorem modelrow_agrees_9802 : modelRow 9802 = row 9802 := model_count_9802.trans row_count_9802.symm
def row_9803 : RowCertificate 9803 where
  first := false
  first_correct := by
    have composite : ¬ Nat.Prime (corner 9803 1) := by change ¬ Nat.Prime 384414843; exact PrattCertificates.composite_of_product (factor := 3) (cofactor := 128138281) rfl (by decide) (by decide)
    exact decide_eq_false composite
  second := false
  second_correct := by
    have composite : ¬ Nat.Prime (corner 9803 2) := by change ¬ Nat.Prime 384395237; exact PrattCertificates.composite_of_product (factor := 9601) (cofactor := 40037) rfl (by decide) (by decide)
    exact decide_eq_false composite
  third := false
  third_correct := by
    have composite : ¬ Nat.Prime (corner 9803 3) := by change ¬ Nat.Prime 384375631; exact PrattCertificates.composite_of_product (factor := 541) (cofactor := 710491) rfl (by decide) (by decide)
    exact decide_eq_false composite
theorem row_count_9803 : row 9803 = 0 := (row_9803.checked).trans rfl
theorem model_count_9803 : modelRow 9803 = 0 := by decide
theorem modelrow_agrees_9803 : modelRow 9803 = row 9803 := model_count_9803.trans row_count_9803.symm
def row_9804 : RowCertificate 9804 where
  first := false
  first_correct := by
    have composite : ¬ Nat.Prime (corner 9804 1) := by change ¬ Nat.Prime 384493273; exact PrattCertificates.composite_of_product (factor := 199) (cofactor := 1932127) rfl (by decide) (by decide)
    exact decide_eq_false composite
  second := false
  second_correct := by
    have composite : ¬ Nat.Prime (corner 9804 2) := by change ¬ Nat.Prime 384473665; exact PrattCertificates.composite_of_product (factor := 5) (cofactor := 76894733) rfl (by decide) (by decide)
    exact decide_eq_false composite
  third := false
  third_correct := by
    have composite : ¬ Nat.Prime (corner 9804 3) := by change ¬ Nat.Prime 384454057; exact PrattCertificates.composite_of_product (factor := 13) (cofactor := 29573389) rfl (by decide) (by decide)
    exact decide_eq_false composite
theorem row_count_9804 : row 9804 = 0 := (row_9804.checked).trans rfl
theorem model_count_9804 : modelRow 9804 = 0 := by decide
theorem modelrow_agrees_9804 : modelRow 9804 = row 9804 := model_count_9804.trans row_count_9804.symm
def row_9805 : RowCertificate 9805 where
  first := true
  first_correct := by
    have prime : Nat.Prime (corner 9805 1) := by change Nat.Prime 384571711; exact ProjectEulerSolutions.P58PrimeNodes.p_384571711.prime
    exact decide_eq_true prime
  second := false
  second_correct := by
    have composite : ¬ Nat.Prime (corner 9805 2) := by change ¬ Nat.Prime 384552101; exact PrattCertificates.composite_of_product (factor := 73) (cofactor := 5267837) rfl (by decide) (by decide)
    exact decide_eq_false composite
  third := false
  third_correct := by
    have composite : ¬ Nat.Prime (corner 9805 3) := by change ¬ Nat.Prime 384532491; exact PrattCertificates.composite_of_product (factor := 3) (cofactor := 128177497) rfl (by decide) (by decide)
    exact decide_eq_false composite
theorem row_count_9805 : row 9805 = 1 := (row_9805.checked).trans rfl
theorem model_count_9805 : modelRow 9805 = 1 := by decide
theorem modelrow_agrees_9805 : modelRow 9805 = row 9805 := model_count_9805.trans row_count_9805.symm
def row_9806 : RowCertificate 9806 where
  first := false
  first_correct := by
    have composite : ¬ Nat.Prime (corner 9806 1) := by change ¬ Nat.Prime 384650157; exact PrattCertificates.composite_of_product (factor := 3) (cofactor := 128216719) rfl (by decide) (by decide)
    exact decide_eq_false composite
  second := false
  second_correct := by
    have composite : ¬ Nat.Prime (corner 9806 2) := by change ¬ Nat.Prime 384630545; exact PrattCertificates.composite_of_product (factor := 5) (cofactor := 76926109) rfl (by decide) (by decide)
    exact decide_eq_false composite
  third := false
  third_correct := by
    have composite : ¬ Nat.Prime (corner 9806 3) := by change ¬ Nat.Prime 384610933; exact PrattCertificates.composite_of_product (factor := 7) (cofactor := 54944419) rfl (by decide) (by decide)
    exact decide_eq_false composite
theorem row_count_9806 : row 9806 = 0 := (row_9806.checked).trans rfl
theorem model_count_9806 : modelRow 9806 = 0 := by decide
theorem modelrow_agrees_9806 : modelRow 9806 = row 9806 := model_count_9806.trans row_count_9806.symm
def row_9807 : RowCertificate 9807 where
  first := false
  first_correct := by
    have composite : ¬ Nat.Prime (corner 9807 1) := by change ¬ Nat.Prime 384728611; exact PrattCertificates.composite_of_product (factor := 43) (cofactor := 8947177) rfl (by decide) (by decide)
    exact decide_eq_false composite
  second := false
  second_correct := by
    have composite : ¬ Nat.Prime (corner 9807 2) := by change ¬ Nat.Prime 384708997; exact PrattCertificates.composite_of_product (factor := 17) (cofactor := 22629941) rfl (by decide) (by decide)
    exact decide_eq_false composite
  third := false
  third_correct := by
    have composite : ¬ Nat.Prime (corner 9807 3) := by change ¬ Nat.Prime 384689383; exact PrattCertificates.composite_of_product (factor := 13) (cofactor := 29591491) rfl (by decide) (by decide)
    exact decide_eq_false composite
theorem row_count_9807 : row 9807 = 0 := (row_9807.checked).trans rfl
theorem model_count_9807 : modelRow 9807 = 0 := by decide
theorem modelrow_agrees_9807 : modelRow 9807 = row 9807 := model_count_9807.trans row_count_9807.symm
def row_9808 : RowCertificate 9808 where
  first := false
  first_correct := by
    have composite : ¬ Nat.Prime (corner 9808 1) := by change ¬ Nat.Prime 384807073; exact PrattCertificates.composite_of_product (factor := 7) (cofactor := 54972439) rfl (by decide) (by decide)
    exact decide_eq_false composite
  second := false
  second_correct := by
    have composite : ¬ Nat.Prime (corner 9808 2) := by change ¬ Nat.Prime 384787457; exact PrattCertificates.composite_of_product (factor := 29) (cofactor := 13268533) rfl (by decide) (by decide)
    exact decide_eq_false composite
  third := false
  third_correct := by
    have composite : ¬ Nat.Prime (corner 9808 3) := by change ¬ Nat.Prime 384767841; exact PrattCertificates.composite_of_product (factor := 3) (cofactor := 128255947) rfl (by decide) (by decide)
    exact decide_eq_false composite
theorem row_count_9808 : row 9808 = 0 := (row_9808.checked).trans rfl
theorem model_count_9808 : modelRow 9808 = 0 := by decide
theorem modelrow_agrees_9808 : modelRow 9808 = row 9808 := model_count_9808.trans row_count_9808.symm
def row_9809 : RowCertificate 9809 where
  first := false
  first_correct := by
    have composite : ¬ Nat.Prime (corner 9809 1) := by change ¬ Nat.Prime 384885543; exact PrattCertificates.composite_of_product (factor := 3) (cofactor := 128295181) rfl (by decide) (by decide)
    exact decide_eq_false composite
  second := false
  second_correct := by
    have composite : ¬ Nat.Prime (corner 9809 2) := by change ¬ Nat.Prime 384865925; exact PrattCertificates.composite_of_product (factor := 5) (cofactor := 76973185) rfl (by decide) (by decide)
    exact decide_eq_false composite
  third := false
  third_correct := by
    have composite : ¬ Nat.Prime (corner 9809 3) := by change ¬ Nat.Prime 384846307; exact PrattCertificates.composite_of_product (factor := 31) (cofactor := 12414397) rfl (by decide) (by decide)
    exact decide_eq_false composite
theorem row_count_9809 : row 9809 = 0 := (row_9809.checked).trans rfl
theorem model_count_9809 : modelRow 9809 = 0 := by decide
theorem modelrow_agrees_9809 : modelRow 9809 = row 9809 := model_count_9809.trans row_count_9809.symm
def row_9810 : RowCertificate 9810 where
  first := false
  first_correct := by
    have composite : ¬ Nat.Prime (corner 9810 1) := by change ¬ Nat.Prime 384964021; exact PrattCertificates.composite_of_product (factor := 13) (cofactor := 29612617) rfl (by decide) (by decide)
    exact decide_eq_false composite
  second := true
  second_correct := by
    have prime : Nat.Prime (corner 9810 2) := by change Nat.Prime 384944401; exact ProjectEulerSolutions.P58PrimeNodes.p_384944401.prime
    exact decide_eq_true prime
  third := false
  third_correct := by
    have composite : ¬ Nat.Prime (corner 9810 3) := by change ¬ Nat.Prime 384924781; exact PrattCertificates.composite_of_product (factor := 19) (cofactor := 20259199) rfl (by decide) (by decide)
    exact decide_eq_false composite
theorem row_count_9810 : row 9810 = 1 := (row_9810.checked).trans rfl
theorem model_count_9810 : modelRow 9810 = 1 := by decide
theorem modelrow_agrees_9810 : modelRow 9810 = row 9810 := model_count_9810.trans row_count_9810.symm
def row_9811 : RowCertificate 9811 where
  first := true
  first_correct := by
    have prime : Nat.Prime (corner 9811 1) := by change Nat.Prime 385042507; exact ProjectEulerSolutions.P58PrimeNodes.p_385042507.prime
    exact decide_eq_true prime
  second := false
  second_correct := by
    have composite : ¬ Nat.Prime (corner 9811 2) := by change ¬ Nat.Prime 385022885; exact PrattCertificates.composite_of_product (factor := 5) (cofactor := 77004577) rfl (by decide) (by decide)
    exact decide_eq_false composite
  third := false
  third_correct := by
    have composite : ¬ Nat.Prime (corner 9811 3) := by change ¬ Nat.Prime 385003263; exact PrattCertificates.composite_of_product (factor := 3) (cofactor := 128334421) rfl (by decide) (by decide)
    exact decide_eq_false composite
theorem row_count_9811 : row 9811 = 1 := (row_9811.checked).trans rfl
theorem model_count_9811 : modelRow 9811 = 1 := by decide
theorem modelrow_agrees_9811 : modelRow 9811 = row 9811 := model_count_9811.trans row_count_9811.symm
def row_9812 : RowCertificate 9812 where
  first := false
  first_correct := by
    have composite : ¬ Nat.Prime (corner 9812 1) := by change ¬ Nat.Prime 385121001; exact PrattCertificates.composite_of_product (factor := 3) (cofactor := 128373667) rfl (by decide) (by decide)
    exact decide_eq_false composite
  second := false
  second_correct := by
    have composite : ¬ Nat.Prime (corner 9812 2) := by change ¬ Nat.Prime 385101377; exact PrattCertificates.composite_of_product (factor := 149) (cofactor := 2584573) rfl (by decide) (by decide)
    exact decide_eq_false composite
  third := false
  third_correct := by
    have composite : ¬ Nat.Prime (corner 9812 3) := by change ¬ Nat.Prime 385081753; exact PrattCertificates.composite_of_product (factor := 7) (cofactor := 55011679) rfl (by decide) (by decide)
    exact decide_eq_false composite
theorem row_count_9812 : row 9812 = 0 := (row_9812.checked).trans rfl
theorem model_count_9812 : modelRow 9812 = 0 := by decide
theorem modelrow_agrees_9812 : modelRow 9812 = row 9812 := model_count_9812.trans row_count_9812.symm
def row_9813 : RowCertificate 9813 where
  first := false
  first_correct := by
    have composite : ¬ Nat.Prime (corner 9813 1) := by change ¬ Nat.Prime 385199503; exact PrattCertificates.composite_of_product (factor := 13) (cofactor := 29630731) rfl (by decide) (by decide)
    exact decide_eq_false composite
  second := true
  second_correct := by
    have prime : Nat.Prime (corner 9813 2) := by change Nat.Prime 385179877; exact ProjectEulerSolutions.P58PrimeNodes.p_385179877.prime
    exact decide_eq_true prime
  third := false
  third_correct := by
    have composite : ¬ Nat.Prime (corner 9813 3) := by change ¬ Nat.Prime 385160251; exact PrattCertificates.composite_of_product (factor := 7) (cofactor := 55022893) rfl (by decide) (by decide)
    exact decide_eq_false composite
theorem row_count_9813 : row 9813 = 1 := (row_9813.checked).trans rfl
theorem model_count_9813 : modelRow 9813 = 1 := by decide
theorem modelrow_agrees_9813 : modelRow 9813 = row 9813 := model_count_9813.trans row_count_9813.symm
def row_9814 : RowCertificate 9814 where
  first := false
  first_correct := by
    have composite : ¬ Nat.Prime (corner 9814 1) := by change ¬ Nat.Prime 385278013; exact PrattCertificates.composite_of_product (factor := 31) (cofactor := 12428323) rfl (by decide) (by decide)
    exact decide_eq_false composite
  second := false
  second_correct := by
    have composite : ¬ Nat.Prime (corner 9814 2) := by change ¬ Nat.Prime 385258385; exact PrattCertificates.composite_of_product (factor := 5) (cofactor := 77051677) rfl (by decide) (by decide)
    exact decide_eq_false composite
  third := false
  third_correct := by
    have composite : ¬ Nat.Prime (corner 9814 3) := by change ¬ Nat.Prime 385238757; exact PrattCertificates.composite_of_product (factor := 3) (cofactor := 128412919) rfl (by decide) (by decide)
    exact decide_eq_false composite
theorem row_count_9814 : row 9814 = 0 := (row_9814.checked).trans rfl
theorem model_count_9814 : modelRow 9814 = 0 := by decide
theorem modelrow_agrees_9814 : modelRow 9814 = row 9814 := model_count_9814.trans row_count_9814.symm
def row_9815 : RowCertificate 9815 where
  first := false
  first_correct := by
    have composite : ¬ Nat.Prime (corner 9815 1) := by change ¬ Nat.Prime 385356531; exact PrattCertificates.composite_of_product (factor := 3) (cofactor := 128452177) rfl (by decide) (by decide)
    exact decide_eq_false composite
  second := false
  second_correct := by
    have composite : ¬ Nat.Prime (corner 9815 2) := by change ¬ Nat.Prime 385336901; exact PrattCertificates.composite_of_product (factor := 41) (cofactor := 9398461) rfl (by decide) (by decide)
    exact decide_eq_false composite
  third := false
  third_correct := by
    have composite : ¬ Nat.Prime (corner 9815 3) := by change ¬ Nat.Prime 385317271; exact PrattCertificates.composite_of_product (factor := 97) (cofactor := 3972343) rfl (by decide) (by decide)
    exact decide_eq_false composite
theorem row_count_9815 : row 9815 = 0 := (row_9815.checked).trans rfl
theorem model_count_9815 : modelRow 9815 = 0 := by decide
theorem modelrow_agrees_9815 : modelRow 9815 = row 9815 := model_count_9815.trans row_count_9815.symm
def row_9816 : RowCertificate 9816 where
  first := false
  first_correct := by
    have composite : ¬ Nat.Prime (corner 9816 1) := by change ¬ Nat.Prime 385435057; exact PrattCertificates.composite_of_product (factor := 7) (cofactor := 55062151) rfl (by decide) (by decide)
    exact decide_eq_false composite
  second := false
  second_correct := by
    have composite : ¬ Nat.Prime (corner 9816 2) := by change ¬ Nat.Prime 385415425; exact PrattCertificates.composite_of_product (factor := 5) (cofactor := 77083085) rfl (by decide) (by decide)
    exact decide_eq_false composite
  third := true
  third_correct := by
    have prime : Nat.Prime (corner 9816 3) := by change Nat.Prime 385395793; exact ProjectEulerSolutions.P58PrimeNodes.p_385395793.prime
    exact decide_eq_true prime
theorem row_count_9816 : row 9816 = 1 := (row_9816.checked).trans rfl
theorem model_count_9816 : modelRow 9816 = 1 := by decide
theorem modelrow_agrees_9816 : modelRow 9816 = row 9816 := model_count_9816.trans row_count_9816.symm
def row_9817 : RowCertificate 9817 where
  first := false
  first_correct := by
    have composite : ¬ Nat.Prime (corner 9817 1) := by change ¬ Nat.Prime 385513591; exact PrattCertificates.composite_of_product (factor := 19) (cofactor := 20290189) rfl (by decide) (by decide)
    exact decide_eq_false composite
  second := true
  second_correct := by
    have prime : Nat.Prime (corner 9817 2) := by change Nat.Prime 385493957; exact ProjectEulerSolutions.P58PrimeNodes.p_385493957.prime
    exact decide_eq_true prime
  third := false
  third_correct := by
    have composite : ¬ Nat.Prime (corner 9817 3) := by change ¬ Nat.Prime 385474323; exact PrattCertificates.composite_of_product (factor := 3) (cofactor := 128491441) rfl (by decide) (by decide)
    exact decide_eq_false composite
theorem row_count_9817 : row 9817 = 1 := (row_9817.checked).trans rfl
theorem model_count_9817 : modelRow 9817 = 1 := by decide
theorem modelrow_agrees_9817 : modelRow 9817 = row 9817 := model_count_9817.trans row_count_9817.symm
def row_9818 : RowCertificate 9818 where
  first := false
  first_correct := by
    have composite : ¬ Nat.Prime (corner 9818 1) := by change ¬ Nat.Prime 385592133; exact PrattCertificates.composite_of_product (factor := 3) (cofactor := 128530711) rfl (by decide) (by decide)
    exact decide_eq_false composite
  second := true
  second_correct := by
    have prime : Nat.Prime (corner 9818 2) := by change Nat.Prime 385572497; exact ProjectEulerSolutions.P58PrimeNodes.p_385572497.prime
    exact decide_eq_true prime
  third := false
  third_correct := by
    have composite : ¬ Nat.Prime (corner 9818 3) := by change ¬ Nat.Prime 385552861; exact PrattCertificates.composite_of_product (factor := 997) (cofactor := 386713) rfl (by decide) (by decide)
    exact decide_eq_false composite
theorem row_count_9818 : row 9818 = 1 := (row_9818.checked).trans rfl
theorem model_count_9818 : modelRow 9818 = 1 := by decide
theorem modelrow_agrees_9818 : modelRow 9818 = row 9818 := model_count_9818.trans row_count_9818.symm
def row_9819 : RowCertificate 9819 where
  first := false
  first_correct := by
    have composite : ¬ Nat.Prime (corner 9819 1) := by change ¬ Nat.Prime 385670683; exact PrattCertificates.composite_of_product (factor := 19) (cofactor := 20298457) rfl (by decide) (by decide)
    exact decide_eq_false composite
  second := false
  second_correct := by
    have composite : ¬ Nat.Prime (corner 9819 2) := by change ¬ Nat.Prime 385651045; exact PrattCertificates.composite_of_product (factor := 5) (cofactor := 77130209) rfl (by decide) (by decide)
    exact decide_eq_false composite
  third := false
  third_correct := by
    have composite : ¬ Nat.Prime (corner 9819 3) := by change ¬ Nat.Prime 385631407; exact PrattCertificates.composite_of_product (factor := 7) (cofactor := 55090201) rfl (by decide) (by decide)
    exact decide_eq_false composite
theorem row_count_9819 : row 9819 = 0 := (row_9819.checked).trans rfl
theorem model_count_9819 : modelRow 9819 = 0 := by decide
theorem modelrow_agrees_9819 : modelRow 9819 = row 9819 := model_count_9819.trans row_count_9819.symm
def row_9820 : RowCertificate 9820 where
  first := false
  first_correct := by
    have composite : ¬ Nat.Prime (corner 9820 1) := by change ¬ Nat.Prime 385749241; exact PrattCertificates.composite_of_product (factor := 1303) (cofactor := 296047) rfl (by decide) (by decide)
    exact decide_eq_false composite
  second := false
  second_correct := by
    have composite : ¬ Nat.Prime (corner 9820 2) := by change ¬ Nat.Prime 385729601; exact PrattCertificates.composite_of_product (factor := 53) (cofactor := 7277917) rfl (by decide) (by decide)
    exact decide_eq_false composite
  third := false
  third_correct := by
    have composite : ¬ Nat.Prime (corner 9820 3) := by change ¬ Nat.Prime 385709961; exact PrattCertificates.composite_of_product (factor := 3) (cofactor := 128569987) rfl (by decide) (by decide)
    exact decide_eq_false composite
theorem row_count_9820 : row 9820 = 0 := (row_9820.checked).trans rfl
theorem model_count_9820 : modelRow 9820 = 0 := by decide
theorem modelrow_agrees_9820 : modelRow 9820 = row 9820 := model_count_9820.trans row_count_9820.symm
def row_9821 : RowCertificate 9821 where
  first := false
  first_correct := by
    have composite : ¬ Nat.Prime (corner 9821 1) := by change ¬ Nat.Prime 385827807; exact PrattCertificates.composite_of_product (factor := 3) (cofactor := 128609269) rfl (by decide) (by decide)
    exact decide_eq_false composite
  second := false
  second_correct := by
    have composite : ¬ Nat.Prime (corner 9821 2) := by change ¬ Nat.Prime 385808165; exact PrattCertificates.composite_of_product (factor := 5) (cofactor := 77161633) rfl (by decide) (by decide)
    exact decide_eq_false composite
  third := false
  third_correct := by
    have composite : ¬ Nat.Prime (corner 9821 3) := by change ¬ Nat.Prime 385788523; exact PrattCertificates.composite_of_product (factor := 139) (cofactor := 2775457) rfl (by decide) (by decide)
    exact decide_eq_false composite
theorem row_count_9821 : row 9821 = 0 := (row_9821.checked).trans rfl
theorem model_count_9821 : modelRow 9821 = 0 := by decide
theorem modelrow_agrees_9821 : modelRow 9821 = row 9821 := model_count_9821.trans row_count_9821.symm
def row_9822 : RowCertificate 9822 where
  first := false
  first_correct := by
    have composite : ¬ Nat.Prime (corner 9822 1) := by change ¬ Nat.Prime 385906381; exact PrattCertificates.composite_of_product (factor := 7) (cofactor := 55129483) rfl (by decide) (by decide)
    exact decide_eq_false composite
  second := false
  second_correct := by
    have composite : ¬ Nat.Prime (corner 9822 2) := by change ¬ Nat.Prime 385886737; exact PrattCertificates.composite_of_product (factor := 11549) (cofactor := 33413) rfl (by decide) (by decide)
    exact decide_eq_false composite
  third := false
  third_correct := by
    have composite : ¬ Nat.Prime (corner 9822 3) := by change ¬ Nat.Prime 385867093; exact PrattCertificates.composite_of_product (factor := 5209) (cofactor := 74077) rfl (by decide) (by decide)
    exact decide_eq_false composite
theorem row_count_9822 : row 9822 = 0 := (row_9822.checked).trans rfl
theorem model_count_9822 : modelRow 9822 = 0 := by decide
theorem modelrow_agrees_9822 : modelRow 9822 = row 9822 := model_count_9822.trans row_count_9822.symm
def row_9823 : RowCertificate 9823 where
  first := false
  first_correct := by
    have composite : ¬ Nat.Prime (corner 9823 1) := by change ¬ Nat.Prime 385984963; exact PrattCertificates.composite_of_product (factor := 7) (cofactor := 55140709) rfl (by decide) (by decide)
    exact decide_eq_false composite
  second := false
  second_correct := by
    have composite : ¬ Nat.Prime (corner 9823 2) := by change ¬ Nat.Prime 385965317; exact PrattCertificates.composite_of_product (factor := 593) (cofactor := 650869) rfl (by decide) (by decide)
    exact decide_eq_false composite
  third := false
  third_correct := by
    have composite : ¬ Nat.Prime (corner 9823 3) := by change ¬ Nat.Prime 385945671; exact PrattCertificates.composite_of_product (factor := 3) (cofactor := 128648557) rfl (by decide) (by decide)
    exact decide_eq_false composite
theorem row_count_9823 : row 9823 = 0 := (row_9823.checked).trans rfl
theorem model_count_9823 : modelRow 9823 = 0 := by decide
theorem modelrow_agrees_9823 : modelRow 9823 = row 9823 := model_count_9823.trans row_count_9823.symm
def row_9824 : RowCertificate 9824 where
  first := false
  first_correct := by
    have composite : ¬ Nat.Prime (corner 9824 1) := by change ¬ Nat.Prime 386063553; exact PrattCertificates.composite_of_product (factor := 3) (cofactor := 128687851) rfl (by decide) (by decide)
    exact decide_eq_false composite
  second := false
  second_correct := by
    have composite : ¬ Nat.Prime (corner 9824 2) := by change ¬ Nat.Prime 386043905; exact PrattCertificates.composite_of_product (factor := 5) (cofactor := 77208781) rfl (by decide) (by decide)
    exact decide_eq_false composite
  third := false
  third_correct := by
    have composite : ¬ Nat.Prime (corner 9824 3) := by change ¬ Nat.Prime 386024257; exact PrattCertificates.composite_of_product (factor := 79) (cofactor := 4886383) rfl (by decide) (by decide)
    exact decide_eq_false composite
theorem row_count_9824 : row 9824 = 0 := (row_9824.checked).trans rfl
theorem model_count_9824 : modelRow 9824 = 0 := by decide
theorem modelrow_agrees_9824 : modelRow 9824 = row 9824 := model_count_9824.trans row_count_9824.symm
def row_9825 : RowCertificate 9825 where
  first := true
  first_correct := by
    have prime : Nat.Prime (corner 9825 1) := by change Nat.Prime 386142151; exact ProjectEulerSolutions.P58PrimeNodes.p_386142151.prime
    exact decide_eq_true prime
  second := false
  second_correct := by
    have composite : ¬ Nat.Prime (corner 9825 2) := by change ¬ Nat.Prime 386122501; exact PrattCertificates.composite_of_product (factor := 29) (cofactor := 13314569) rfl (by decide) (by decide)
    exact decide_eq_false composite
  third := true
  third_correct := by
    have prime : Nat.Prime (corner 9825 3) := by change Nat.Prime 386102851; exact ProjectEulerSolutions.P58PrimeNodes.p_386102851.prime
    exact decide_eq_true prime
theorem row_count_9825 : row 9825 = 2 := (row_9825.checked).trans rfl
theorem model_count_9825 : modelRow 9825 = 2 := by decide
theorem modelrow_agrees_9825 : modelRow 9825 = row 9825 := model_count_9825.trans row_count_9825.symm
def row_9826 : RowCertificate 9826 where
  first := false
  first_correct := by
    have composite : ¬ Nat.Prime (corner 9826 1) := by change ¬ Nat.Prime 386220757; exact PrattCertificates.composite_of_product (factor := 13) (cofactor := 29709289) rfl (by decide) (by decide)
    exact decide_eq_false composite
  second := false
  second_correct := by
    have composite : ¬ Nat.Prime (corner 9826 2) := by change ¬ Nat.Prime 386201105; exact PrattCertificates.composite_of_product (factor := 5) (cofactor := 77240221) rfl (by decide) (by decide)
    exact decide_eq_false composite
  third := false
  third_correct := by
    have composite : ¬ Nat.Prime (corner 9826 3) := by change ¬ Nat.Prime 386181453; exact PrattCertificates.composite_of_product (factor := 3) (cofactor := 128727151) rfl (by decide) (by decide)
    exact decide_eq_false composite
theorem row_count_9826 : row 9826 = 0 := (row_9826.checked).trans rfl
theorem model_count_9826 : modelRow 9826 = 0 := by decide
theorem modelrow_agrees_9826 : modelRow 9826 = row 9826 := model_count_9826.trans row_count_9826.symm
def row_9827 : RowCertificate 9827 where
  first := false
  first_correct := by
    have composite : ¬ Nat.Prime (corner 9827 1) := by change ¬ Nat.Prime 386299371; exact PrattCertificates.composite_of_product (factor := 3) (cofactor := 128766457) rfl (by decide) (by decide)
    exact decide_eq_false composite
  second := false
  second_correct := by
    have composite : ¬ Nat.Prime (corner 9827 2) := by change ¬ Nat.Prime 386279717; exact PrattCertificates.composite_of_product (factor := 3001) (cofactor := 128717) rfl (by decide) (by decide)
    exact decide_eq_false composite
  third := false
  third_correct := by
    have composite : ¬ Nat.Prime (corner 9827 3) := by change ¬ Nat.Prime 386260063; exact PrattCertificates.composite_of_product (factor := 7) (cofactor := 55180009) rfl (by decide) (by decide)
    exact decide_eq_false composite
theorem row_count_9827 : row 9827 = 0 := (row_9827.checked).trans rfl
theorem model_count_9827 : modelRow 9827 = 0 := by decide
theorem modelrow_agrees_9827 : modelRow 9827 = row 9827 := model_count_9827.trans row_count_9827.symm
def row_9828 : RowCertificate 9828 where
  first := false
  first_correct := by
    have composite : ¬ Nat.Prime (corner 9828 1) := by change ¬ Nat.Prime 386377993; exact PrattCertificates.composite_of_product (factor := 379) (cofactor := 1019467) rfl (by decide) (by decide)
    exact decide_eq_false composite
  second := false
  second_correct := by
    have composite : ¬ Nat.Prime (corner 9828 2) := by change ¬ Nat.Prime 386358337; exact PrattCertificates.composite_of_product (factor := 17) (cofactor := 22726961) rfl (by decide) (by decide)
    exact decide_eq_false composite
  third := false
  third_correct := by
    have composite : ¬ Nat.Prime (corner 9828 3) := by change ¬ Nat.Prime 386338681; exact PrattCertificates.composite_of_product (factor := 61) (cofactor := 6333421) rfl (by decide) (by decide)
    exact decide_eq_false composite
theorem row_count_9828 : row 9828 = 0 := (row_9828.checked).trans rfl
theorem model_count_9828 : modelRow 9828 = 0 := by decide
theorem modelrow_agrees_9828 : modelRow 9828 = row 9828 := model_count_9828.trans row_count_9828.symm
def row_9829 : RowCertificate 9829 where
  first := false
  first_correct := by
    have composite : ¬ Nat.Prime (corner 9829 1) := by change ¬ Nat.Prime 386456623; exact PrattCertificates.composite_of_product (factor := 7) (cofactor := 55208089) rfl (by decide) (by decide)
    exact decide_eq_false composite
  second := false
  second_correct := by
    have composite : ¬ Nat.Prime (corner 9829 2) := by change ¬ Nat.Prime 386436965; exact PrattCertificates.composite_of_product (factor := 5) (cofactor := 77287393) rfl (by decide) (by decide)
    exact decide_eq_false composite
  third := false
  third_correct := by
    have composite : ¬ Nat.Prime (corner 9829 3) := by change ¬ Nat.Prime 386417307; exact PrattCertificates.composite_of_product (factor := 3) (cofactor := 128805769) rfl (by decide) (by decide)
    exact decide_eq_false composite
theorem row_count_9829 : row 9829 = 0 := (row_9829.checked).trans rfl
theorem model_count_9829 : modelRow 9829 = 0 := by decide
theorem modelrow_agrees_9829 : modelRow 9829 = row 9829 := model_count_9829.trans row_count_9829.symm
def row_9830 : RowCertificate 9830 where
  first := false
  first_correct := by
    have composite : ¬ Nat.Prime (corner 9830 1) := by change ¬ Nat.Prime 386535261; exact PrattCertificates.composite_of_product (factor := 3) (cofactor := 128845087) rfl (by decide) (by decide)
    exact decide_eq_false composite
  second := false
  second_correct := by
    have composite : ¬ Nat.Prime (corner 9830 2) := by change ¬ Nat.Prime 386515601; exact PrattCertificates.composite_of_product (factor := 7681) (cofactor := 50321) rfl (by decide) (by decide)
    exact decide_eq_false composite
  third := false
  third_correct := by
    have composite : ¬ Nat.Prime (corner 9830 3) := by change ¬ Nat.Prime 386495941; exact PrattCertificates.composite_of_product (factor := 13) (cofactor := 29730457) rfl (by decide) (by decide)
    exact decide_eq_false composite
theorem row_count_9830 : row 9830 = 0 := (row_9830.checked).trans rfl
theorem model_count_9830 : modelRow 9830 = 0 := by decide
theorem modelrow_agrees_9830 : modelRow 9830 = row 9830 := model_count_9830.trans row_count_9830.symm
def row_9831 : RowCertificate 9831 where
  first := false
  first_correct := by
    have composite : ¬ Nat.Prime (corner 9831 1) := by change ¬ Nat.Prime 386613907; exact PrattCertificates.composite_of_product (factor := 151) (cofactor := 2560357) rfl (by decide) (by decide)
    exact decide_eq_false composite
  second := false
  second_correct := by
    have composite : ¬ Nat.Prime (corner 9831 2) := by change ¬ Nat.Prime 386594245; exact PrattCertificates.composite_of_product (factor := 5) (cofactor := 77318849) rfl (by decide) (by decide)
    exact decide_eq_false composite
  third := true
  third_correct := by
    have prime : Nat.Prime (corner 9831 3) := by change Nat.Prime 386574583; exact ProjectEulerSolutions.P58PrimeNodes.p_386574583.prime
    exact decide_eq_true prime
theorem row_count_9831 : row 9831 = 1 := (row_9831.checked).trans rfl
theorem model_count_9831 : modelRow 9831 = 1 := by decide
theorem modelrow_agrees_9831 : modelRow 9831 = row 9831 := model_count_9831.trans row_count_9831.symm
def row_9832 : RowCertificate 9832 where
  first := false
  first_correct := by
    have composite : ¬ Nat.Prime (corner 9832 1) := by change ¬ Nat.Prime 386692561; exact PrattCertificates.composite_of_product (factor := 163) (cofactor := 2372347) rfl (by decide) (by decide)
    exact decide_eq_false composite
  second := false
  second_correct := by
    have composite : ¬ Nat.Prime (corner 9832 2) := by change ¬ Nat.Prime 386672897; exact PrattCertificates.composite_of_product (factor := 13) (cofactor := 29744069) rfl (by decide) (by decide)
    exact decide_eq_false composite
  third := false
  third_correct := by
    have composite : ¬ Nat.Prime (corner 9832 3) := by change ¬ Nat.Prime 386653233; exact PrattCertificates.composite_of_product (factor := 3) (cofactor := 128884411) rfl (by decide) (by decide)
    exact decide_eq_false composite
theorem row_count_9832 : row 9832 = 0 := (row_9832.checked).trans rfl
theorem model_count_9832 : modelRow 9832 = 0 := by decide
theorem modelrow_agrees_9832 : modelRow 9832 = row 9832 := model_count_9832.trans row_count_9832.symm
def row_9833 : RowCertificate 9833 where
  first := false
  first_correct := by
    have composite : ¬ Nat.Prime (corner 9833 1) := by change ¬ Nat.Prime 386771223; exact PrattCertificates.composite_of_product (factor := 3) (cofactor := 128923741) rfl (by decide) (by decide)
    exact decide_eq_false composite
  second := true
  second_correct := by
    have prime : Nat.Prime (corner 9833 2) := by change Nat.Prime 386751557; exact ProjectEulerSolutions.P58PrimeNodes.p_386751557.prime
    exact decide_eq_true prime
  third := false
  third_correct := by
    have composite : ¬ Nat.Prime (corner 9833 3) := by change ¬ Nat.Prime 386731891; exact PrattCertificates.composite_of_product (factor := 7) (cofactor := 55247413) rfl (by decide) (by decide)
    exact decide_eq_false composite
theorem row_count_9833 : row 9833 = 1 := (row_9833.checked).trans rfl
theorem model_count_9833 : modelRow 9833 = 1 := by decide
theorem modelrow_agrees_9833 : modelRow 9833 = row 9833 := model_count_9833.trans row_count_9833.symm
def row_9834 : RowCertificate 9834 where
  first := false
  first_correct := by
    have composite : ¬ Nat.Prime (corner 9834 1) := by change ¬ Nat.Prime 386849893; exact PrattCertificates.composite_of_product (factor := 67) (cofactor := 5773879) rfl (by decide) (by decide)
    exact decide_eq_false composite
  second := false
  second_correct := by
    have composite : ¬ Nat.Prime (corner 9834 2) := by change ¬ Nat.Prime 386830225; exact PrattCertificates.composite_of_product (factor := 5) (cofactor := 77366045) rfl (by decide) (by decide)
    exact decide_eq_false composite
  third := false
  third_correct := by
    have composite : ¬ Nat.Prime (corner 9834 3) := by change ¬ Nat.Prime 386810557; exact PrattCertificates.composite_of_product (factor := 7) (cofactor := 55258651) rfl (by decide) (by decide)
    exact decide_eq_false composite
theorem row_count_9834 : row 9834 = 0 := (row_9834.checked).trans rfl
theorem model_count_9834 : modelRow 9834 = 0 := by decide
theorem modelrow_agrees_9834 : modelRow 9834 = row 9834 := model_count_9834.trans row_count_9834.symm
def row_9835 : RowCertificate 9835 where
  first := true
  first_correct := by
    have prime : Nat.Prime (corner 9835 1) := by change Nat.Prime 386928571; exact ProjectEulerSolutions.P58PrimeNodes.p_386928571.prime
    exact decide_eq_true prime
  second := false
  second_correct := by
    have composite : ¬ Nat.Prime (corner 9835 2) := by change ¬ Nat.Prime 386908901; exact PrattCertificates.composite_of_product (factor := 6793) (cofactor := 56957) rfl (by decide) (by decide)
    exact decide_eq_false composite
  third := false
  third_correct := by
    have composite : ¬ Nat.Prime (corner 9835 3) := by change ¬ Nat.Prime 386889231; exact PrattCertificates.composite_of_product (factor := 3) (cofactor := 128963077) rfl (by decide) (by decide)
    exact decide_eq_false composite
theorem row_count_9835 : row 9835 = 1 := (row_9835.checked).trans rfl
theorem model_count_9835 : modelRow 9835 = 1 := by decide
theorem modelrow_agrees_9835 : modelRow 9835 = row 9835 := model_count_9835.trans row_count_9835.symm
def row_9836 : RowCertificate 9836 where
  first := false
  first_correct := by
    have composite : ¬ Nat.Prime (corner 9836 1) := by change ¬ Nat.Prime 387007257; exact PrattCertificates.composite_of_product (factor := 3) (cofactor := 129002419) rfl (by decide) (by decide)
    exact decide_eq_false composite
  second := false
  second_correct := by
    have composite : ¬ Nat.Prime (corner 9836 2) := by change ¬ Nat.Prime 386987585; exact PrattCertificates.composite_of_product (factor := 5) (cofactor := 77397517) rfl (by decide) (by decide)
    exact decide_eq_false composite
  third := true
  third_correct := by
    have prime : Nat.Prime (corner 9836 3) := by change Nat.Prime 386967913; exact ProjectEulerSolutions.P58PrimeNodes.p_386967913.prime
    exact decide_eq_true prime
theorem row_count_9836 : row 9836 = 1 := (row_9836.checked).trans rfl
theorem model_count_9836 : modelRow 9836 = 1 := by decide
theorem modelrow_agrees_9836 : modelRow 9836 = row 9836 := model_count_9836.trans row_count_9836.symm
def row_9837 : RowCertificate 9837 where
  first := false
  first_correct := by
    have composite : ¬ Nat.Prime (corner 9837 1) := by change ¬ Nat.Prime 387085951; exact PrattCertificates.composite_of_product (factor := 7) (cofactor := 55297993) rfl (by decide) (by decide)
    exact decide_eq_false composite
  second := false
  second_correct := by
    have composite : ¬ Nat.Prime (corner 9837 2) := by change ¬ Nat.Prime 387066277; exact PrattCertificates.composite_of_product (factor := 13) (cofactor := 29774329) rfl (by decide) (by decide)
    exact decide_eq_false composite
  third := false
  third_correct := by
    have composite : ¬ Nat.Prime (corner 9837 3) := by change ¬ Nat.Prime 387046603; exact PrattCertificates.composite_of_product (factor := 37) (cofactor := 10460719) rfl (by decide) (by decide)
    exact decide_eq_false composite
theorem row_count_9837 : row 9837 = 0 := (row_9837.checked).trans rfl
theorem model_count_9837 : modelRow 9837 = 0 := by decide
theorem modelrow_agrees_9837 : modelRow 9837 = row 9837 := model_count_9837.trans row_count_9837.symm
def row_9838 : RowCertificate 9838 where
  first := false
  first_correct := by
    have composite : ¬ Nat.Prime (corner 9838 1) := by change ¬ Nat.Prime 387164653; exact PrattCertificates.composite_of_product (factor := 19) (cofactor := 20377087) rfl (by decide) (by decide)
    exact decide_eq_false composite
  second := false
  second_correct := by
    have composite : ¬ Nat.Prime (corner 9838 2) := by change ¬ Nat.Prime 387144977; exact PrattCertificates.composite_of_product (factor := 12197) (cofactor := 31741) rfl (by decide) (by decide)
    exact decide_eq_false composite
  third := false
  third_correct := by
    have composite : ¬ Nat.Prime (corner 9838 3) := by change ¬ Nat.Prime 387125301; exact PrattCertificates.composite_of_product (factor := 3) (cofactor := 129041767) rfl (by decide) (by decide)
    exact decide_eq_false composite
theorem row_count_9838 : row 9838 = 0 := (row_9838.checked).trans rfl
theorem model_count_9838 : modelRow 9838 = 0 := by decide
theorem modelrow_agrees_9838 : modelRow 9838 = row 9838 := model_count_9838.trans row_count_9838.symm
def row_9839 : RowCertificate 9839 where
  first := false
  first_correct := by
    have composite : ¬ Nat.Prime (corner 9839 1) := by change ¬ Nat.Prime 387243363; exact PrattCertificates.composite_of_product (factor := 3) (cofactor := 129081121) rfl (by decide) (by decide)
    exact decide_eq_false composite
  second := false
  second_correct := by
    have composite : ¬ Nat.Prime (corner 9839 2) := by change ¬ Nat.Prime 387223685; exact PrattCertificates.composite_of_product (factor := 5) (cofactor := 77444737) rfl (by decide) (by decide)
    exact decide_eq_false composite
  third := true
  third_correct := by
    have prime : Nat.Prime (corner 9839 3) := by change Nat.Prime 387204007; exact ProjectEulerSolutions.P58PrimeNodes.p_387204007.prime
    exact decide_eq_true prime
theorem row_count_9839 : row 9839 = 1 := (row_9839.checked).trans rfl
theorem model_count_9839 : modelRow 9839 = 1 := by decide
theorem modelrow_agrees_9839 : modelRow 9839 = row 9839 := model_count_9839.trans row_count_9839.symm
def row_9840 : RowCertificate 9840 where
  first := false
  first_correct := by
    have composite : ¬ Nat.Prime (corner 9840 1) := by change ¬ Nat.Prime 387322081; exact PrattCertificates.composite_of_product (factor := 181) (cofactor := 2139901) rfl (by decide) (by decide)
    exact decide_eq_false composite
  second := false
  second_correct := by
    have composite : ¬ Nat.Prime (corner 9840 2) := by change ¬ Nat.Prime 387302401; exact PrattCertificates.composite_of_product (factor := 349) (cofactor := 1109749) rfl (by decide) (by decide)
    exact decide_eq_false composite
  third := false
  third_correct := by
    have composite : ¬ Nat.Prime (corner 9840 3) := by change ¬ Nat.Prime 387282721; exact PrattCertificates.composite_of_product (factor := 7) (cofactor := 55326103) rfl (by decide) (by decide)
    exact decide_eq_false composite
theorem row_count_9840 : row 9840 = 0 := (row_9840.checked).trans rfl
theorem model_count_9840 : modelRow 9840 = 0 := by decide
theorem modelrow_agrees_9840 : modelRow 9840 = row 9840 := model_count_9840.trans row_count_9840.symm
def row_9841 : RowCertificate 9841 where
  first := false
  first_correct := by
    have composite : ¬ Nat.Prime (corner 9841 1) := by change ¬ Nat.Prime 387400807; exact PrattCertificates.composite_of_product (factor := 19441) (cofactor := 19927) rfl (by decide) (by decide)
    exact decide_eq_false composite
  second := false
  second_correct := by
    have composite : ¬ Nat.Prime (corner 9841 2) := by change ¬ Nat.Prime 387381125; exact PrattCertificates.composite_of_product (factor := 5) (cofactor := 77476225) rfl (by decide) (by decide)
    exact decide_eq_false composite
  third := false
  third_correct := by
    have composite : ¬ Nat.Prime (corner 9841 3) := by change ¬ Nat.Prime 387361443; exact PrattCertificates.composite_of_product (factor := 3) (cofactor := 129120481) rfl (by decide) (by decide)
    exact decide_eq_false composite
theorem row_count_9841 : row 9841 = 0 := (row_9841.checked).trans rfl
theorem model_count_9841 : modelRow 9841 = 0 := by decide
theorem modelrow_agrees_9841 : modelRow 9841 = row 9841 := model_count_9841.trans row_count_9841.symm
def row_9842 : RowCertificate 9842 where
  first := false
  first_correct := by
    have composite : ¬ Nat.Prime (corner 9842 1) := by change ¬ Nat.Prime 387479541; exact PrattCertificates.composite_of_product (factor := 3) (cofactor := 129159847) rfl (by decide) (by decide)
    exact decide_eq_false composite
  second := false
  second_correct := by
    have composite : ¬ Nat.Prime (corner 9842 2) := by change ¬ Nat.Prime 387459857; exact PrattCertificates.composite_of_product (factor := 9013) (cofactor := 42989) rfl (by decide) (by decide)
    exact decide_eq_false composite
  third := false
  third_correct := by
    have composite : ¬ Nat.Prime (corner 9842 3) := by change ¬ Nat.Prime 387440173; exact PrattCertificates.composite_of_product (factor := 109) (cofactor := 3554497) rfl (by decide) (by decide)
    exact decide_eq_false composite
theorem row_count_9842 : row 9842 = 0 := (row_9842.checked).trans rfl
theorem model_count_9842 : modelRow 9842 = 0 := by decide
theorem modelrow_agrees_9842 : modelRow 9842 = row 9842 := model_count_9842.trans row_count_9842.symm
def row_9843 : RowCertificate 9843 where
  first := false
  first_correct := by
    have composite : ¬ Nat.Prime (corner 9843 1) := by change ¬ Nat.Prime 387558283; exact PrattCertificates.composite_of_product (factor := 7) (cofactor := 55365469) rfl (by decide) (by decide)
    exact decide_eq_false composite
  second := false
  second_correct := by
    have composite : ¬ Nat.Prime (corner 9843 2) := by change ¬ Nat.Prime 387538597; exact PrattCertificates.composite_of_product (factor := 53) (cofactor := 7312049) rfl (by decide) (by decide)
    exact decide_eq_false composite
  third := false
  third_correct := by
    have composite : ¬ Nat.Prime (corner 9843 3) := by change ¬ Nat.Prime 387518911; exact PrattCertificates.composite_of_product (factor := 13) (cofactor := 29809147) rfl (by decide) (by decide)
    exact decide_eq_false composite
theorem row_count_9843 : row 9843 = 0 := (row_9843.checked).trans rfl
theorem model_count_9843 : modelRow 9843 = 0 := by decide
theorem modelrow_agrees_9843 : modelRow 9843 = row 9843 := model_count_9843.trans row_count_9843.symm
def row_9844 : RowCertificate 9844 where
  first := false
  first_correct := by
    have composite : ¬ Nat.Prime (corner 9844 1) := by change ¬ Nat.Prime 387637033; exact PrattCertificates.composite_of_product (factor := 7) (cofactor := 55376719) rfl (by decide) (by decide)
    exact decide_eq_false composite
  second := false
  second_correct := by
    have composite : ¬ Nat.Prime (corner 9844 2) := by change ¬ Nat.Prime 387617345; exact PrattCertificates.composite_of_product (factor := 5) (cofactor := 77523469) rfl (by decide) (by decide)
    exact decide_eq_false composite
  third := false
  third_correct := by
    have composite : ¬ Nat.Prime (corner 9844 3) := by change ¬ Nat.Prime 387597657; exact PrattCertificates.composite_of_product (factor := 3) (cofactor := 129199219) rfl (by decide) (by decide)
    exact decide_eq_false composite
theorem row_count_9844 : row 9844 = 0 := (row_9844.checked).trans rfl
theorem model_count_9844 : modelRow 9844 = 0 := by decide
theorem modelrow_agrees_9844 : modelRow 9844 = row 9844 := model_count_9844.trans row_count_9844.symm
def row_9845 : RowCertificate 9845 where
  first := false
  first_correct := by
    have composite : ¬ Nat.Prime (corner 9845 1) := by change ¬ Nat.Prime 387715791; exact PrattCertificates.composite_of_product (factor := 3) (cofactor := 129238597) rfl (by decide) (by decide)
    exact decide_eq_false composite
  second := false
  second_correct := by
    have composite : ¬ Nat.Prime (corner 9845 2) := by change ¬ Nat.Prime 387696101; exact PrattCertificates.composite_of_product (factor := 13) (cofactor := 29822777) rfl (by decide) (by decide)
    exact decide_eq_false composite
  third := false
  third_correct := by
    have composite : ¬ Nat.Prime (corner 9845 3) := by change ¬ Nat.Prime 387676411; exact PrattCertificates.composite_of_product (factor := 61) (cofactor := 6355351) rfl (by decide) (by decide)
    exact decide_eq_false composite
theorem row_count_9845 : row 9845 = 0 := (row_9845.checked).trans rfl
theorem model_count_9845 : modelRow 9845 = 0 := by decide
theorem modelrow_agrees_9845 : modelRow 9845 = row 9845 := model_count_9845.trans row_count_9845.symm
def row_9846 : RowCertificate 9846 where
  first := true
  first_correct := by
    have prime : Nat.Prime (corner 9846 1) := by change Nat.Prime 387794557; exact ProjectEulerSolutions.P58PrimeNodes.p_387794557.prime
    exact decide_eq_true prime
  second := false
  second_correct := by
    have composite : ¬ Nat.Prime (corner 9846 2) := by change ¬ Nat.Prime 387774865; exact PrattCertificates.composite_of_product (factor := 5) (cofactor := 77554973) rfl (by decide) (by decide)
    exact decide_eq_false composite
  third := false
  third_correct := by
    have composite : ¬ Nat.Prime (corner 9846 3) := by change ¬ Nat.Prime 387755173; exact PrattCertificates.composite_of_product (factor := 13) (cofactor := 29827321) rfl (by decide) (by decide)
    exact decide_eq_false composite
theorem row_count_9846 : row 9846 = 1 := (row_9846.checked).trans rfl
theorem model_count_9846 : modelRow 9846 = 1 := by decide
theorem modelrow_agrees_9846 : modelRow 9846 = row 9846 := model_count_9846.trans row_count_9846.symm
def row_9847 : RowCertificate 9847 where
  first := false
  first_correct := by
    have composite : ¬ Nat.Prime (corner 9847 1) := by change ¬ Nat.Prime 387873331; exact PrattCertificates.composite_of_product (factor := 37) (cofactor := 10483063) rfl (by decide) (by decide)
    exact decide_eq_false composite
  second := false
  second_correct := by
    have composite : ¬ Nat.Prime (corner 9847 2) := by change ¬ Nat.Prime 387853637; exact PrattCertificates.composite_of_product (factor := 313) (cofactor := 1239149) rfl (by decide) (by decide)
    exact decide_eq_false composite
  third := false
  third_correct := by
    have composite : ¬ Nat.Prime (corner 9847 3) := by change ¬ Nat.Prime 387833943; exact PrattCertificates.composite_of_product (factor := 3) (cofactor := 129277981) rfl (by decide) (by decide)
    exact decide_eq_false composite
theorem row_count_9847 : row 9847 = 0 := (row_9847.checked).trans rfl
theorem model_count_9847 : modelRow 9847 = 0 := by decide
theorem modelrow_agrees_9847 : modelRow 9847 = row 9847 := model_count_9847.trans row_count_9847.symm
def row_9848 : RowCertificate 9848 where
  first := false
  first_correct := by
    have composite : ¬ Nat.Prime (corner 9848 1) := by change ¬ Nat.Prime 387952113; exact PrattCertificates.composite_of_product (factor := 3) (cofactor := 129317371) rfl (by decide) (by decide)
    exact decide_eq_false composite
  second := false
  second_correct := by
    have composite : ¬ Nat.Prime (corner 9848 2) := by change ¬ Nat.Prime 387932417; exact PrattCertificates.composite_of_product (factor := 109) (cofactor := 3559013) rfl (by decide) (by decide)
    exact decide_eq_false composite
  third := false
  third_correct := by
    have composite : ¬ Nat.Prime (corner 9848 3) := by change ¬ Nat.Prime 387912721; exact PrattCertificates.composite_of_product (factor := 7) (cofactor := 55416103) rfl (by decide) (by decide)
    exact decide_eq_false composite
theorem row_count_9848 : row 9848 = 0 := (row_9848.checked).trans rfl
theorem model_count_9848 : modelRow 9848 = 0 := by decide
theorem modelrow_agrees_9848 : modelRow 9848 = row 9848 := model_count_9848.trans row_count_9848.symm
def row_9849 : RowCertificate 9849 where
  first := false
  first_correct := by
    have composite : ¬ Nat.Prime (corner 9849 1) := by change ¬ Nat.Prime 388030903; exact PrattCertificates.composite_of_product (factor := 13) (cofactor := 29848531) rfl (by decide) (by decide)
    exact decide_eq_false composite
  second := false
  second_correct := by
    have composite : ¬ Nat.Prime (corner 9849 2) := by change ¬ Nat.Prime 388011205; exact PrattCertificates.composite_of_product (factor := 5) (cofactor := 77602241) rfl (by decide) (by decide)
    exact decide_eq_false composite
  third := false
  third_correct := by
    have composite : ¬ Nat.Prime (corner 9849 3) := by change ¬ Nat.Prime 387991507; exact PrattCertificates.composite_of_product (factor := 2767) (cofactor := 140221) rfl (by decide) (by decide)
    exact decide_eq_false composite
theorem row_count_9849 : row 9849 = 0 := (row_9849.checked).trans rfl
theorem model_count_9849 : modelRow 9849 = 0 := by decide
theorem modelrow_agrees_9849 : modelRow 9849 = row 9849 := model_count_9849.trans row_count_9849.symm
def row_9850 : RowCertificate 9850 where
  first := false
  first_correct := by
    have composite : ¬ Nat.Prime (corner 9850 1) := by change ¬ Nat.Prime 388109701; exact PrattCertificates.composite_of_product (factor := 7) (cofactor := 55444243) rfl (by decide) (by decide)
    exact decide_eq_false composite
  second := false
  second_correct := by
    have composite : ¬ Nat.Prime (corner 9850 2) := by change ¬ Nat.Prime 388090001; exact PrattCertificates.composite_of_product (factor := 13) (cofactor := 29853077) rfl (by decide) (by decide)
    exact decide_eq_false composite
  third := false
  third_correct := by
    have composite : ¬ Nat.Prime (corner 9850 3) := by change ¬ Nat.Prime 388070301; exact PrattCertificates.composite_of_product (factor := 3) (cofactor := 129356767) rfl (by decide) (by decide)
    exact decide_eq_false composite
theorem row_count_9850 : row 9850 = 0 := (row_9850.checked).trans rfl
theorem model_count_9850 : modelRow 9850 = 0 := by decide
theorem modelrow_agrees_9850 : modelRow 9850 = row 9850 := model_count_9850.trans row_count_9850.symm
def row_9851 : RowCertificate 9851 where
  first := false
  first_correct := by
    have composite : ¬ Nat.Prime (corner 9851 1) := by change ¬ Nat.Prime 388188507; exact PrattCertificates.composite_of_product (factor := 3) (cofactor := 129396169) rfl (by decide) (by decide)
    exact decide_eq_false composite
  second := false
  second_correct := by
    have composite : ¬ Nat.Prime (corner 9851 2) := by change ¬ Nat.Prime 388168805; exact PrattCertificates.composite_of_product (factor := 5) (cofactor := 77633761) rfl (by decide) (by decide)
    exact decide_eq_false composite
  third := false
  third_correct := by
    have composite : ¬ Nat.Prime (corner 9851 3) := by change ¬ Nat.Prime 388149103; exact PrattCertificates.composite_of_product (factor := 73) (cofactor := 5317111) rfl (by decide) (by decide)
    exact decide_eq_false composite
theorem row_count_9851 : row 9851 = 0 := (row_9851.checked).trans rfl
theorem model_count_9851 : modelRow 9851 = 0 := by decide
theorem modelrow_agrees_9851 : modelRow 9851 = row 9851 := model_count_9851.trans row_count_9851.symm
def row_9852 : RowCertificate 9852 where
  first := false
  first_correct := by
    have composite : ¬ Nat.Prime (corner 9852 1) := by change ¬ Nat.Prime 388267321; exact PrattCertificates.composite_of_product (factor := 13) (cofactor := 29866717) rfl (by decide) (by decide)
    exact decide_eq_false composite
  second := false
  second_correct := by
    have composite : ¬ Nat.Prime (corner 9852 2) := by change ¬ Nat.Prime 388247617; exact PrattCertificates.composite_of_product (factor := 9437) (cofactor := 41141) rfl (by decide) (by decide)
    exact decide_eq_false composite
  third := true
  third_correct := by
    have prime : Nat.Prime (corner 9852 3) := by change Nat.Prime 388227913; exact ProjectEulerSolutions.P58PrimeNodes.p_388227913.prime
    exact decide_eq_true prime
theorem row_count_9852 : row 9852 = 1 := (row_9852.checked).trans rfl
theorem model_count_9852 : modelRow 9852 = 1 := by decide
theorem modelrow_agrees_9852 : modelRow 9852 = row 9852 := model_count_9852.trans row_count_9852.symm
def row_9853 : RowCertificate 9853 where
  first := false
  first_correct := by
    have composite : ¬ Nat.Prime (corner 9853 1) := by change ¬ Nat.Prime 388346143; exact PrattCertificates.composite_of_product (factor := 991) (cofactor := 391873) rfl (by decide) (by decide)
    exact decide_eq_false composite
  second := true
  second_correct := by
    have prime : Nat.Prime (corner 9853 2) := by change Nat.Prime 388326437; exact ProjectEulerSolutions.P58PrimeNodes.p_388326437.prime
    exact decide_eq_true prime
  third := false
  third_correct := by
    have composite : ¬ Nat.Prime (corner 9853 3) := by change ¬ Nat.Prime 388306731; exact PrattCertificates.composite_of_product (factor := 3) (cofactor := 129435577) rfl (by decide) (by decide)
    exact decide_eq_false composite
theorem row_count_9853 : row 9853 = 1 := (row_9853.checked).trans rfl
theorem model_count_9853 : modelRow 9853 = 1 := by decide
theorem modelrow_agrees_9853 : modelRow 9853 = row 9853 := model_count_9853.trans row_count_9853.symm
def row_9854 : RowCertificate 9854 where
  first := false
  first_correct := by
    have composite : ¬ Nat.Prime (corner 9854 1) := by change ¬ Nat.Prime 388424973; exact PrattCertificates.composite_of_product (factor := 3) (cofactor := 129474991) rfl (by decide) (by decide)
    exact decide_eq_false composite
  second := false
  second_correct := by
    have composite : ¬ Nat.Prime (corner 9854 2) := by change ¬ Nat.Prime 388405265; exact PrattCertificates.composite_of_product (factor := 5) (cofactor := 77681053) rfl (by decide) (by decide)
    exact decide_eq_false composite
  third := false
  third_correct := by
    have composite : ¬ Nat.Prime (corner 9854 3) := by change ¬ Nat.Prime 388385557; exact PrattCertificates.composite_of_product (factor := 7) (cofactor := 55483651) rfl (by decide) (by decide)
    exact decide_eq_false composite
theorem row_count_9854 : row 9854 = 0 := (row_9854.checked).trans rfl
theorem model_count_9854 : modelRow 9854 = 0 := by decide
theorem modelrow_agrees_9854 : modelRow 9854 = row 9854 := model_count_9854.trans row_count_9854.symm
def row_9855 : RowCertificate 9855 where
  first := false
  first_correct := by
    have composite : ¬ Nat.Prime (corner 9855 1) := by change ¬ Nat.Prime 388503811; exact PrattCertificates.composite_of_product (factor := 19) (cofactor := 20447569) rfl (by decide) (by decide)
    exact decide_eq_false composite
  second := false
  second_correct := by
    have composite : ¬ Nat.Prime (corner 9855 2) := by change ¬ Nat.Prime 388484101; exact PrattCertificates.composite_of_product (factor := 181) (cofactor := 2146321) rfl (by decide) (by decide)
    exact decide_eq_false composite
  third := false
  third_correct := by
    have composite : ¬ Nat.Prime (corner 9855 3) := by change ¬ Nat.Prime 388464391; exact PrattCertificates.composite_of_product (factor := 7) (cofactor := 55494913) rfl (by decide) (by decide)
    exact decide_eq_false composite
theorem row_count_9855 : row 9855 = 0 := (row_9855.checked).trans rfl
theorem model_count_9855 : modelRow 9855 = 0 := by decide
theorem modelrow_agrees_9855 : modelRow 9855 = row 9855 := model_count_9855.trans row_count_9855.symm
def row_9856 : RowCertificate 9856 where
  first := false
  first_correct := by
    have composite : ¬ Nat.Prime (corner 9856 1) := by change ¬ Nat.Prime 388582657; exact PrattCertificates.composite_of_product (factor := 487) (cofactor := 797911) rfl (by decide) (by decide)
    exact decide_eq_false composite
  second := false
  second_correct := by
    have composite : ¬ Nat.Prime (corner 9856 2) := by change ¬ Nat.Prime 388562945; exact PrattCertificates.composite_of_product (factor := 5) (cofactor := 77712589) rfl (by decide) (by decide)
    exact decide_eq_false composite
  third := false
  third_correct := by
    have composite : ¬ Nat.Prime (corner 9856 3) := by change ¬ Nat.Prime 388543233; exact PrattCertificates.composite_of_product (factor := 3) (cofactor := 129514411) rfl (by decide) (by decide)
    exact decide_eq_false composite
theorem row_count_9856 : row 9856 = 0 := (row_9856.checked).trans rfl
theorem model_count_9856 : modelRow 9856 = 0 := by decide
theorem modelrow_agrees_9856 : modelRow 9856 = row 9856 := model_count_9856.trans row_count_9856.symm
private def atomic_9729 : CountSegment 1 10 13120 9728 9729 4040 4041 := CountSegment.single (first := 9728) (initial := 4040) row_count_9729 (by decide)
private theorem atomic_9729_agreement : RowAgreement 9728 9729 := RowAgreement.single (first := 9728) modelrow_agrees_9729
private def atomic_9730 : CountSegment 1 10 13120 9729 9730 4041 4042 := CountSegment.single (first := 9729) (initial := 4041) row_count_9730 (by decide)
private theorem atomic_9730_agreement : RowAgreement 9729 9730 := RowAgreement.single (first := 9729) modelrow_agrees_9730
private def atomic_9731 : CountSegment 1 10 13120 9730 9731 4042 4042 := CountSegment.single (first := 9730) (initial := 4042) row_count_9731 (by decide)
private theorem atomic_9731_agreement : RowAgreement 9730 9731 := RowAgreement.single (first := 9730) modelrow_agrees_9731
private def atomic_9732 : CountSegment 1 10 13120 9731 9732 4042 4042 := CountSegment.single (first := 9731) (initial := 4042) row_count_9732 (by decide)
private theorem atomic_9732_agreement : RowAgreement 9731 9732 := RowAgreement.single (first := 9731) modelrow_agrees_9732
private def atomic_9733 : CountSegment 1 10 13120 9732 9733 4042 4043 := CountSegment.single (first := 9732) (initial := 4042) row_count_9733 (by decide)
private theorem atomic_9733_agreement : RowAgreement 9732 9733 := RowAgreement.single (first := 9732) modelrow_agrees_9733
private def atomic_9734 : CountSegment 1 10 13120 9733 9734 4043 4043 := CountSegment.single (first := 9733) (initial := 4043) row_count_9734 (by decide)
private theorem atomic_9734_agreement : RowAgreement 9733 9734 := RowAgreement.single (first := 9733) modelrow_agrees_9734
private def atomic_9735 : CountSegment 1 10 13120 9734 9735 4043 4043 := CountSegment.single (first := 9734) (initial := 4043) row_count_9735 (by decide)
private theorem atomic_9735_agreement : RowAgreement 9734 9735 := RowAgreement.single (first := 9734) modelrow_agrees_9735
private def atomic_9736 : CountSegment 1 10 13120 9735 9736 4043 4043 := CountSegment.single (first := 9735) (initial := 4043) row_count_9736 (by decide)
private theorem atomic_9736_agreement : RowAgreement 9735 9736 := RowAgreement.single (first := 9735) modelrow_agrees_9736
private def atomic_9737 : CountSegment 1 10 13120 9736 9737 4043 4043 := CountSegment.single (first := 9736) (initial := 4043) row_count_9737 (by decide)
private theorem atomic_9737_agreement : RowAgreement 9736 9737 := RowAgreement.single (first := 9736) modelrow_agrees_9737
private def atomic_9738 : CountSegment 1 10 13120 9737 9738 4043 4044 := CountSegment.single (first := 9737) (initial := 4043) row_count_9738 (by decide)
private theorem atomic_9738_agreement : RowAgreement 9737 9738 := RowAgreement.single (first := 9737) modelrow_agrees_9738
private def atomic_9739 : CountSegment 1 10 13120 9738 9739 4044 4044 := CountSegment.single (first := 9738) (initial := 4044) row_count_9739 (by decide)
private theorem atomic_9739_agreement : RowAgreement 9738 9739 := RowAgreement.single (first := 9738) modelrow_agrees_9739
private def atomic_9740 : CountSegment 1 10 13120 9739 9740 4044 4046 := CountSegment.single (first := 9739) (initial := 4044) row_count_9740 (by decide)
private theorem atomic_9740_agreement : RowAgreement 9739 9740 := RowAgreement.single (first := 9739) modelrow_agrees_9740
private def atomic_9741 : CountSegment 1 10 13120 9740 9741 4046 4047 := CountSegment.single (first := 9740) (initial := 4046) row_count_9741 (by decide)
private theorem atomic_9741_agreement : RowAgreement 9740 9741 := RowAgreement.single (first := 9740) modelrow_agrees_9741
private def atomic_9742 : CountSegment 1 10 13120 9741 9742 4047 4047 := CountSegment.single (first := 9741) (initial := 4047) row_count_9742 (by decide)
private theorem atomic_9742_agreement : RowAgreement 9741 9742 := RowAgreement.single (first := 9741) modelrow_agrees_9742
private def atomic_9743 : CountSegment 1 10 13120 9742 9743 4047 4047 := CountSegment.single (first := 9742) (initial := 4047) row_count_9743 (by decide)
private theorem atomic_9743_agreement : RowAgreement 9742 9743 := RowAgreement.single (first := 9742) modelrow_agrees_9743
private def atomic_9744 : CountSegment 1 10 13120 9743 9744 4047 4047 := CountSegment.single (first := 9743) (initial := 4047) row_count_9744 (by decide)
private theorem atomic_9744_agreement : RowAgreement 9743 9744 := RowAgreement.single (first := 9743) modelrow_agrees_9744
private def atomic_9745 : CountSegment 1 10 13120 9744 9745 4047 4048 := CountSegment.single (first := 9744) (initial := 4047) row_count_9745 (by decide)
private theorem atomic_9745_agreement : RowAgreement 9744 9745 := RowAgreement.single (first := 9744) modelrow_agrees_9745
private def atomic_9746 : CountSegment 1 10 13120 9745 9746 4048 4048 := CountSegment.single (first := 9745) (initial := 4048) row_count_9746 (by decide)
private theorem atomic_9746_agreement : RowAgreement 9745 9746 := RowAgreement.single (first := 9745) modelrow_agrees_9746
private def atomic_9747 : CountSegment 1 10 13120 9746 9747 4048 4050 := CountSegment.single (first := 9746) (initial := 4048) row_count_9747 (by decide)
private theorem atomic_9747_agreement : RowAgreement 9746 9747 := RowAgreement.single (first := 9746) modelrow_agrees_9747
private def atomic_9748 : CountSegment 1 10 13120 9747 9748 4050 4050 := CountSegment.single (first := 9747) (initial := 4050) row_count_9748 (by decide)
private theorem atomic_9748_agreement : RowAgreement 9747 9748 := RowAgreement.single (first := 9747) modelrow_agrees_9748
private def atomic_9749 : CountSegment 1 10 13120 9748 9749 4050 4050 := CountSegment.single (first := 9748) (initial := 4050) row_count_9749 (by decide)
private theorem atomic_9749_agreement : RowAgreement 9748 9749 := RowAgreement.single (first := 9748) modelrow_agrees_9749
private def atomic_9750 : CountSegment 1 10 13120 9749 9750 4050 4051 := CountSegment.single (first := 9749) (initial := 4050) row_count_9750 (by decide)
private theorem atomic_9750_agreement : RowAgreement 9749 9750 := RowAgreement.single (first := 9749) modelrow_agrees_9750
private def atomic_9751 : CountSegment 1 10 13120 9750 9751 4051 4051 := CountSegment.single (first := 9750) (initial := 4051) row_count_9751 (by decide)
private theorem atomic_9751_agreement : RowAgreement 9750 9751 := RowAgreement.single (first := 9750) modelrow_agrees_9751
private def atomic_9752 : CountSegment 1 10 13120 9751 9752 4051 4051 := CountSegment.single (first := 9751) (initial := 4051) row_count_9752 (by decide)
private theorem atomic_9752_agreement : RowAgreement 9751 9752 := RowAgreement.single (first := 9751) modelrow_agrees_9752
private def atomic_9753 : CountSegment 1 10 13120 9752 9753 4051 4052 := CountSegment.single (first := 9752) (initial := 4051) row_count_9753 (by decide)
private theorem atomic_9753_agreement : RowAgreement 9752 9753 := RowAgreement.single (first := 9752) modelrow_agrees_9753
private def atomic_9754 : CountSegment 1 10 13120 9753 9754 4052 4053 := CountSegment.single (first := 9753) (initial := 4052) row_count_9754 (by decide)
private theorem atomic_9754_agreement : RowAgreement 9753 9754 := RowAgreement.single (first := 9753) modelrow_agrees_9754
private def atomic_9755 : CountSegment 1 10 13120 9754 9755 4053 4054 := CountSegment.single (first := 9754) (initial := 4053) row_count_9755 (by decide)
private theorem atomic_9755_agreement : RowAgreement 9754 9755 := RowAgreement.single (first := 9754) modelrow_agrees_9755
private def atomic_9756 : CountSegment 1 10 13120 9755 9756 4054 4055 := CountSegment.single (first := 9755) (initial := 4054) row_count_9756 (by decide)
private theorem atomic_9756_agreement : RowAgreement 9755 9756 := RowAgreement.single (first := 9755) modelrow_agrees_9756
private def atomic_9757 : CountSegment 1 10 13120 9756 9757 4055 4055 := CountSegment.single (first := 9756) (initial := 4055) row_count_9757 (by decide)
private theorem atomic_9757_agreement : RowAgreement 9756 9757 := RowAgreement.single (first := 9756) modelrow_agrees_9757
private def atomic_9758 : CountSegment 1 10 13120 9757 9758 4055 4055 := CountSegment.single (first := 9757) (initial := 4055) row_count_9758 (by decide)
private theorem atomic_9758_agreement : RowAgreement 9757 9758 := RowAgreement.single (first := 9757) modelrow_agrees_9758
private def atomic_9759 : CountSegment 1 10 13120 9758 9759 4055 4055 := CountSegment.single (first := 9758) (initial := 4055) row_count_9759 (by decide)
private theorem atomic_9759_agreement : RowAgreement 9758 9759 := RowAgreement.single (first := 9758) modelrow_agrees_9759
private def atomic_9760 : CountSegment 1 10 13120 9759 9760 4055 4055 := CountSegment.single (first := 9759) (initial := 4055) row_count_9760 (by decide)
private theorem atomic_9760_agreement : RowAgreement 9759 9760 := RowAgreement.single (first := 9759) modelrow_agrees_9760
private def atomic_9761 : CountSegment 1 10 13120 9760 9761 4055 4056 := CountSegment.single (first := 9760) (initial := 4055) row_count_9761 (by decide)
private theorem atomic_9761_agreement : RowAgreement 9760 9761 := RowAgreement.single (first := 9760) modelrow_agrees_9761
private def atomic_9762 : CountSegment 1 10 13120 9761 9762 4056 4057 := CountSegment.single (first := 9761) (initial := 4056) row_count_9762 (by decide)
private theorem atomic_9762_agreement : RowAgreement 9761 9762 := RowAgreement.single (first := 9761) modelrow_agrees_9762
private def atomic_9763 : CountSegment 1 10 13120 9762 9763 4057 4057 := CountSegment.single (first := 9762) (initial := 4057) row_count_9763 (by decide)
private theorem atomic_9763_agreement : RowAgreement 9762 9763 := RowAgreement.single (first := 9762) modelrow_agrees_9763
private def atomic_9764 : CountSegment 1 10 13120 9763 9764 4057 4057 := CountSegment.single (first := 9763) (initial := 4057) row_count_9764 (by decide)
private theorem atomic_9764_agreement : RowAgreement 9763 9764 := RowAgreement.single (first := 9763) modelrow_agrees_9764
private def atomic_9765 : CountSegment 1 10 13120 9764 9765 4057 4058 := CountSegment.single (first := 9764) (initial := 4057) row_count_9765 (by decide)
private theorem atomic_9765_agreement : RowAgreement 9764 9765 := RowAgreement.single (first := 9764) modelrow_agrees_9765
private def atomic_9766 : CountSegment 1 10 13120 9765 9766 4058 4058 := CountSegment.single (first := 9765) (initial := 4058) row_count_9766 (by decide)
private theorem atomic_9766_agreement : RowAgreement 9765 9766 := RowAgreement.single (first := 9765) modelrow_agrees_9766
private def atomic_9767 : CountSegment 1 10 13120 9766 9767 4058 4058 := CountSegment.single (first := 9766) (initial := 4058) row_count_9767 (by decide)
private theorem atomic_9767_agreement : RowAgreement 9766 9767 := RowAgreement.single (first := 9766) modelrow_agrees_9767
private def atomic_9768 : CountSegment 1 10 13120 9767 9768 4058 4059 := CountSegment.single (first := 9767) (initial := 4058) row_count_9768 (by decide)
private theorem atomic_9768_agreement : RowAgreement 9767 9768 := RowAgreement.single (first := 9767) modelrow_agrees_9768
private def atomic_9769 : CountSegment 1 10 13120 9768 9769 4059 4059 := CountSegment.single (first := 9768) (initial := 4059) row_count_9769 (by decide)
private theorem atomic_9769_agreement : RowAgreement 9768 9769 := RowAgreement.single (first := 9768) modelrow_agrees_9769
private def atomic_9770 : CountSegment 1 10 13120 9769 9770 4059 4059 := CountSegment.single (first := 9769) (initial := 4059) row_count_9770 (by decide)
private theorem atomic_9770_agreement : RowAgreement 9769 9770 := RowAgreement.single (first := 9769) modelrow_agrees_9770
private def atomic_9771 : CountSegment 1 10 13120 9770 9771 4059 4059 := CountSegment.single (first := 9770) (initial := 4059) row_count_9771 (by decide)
private theorem atomic_9771_agreement : RowAgreement 9770 9771 := RowAgreement.single (first := 9770) modelrow_agrees_9771
private def atomic_9772 : CountSegment 1 10 13120 9771 9772 4059 4060 := CountSegment.single (first := 9771) (initial := 4059) row_count_9772 (by decide)
private theorem atomic_9772_agreement : RowAgreement 9771 9772 := RowAgreement.single (first := 9771) modelrow_agrees_9772
private def atomic_9773 : CountSegment 1 10 13120 9772 9773 4060 4060 := CountSegment.single (first := 9772) (initial := 4060) row_count_9773 (by decide)
private theorem atomic_9773_agreement : RowAgreement 9772 9773 := RowAgreement.single (first := 9772) modelrow_agrees_9773
private def atomic_9774 : CountSegment 1 10 13120 9773 9774 4060 4060 := CountSegment.single (first := 9773) (initial := 4060) row_count_9774 (by decide)
private theorem atomic_9774_agreement : RowAgreement 9773 9774 := RowAgreement.single (first := 9773) modelrow_agrees_9774
private def atomic_9775 : CountSegment 1 10 13120 9774 9775 4060 4060 := CountSegment.single (first := 9774) (initial := 4060) row_count_9775 (by decide)
private theorem atomic_9775_agreement : RowAgreement 9774 9775 := RowAgreement.single (first := 9774) modelrow_agrees_9775
private def atomic_9776 : CountSegment 1 10 13120 9775 9776 4060 4061 := CountSegment.single (first := 9775) (initial := 4060) row_count_9776 (by decide)
private theorem atomic_9776_agreement : RowAgreement 9775 9776 := RowAgreement.single (first := 9775) modelrow_agrees_9776
private def atomic_9777 : CountSegment 1 10 13120 9776 9777 4061 4062 := CountSegment.single (first := 9776) (initial := 4061) row_count_9777 (by decide)
private theorem atomic_9777_agreement : RowAgreement 9776 9777 := RowAgreement.single (first := 9776) modelrow_agrees_9777
private def atomic_9778 : CountSegment 1 10 13120 9777 9778 4062 4063 := CountSegment.single (first := 9777) (initial := 4062) row_count_9778 (by decide)
private theorem atomic_9778_agreement : RowAgreement 9777 9778 := RowAgreement.single (first := 9777) modelrow_agrees_9778
private def atomic_9779 : CountSegment 1 10 13120 9778 9779 4063 4063 := CountSegment.single (first := 9778) (initial := 4063) row_count_9779 (by decide)
private theorem atomic_9779_agreement : RowAgreement 9778 9779 := RowAgreement.single (first := 9778) modelrow_agrees_9779
private def atomic_9780 : CountSegment 1 10 13120 9779 9780 4063 4063 := CountSegment.single (first := 9779) (initial := 4063) row_count_9780 (by decide)
private theorem atomic_9780_agreement : RowAgreement 9779 9780 := RowAgreement.single (first := 9779) modelrow_agrees_9780
private def atomic_9781 : CountSegment 1 10 13120 9780 9781 4063 4063 := CountSegment.single (first := 9780) (initial := 4063) row_count_9781 (by decide)
private theorem atomic_9781_agreement : RowAgreement 9780 9781 := RowAgreement.single (first := 9780) modelrow_agrees_9781
private def atomic_9782 : CountSegment 1 10 13120 9781 9782 4063 4063 := CountSegment.single (first := 9781) (initial := 4063) row_count_9782 (by decide)
private theorem atomic_9782_agreement : RowAgreement 9781 9782 := RowAgreement.single (first := 9781) modelrow_agrees_9782
private def atomic_9783 : CountSegment 1 10 13120 9782 9783 4063 4063 := CountSegment.single (first := 9782) (initial := 4063) row_count_9783 (by decide)
private theorem atomic_9783_agreement : RowAgreement 9782 9783 := RowAgreement.single (first := 9782) modelrow_agrees_9783
private def atomic_9784 : CountSegment 1 10 13120 9783 9784 4063 4063 := CountSegment.single (first := 9783) (initial := 4063) row_count_9784 (by decide)
private theorem atomic_9784_agreement : RowAgreement 9783 9784 := RowAgreement.single (first := 9783) modelrow_agrees_9784
private def atomic_9785 : CountSegment 1 10 13120 9784 9785 4063 4063 := CountSegment.single (first := 9784) (initial := 4063) row_count_9785 (by decide)
private theorem atomic_9785_agreement : RowAgreement 9784 9785 := RowAgreement.single (first := 9784) modelrow_agrees_9785
private def atomic_9786 : CountSegment 1 10 13120 9785 9786 4063 4063 := CountSegment.single (first := 9785) (initial := 4063) row_count_9786 (by decide)
private theorem atomic_9786_agreement : RowAgreement 9785 9786 := RowAgreement.single (first := 9785) modelrow_agrees_9786
private def atomic_9787 : CountSegment 1 10 13120 9786 9787 4063 4063 := CountSegment.single (first := 9786) (initial := 4063) row_count_9787 (by decide)
private theorem atomic_9787_agreement : RowAgreement 9786 9787 := RowAgreement.single (first := 9786) modelrow_agrees_9787
private def atomic_9788 : CountSegment 1 10 13120 9787 9788 4063 4065 := CountSegment.single (first := 9787) (initial := 4063) row_count_9788 (by decide)
private theorem atomic_9788_agreement : RowAgreement 9787 9788 := RowAgreement.single (first := 9787) modelrow_agrees_9788
private def atomic_9789 : CountSegment 1 10 13120 9788 9789 4065 4065 := CountSegment.single (first := 9788) (initial := 4065) row_count_9789 (by decide)
private theorem atomic_9789_agreement : RowAgreement 9788 9789 := RowAgreement.single (first := 9788) modelrow_agrees_9789
private def atomic_9790 : CountSegment 1 10 13120 9789 9790 4065 4065 := CountSegment.single (first := 9789) (initial := 4065) row_count_9790 (by decide)
private theorem atomic_9790_agreement : RowAgreement 9789 9790 := RowAgreement.single (first := 9789) modelrow_agrees_9790
private def atomic_9791 : CountSegment 1 10 13120 9790 9791 4065 4065 := CountSegment.single (first := 9790) (initial := 4065) row_count_9791 (by decide)
private theorem atomic_9791_agreement : RowAgreement 9790 9791 := RowAgreement.single (first := 9790) modelrow_agrees_9791
private def atomic_9792 : CountSegment 1 10 13120 9791 9792 4065 4065 := CountSegment.single (first := 9791) (initial := 4065) row_count_9792 (by decide)
private theorem atomic_9792_agreement : RowAgreement 9791 9792 := RowAgreement.single (first := 9791) modelrow_agrees_9792
private def atomic_9793 : CountSegment 1 10 13120 9792 9793 4065 4065 := CountSegment.single (first := 9792) (initial := 4065) row_count_9793 (by decide)
private theorem atomic_9793_agreement : RowAgreement 9792 9793 := RowAgreement.single (first := 9792) modelrow_agrees_9793
private def atomic_9794 : CountSegment 1 10 13120 9793 9794 4065 4065 := CountSegment.single (first := 9793) (initial := 4065) row_count_9794 (by decide)
private theorem atomic_9794_agreement : RowAgreement 9793 9794 := RowAgreement.single (first := 9793) modelrow_agrees_9794
private def atomic_9795 : CountSegment 1 10 13120 9794 9795 4065 4066 := CountSegment.single (first := 9794) (initial := 4065) row_count_9795 (by decide)
private theorem atomic_9795_agreement : RowAgreement 9794 9795 := RowAgreement.single (first := 9794) modelrow_agrees_9795
private def atomic_9796 : CountSegment 1 10 13120 9795 9796 4066 4067 := CountSegment.single (first := 9795) (initial := 4066) row_count_9796 (by decide)
private theorem atomic_9796_agreement : RowAgreement 9795 9796 := RowAgreement.single (first := 9795) modelrow_agrees_9796
private def atomic_9797 : CountSegment 1 10 13120 9796 9797 4067 4067 := CountSegment.single (first := 9796) (initial := 4067) row_count_9797 (by decide)
private theorem atomic_9797_agreement : RowAgreement 9796 9797 := RowAgreement.single (first := 9796) modelrow_agrees_9797
private def atomic_9798 : CountSegment 1 10 13120 9797 9798 4067 4067 := CountSegment.single (first := 9797) (initial := 4067) row_count_9798 (by decide)
private theorem atomic_9798_agreement : RowAgreement 9797 9798 := RowAgreement.single (first := 9797) modelrow_agrees_9798
private def atomic_9799 : CountSegment 1 10 13120 9798 9799 4067 4067 := CountSegment.single (first := 9798) (initial := 4067) row_count_9799 (by decide)
private theorem atomic_9799_agreement : RowAgreement 9798 9799 := RowAgreement.single (first := 9798) modelrow_agrees_9799
private def atomic_9800 : CountSegment 1 10 13120 9799 9800 4067 4068 := CountSegment.single (first := 9799) (initial := 4067) row_count_9800 (by decide)
private theorem atomic_9800_agreement : RowAgreement 9799 9800 := RowAgreement.single (first := 9799) modelrow_agrees_9800
private def atomic_9801 : CountSegment 1 10 13120 9800 9801 4068 4068 := CountSegment.single (first := 9800) (initial := 4068) row_count_9801 (by decide)
private theorem atomic_9801_agreement : RowAgreement 9800 9801 := RowAgreement.single (first := 9800) modelrow_agrees_9801
private def atomic_9802 : CountSegment 1 10 13120 9801 9802 4068 4068 := CountSegment.single (first := 9801) (initial := 4068) row_count_9802 (by decide)
private theorem atomic_9802_agreement : RowAgreement 9801 9802 := RowAgreement.single (first := 9801) modelrow_agrees_9802
private def atomic_9803 : CountSegment 1 10 13120 9802 9803 4068 4068 := CountSegment.single (first := 9802) (initial := 4068) row_count_9803 (by decide)
private theorem atomic_9803_agreement : RowAgreement 9802 9803 := RowAgreement.single (first := 9802) modelrow_agrees_9803
private def atomic_9804 : CountSegment 1 10 13120 9803 9804 4068 4068 := CountSegment.single (first := 9803) (initial := 4068) row_count_9804 (by decide)
private theorem atomic_9804_agreement : RowAgreement 9803 9804 := RowAgreement.single (first := 9803) modelrow_agrees_9804
private def atomic_9805 : CountSegment 1 10 13120 9804 9805 4068 4069 := CountSegment.single (first := 9804) (initial := 4068) row_count_9805 (by decide)
private theorem atomic_9805_agreement : RowAgreement 9804 9805 := RowAgreement.single (first := 9804) modelrow_agrees_9805
private def atomic_9806 : CountSegment 1 10 13120 9805 9806 4069 4069 := CountSegment.single (first := 9805) (initial := 4069) row_count_9806 (by decide)
private theorem atomic_9806_agreement : RowAgreement 9805 9806 := RowAgreement.single (first := 9805) modelrow_agrees_9806
private def atomic_9807 : CountSegment 1 10 13120 9806 9807 4069 4069 := CountSegment.single (first := 9806) (initial := 4069) row_count_9807 (by decide)
private theorem atomic_9807_agreement : RowAgreement 9806 9807 := RowAgreement.single (first := 9806) modelrow_agrees_9807
private def atomic_9808 : CountSegment 1 10 13120 9807 9808 4069 4069 := CountSegment.single (first := 9807) (initial := 4069) row_count_9808 (by decide)
private theorem atomic_9808_agreement : RowAgreement 9807 9808 := RowAgreement.single (first := 9807) modelrow_agrees_9808
private def atomic_9809 : CountSegment 1 10 13120 9808 9809 4069 4069 := CountSegment.single (first := 9808) (initial := 4069) row_count_9809 (by decide)
private theorem atomic_9809_agreement : RowAgreement 9808 9809 := RowAgreement.single (first := 9808) modelrow_agrees_9809
private def atomic_9810 : CountSegment 1 10 13120 9809 9810 4069 4070 := CountSegment.single (first := 9809) (initial := 4069) row_count_9810 (by decide)
private theorem atomic_9810_agreement : RowAgreement 9809 9810 := RowAgreement.single (first := 9809) modelrow_agrees_9810
private def atomic_9811 : CountSegment 1 10 13120 9810 9811 4070 4071 := CountSegment.single (first := 9810) (initial := 4070) row_count_9811 (by decide)
private theorem atomic_9811_agreement : RowAgreement 9810 9811 := RowAgreement.single (first := 9810) modelrow_agrees_9811
private def atomic_9812 : CountSegment 1 10 13120 9811 9812 4071 4071 := CountSegment.single (first := 9811) (initial := 4071) row_count_9812 (by decide)
private theorem atomic_9812_agreement : RowAgreement 9811 9812 := RowAgreement.single (first := 9811) modelrow_agrees_9812
private def atomic_9813 : CountSegment 1 10 13120 9812 9813 4071 4072 := CountSegment.single (first := 9812) (initial := 4071) row_count_9813 (by decide)
private theorem atomic_9813_agreement : RowAgreement 9812 9813 := RowAgreement.single (first := 9812) modelrow_agrees_9813
private def atomic_9814 : CountSegment 1 10 13120 9813 9814 4072 4072 := CountSegment.single (first := 9813) (initial := 4072) row_count_9814 (by decide)
private theorem atomic_9814_agreement : RowAgreement 9813 9814 := RowAgreement.single (first := 9813) modelrow_agrees_9814
private def atomic_9815 : CountSegment 1 10 13120 9814 9815 4072 4072 := CountSegment.single (first := 9814) (initial := 4072) row_count_9815 (by decide)
private theorem atomic_9815_agreement : RowAgreement 9814 9815 := RowAgreement.single (first := 9814) modelrow_agrees_9815
private def atomic_9816 : CountSegment 1 10 13120 9815 9816 4072 4073 := CountSegment.single (first := 9815) (initial := 4072) row_count_9816 (by decide)
private theorem atomic_9816_agreement : RowAgreement 9815 9816 := RowAgreement.single (first := 9815) modelrow_agrees_9816
private def atomic_9817 : CountSegment 1 10 13120 9816 9817 4073 4074 := CountSegment.single (first := 9816) (initial := 4073) row_count_9817 (by decide)
private theorem atomic_9817_agreement : RowAgreement 9816 9817 := RowAgreement.single (first := 9816) modelrow_agrees_9817
private def atomic_9818 : CountSegment 1 10 13120 9817 9818 4074 4075 := CountSegment.single (first := 9817) (initial := 4074) row_count_9818 (by decide)
private theorem atomic_9818_agreement : RowAgreement 9817 9818 := RowAgreement.single (first := 9817) modelrow_agrees_9818
private def atomic_9819 : CountSegment 1 10 13120 9818 9819 4075 4075 := CountSegment.single (first := 9818) (initial := 4075) row_count_9819 (by decide)
private theorem atomic_9819_agreement : RowAgreement 9818 9819 := RowAgreement.single (first := 9818) modelrow_agrees_9819
private def atomic_9820 : CountSegment 1 10 13120 9819 9820 4075 4075 := CountSegment.single (first := 9819) (initial := 4075) row_count_9820 (by decide)
private theorem atomic_9820_agreement : RowAgreement 9819 9820 := RowAgreement.single (first := 9819) modelrow_agrees_9820
private def atomic_9821 : CountSegment 1 10 13120 9820 9821 4075 4075 := CountSegment.single (first := 9820) (initial := 4075) row_count_9821 (by decide)
private theorem atomic_9821_agreement : RowAgreement 9820 9821 := RowAgreement.single (first := 9820) modelrow_agrees_9821
private def atomic_9822 : CountSegment 1 10 13120 9821 9822 4075 4075 := CountSegment.single (first := 9821) (initial := 4075) row_count_9822 (by decide)
private theorem atomic_9822_agreement : RowAgreement 9821 9822 := RowAgreement.single (first := 9821) modelrow_agrees_9822
private def atomic_9823 : CountSegment 1 10 13120 9822 9823 4075 4075 := CountSegment.single (first := 9822) (initial := 4075) row_count_9823 (by decide)
private theorem atomic_9823_agreement : RowAgreement 9822 9823 := RowAgreement.single (first := 9822) modelrow_agrees_9823
private def atomic_9824 : CountSegment 1 10 13120 9823 9824 4075 4075 := CountSegment.single (first := 9823) (initial := 4075) row_count_9824 (by decide)
private theorem atomic_9824_agreement : RowAgreement 9823 9824 := RowAgreement.single (first := 9823) modelrow_agrees_9824
private def atomic_9825 : CountSegment 1 10 13120 9824 9825 4075 4077 := CountSegment.single (first := 9824) (initial := 4075) row_count_9825 (by decide)
private theorem atomic_9825_agreement : RowAgreement 9824 9825 := RowAgreement.single (first := 9824) modelrow_agrees_9825
private def atomic_9826 : CountSegment 1 10 13120 9825 9826 4077 4077 := CountSegment.single (first := 9825) (initial := 4077) row_count_9826 (by decide)
private theorem atomic_9826_agreement : RowAgreement 9825 9826 := RowAgreement.single (first := 9825) modelrow_agrees_9826
private def atomic_9827 : CountSegment 1 10 13120 9826 9827 4077 4077 := CountSegment.single (first := 9826) (initial := 4077) row_count_9827 (by decide)
private theorem atomic_9827_agreement : RowAgreement 9826 9827 := RowAgreement.single (first := 9826) modelrow_agrees_9827
private def atomic_9828 : CountSegment 1 10 13120 9827 9828 4077 4077 := CountSegment.single (first := 9827) (initial := 4077) row_count_9828 (by decide)
private theorem atomic_9828_agreement : RowAgreement 9827 9828 := RowAgreement.single (first := 9827) modelrow_agrees_9828
private def atomic_9829 : CountSegment 1 10 13120 9828 9829 4077 4077 := CountSegment.single (first := 9828) (initial := 4077) row_count_9829 (by decide)
private theorem atomic_9829_agreement : RowAgreement 9828 9829 := RowAgreement.single (first := 9828) modelrow_agrees_9829
private def atomic_9830 : CountSegment 1 10 13120 9829 9830 4077 4077 := CountSegment.single (first := 9829) (initial := 4077) row_count_9830 (by decide)
private theorem atomic_9830_agreement : RowAgreement 9829 9830 := RowAgreement.single (first := 9829) modelrow_agrees_9830
private def atomic_9831 : CountSegment 1 10 13120 9830 9831 4077 4078 := CountSegment.single (first := 9830) (initial := 4077) row_count_9831 (by decide)
private theorem atomic_9831_agreement : RowAgreement 9830 9831 := RowAgreement.single (first := 9830) modelrow_agrees_9831
private def atomic_9832 : CountSegment 1 10 13120 9831 9832 4078 4078 := CountSegment.single (first := 9831) (initial := 4078) row_count_9832 (by decide)
private theorem atomic_9832_agreement : RowAgreement 9831 9832 := RowAgreement.single (first := 9831) modelrow_agrees_9832
private def atomic_9833 : CountSegment 1 10 13120 9832 9833 4078 4079 := CountSegment.single (first := 9832) (initial := 4078) row_count_9833 (by decide)
private theorem atomic_9833_agreement : RowAgreement 9832 9833 := RowAgreement.single (first := 9832) modelrow_agrees_9833
private def atomic_9834 : CountSegment 1 10 13120 9833 9834 4079 4079 := CountSegment.single (first := 9833) (initial := 4079) row_count_9834 (by decide)
private theorem atomic_9834_agreement : RowAgreement 9833 9834 := RowAgreement.single (first := 9833) modelrow_agrees_9834
private def atomic_9835 : CountSegment 1 10 13120 9834 9835 4079 4080 := CountSegment.single (first := 9834) (initial := 4079) row_count_9835 (by decide)
private theorem atomic_9835_agreement : RowAgreement 9834 9835 := RowAgreement.single (first := 9834) modelrow_agrees_9835
private def atomic_9836 : CountSegment 1 10 13120 9835 9836 4080 4081 := CountSegment.single (first := 9835) (initial := 4080) row_count_9836 (by decide)
private theorem atomic_9836_agreement : RowAgreement 9835 9836 := RowAgreement.single (first := 9835) modelrow_agrees_9836
private def atomic_9837 : CountSegment 1 10 13120 9836 9837 4081 4081 := CountSegment.single (first := 9836) (initial := 4081) row_count_9837 (by decide)
private theorem atomic_9837_agreement : RowAgreement 9836 9837 := RowAgreement.single (first := 9836) modelrow_agrees_9837
private def atomic_9838 : CountSegment 1 10 13120 9837 9838 4081 4081 := CountSegment.single (first := 9837) (initial := 4081) row_count_9838 (by decide)
private theorem atomic_9838_agreement : RowAgreement 9837 9838 := RowAgreement.single (first := 9837) modelrow_agrees_9838
private def atomic_9839 : CountSegment 1 10 13120 9838 9839 4081 4082 := CountSegment.single (first := 9838) (initial := 4081) row_count_9839 (by decide)
private theorem atomic_9839_agreement : RowAgreement 9838 9839 := RowAgreement.single (first := 9838) modelrow_agrees_9839
private def atomic_9840 : CountSegment 1 10 13120 9839 9840 4082 4082 := CountSegment.single (first := 9839) (initial := 4082) row_count_9840 (by decide)
private theorem atomic_9840_agreement : RowAgreement 9839 9840 := RowAgreement.single (first := 9839) modelrow_agrees_9840
private def atomic_9841 : CountSegment 1 10 13120 9840 9841 4082 4082 := CountSegment.single (first := 9840) (initial := 4082) row_count_9841 (by decide)
private theorem atomic_9841_agreement : RowAgreement 9840 9841 := RowAgreement.single (first := 9840) modelrow_agrees_9841
private def atomic_9842 : CountSegment 1 10 13120 9841 9842 4082 4082 := CountSegment.single (first := 9841) (initial := 4082) row_count_9842 (by decide)
private theorem atomic_9842_agreement : RowAgreement 9841 9842 := RowAgreement.single (first := 9841) modelrow_agrees_9842
private def atomic_9843 : CountSegment 1 10 13120 9842 9843 4082 4082 := CountSegment.single (first := 9842) (initial := 4082) row_count_9843 (by decide)
private theorem atomic_9843_agreement : RowAgreement 9842 9843 := RowAgreement.single (first := 9842) modelrow_agrees_9843
private def atomic_9844 : CountSegment 1 10 13120 9843 9844 4082 4082 := CountSegment.single (first := 9843) (initial := 4082) row_count_9844 (by decide)
private theorem atomic_9844_agreement : RowAgreement 9843 9844 := RowAgreement.single (first := 9843) modelrow_agrees_9844
private def atomic_9845 : CountSegment 1 10 13120 9844 9845 4082 4082 := CountSegment.single (first := 9844) (initial := 4082) row_count_9845 (by decide)
private theorem atomic_9845_agreement : RowAgreement 9844 9845 := RowAgreement.single (first := 9844) modelrow_agrees_9845
private def atomic_9846 : CountSegment 1 10 13120 9845 9846 4082 4083 := CountSegment.single (first := 9845) (initial := 4082) row_count_9846 (by decide)
private theorem atomic_9846_agreement : RowAgreement 9845 9846 := RowAgreement.single (first := 9845) modelrow_agrees_9846
private def atomic_9847 : CountSegment 1 10 13120 9846 9847 4083 4083 := CountSegment.single (first := 9846) (initial := 4083) row_count_9847 (by decide)
private theorem atomic_9847_agreement : RowAgreement 9846 9847 := RowAgreement.single (first := 9846) modelrow_agrees_9847
private def atomic_9848 : CountSegment 1 10 13120 9847 9848 4083 4083 := CountSegment.single (first := 9847) (initial := 4083) row_count_9848 (by decide)
private theorem atomic_9848_agreement : RowAgreement 9847 9848 := RowAgreement.single (first := 9847) modelrow_agrees_9848
private def atomic_9849 : CountSegment 1 10 13120 9848 9849 4083 4083 := CountSegment.single (first := 9848) (initial := 4083) row_count_9849 (by decide)
private theorem atomic_9849_agreement : RowAgreement 9848 9849 := RowAgreement.single (first := 9848) modelrow_agrees_9849
private def atomic_9850 : CountSegment 1 10 13120 9849 9850 4083 4083 := CountSegment.single (first := 9849) (initial := 4083) row_count_9850 (by decide)
private theorem atomic_9850_agreement : RowAgreement 9849 9850 := RowAgreement.single (first := 9849) modelrow_agrees_9850
private def atomic_9851 : CountSegment 1 10 13120 9850 9851 4083 4083 := CountSegment.single (first := 9850) (initial := 4083) row_count_9851 (by decide)
private theorem atomic_9851_agreement : RowAgreement 9850 9851 := RowAgreement.single (first := 9850) modelrow_agrees_9851
private def atomic_9852 : CountSegment 1 10 13120 9851 9852 4083 4084 := CountSegment.single (first := 9851) (initial := 4083) row_count_9852 (by decide)
private theorem atomic_9852_agreement : RowAgreement 9851 9852 := RowAgreement.single (first := 9851) modelrow_agrees_9852
private def atomic_9853 : CountSegment 1 10 13120 9852 9853 4084 4085 := CountSegment.single (first := 9852) (initial := 4084) row_count_9853 (by decide)
private theorem atomic_9853_agreement : RowAgreement 9852 9853 := RowAgreement.single (first := 9852) modelrow_agrees_9853
private def atomic_9854 : CountSegment 1 10 13120 9853 9854 4085 4085 := CountSegment.single (first := 9853) (initial := 4085) row_count_9854 (by decide)
private theorem atomic_9854_agreement : RowAgreement 9853 9854 := RowAgreement.single (first := 9853) modelrow_agrees_9854
private def atomic_9855 : CountSegment 1 10 13120 9854 9855 4085 4085 := CountSegment.single (first := 9854) (initial := 4085) row_count_9855 (by decide)
private theorem atomic_9855_agreement : RowAgreement 9854 9855 := RowAgreement.single (first := 9854) modelrow_agrees_9855
private def atomic_9856 : CountSegment 1 10 13120 9855 9856 4085 4085 := CountSegment.single (first := 9855) (initial := 4085) row_count_9856 (by decide)
private theorem atomic_9856_agreement : RowAgreement 9855 9856 := RowAgreement.single (first := 9855) modelrow_agrees_9856
private def within_076_000 : CountSegment 1 10 13120 9728 9730 4040 4042 := CountSegment.join atomic_9729 atomic_9730
private theorem within_076_000_agreement : RowAgreement 9728 9730 := RowAgreement.join atomic_9729_agreement atomic_9730_agreement
private def within_076_001 : CountSegment 1 10 13120 9730 9732 4042 4042 := CountSegment.join atomic_9731 atomic_9732
private theorem within_076_001_agreement : RowAgreement 9730 9732 := RowAgreement.join atomic_9731_agreement atomic_9732_agreement
private def within_076_002 : CountSegment 1 10 13120 9728 9732 4040 4042 := CountSegment.join within_076_000 within_076_001
private theorem within_076_002_agreement : RowAgreement 9728 9732 := RowAgreement.join within_076_000_agreement within_076_001_agreement
private def within_076_003 : CountSegment 1 10 13120 9732 9734 4042 4043 := CountSegment.join atomic_9733 atomic_9734
private theorem within_076_003_agreement : RowAgreement 9732 9734 := RowAgreement.join atomic_9733_agreement atomic_9734_agreement
private def within_076_004 : CountSegment 1 10 13120 9734 9736 4043 4043 := CountSegment.join atomic_9735 atomic_9736
private theorem within_076_004_agreement : RowAgreement 9734 9736 := RowAgreement.join atomic_9735_agreement atomic_9736_agreement
private def within_076_005 : CountSegment 1 10 13120 9732 9736 4042 4043 := CountSegment.join within_076_003 within_076_004
private theorem within_076_005_agreement : RowAgreement 9732 9736 := RowAgreement.join within_076_003_agreement within_076_004_agreement
private def within_076_006 : CountSegment 1 10 13120 9728 9736 4040 4043 := CountSegment.join within_076_002 within_076_005
private theorem within_076_006_agreement : RowAgreement 9728 9736 := RowAgreement.join within_076_002_agreement within_076_005_agreement
private def within_076_007 : CountSegment 1 10 13120 9736 9738 4043 4044 := CountSegment.join atomic_9737 atomic_9738
private theorem within_076_007_agreement : RowAgreement 9736 9738 := RowAgreement.join atomic_9737_agreement atomic_9738_agreement
private def within_076_008 : CountSegment 1 10 13120 9738 9740 4044 4046 := CountSegment.join atomic_9739 atomic_9740
private theorem within_076_008_agreement : RowAgreement 9738 9740 := RowAgreement.join atomic_9739_agreement atomic_9740_agreement
private def within_076_009 : CountSegment 1 10 13120 9736 9740 4043 4046 := CountSegment.join within_076_007 within_076_008
private theorem within_076_009_agreement : RowAgreement 9736 9740 := RowAgreement.join within_076_007_agreement within_076_008_agreement
private def within_076_010 : CountSegment 1 10 13120 9740 9742 4046 4047 := CountSegment.join atomic_9741 atomic_9742
private theorem within_076_010_agreement : RowAgreement 9740 9742 := RowAgreement.join atomic_9741_agreement atomic_9742_agreement
private def within_076_011 : CountSegment 1 10 13120 9742 9744 4047 4047 := CountSegment.join atomic_9743 atomic_9744
private theorem within_076_011_agreement : RowAgreement 9742 9744 := RowAgreement.join atomic_9743_agreement atomic_9744_agreement
private def within_076_012 : CountSegment 1 10 13120 9740 9744 4046 4047 := CountSegment.join within_076_010 within_076_011
private theorem within_076_012_agreement : RowAgreement 9740 9744 := RowAgreement.join within_076_010_agreement within_076_011_agreement
private def within_076_013 : CountSegment 1 10 13120 9736 9744 4043 4047 := CountSegment.join within_076_009 within_076_012
private theorem within_076_013_agreement : RowAgreement 9736 9744 := RowAgreement.join within_076_009_agreement within_076_012_agreement
private def within_076_014 : CountSegment 1 10 13120 9728 9744 4040 4047 := CountSegment.join within_076_006 within_076_013
private theorem within_076_014_agreement : RowAgreement 9728 9744 := RowAgreement.join within_076_006_agreement within_076_013_agreement
private def within_076_015 : CountSegment 1 10 13120 9744 9746 4047 4048 := CountSegment.join atomic_9745 atomic_9746
private theorem within_076_015_agreement : RowAgreement 9744 9746 := RowAgreement.join atomic_9745_agreement atomic_9746_agreement
private def within_076_016 : CountSegment 1 10 13120 9746 9748 4048 4050 := CountSegment.join atomic_9747 atomic_9748
private theorem within_076_016_agreement : RowAgreement 9746 9748 := RowAgreement.join atomic_9747_agreement atomic_9748_agreement
private def within_076_017 : CountSegment 1 10 13120 9744 9748 4047 4050 := CountSegment.join within_076_015 within_076_016
private theorem within_076_017_agreement : RowAgreement 9744 9748 := RowAgreement.join within_076_015_agreement within_076_016_agreement
private def within_076_018 : CountSegment 1 10 13120 9748 9750 4050 4051 := CountSegment.join atomic_9749 atomic_9750
private theorem within_076_018_agreement : RowAgreement 9748 9750 := RowAgreement.join atomic_9749_agreement atomic_9750_agreement
private def within_076_019 : CountSegment 1 10 13120 9750 9752 4051 4051 := CountSegment.join atomic_9751 atomic_9752
private theorem within_076_019_agreement : RowAgreement 9750 9752 := RowAgreement.join atomic_9751_agreement atomic_9752_agreement
private def within_076_020 : CountSegment 1 10 13120 9748 9752 4050 4051 := CountSegment.join within_076_018 within_076_019
private theorem within_076_020_agreement : RowAgreement 9748 9752 := RowAgreement.join within_076_018_agreement within_076_019_agreement
private def within_076_021 : CountSegment 1 10 13120 9744 9752 4047 4051 := CountSegment.join within_076_017 within_076_020
private theorem within_076_021_agreement : RowAgreement 9744 9752 := RowAgreement.join within_076_017_agreement within_076_020_agreement
private def within_076_022 : CountSegment 1 10 13120 9752 9754 4051 4053 := CountSegment.join atomic_9753 atomic_9754
private theorem within_076_022_agreement : RowAgreement 9752 9754 := RowAgreement.join atomic_9753_agreement atomic_9754_agreement
private def within_076_023 : CountSegment 1 10 13120 9754 9756 4053 4055 := CountSegment.join atomic_9755 atomic_9756
private theorem within_076_023_agreement : RowAgreement 9754 9756 := RowAgreement.join atomic_9755_agreement atomic_9756_agreement
private def within_076_024 : CountSegment 1 10 13120 9752 9756 4051 4055 := CountSegment.join within_076_022 within_076_023
private theorem within_076_024_agreement : RowAgreement 9752 9756 := RowAgreement.join within_076_022_agreement within_076_023_agreement
private def within_076_025 : CountSegment 1 10 13120 9756 9758 4055 4055 := CountSegment.join atomic_9757 atomic_9758
private theorem within_076_025_agreement : RowAgreement 9756 9758 := RowAgreement.join atomic_9757_agreement atomic_9758_agreement
private def within_076_026 : CountSegment 1 10 13120 9758 9760 4055 4055 := CountSegment.join atomic_9759 atomic_9760
private theorem within_076_026_agreement : RowAgreement 9758 9760 := RowAgreement.join atomic_9759_agreement atomic_9760_agreement
private def within_076_027 : CountSegment 1 10 13120 9756 9760 4055 4055 := CountSegment.join within_076_025 within_076_026
private theorem within_076_027_agreement : RowAgreement 9756 9760 := RowAgreement.join within_076_025_agreement within_076_026_agreement
private def within_076_028 : CountSegment 1 10 13120 9752 9760 4051 4055 := CountSegment.join within_076_024 within_076_027
private theorem within_076_028_agreement : RowAgreement 9752 9760 := RowAgreement.join within_076_024_agreement within_076_027_agreement
private def within_076_029 : CountSegment 1 10 13120 9744 9760 4047 4055 := CountSegment.join within_076_021 within_076_028
private theorem within_076_029_agreement : RowAgreement 9744 9760 := RowAgreement.join within_076_021_agreement within_076_028_agreement
private def within_076_030 : CountSegment 1 10 13120 9728 9760 4040 4055 := CountSegment.join within_076_014 within_076_029
private theorem within_076_030_agreement : RowAgreement 9728 9760 := RowAgreement.join within_076_014_agreement within_076_029_agreement
private def within_076_031 : CountSegment 1 10 13120 9760 9762 4055 4057 := CountSegment.join atomic_9761 atomic_9762
private theorem within_076_031_agreement : RowAgreement 9760 9762 := RowAgreement.join atomic_9761_agreement atomic_9762_agreement
private def within_076_032 : CountSegment 1 10 13120 9762 9764 4057 4057 := CountSegment.join atomic_9763 atomic_9764
private theorem within_076_032_agreement : RowAgreement 9762 9764 := RowAgreement.join atomic_9763_agreement atomic_9764_agreement
private def within_076_033 : CountSegment 1 10 13120 9760 9764 4055 4057 := CountSegment.join within_076_031 within_076_032
private theorem within_076_033_agreement : RowAgreement 9760 9764 := RowAgreement.join within_076_031_agreement within_076_032_agreement
private def within_076_034 : CountSegment 1 10 13120 9764 9766 4057 4058 := CountSegment.join atomic_9765 atomic_9766
private theorem within_076_034_agreement : RowAgreement 9764 9766 := RowAgreement.join atomic_9765_agreement atomic_9766_agreement
private def within_076_035 : CountSegment 1 10 13120 9766 9768 4058 4059 := CountSegment.join atomic_9767 atomic_9768
private theorem within_076_035_agreement : RowAgreement 9766 9768 := RowAgreement.join atomic_9767_agreement atomic_9768_agreement
private def within_076_036 : CountSegment 1 10 13120 9764 9768 4057 4059 := CountSegment.join within_076_034 within_076_035
private theorem within_076_036_agreement : RowAgreement 9764 9768 := RowAgreement.join within_076_034_agreement within_076_035_agreement
private def within_076_037 : CountSegment 1 10 13120 9760 9768 4055 4059 := CountSegment.join within_076_033 within_076_036
private theorem within_076_037_agreement : RowAgreement 9760 9768 := RowAgreement.join within_076_033_agreement within_076_036_agreement
private def within_076_038 : CountSegment 1 10 13120 9768 9770 4059 4059 := CountSegment.join atomic_9769 atomic_9770
private theorem within_076_038_agreement : RowAgreement 9768 9770 := RowAgreement.join atomic_9769_agreement atomic_9770_agreement
private def within_076_039 : CountSegment 1 10 13120 9770 9772 4059 4060 := CountSegment.join atomic_9771 atomic_9772
private theorem within_076_039_agreement : RowAgreement 9770 9772 := RowAgreement.join atomic_9771_agreement atomic_9772_agreement
private def within_076_040 : CountSegment 1 10 13120 9768 9772 4059 4060 := CountSegment.join within_076_038 within_076_039
private theorem within_076_040_agreement : RowAgreement 9768 9772 := RowAgreement.join within_076_038_agreement within_076_039_agreement
private def within_076_041 : CountSegment 1 10 13120 9772 9774 4060 4060 := CountSegment.join atomic_9773 atomic_9774
private theorem within_076_041_agreement : RowAgreement 9772 9774 := RowAgreement.join atomic_9773_agreement atomic_9774_agreement
private def within_076_042 : CountSegment 1 10 13120 9774 9776 4060 4061 := CountSegment.join atomic_9775 atomic_9776
private theorem within_076_042_agreement : RowAgreement 9774 9776 := RowAgreement.join atomic_9775_agreement atomic_9776_agreement
private def within_076_043 : CountSegment 1 10 13120 9772 9776 4060 4061 := CountSegment.join within_076_041 within_076_042
private theorem within_076_043_agreement : RowAgreement 9772 9776 := RowAgreement.join within_076_041_agreement within_076_042_agreement
private def within_076_044 : CountSegment 1 10 13120 9768 9776 4059 4061 := CountSegment.join within_076_040 within_076_043
private theorem within_076_044_agreement : RowAgreement 9768 9776 := RowAgreement.join within_076_040_agreement within_076_043_agreement
private def within_076_045 : CountSegment 1 10 13120 9760 9776 4055 4061 := CountSegment.join within_076_037 within_076_044
private theorem within_076_045_agreement : RowAgreement 9760 9776 := RowAgreement.join within_076_037_agreement within_076_044_agreement
private def within_076_046 : CountSegment 1 10 13120 9776 9778 4061 4063 := CountSegment.join atomic_9777 atomic_9778
private theorem within_076_046_agreement : RowAgreement 9776 9778 := RowAgreement.join atomic_9777_agreement atomic_9778_agreement
private def within_076_047 : CountSegment 1 10 13120 9778 9780 4063 4063 := CountSegment.join atomic_9779 atomic_9780
private theorem within_076_047_agreement : RowAgreement 9778 9780 := RowAgreement.join atomic_9779_agreement atomic_9780_agreement
private def within_076_048 : CountSegment 1 10 13120 9776 9780 4061 4063 := CountSegment.join within_076_046 within_076_047
private theorem within_076_048_agreement : RowAgreement 9776 9780 := RowAgreement.join within_076_046_agreement within_076_047_agreement
private def within_076_049 : CountSegment 1 10 13120 9780 9782 4063 4063 := CountSegment.join atomic_9781 atomic_9782
private theorem within_076_049_agreement : RowAgreement 9780 9782 := RowAgreement.join atomic_9781_agreement atomic_9782_agreement
private def within_076_050 : CountSegment 1 10 13120 9782 9784 4063 4063 := CountSegment.join atomic_9783 atomic_9784
private theorem within_076_050_agreement : RowAgreement 9782 9784 := RowAgreement.join atomic_9783_agreement atomic_9784_agreement
private def within_076_051 : CountSegment 1 10 13120 9780 9784 4063 4063 := CountSegment.join within_076_049 within_076_050
private theorem within_076_051_agreement : RowAgreement 9780 9784 := RowAgreement.join within_076_049_agreement within_076_050_agreement
private def within_076_052 : CountSegment 1 10 13120 9776 9784 4061 4063 := CountSegment.join within_076_048 within_076_051
private theorem within_076_052_agreement : RowAgreement 9776 9784 := RowAgreement.join within_076_048_agreement within_076_051_agreement
private def within_076_053 : CountSegment 1 10 13120 9784 9786 4063 4063 := CountSegment.join atomic_9785 atomic_9786
private theorem within_076_053_agreement : RowAgreement 9784 9786 := RowAgreement.join atomic_9785_agreement atomic_9786_agreement
private def within_076_054 : CountSegment 1 10 13120 9786 9788 4063 4065 := CountSegment.join atomic_9787 atomic_9788
private theorem within_076_054_agreement : RowAgreement 9786 9788 := RowAgreement.join atomic_9787_agreement atomic_9788_agreement
private def within_076_055 : CountSegment 1 10 13120 9784 9788 4063 4065 := CountSegment.join within_076_053 within_076_054
private theorem within_076_055_agreement : RowAgreement 9784 9788 := RowAgreement.join within_076_053_agreement within_076_054_agreement
private def within_076_056 : CountSegment 1 10 13120 9788 9790 4065 4065 := CountSegment.join atomic_9789 atomic_9790
private theorem within_076_056_agreement : RowAgreement 9788 9790 := RowAgreement.join atomic_9789_agreement atomic_9790_agreement
private def within_076_057 : CountSegment 1 10 13120 9790 9792 4065 4065 := CountSegment.join atomic_9791 atomic_9792
private theorem within_076_057_agreement : RowAgreement 9790 9792 := RowAgreement.join atomic_9791_agreement atomic_9792_agreement
private def within_076_058 : CountSegment 1 10 13120 9788 9792 4065 4065 := CountSegment.join within_076_056 within_076_057
private theorem within_076_058_agreement : RowAgreement 9788 9792 := RowAgreement.join within_076_056_agreement within_076_057_agreement
private def within_076_059 : CountSegment 1 10 13120 9784 9792 4063 4065 := CountSegment.join within_076_055 within_076_058
private theorem within_076_059_agreement : RowAgreement 9784 9792 := RowAgreement.join within_076_055_agreement within_076_058_agreement
private def within_076_060 : CountSegment 1 10 13120 9776 9792 4061 4065 := CountSegment.join within_076_052 within_076_059
private theorem within_076_060_agreement : RowAgreement 9776 9792 := RowAgreement.join within_076_052_agreement within_076_059_agreement
private def within_076_061 : CountSegment 1 10 13120 9760 9792 4055 4065 := CountSegment.join within_076_045 within_076_060
private theorem within_076_061_agreement : RowAgreement 9760 9792 := RowAgreement.join within_076_045_agreement within_076_060_agreement
private def within_076_062 : CountSegment 1 10 13120 9728 9792 4040 4065 := CountSegment.join within_076_030 within_076_061
private theorem within_076_062_agreement : RowAgreement 9728 9792 := RowAgreement.join within_076_030_agreement within_076_061_agreement
private def within_076_063 : CountSegment 1 10 13120 9792 9794 4065 4065 := CountSegment.join atomic_9793 atomic_9794
private theorem within_076_063_agreement : RowAgreement 9792 9794 := RowAgreement.join atomic_9793_agreement atomic_9794_agreement
private def within_076_064 : CountSegment 1 10 13120 9794 9796 4065 4067 := CountSegment.join atomic_9795 atomic_9796
private theorem within_076_064_agreement : RowAgreement 9794 9796 := RowAgreement.join atomic_9795_agreement atomic_9796_agreement
private def within_076_065 : CountSegment 1 10 13120 9792 9796 4065 4067 := CountSegment.join within_076_063 within_076_064
private theorem within_076_065_agreement : RowAgreement 9792 9796 := RowAgreement.join within_076_063_agreement within_076_064_agreement
private def within_076_066 : CountSegment 1 10 13120 9796 9798 4067 4067 := CountSegment.join atomic_9797 atomic_9798
private theorem within_076_066_agreement : RowAgreement 9796 9798 := RowAgreement.join atomic_9797_agreement atomic_9798_agreement
private def within_076_067 : CountSegment 1 10 13120 9798 9800 4067 4068 := CountSegment.join atomic_9799 atomic_9800
private theorem within_076_067_agreement : RowAgreement 9798 9800 := RowAgreement.join atomic_9799_agreement atomic_9800_agreement
private def within_076_068 : CountSegment 1 10 13120 9796 9800 4067 4068 := CountSegment.join within_076_066 within_076_067
private theorem within_076_068_agreement : RowAgreement 9796 9800 := RowAgreement.join within_076_066_agreement within_076_067_agreement
private def within_076_069 : CountSegment 1 10 13120 9792 9800 4065 4068 := CountSegment.join within_076_065 within_076_068
private theorem within_076_069_agreement : RowAgreement 9792 9800 := RowAgreement.join within_076_065_agreement within_076_068_agreement
private def within_076_070 : CountSegment 1 10 13120 9800 9802 4068 4068 := CountSegment.join atomic_9801 atomic_9802
private theorem within_076_070_agreement : RowAgreement 9800 9802 := RowAgreement.join atomic_9801_agreement atomic_9802_agreement
private def within_076_071 : CountSegment 1 10 13120 9802 9804 4068 4068 := CountSegment.join atomic_9803 atomic_9804
private theorem within_076_071_agreement : RowAgreement 9802 9804 := RowAgreement.join atomic_9803_agreement atomic_9804_agreement
private def within_076_072 : CountSegment 1 10 13120 9800 9804 4068 4068 := CountSegment.join within_076_070 within_076_071
private theorem within_076_072_agreement : RowAgreement 9800 9804 := RowAgreement.join within_076_070_agreement within_076_071_agreement
private def within_076_073 : CountSegment 1 10 13120 9804 9806 4068 4069 := CountSegment.join atomic_9805 atomic_9806
private theorem within_076_073_agreement : RowAgreement 9804 9806 := RowAgreement.join atomic_9805_agreement atomic_9806_agreement
private def within_076_074 : CountSegment 1 10 13120 9806 9808 4069 4069 := CountSegment.join atomic_9807 atomic_9808
private theorem within_076_074_agreement : RowAgreement 9806 9808 := RowAgreement.join atomic_9807_agreement atomic_9808_agreement
private def within_076_075 : CountSegment 1 10 13120 9804 9808 4068 4069 := CountSegment.join within_076_073 within_076_074
private theorem within_076_075_agreement : RowAgreement 9804 9808 := RowAgreement.join within_076_073_agreement within_076_074_agreement
private def within_076_076 : CountSegment 1 10 13120 9800 9808 4068 4069 := CountSegment.join within_076_072 within_076_075
private theorem within_076_076_agreement : RowAgreement 9800 9808 := RowAgreement.join within_076_072_agreement within_076_075_agreement
private def within_076_077 : CountSegment 1 10 13120 9792 9808 4065 4069 := CountSegment.join within_076_069 within_076_076
private theorem within_076_077_agreement : RowAgreement 9792 9808 := RowAgreement.join within_076_069_agreement within_076_076_agreement
private def within_076_078 : CountSegment 1 10 13120 9808 9810 4069 4070 := CountSegment.join atomic_9809 atomic_9810
private theorem within_076_078_agreement : RowAgreement 9808 9810 := RowAgreement.join atomic_9809_agreement atomic_9810_agreement
private def within_076_079 : CountSegment 1 10 13120 9810 9812 4070 4071 := CountSegment.join atomic_9811 atomic_9812
private theorem within_076_079_agreement : RowAgreement 9810 9812 := RowAgreement.join atomic_9811_agreement atomic_9812_agreement
private def within_076_080 : CountSegment 1 10 13120 9808 9812 4069 4071 := CountSegment.join within_076_078 within_076_079
private theorem within_076_080_agreement : RowAgreement 9808 9812 := RowAgreement.join within_076_078_agreement within_076_079_agreement
private def within_076_081 : CountSegment 1 10 13120 9812 9814 4071 4072 := CountSegment.join atomic_9813 atomic_9814
private theorem within_076_081_agreement : RowAgreement 9812 9814 := RowAgreement.join atomic_9813_agreement atomic_9814_agreement
private def within_076_082 : CountSegment 1 10 13120 9814 9816 4072 4073 := CountSegment.join atomic_9815 atomic_9816
private theorem within_076_082_agreement : RowAgreement 9814 9816 := RowAgreement.join atomic_9815_agreement atomic_9816_agreement
private def within_076_083 : CountSegment 1 10 13120 9812 9816 4071 4073 := CountSegment.join within_076_081 within_076_082
private theorem within_076_083_agreement : RowAgreement 9812 9816 := RowAgreement.join within_076_081_agreement within_076_082_agreement
private def within_076_084 : CountSegment 1 10 13120 9808 9816 4069 4073 := CountSegment.join within_076_080 within_076_083
private theorem within_076_084_agreement : RowAgreement 9808 9816 := RowAgreement.join within_076_080_agreement within_076_083_agreement
private def within_076_085 : CountSegment 1 10 13120 9816 9818 4073 4075 := CountSegment.join atomic_9817 atomic_9818
private theorem within_076_085_agreement : RowAgreement 9816 9818 := RowAgreement.join atomic_9817_agreement atomic_9818_agreement
private def within_076_086 : CountSegment 1 10 13120 9818 9820 4075 4075 := CountSegment.join atomic_9819 atomic_9820
private theorem within_076_086_agreement : RowAgreement 9818 9820 := RowAgreement.join atomic_9819_agreement atomic_9820_agreement
private def within_076_087 : CountSegment 1 10 13120 9816 9820 4073 4075 := CountSegment.join within_076_085 within_076_086
private theorem within_076_087_agreement : RowAgreement 9816 9820 := RowAgreement.join within_076_085_agreement within_076_086_agreement
private def within_076_088 : CountSegment 1 10 13120 9820 9822 4075 4075 := CountSegment.join atomic_9821 atomic_9822
private theorem within_076_088_agreement : RowAgreement 9820 9822 := RowAgreement.join atomic_9821_agreement atomic_9822_agreement
private def within_076_089 : CountSegment 1 10 13120 9822 9824 4075 4075 := CountSegment.join atomic_9823 atomic_9824
private theorem within_076_089_agreement : RowAgreement 9822 9824 := RowAgreement.join atomic_9823_agreement atomic_9824_agreement
private def within_076_090 : CountSegment 1 10 13120 9820 9824 4075 4075 := CountSegment.join within_076_088 within_076_089
private theorem within_076_090_agreement : RowAgreement 9820 9824 := RowAgreement.join within_076_088_agreement within_076_089_agreement
private def within_076_091 : CountSegment 1 10 13120 9816 9824 4073 4075 := CountSegment.join within_076_087 within_076_090
private theorem within_076_091_agreement : RowAgreement 9816 9824 := RowAgreement.join within_076_087_agreement within_076_090_agreement
private def within_076_092 : CountSegment 1 10 13120 9808 9824 4069 4075 := CountSegment.join within_076_084 within_076_091
private theorem within_076_092_agreement : RowAgreement 9808 9824 := RowAgreement.join within_076_084_agreement within_076_091_agreement
private def within_076_093 : CountSegment 1 10 13120 9792 9824 4065 4075 := CountSegment.join within_076_077 within_076_092
private theorem within_076_093_agreement : RowAgreement 9792 9824 := RowAgreement.join within_076_077_agreement within_076_092_agreement
private def within_076_094 : CountSegment 1 10 13120 9824 9826 4075 4077 := CountSegment.join atomic_9825 atomic_9826
private theorem within_076_094_agreement : RowAgreement 9824 9826 := RowAgreement.join atomic_9825_agreement atomic_9826_agreement
private def within_076_095 : CountSegment 1 10 13120 9826 9828 4077 4077 := CountSegment.join atomic_9827 atomic_9828
private theorem within_076_095_agreement : RowAgreement 9826 9828 := RowAgreement.join atomic_9827_agreement atomic_9828_agreement
private def within_076_096 : CountSegment 1 10 13120 9824 9828 4075 4077 := CountSegment.join within_076_094 within_076_095
private theorem within_076_096_agreement : RowAgreement 9824 9828 := RowAgreement.join within_076_094_agreement within_076_095_agreement
private def within_076_097 : CountSegment 1 10 13120 9828 9830 4077 4077 := CountSegment.join atomic_9829 atomic_9830
private theorem within_076_097_agreement : RowAgreement 9828 9830 := RowAgreement.join atomic_9829_agreement atomic_9830_agreement
private def within_076_098 : CountSegment 1 10 13120 9830 9832 4077 4078 := CountSegment.join atomic_9831 atomic_9832
private theorem within_076_098_agreement : RowAgreement 9830 9832 := RowAgreement.join atomic_9831_agreement atomic_9832_agreement
private def within_076_099 : CountSegment 1 10 13120 9828 9832 4077 4078 := CountSegment.join within_076_097 within_076_098
private theorem within_076_099_agreement : RowAgreement 9828 9832 := RowAgreement.join within_076_097_agreement within_076_098_agreement
private def within_076_100 : CountSegment 1 10 13120 9824 9832 4075 4078 := CountSegment.join within_076_096 within_076_099
private theorem within_076_100_agreement : RowAgreement 9824 9832 := RowAgreement.join within_076_096_agreement within_076_099_agreement
private def within_076_101 : CountSegment 1 10 13120 9832 9834 4078 4079 := CountSegment.join atomic_9833 atomic_9834
private theorem within_076_101_agreement : RowAgreement 9832 9834 := RowAgreement.join atomic_9833_agreement atomic_9834_agreement
private def within_076_102 : CountSegment 1 10 13120 9834 9836 4079 4081 := CountSegment.join atomic_9835 atomic_9836
private theorem within_076_102_agreement : RowAgreement 9834 9836 := RowAgreement.join atomic_9835_agreement atomic_9836_agreement
private def within_076_103 : CountSegment 1 10 13120 9832 9836 4078 4081 := CountSegment.join within_076_101 within_076_102
private theorem within_076_103_agreement : RowAgreement 9832 9836 := RowAgreement.join within_076_101_agreement within_076_102_agreement
private def within_076_104 : CountSegment 1 10 13120 9836 9838 4081 4081 := CountSegment.join atomic_9837 atomic_9838
private theorem within_076_104_agreement : RowAgreement 9836 9838 := RowAgreement.join atomic_9837_agreement atomic_9838_agreement
private def within_076_105 : CountSegment 1 10 13120 9838 9840 4081 4082 := CountSegment.join atomic_9839 atomic_9840
private theorem within_076_105_agreement : RowAgreement 9838 9840 := RowAgreement.join atomic_9839_agreement atomic_9840_agreement
private def within_076_106 : CountSegment 1 10 13120 9836 9840 4081 4082 := CountSegment.join within_076_104 within_076_105
private theorem within_076_106_agreement : RowAgreement 9836 9840 := RowAgreement.join within_076_104_agreement within_076_105_agreement
private def within_076_107 : CountSegment 1 10 13120 9832 9840 4078 4082 := CountSegment.join within_076_103 within_076_106
private theorem within_076_107_agreement : RowAgreement 9832 9840 := RowAgreement.join within_076_103_agreement within_076_106_agreement
private def within_076_108 : CountSegment 1 10 13120 9824 9840 4075 4082 := CountSegment.join within_076_100 within_076_107
private theorem within_076_108_agreement : RowAgreement 9824 9840 := RowAgreement.join within_076_100_agreement within_076_107_agreement
private def within_076_109 : CountSegment 1 10 13120 9840 9842 4082 4082 := CountSegment.join atomic_9841 atomic_9842
private theorem within_076_109_agreement : RowAgreement 9840 9842 := RowAgreement.join atomic_9841_agreement atomic_9842_agreement
private def within_076_110 : CountSegment 1 10 13120 9842 9844 4082 4082 := CountSegment.join atomic_9843 atomic_9844
private theorem within_076_110_agreement : RowAgreement 9842 9844 := RowAgreement.join atomic_9843_agreement atomic_9844_agreement
private def within_076_111 : CountSegment 1 10 13120 9840 9844 4082 4082 := CountSegment.join within_076_109 within_076_110
private theorem within_076_111_agreement : RowAgreement 9840 9844 := RowAgreement.join within_076_109_agreement within_076_110_agreement
private def within_076_112 : CountSegment 1 10 13120 9844 9846 4082 4083 := CountSegment.join atomic_9845 atomic_9846
private theorem within_076_112_agreement : RowAgreement 9844 9846 := RowAgreement.join atomic_9845_agreement atomic_9846_agreement
private def within_076_113 : CountSegment 1 10 13120 9846 9848 4083 4083 := CountSegment.join atomic_9847 atomic_9848
private theorem within_076_113_agreement : RowAgreement 9846 9848 := RowAgreement.join atomic_9847_agreement atomic_9848_agreement
private def within_076_114 : CountSegment 1 10 13120 9844 9848 4082 4083 := CountSegment.join within_076_112 within_076_113
private theorem within_076_114_agreement : RowAgreement 9844 9848 := RowAgreement.join within_076_112_agreement within_076_113_agreement
private def within_076_115 : CountSegment 1 10 13120 9840 9848 4082 4083 := CountSegment.join within_076_111 within_076_114
private theorem within_076_115_agreement : RowAgreement 9840 9848 := RowAgreement.join within_076_111_agreement within_076_114_agreement
private def within_076_116 : CountSegment 1 10 13120 9848 9850 4083 4083 := CountSegment.join atomic_9849 atomic_9850
private theorem within_076_116_agreement : RowAgreement 9848 9850 := RowAgreement.join atomic_9849_agreement atomic_9850_agreement
private def within_076_117 : CountSegment 1 10 13120 9850 9852 4083 4084 := CountSegment.join atomic_9851 atomic_9852
private theorem within_076_117_agreement : RowAgreement 9850 9852 := RowAgreement.join atomic_9851_agreement atomic_9852_agreement
private def within_076_118 : CountSegment 1 10 13120 9848 9852 4083 4084 := CountSegment.join within_076_116 within_076_117
private theorem within_076_118_agreement : RowAgreement 9848 9852 := RowAgreement.join within_076_116_agreement within_076_117_agreement
private def within_076_119 : CountSegment 1 10 13120 9852 9854 4084 4085 := CountSegment.join atomic_9853 atomic_9854
private theorem within_076_119_agreement : RowAgreement 9852 9854 := RowAgreement.join atomic_9853_agreement atomic_9854_agreement
private def within_076_120 : CountSegment 1 10 13120 9854 9856 4085 4085 := CountSegment.join atomic_9855 atomic_9856
private theorem within_076_120_agreement : RowAgreement 9854 9856 := RowAgreement.join atomic_9855_agreement atomic_9856_agreement
private def within_076_121 : CountSegment 1 10 13120 9852 9856 4084 4085 := CountSegment.join within_076_119 within_076_120
private theorem within_076_121_agreement : RowAgreement 9852 9856 := RowAgreement.join within_076_119_agreement within_076_120_agreement
private def within_076_122 : CountSegment 1 10 13120 9848 9856 4083 4085 := CountSegment.join within_076_118 within_076_121
private theorem within_076_122_agreement : RowAgreement 9848 9856 := RowAgreement.join within_076_118_agreement within_076_121_agreement
private def within_076_123 : CountSegment 1 10 13120 9840 9856 4082 4085 := CountSegment.join within_076_115 within_076_122
private theorem within_076_123_agreement : RowAgreement 9840 9856 := RowAgreement.join within_076_115_agreement within_076_122_agreement
private def within_076_124 : CountSegment 1 10 13120 9824 9856 4075 4085 := CountSegment.join within_076_108 within_076_123
private theorem within_076_124_agreement : RowAgreement 9824 9856 := RowAgreement.join within_076_108_agreement within_076_123_agreement
private def within_076_125 : CountSegment 1 10 13120 9792 9856 4065 4085 := CountSegment.join within_076_093 within_076_124
private theorem within_076_125_agreement : RowAgreement 9792 9856 := RowAgreement.join within_076_093_agreement within_076_124_agreement
private def within_076_126 : CountSegment 1 10 13120 9728 9856 4040 4085 := CountSegment.join within_076_062 within_076_125
private theorem within_076_126_agreement : RowAgreement 9728 9856 := RowAgreement.join within_076_062_agreement within_076_125_agreement
def segment_076 : CountSegment 1 10 13120 9728 9856 4040 4085 := within_076_126
theorem agreement_076 : RowAgreement 9728 9856 := within_076_126_agreement
end ProjectEulerSolutions.P58Numerical
