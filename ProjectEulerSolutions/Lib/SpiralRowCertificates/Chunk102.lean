import ProjectEulerSolutions.Lib.SpiralPrimeSegments
import ProjectEulerSolutions.Lib.SpiralPrimeCertificates.All
import ProjectEulerSolutions.Lib.MillerRabin
import ProjectEulerSolutions.Lib.SpiralMillerAgreement
/-! Finite spiral row and counter certificates. Python execution is checked separately. -/
namespace ProjectEulerSolutions.P58Numerical
open SpiralPrimes
set_option Elab.async false
set_option maxRecDepth 4096
def row_13057 : RowCertificate 13057 where
  first := false
  first_correct := by
    have composite : ¬ Nat.Prime (corner 13057 1) := by change ¬ Nat.Prime 681967111; exact PrattCertificates.composite_of_product (factor := 7) (cofactor := 97423873) rfl (by decide) (by decide)
    exact decide_eq_false composite
  second := true
  second_correct := by
    have prime : Nat.Prime (corner 13057 2) := by change Nat.Prime 681940997; exact ProjectEulerSolutions.P58PrimeNodes.p_681940997.prime
    exact decide_eq_true prime
  third := false
  third_correct := by
    have composite : ¬ Nat.Prime (corner 13057 3) := by change ¬ Nat.Prime 681914883; exact PrattCertificates.composite_of_product (factor := 3) (cofactor := 227304961) rfl (by decide) (by decide)
    exact decide_eq_false composite
theorem row_count_13057 : row 13057 = 1 := (row_13057.checked).trans rfl
theorem model_count_13057 : modelRow 13057 = 1 := by decide
theorem modelrow_agrees_13057 : modelRow 13057 = row 13057 := model_count_13057.trans row_count_13057.symm
def row_13058 : RowCertificate 13058 where
  first := false
  first_correct := by
    have composite : ¬ Nat.Prime (corner 13058 1) := by change ¬ Nat.Prime 682071573; exact PrattCertificates.composite_of_product (factor := 3) (cofactor := 227357191) rfl (by decide) (by decide)
    exact decide_eq_false composite
  second := false
  second_correct := by
    have composite : ¬ Nat.Prime (corner 13058 2) := by change ¬ Nat.Prime 682045457; exact PrattCertificates.composite_of_product (factor := 17) (cofactor := 40120321) rfl (by decide) (by decide)
    exact decide_eq_false composite
  third := false
  third_correct := by
    have composite : ¬ Nat.Prime (corner 13058 3) := by change ¬ Nat.Prime 682019341; exact PrattCertificates.composite_of_product (factor := 103) (cofactor := 6621547) rfl (by decide) (by decide)
    exact decide_eq_false composite
theorem row_count_13058 : row 13058 = 0 := (row_13058.checked).trans rfl
theorem model_count_13058 : modelRow 13058 = 0 := by decide
theorem modelrow_agrees_13058 : modelRow 13058 = row 13058 := model_count_13058.trans row_count_13058.symm
def row_13059 : RowCertificate 13059 where
  first := true
  first_correct := by
    have prime : Nat.Prime (corner 13059 1) := by change Nat.Prime 682176043; exact ProjectEulerSolutions.P58PrimeNodes.p_682176043.prime
    exact decide_eq_true prime
  second := false
  second_correct := by
    have composite : ¬ Nat.Prime (corner 13059 2) := by change ¬ Nat.Prime 682149925; exact PrattCertificates.composite_of_product (factor := 5) (cofactor := 136429985) rfl (by decide) (by decide)
    exact decide_eq_false composite
  third := false
  third_correct := by
    have composite : ¬ Nat.Prime (corner 13059 3) := by change ¬ Nat.Prime 682123807; exact PrattCertificates.composite_of_product (factor := 19) (cofactor := 35901253) rfl (by decide) (by decide)
    exact decide_eq_false composite
theorem row_count_13059 : row 13059 = 1 := (row_13059.checked).trans rfl
theorem model_count_13059 : modelRow 13059 = 1 := by decide
theorem modelrow_agrees_13059 : modelRow 13059 = row 13059 := model_count_13059.trans row_count_13059.symm
def row_13060 : RowCertificate 13060 where
  first := false
  first_correct := by
    have composite : ¬ Nat.Prime (corner 13060 1) := by change ¬ Nat.Prime 682280521; exact PrattCertificates.composite_of_product (factor := 13) (cofactor := 52483117) rfl (by decide) (by decide)
    exact decide_eq_false composite
  second := true
  second_correct := by
    have prime : Nat.Prime (corner 13060 2) := by change Nat.Prime 682254401; exact ProjectEulerSolutions.P58PrimeNodes.p_682254401.prime
    exact decide_eq_true prime
  third := false
  third_correct := by
    have composite : ¬ Nat.Prime (corner 13060 3) := by change ¬ Nat.Prime 682228281; exact PrattCertificates.composite_of_product (factor := 3) (cofactor := 227409427) rfl (by decide) (by decide)
    exact decide_eq_false composite
theorem row_count_13060 : row 13060 = 1 := (row_13060.checked).trans rfl
theorem model_count_13060 : modelRow 13060 = 1 := by decide
theorem modelrow_agrees_13060 : modelRow 13060 = row 13060 := model_count_13060.trans row_count_13060.symm
def row_13061 : RowCertificate 13061 where
  first := false
  first_correct := by
    have composite : ¬ Nat.Prime (corner 13061 1) := by change ¬ Nat.Prime 682385007; exact PrattCertificates.composite_of_product (factor := 3) (cofactor := 227461669) rfl (by decide) (by decide)
    exact decide_eq_false composite
  second := false
  second_correct := by
    have composite : ¬ Nat.Prime (corner 13061 2) := by change ¬ Nat.Prime 682358885; exact PrattCertificates.composite_of_product (factor := 5) (cofactor := 136471777) rfl (by decide) (by decide)
    exact decide_eq_false composite
  third := false
  third_correct := by
    have composite : ¬ Nat.Prime (corner 13061 3) := by change ¬ Nat.Prime 682332763; exact PrattCertificates.composite_of_product (factor := 7) (cofactor := 97476109) rfl (by decide) (by decide)
    exact decide_eq_false composite
theorem row_count_13061 : row 13061 = 0 := (row_13061.checked).trans rfl
theorem model_count_13061 : modelRow 13061 = 0 := by decide
theorem modelrow_agrees_13061 : modelRow 13061 = row 13061 := model_count_13061.trans row_count_13061.symm
def row_13062 : RowCertificate 13062 where
  first := true
  first_correct := by
    have prime : Nat.Prime (corner 13062 1) := by change Nat.Prime 682489501; exact ProjectEulerSolutions.P58PrimeNodes.p_682489501.prime
    exact decide_eq_true prime
  second := true
  second_correct := by
    have prime : Nat.Prime (corner 13062 2) := by change Nat.Prime 682463377; exact ProjectEulerSolutions.P58PrimeNodes.p_682463377.prime
    exact decide_eq_true prime
  third := false
  third_correct := by
    have composite : ¬ Nat.Prime (corner 13062 3) := by change ¬ Nat.Prime 682437253; exact PrattCertificates.composite_of_product (factor := 9679) (cofactor := 70507) rfl (by decide) (by decide)
    exact decide_eq_false composite
theorem row_count_13062 : row 13062 = 2 := (row_13062.checked).trans rfl
theorem model_count_13062 : modelRow 13062 = 2 := by decide
theorem modelrow_agrees_13062 : modelRow 13062 = row 13062 := model_count_13062.trans row_count_13062.symm
def row_13063 : RowCertificate 13063 where
  first := false
  first_correct := by
    have composite : ¬ Nat.Prime (corner 13063 1) := by change ¬ Nat.Prime 682594003; exact PrattCertificates.composite_of_product (factor := 7) (cofactor := 97513429) rfl (by decide) (by decide)
    exact decide_eq_false composite
  second := false
  second_correct := by
    have composite : ¬ Nat.Prime (corner 13063 2) := by change ¬ Nat.Prime 682567877; exact PrattCertificates.composite_of_product (factor := 41) (cofactor := 16647997) rfl (by decide) (by decide)
    exact decide_eq_false composite
  third := false
  third_correct := by
    have composite : ¬ Nat.Prime (corner 13063 3) := by change ¬ Nat.Prime 682541751; exact PrattCertificates.composite_of_product (factor := 3) (cofactor := 227513917) rfl (by decide) (by decide)
    exact decide_eq_false composite
theorem row_count_13063 : row 13063 = 0 := (row_13063.checked).trans rfl
theorem model_count_13063 : modelRow 13063 = 0 := by decide
theorem modelrow_agrees_13063 : modelRow 13063 = row 13063 := model_count_13063.trans row_count_13063.symm
def row_13064 : RowCertificate 13064 where
  first := false
  first_correct := by
    have composite : ¬ Nat.Prime (corner 13064 1) := by change ¬ Nat.Prime 682698513; exact PrattCertificates.composite_of_product (factor := 3) (cofactor := 227566171) rfl (by decide) (by decide)
    exact decide_eq_false composite
  second := false
  second_correct := by
    have composite : ¬ Nat.Prime (corner 13064 2) := by change ¬ Nat.Prime 682672385; exact PrattCertificates.composite_of_product (factor := 5) (cofactor := 136534477) rfl (by decide) (by decide)
    exact decide_eq_false composite
  third := false
  third_correct := by
    have composite : ¬ Nat.Prime (corner 13064 3) := by change ¬ Nat.Prime 682646257; exact PrattCertificates.composite_of_product (factor := 31) (cofactor := 22020847) rfl (by decide) (by decide)
    exact decide_eq_false composite
theorem row_count_13064 : row 13064 = 0 := (row_13064.checked).trans rfl
theorem model_count_13064 : modelRow 13064 = 0 := by decide
theorem modelrow_agrees_13064 : modelRow 13064 = row 13064 := model_count_13064.trans row_count_13064.symm
def row_13065 : RowCertificate 13065 where
  first := false
  first_correct := by
    have composite : ¬ Nat.Prime (corner 13065 1) := by change ¬ Nat.Prime 682803031; exact PrattCertificates.composite_of_product (factor := 223) (cofactor := 3061897) rfl (by decide) (by decide)
    exact decide_eq_false composite
  second := false
  second_correct := by
    have composite : ¬ Nat.Prime (corner 13065 2) := by change ¬ Nat.Prime 682776901; exact PrattCertificates.composite_of_product (factor := 137) (cofactor := 4983773) rfl (by decide) (by decide)
    exact decide_eq_false composite
  third := true
  third_correct := by
    have prime : Nat.Prime (corner 13065 3) := by change Nat.Prime 682750771; exact ProjectEulerSolutions.P58PrimeNodes.p_682750771.prime
    exact decide_eq_true prime
theorem row_count_13065 : row 13065 = 1 := (row_13065.checked).trans rfl
theorem model_count_13065 : modelRow 13065 = 1 := by decide
theorem modelrow_agrees_13065 : modelRow 13065 = row 13065 := model_count_13065.trans row_count_13065.symm
def row_13066 : RowCertificate 13066 where
  first := false
  first_correct := by
    have composite : ¬ Nat.Prime (corner 13066 1) := by change ¬ Nat.Prime 682907557; exact PrattCertificates.composite_of_product (factor := 19) (cofactor := 35942503) rfl (by decide) (by decide)
    exact decide_eq_false composite
  second := false
  second_correct := by
    have composite : ¬ Nat.Prime (corner 13066 2) := by change ¬ Nat.Prime 682881425; exact PrattCertificates.composite_of_product (factor := 5) (cofactor := 136576285) rfl (by decide) (by decide)
    exact decide_eq_false composite
  third := false
  third_correct := by
    have composite : ¬ Nat.Prime (corner 13066 3) := by change ¬ Nat.Prime 682855293; exact PrattCertificates.composite_of_product (factor := 3) (cofactor := 227618431) rfl (by decide) (by decide)
    exact decide_eq_false composite
theorem row_count_13066 : row 13066 = 0 := (row_13066.checked).trans rfl
theorem model_count_13066 : modelRow 13066 = 0 := by decide
theorem modelrow_agrees_13066 : modelRow 13066 = row 13066 := model_count_13066.trans row_count_13066.symm
def row_13067 : RowCertificate 13067 where
  first := false
  first_correct := by
    have composite : ¬ Nat.Prime (corner 13067 1) := by change ¬ Nat.Prime 683012091; exact PrattCertificates.composite_of_product (factor := 3) (cofactor := 227670697) rfl (by decide) (by decide)
    exact decide_eq_false composite
  second := true
  second_correct := by
    have prime : Nat.Prime (corner 13067 2) := by change Nat.Prime 682985957; exact ProjectEulerSolutions.P58PrimeNodes.p_682985957.prime
    exact decide_eq_true prime
  third := false
  third_correct := by
    have composite : ¬ Nat.Prime (corner 13067 3) := by change ¬ Nat.Prime 682959823; exact PrattCertificates.composite_of_product (factor := 7) (cofactor := 97565689) rfl (by decide) (by decide)
    exact decide_eq_false composite
theorem row_count_13067 : row 13067 = 1 := (row_13067.checked).trans rfl
theorem model_count_13067 : modelRow 13067 = 1 := by decide
theorem modelrow_agrees_13067 : modelRow 13067 = row 13067 := model_count_13067.trans row_count_13067.symm
def row_13068 : RowCertificate 13068 where
  first := false
  first_correct := by
    have composite : ¬ Nat.Prime (corner 13068 1) := by change ¬ Nat.Prime 683116633; exact PrattCertificates.composite_of_product (factor := 19) (cofactor := 35953507) rfl (by decide) (by decide)
    exact decide_eq_false composite
  second := false
  second_correct := by
    have composite : ¬ Nat.Prime (corner 13068 2) := by change ¬ Nat.Prime 683090497; exact PrattCertificates.composite_of_product (factor := 193) (cofactor := 3539329) rfl (by decide) (by decide)
    exact decide_eq_false composite
  third := false
  third_correct := by
    have composite : ¬ Nat.Prime (corner 13068 3) := by change ¬ Nat.Prime 683064361; exact PrattCertificates.composite_of_product (factor := 7) (cofactor := 97580623) rfl (by decide) (by decide)
    exact decide_eq_false composite
theorem row_count_13068 : row 13068 = 0 := (row_13068.checked).trans rfl
theorem model_count_13068 : modelRow 13068 = 0 := by decide
theorem modelrow_agrees_13068 : modelRow 13068 = row 13068 := model_count_13068.trans row_count_13068.symm
def row_13069 : RowCertificate 13069 where
  first := false
  first_correct := by
    have composite : ¬ Nat.Prime (corner 13069 1) := by change ¬ Nat.Prime 683221183; exact PrattCertificates.composite_of_product (factor := 31) (cofactor := 22039393) rfl (by decide) (by decide)
    exact decide_eq_false composite
  second := false
  second_correct := by
    have composite : ¬ Nat.Prime (corner 13069 2) := by change ¬ Nat.Prime 683195045; exact PrattCertificates.composite_of_product (factor := 5) (cofactor := 136639009) rfl (by decide) (by decide)
    exact decide_eq_false composite
  third := false
  third_correct := by
    have composite : ¬ Nat.Prime (corner 13069 3) := by change ¬ Nat.Prime 683168907; exact PrattCertificates.composite_of_product (factor := 3) (cofactor := 227722969) rfl (by decide) (by decide)
    exact decide_eq_false composite
theorem row_count_13069 : row 13069 = 0 := (row_13069.checked).trans rfl
theorem model_count_13069 : modelRow 13069 = 0 := by decide
theorem modelrow_agrees_13069 : modelRow 13069 = row 13069 := model_count_13069.trans row_count_13069.symm
def row_13070 : RowCertificate 13070 where
  first := false
  first_correct := by
    have composite : ¬ Nat.Prime (corner 13070 1) := by change ¬ Nat.Prime 683325741; exact PrattCertificates.composite_of_product (factor := 3) (cofactor := 227775247) rfl (by decide) (by decide)
    exact decide_eq_false composite
  second := true
  second_correct := by
    have prime : Nat.Prime (corner 13070 2) := by change Nat.Prime 683299601; exact ProjectEulerSolutions.P58PrimeNodes.p_683299601.prime
    exact decide_eq_true prime
  third := false
  third_correct := by
    have composite : ¬ Nat.Prime (corner 13070 3) := by change ¬ Nat.Prime 683273461; exact PrattCertificates.composite_of_product (factor := 13) (cofactor := 52559497) rfl (by decide) (by decide)
    exact decide_eq_false composite
theorem row_count_13070 : row 13070 = 1 := (row_13070.checked).trans rfl
theorem model_count_13070 : modelRow 13070 = 1 := by decide
theorem modelrow_agrees_13070 : modelRow 13070 = row 13070 := model_count_13070.trans row_count_13070.symm
def row_13071 : RowCertificate 13071 where
  first := false
  first_correct := by
    have composite : ¬ Nat.Prime (corner 13071 1) := by change ¬ Nat.Prime 683430307; exact PrattCertificates.composite_of_product (factor := 7) (cofactor := 97632901) rfl (by decide) (by decide)
    exact decide_eq_false composite
  second := false
  second_correct := by
    have composite : ¬ Nat.Prime (corner 13071 2) := by change ¬ Nat.Prime 683404165; exact PrattCertificates.composite_of_product (factor := 5) (cofactor := 136680833) rfl (by decide) (by decide)
    exact decide_eq_false composite
  third := false
  third_correct := by
    have composite : ¬ Nat.Prime (corner 13071 3) := by change ¬ Nat.Prime 683378023; exact PrattCertificates.composite_of_product (factor := 25411) (cofactor := 26893) rfl (by decide) (by decide)
    exact decide_eq_false composite
theorem row_count_13071 : row 13071 = 0 := (row_13071.checked).trans rfl
theorem model_count_13071 : modelRow 13071 = 0 := by decide
theorem modelrow_agrees_13071 : modelRow 13071 = row 13071 := model_count_13071.trans row_count_13071.symm
def row_13072 : RowCertificate 13072 where
  first := false
  first_correct := by
    have composite : ¬ Nat.Prime (corner 13072 1) := by change ¬ Nat.Prime 683534881; exact PrattCertificates.composite_of_product (factor := 24631) (cofactor := 27751) rfl (by decide) (by decide)
    exact decide_eq_false composite
  second := true
  second_correct := by
    have prime : Nat.Prime (corner 13072 2) := by change Nat.Prime 683508737; exact ProjectEulerSolutions.P58PrimeNodes.p_683508737.prime
    exact decide_eq_true prime
  third := false
  third_correct := by
    have composite : ¬ Nat.Prime (corner 13072 3) := by change ¬ Nat.Prime 683482593; exact PrattCertificates.composite_of_product (factor := 3) (cofactor := 227827531) rfl (by decide) (by decide)
    exact decide_eq_false composite
theorem row_count_13072 : row 13072 = 1 := (row_13072.checked).trans rfl
theorem model_count_13072 : modelRow 13072 = 1 := by decide
theorem modelrow_agrees_13072 : modelRow 13072 = row 13072 := model_count_13072.trans row_count_13072.symm
def row_13073 : RowCertificate 13073 where
  first := false
  first_correct := by
    have composite : ¬ Nat.Prime (corner 13073 1) := by change ¬ Nat.Prime 683639463; exact PrattCertificates.composite_of_product (factor := 3) (cofactor := 227879821) rfl (by decide) (by decide)
    exact decide_eq_false composite
  second := false
  second_correct := by
    have composite : ¬ Nat.Prime (corner 13073 2) := by change ¬ Nat.Prime 683613317; exact PrattCertificates.composite_of_product (factor := 29) (cofactor := 23572873) rfl (by decide) (by decide)
    exact decide_eq_false composite
  third := true
  third_correct := by
    have prime : Nat.Prime (corner 13073 3) := by change Nat.Prime 683587171; exact ProjectEulerSolutions.P58PrimeNodes.p_683587171.prime
    exact decide_eq_true prime
theorem row_count_13073 : row 13073 = 1 := (row_13073.checked).trans rfl
theorem model_count_13073 : modelRow 13073 = 1 := by decide
theorem modelrow_agrees_13073 : modelRow 13073 = row 13073 := model_count_13073.trans row_count_13073.symm
def row_13074 : RowCertificate 13074 where
  first := false
  first_correct := by
    have composite : ¬ Nat.Prime (corner 13074 1) := by change ¬ Nat.Prime 683744053; exact PrattCertificates.composite_of_product (factor := 37) (cofactor := 18479569) rfl (by decide) (by decide)
    exact decide_eq_false composite
  second := false
  second_correct := by
    have composite : ¬ Nat.Prime (corner 13074 2) := by change ¬ Nat.Prime 683717905; exact PrattCertificates.composite_of_product (factor := 5) (cofactor := 136743581) rfl (by decide) (by decide)
    exact decide_eq_false composite
  third := false
  third_correct := by
    have composite : ¬ Nat.Prime (corner 13074 3) := by change ¬ Nat.Prime 683691757; exact PrattCertificates.composite_of_product (factor := 7) (cofactor := 97670251) rfl (by decide) (by decide)
    exact decide_eq_false composite
theorem row_count_13074 : row 13074 = 0 := (row_13074.checked).trans rfl
theorem model_count_13074 : modelRow 13074 = 0 := by decide
theorem modelrow_agrees_13074 : modelRow 13074 = row 13074 := model_count_13074.trans row_count_13074.symm
def row_13075 : RowCertificate 13075 where
  first := false
  first_correct := by
    have composite : ¬ Nat.Prime (corner 13075 1) := by change ¬ Nat.Prime 683848651; exact PrattCertificates.composite_of_product (factor := 43) (cofactor := 15903457) rfl (by decide) (by decide)
    exact decide_eq_false composite
  second := false
  second_correct := by
    have composite : ¬ Nat.Prime (corner 13075 2) := by change ¬ Nat.Prime 683822501; exact PrattCertificates.composite_of_product (factor := 17) (cofactor := 40224853) rfl (by decide) (by decide)
    exact decide_eq_false composite
  third := false
  third_correct := by
    have composite : ¬ Nat.Prime (corner 13075 3) := by change ¬ Nat.Prime 683796351; exact PrattCertificates.composite_of_product (factor := 3) (cofactor := 227932117) rfl (by decide) (by decide)
    exact decide_eq_false composite
theorem row_count_13075 : row 13075 = 0 := (row_13075.checked).trans rfl
theorem model_count_13075 : modelRow 13075 = 0 := by decide
theorem modelrow_agrees_13075 : modelRow 13075 = row 13075 := model_count_13075.trans row_count_13075.symm
def row_13076 : RowCertificate 13076 where
  first := false
  first_correct := by
    have composite : ¬ Nat.Prime (corner 13076 1) := by change ¬ Nat.Prime 683953257; exact PrattCertificates.composite_of_product (factor := 3) (cofactor := 227984419) rfl (by decide) (by decide)
    exact decide_eq_false composite
  second := false
  second_correct := by
    have composite : ¬ Nat.Prime (corner 13076 2) := by change ¬ Nat.Prime 683927105; exact PrattCertificates.composite_of_product (factor := 5) (cofactor := 136785421) rfl (by decide) (by decide)
    exact decide_eq_false composite
  third := false
  third_correct := by
    have composite : ¬ Nat.Prime (corner 13076 3) := by change ¬ Nat.Prime 683900953; exact PrattCertificates.composite_of_product (factor := 19) (cofactor := 35994787) rfl (by decide) (by decide)
    exact decide_eq_false composite
theorem row_count_13076 : row 13076 = 0 := (row_13076.checked).trans rfl
theorem model_count_13076 : modelRow 13076 = 0 := by decide
theorem modelrow_agrees_13076 : modelRow 13076 = row 13076 := model_count_13076.trans row_count_13076.symm
def row_13077 : RowCertificate 13077 where
  first := false
  first_correct := by
    have composite : ¬ Nat.Prime (corner 13077 1) := by change ¬ Nat.Prime 684057871; exact PrattCertificates.composite_of_product (factor := 7) (cofactor := 97722553) rfl (by decide) (by decide)
    exact decide_eq_false composite
  second := true
  second_correct := by
    have prime : Nat.Prime (corner 13077 2) := by change Nat.Prime 684031717; exact ProjectEulerSolutions.P58PrimeNodes.p_684031717.prime
    exact decide_eq_true prime
  third := false
  third_correct := by
    have composite : ¬ Nat.Prime (corner 13077 3) := by change ¬ Nat.Prime 684005563; exact PrattCertificates.composite_of_product (factor := 23053) (cofactor := 29671) rfl (by decide) (by decide)
    exact decide_eq_false composite
theorem row_count_13077 : row 13077 = 1 := (row_13077.checked).trans rfl
theorem model_count_13077 : modelRow 13077 = 1 := by decide
theorem modelrow_agrees_13077 : modelRow 13077 = row 13077 := model_count_13077.trans row_count_13077.symm
def row_13078 : RowCertificate 13078 where
  first := false
  first_correct := by
    have composite : ¬ Nat.Prime (corner 13078 1) := by change ¬ Nat.Prime 684162493; exact PrattCertificates.composite_of_product (factor := 7) (cofactor := 97737499) rfl (by decide) (by decide)
    exact decide_eq_false composite
  second := false
  second_correct := by
    have composite : ¬ Nat.Prime (corner 13078 2) := by change ¬ Nat.Prime 684136337; exact PrattCertificates.composite_of_product (factor := 3209) (cofactor := 213193) rfl (by decide) (by decide)
    exact decide_eq_false composite
  third := false
  third_correct := by
    have composite : ¬ Nat.Prime (corner 13078 3) := by change ¬ Nat.Prime 684110181; exact PrattCertificates.composite_of_product (factor := 3) (cofactor := 228036727) rfl (by decide) (by decide)
    exact decide_eq_false composite
theorem row_count_13078 : row 13078 = 0 := (row_13078.checked).trans rfl
theorem model_count_13078 : modelRow 13078 = 0 := by decide
theorem modelrow_agrees_13078 : modelRow 13078 = row 13078 := model_count_13078.trans row_count_13078.symm
def row_13079 : RowCertificate 13079 where
  first := false
  first_correct := by
    have composite : ¬ Nat.Prime (corner 13079 1) := by change ¬ Nat.Prime 684267123; exact PrattCertificates.composite_of_product (factor := 3) (cofactor := 228089041) rfl (by decide) (by decide)
    exact decide_eq_false composite
  second := false
  second_correct := by
    have composite : ¬ Nat.Prime (corner 13079 2) := by change ¬ Nat.Prime 684240965; exact PrattCertificates.composite_of_product (factor := 5) (cofactor := 136848193) rfl (by decide) (by decide)
    exact decide_eq_false composite
  third := false
  third_correct := by
    have composite : ¬ Nat.Prime (corner 13079 3) := by change ¬ Nat.Prime 684214807; exact PrattCertificates.composite_of_product (factor := 337) (cofactor := 2030311) rfl (by decide) (by decide)
    exact decide_eq_false composite
theorem row_count_13079 : row 13079 = 0 := (row_13079.checked).trans rfl
theorem model_count_13079 : modelRow 13079 = 0 := by decide
theorem modelrow_agrees_13079 : modelRow 13079 = row 13079 := model_count_13079.trans row_count_13079.symm
def row_13080 : RowCertificate 13080 where
  first := false
  first_correct := by
    have composite : ¬ Nat.Prime (corner 13080 1) := by change ¬ Nat.Prime 684371761; exact PrattCertificates.composite_of_product (factor := 5641) (cofactor := 121321) rfl (by decide) (by decide)
    exact decide_eq_false composite
  second := true
  second_correct := by
    have prime : Nat.Prime (corner 13080 2) := by change Nat.Prime 684345601; exact ProjectEulerSolutions.P58PrimeNodes.p_684345601.prime
    exact decide_eq_true prime
  third := false
  third_correct := by
    have composite : ¬ Nat.Prime (corner 13080 3) := by change ¬ Nat.Prime 684319441; exact PrattCertificates.composite_of_product (factor := 13) (cofactor := 52639957) rfl (by decide) (by decide)
    exact decide_eq_false composite
theorem row_count_13080 : row 13080 = 1 := (row_13080.checked).trans rfl
theorem model_count_13080 : modelRow 13080 = 1 := by decide
theorem modelrow_agrees_13080 : modelRow 13080 = row 13080 := model_count_13080.trans row_count_13080.symm
def row_13081 : RowCertificate 13081 where
  first := true
  first_correct := by
    have prime : Nat.Prime (corner 13081 1) := by change Nat.Prime 684476407; exact ProjectEulerSolutions.P58PrimeNodes.p_684476407.prime
    exact decide_eq_true prime
  second := false
  second_correct := by
    have composite : ¬ Nat.Prime (corner 13081 2) := by change ¬ Nat.Prime 684450245; exact PrattCertificates.composite_of_product (factor := 5) (cofactor := 136890049) rfl (by decide) (by decide)
    exact decide_eq_false composite
  third := false
  third_correct := by
    have composite : ¬ Nat.Prime (corner 13081 3) := by change ¬ Nat.Prime 684424083; exact PrattCertificates.composite_of_product (factor := 3) (cofactor := 228141361) rfl (by decide) (by decide)
    exact decide_eq_false composite
theorem row_count_13081 : row 13081 = 1 := (row_13081.checked).trans rfl
theorem model_count_13081 : modelRow 13081 = 1 := by decide
theorem modelrow_agrees_13081 : modelRow 13081 = row 13081 := model_count_13081.trans row_count_13081.symm
def row_13082 : RowCertificate 13082 where
  first := false
  first_correct := by
    have composite : ¬ Nat.Prime (corner 13082 1) := by change ¬ Nat.Prime 684581061; exact PrattCertificates.composite_of_product (factor := 3) (cofactor := 228193687) rfl (by decide) (by decide)
    exact decide_eq_false composite
  second := false
  second_correct := by
    have composite : ¬ Nat.Prime (corner 13082 2) := by change ¬ Nat.Prime 684554897; exact PrattCertificates.composite_of_product (factor := 13) (cofactor := 52658069) rfl (by decide) (by decide)
    exact decide_eq_false composite
  third := false
  third_correct := by
    have composite : ¬ Nat.Prime (corner 13082 3) := by change ¬ Nat.Prime 684528733; exact PrattCertificates.composite_of_product (factor := 7) (cofactor := 97789819) rfl (by decide) (by decide)
    exact decide_eq_false composite
theorem row_count_13082 : row 13082 = 0 := (row_13082.checked).trans rfl
theorem model_count_13082 : modelRow 13082 = 0 := by decide
theorem modelrow_agrees_13082 : modelRow 13082 = row 13082 := model_count_13082.trans row_count_13082.symm
def row_13083 : RowCertificate 13083 where
  first := false
  first_correct := by
    have composite : ¬ Nat.Prime (corner 13083 1) := by change ¬ Nat.Prime 684685723; exact PrattCertificates.composite_of_product (factor := 1291) (cofactor := 530353) rfl (by decide) (by decide)
    exact decide_eq_false composite
  second := false
  second_correct := by
    have composite : ¬ Nat.Prime (corner 13083 2) := by change ¬ Nat.Prime 684659557; exact PrattCertificates.composite_of_product (factor := 8821) (cofactor := 77617) rfl (by decide) (by decide)
    exact decide_eq_false composite
  third := false
  third_correct := by
    have composite : ¬ Nat.Prime (corner 13083 3) := by change ¬ Nat.Prime 684633391; exact PrattCertificates.composite_of_product (factor := 13) (cofactor := 52664107) rfl (by decide) (by decide)
    exact decide_eq_false composite
theorem row_count_13083 : row 13083 = 0 := (row_13083.checked).trans rfl
theorem model_count_13083 : modelRow 13083 = 0 := by decide
theorem modelrow_agrees_13083 : modelRow 13083 = row 13083 := model_count_13083.trans row_count_13083.symm
def row_13084 : RowCertificate 13084 where
  first := false
  first_correct := by
    have composite : ¬ Nat.Prime (corner 13084 1) := by change ¬ Nat.Prime 684790393; exact PrattCertificates.composite_of_product (factor := 7) (cofactor := 97827199) rfl (by decide) (by decide)
    exact decide_eq_false composite
  second := false
  second_correct := by
    have composite : ¬ Nat.Prime (corner 13084 2) := by change ¬ Nat.Prime 684764225; exact PrattCertificates.composite_of_product (factor := 5) (cofactor := 136952845) rfl (by decide) (by decide)
    exact decide_eq_false composite
  third := false
  third_correct := by
    have composite : ¬ Nat.Prime (corner 13084 3) := by change ¬ Nat.Prime 684738057; exact PrattCertificates.composite_of_product (factor := 3) (cofactor := 228246019) rfl (by decide) (by decide)
    exact decide_eq_false composite
theorem row_count_13084 : row 13084 = 0 := (row_13084.checked).trans rfl
theorem model_count_13084 : modelRow 13084 = 0 := by decide
theorem modelrow_agrees_13084 : modelRow 13084 = row 13084 := model_count_13084.trans row_count_13084.symm
def row_13085 : RowCertificate 13085 where
  first := false
  first_correct := by
    have composite : ¬ Nat.Prime (corner 13085 1) := by change ¬ Nat.Prime 684895071; exact PrattCertificates.composite_of_product (factor := 3) (cofactor := 228298357) rfl (by decide) (by decide)
    exact decide_eq_false composite
  second := false
  second_correct := by
    have composite : ¬ Nat.Prime (corner 13085 2) := by change ¬ Nat.Prime 684868901; exact PrattCertificates.composite_of_product (factor := 29) (cofactor := 23616169) rfl (by decide) (by decide)
    exact decide_eq_false composite
  third := false
  third_correct := by
    have composite : ¬ Nat.Prime (corner 13085 3) := by change ¬ Nat.Prime 684842731; exact PrattCertificates.composite_of_product (factor := 31) (cofactor := 22091701) rfl (by decide) (by decide)
    exact decide_eq_false composite
theorem row_count_13085 : row 13085 = 0 := (row_13085.checked).trans rfl
theorem model_count_13085 : modelRow 13085 = 0 := by decide
theorem modelrow_agrees_13085 : modelRow 13085 = row 13085 := model_count_13085.trans row_count_13085.symm
def row_13086 : RowCertificate 13086 where
  first := false
  first_correct := by
    have composite : ¬ Nat.Prime (corner 13086 1) := by change ¬ Nat.Prime 684999757; exact PrattCertificates.composite_of_product (factor := 13) (cofactor := 52692289) rfl (by decide) (by decide)
    exact decide_eq_false composite
  second := false
  second_correct := by
    have composite : ¬ Nat.Prime (corner 13086 2) := by change ¬ Nat.Prime 684973585; exact PrattCertificates.composite_of_product (factor := 5) (cofactor := 136994717) rfl (by decide) (by decide)
    exact decide_eq_false composite
  third := true
  third_correct := by
    have prime : Nat.Prime (corner 13086 3) := by change Nat.Prime 684947413; exact ProjectEulerSolutions.P58PrimeNodes.p_684947413.prime
    exact decide_eq_true prime
theorem row_count_13086 : row 13086 = 1 := (row_13086.checked).trans rfl
theorem model_count_13086 : modelRow 13086 = 1 := by decide
theorem modelrow_agrees_13086 : modelRow 13086 = row 13086 := model_count_13086.trans row_count_13086.symm
def row_13087 : RowCertificate 13087 where
  first := false
  first_correct := by
    have composite : ¬ Nat.Prime (corner 13087 1) := by change ¬ Nat.Prime 685104451; exact PrattCertificates.composite_of_product (factor := 19) (cofactor := 36058129) rfl (by decide) (by decide)
    exact decide_eq_false composite
  second := false
  second_correct := by
    have composite : ¬ Nat.Prime (corner 13087 2) := by change ¬ Nat.Prime 685078277; exact PrattCertificates.composite_of_product (factor := 13) (cofactor := 52698329) rfl (by decide) (by decide)
    exact decide_eq_false composite
  third := false
  third_correct := by
    have composite : ¬ Nat.Prime (corner 13087 3) := by change ¬ Nat.Prime 685052103; exact PrattCertificates.composite_of_product (factor := 3) (cofactor := 228350701) rfl (by decide) (by decide)
    exact decide_eq_false composite
theorem row_count_13087 : row 13087 = 0 := (row_13087.checked).trans rfl
theorem model_count_13087 : modelRow 13087 = 0 := by decide
theorem modelrow_agrees_13087 : modelRow 13087 = row 13087 := model_count_13087.trans row_count_13087.symm
def row_13088 : RowCertificate 13088 where
  first := false
  first_correct := by
    have composite : ¬ Nat.Prime (corner 13088 1) := by change ¬ Nat.Prime 685209153; exact PrattCertificates.composite_of_product (factor := 3) (cofactor := 228403051) rfl (by decide) (by decide)
    exact decide_eq_false composite
  second := false
  second_correct := by
    have composite : ¬ Nat.Prime (corner 13088 2) := by change ¬ Nat.Prime 685182977; exact PrattCertificates.composite_of_product (factor := 17) (cofactor := 40304881) rfl (by decide) (by decide)
    exact decide_eq_false composite
  third := false
  third_correct := by
    have composite : ¬ Nat.Prime (corner 13088 3) := by change ¬ Nat.Prime 685156801; exact PrattCertificates.composite_of_product (factor := 7) (cofactor := 97879543) rfl (by decide) (by decide)
    exact decide_eq_false composite
theorem row_count_13088 : row 13088 = 0 := (row_13088.checked).trans rfl
theorem model_count_13088 : modelRow 13088 = 0 := by decide
theorem modelrow_agrees_13088 : modelRow 13088 = row 13088 := model_count_13088.trans row_count_13088.symm
def row_13089 : RowCertificate 13089 where
  first := false
  first_correct := by
    have composite : ¬ Nat.Prime (corner 13089 1) := by change ¬ Nat.Prime 685313863; exact PrattCertificates.composite_of_product (factor := 13) (cofactor := 52716451) rfl (by decide) (by decide)
    exact decide_eq_false composite
  second := false
  second_correct := by
    have composite : ¬ Nat.Prime (corner 13089 2) := by change ¬ Nat.Prime 685287685; exact PrattCertificates.composite_of_product (factor := 5) (cofactor := 137057537) rfl (by decide) (by decide)
    exact decide_eq_false composite
  third := false
  third_correct := by
    have composite : ¬ Nat.Prime (corner 13089 3) := by change ¬ Nat.Prime 685261507; exact PrattCertificates.composite_of_product (factor := 7) (cofactor := 97894501) rfl (by decide) (by decide)
    exact decide_eq_false composite
theorem row_count_13089 : row 13089 = 0 := (row_13089.checked).trans rfl
theorem model_count_13089 : modelRow 13089 = 0 := by decide
theorem modelrow_agrees_13089 : modelRow 13089 = row 13089 := model_count_13089.trans row_count_13089.symm
def row_13090 : RowCertificate 13090 where
  first := false
  first_correct := by
    have composite : ¬ Nat.Prime (corner 13090 1) := by change ¬ Nat.Prime 685418581; exact PrattCertificates.composite_of_product (factor := 43) (cofactor := 15939967) rfl (by decide) (by decide)
    exact decide_eq_false composite
  second := false
  second_correct := by
    have composite : ¬ Nat.Prime (corner 13090 2) := by change ¬ Nat.Prime 685392401; exact PrattCertificates.composite_of_product (factor := 61) (cofactor := 11235941) rfl (by decide) (by decide)
    exact decide_eq_false composite
  third := false
  third_correct := by
    have composite : ¬ Nat.Prime (corner 13090 3) := by change ¬ Nat.Prime 685366221; exact PrattCertificates.composite_of_product (factor := 3) (cofactor := 228455407) rfl (by decide) (by decide)
    exact decide_eq_false composite
theorem row_count_13090 : row 13090 = 0 := (row_13090.checked).trans rfl
theorem model_count_13090 : modelRow 13090 = 0 := by decide
theorem modelrow_agrees_13090 : modelRow 13090 = row 13090 := model_count_13090.trans row_count_13090.symm
def row_13091 : RowCertificate 13091 where
  first := false
  first_correct := by
    have composite : ¬ Nat.Prime (corner 13091 1) := by change ¬ Nat.Prime 685523307; exact PrattCertificates.composite_of_product (factor := 3) (cofactor := 228507769) rfl (by decide) (by decide)
    exact decide_eq_false composite
  second := false
  second_correct := by
    have composite : ¬ Nat.Prime (corner 13091 2) := by change ¬ Nat.Prime 685497125; exact PrattCertificates.composite_of_product (factor := 5) (cofactor := 137099425) rfl (by decide) (by decide)
    exact decide_eq_false composite
  third := false
  third_correct := by
    have composite : ¬ Nat.Prime (corner 13091 3) := by change ¬ Nat.Prime 685470943; exact PrattCertificates.composite_of_product (factor := 127) (cofactor := 5397409) rfl (by decide) (by decide)
    exact decide_eq_false composite
theorem row_count_13091 : row 13091 = 0 := (row_13091.checked).trans rfl
theorem model_count_13091 : modelRow 13091 = 0 := by decide
theorem modelrow_agrees_13091 : modelRow 13091 = row 13091 := model_count_13091.trans row_count_13091.symm
def row_13092 : RowCertificate 13092 where
  first := false
  first_correct := by
    have composite : ¬ Nat.Prime (corner 13092 1) := by change ¬ Nat.Prime 685628041; exact PrattCertificates.composite_of_product (factor := 7) (cofactor := 97946863) rfl (by decide) (by decide)
    exact decide_eq_false composite
  second := false
  second_correct := by
    have composite : ¬ Nat.Prime (corner 13092 2) := by change ¬ Nat.Prime 685601857; exact PrattCertificates.composite_of_product (factor := 17) (cofactor := 40329521) rfl (by decide) (by decide)
    exact decide_eq_false composite
  third := true
  third_correct := by
    have prime : Nat.Prime (corner 13092 3) := by change Nat.Prime 685575673; exact ProjectEulerSolutions.P58PrimeNodes.p_685575673.prime
    exact decide_eq_true prime
theorem row_count_13092 : row 13092 = 1 := (row_13092.checked).trans rfl
theorem model_count_13092 : modelRow 13092 = 1 := by decide
theorem modelrow_agrees_13092 : modelRow 13092 = row 13092 := model_count_13092.trans row_count_13092.symm
def row_13093 : RowCertificate 13093 where
  first := true
  first_correct := by
    have prime : Nat.Prime (corner 13093 1) := by change Nat.Prime 685732783; exact ProjectEulerSolutions.P58PrimeNodes.p_685732783.prime
    exact decide_eq_true prime
  second := false
  second_correct := by
    have composite : ¬ Nat.Prime (corner 13093 2) := by change ¬ Nat.Prime 685706597; exact PrattCertificates.composite_of_product (factor := 281) (cofactor := 2440237) rfl (by decide) (by decide)
    exact decide_eq_false composite
  third := false
  third_correct := by
    have composite : ¬ Nat.Prime (corner 13093 3) := by change ¬ Nat.Prime 685680411; exact PrattCertificates.composite_of_product (factor := 3) (cofactor := 228560137) rfl (by decide) (by decide)
    exact decide_eq_false composite
theorem row_count_13093 : row 13093 = 1 := (row_13093.checked).trans rfl
theorem model_count_13093 : modelRow 13093 = 1 := by decide
theorem modelrow_agrees_13093 : modelRow 13093 = row 13093 := model_count_13093.trans row_count_13093.symm
def row_13094 : RowCertificate 13094 where
  first := false
  first_correct := by
    have composite : ¬ Nat.Prime (corner 13094 1) := by change ¬ Nat.Prime 685837533; exact PrattCertificates.composite_of_product (factor := 3) (cofactor := 228612511) rfl (by decide) (by decide)
    exact decide_eq_false composite
  second := false
  second_correct := by
    have composite : ¬ Nat.Prime (corner 13094 2) := by change ¬ Nat.Prime 685811345; exact PrattCertificates.composite_of_product (factor := 5) (cofactor := 137162269) rfl (by decide) (by decide)
    exact decide_eq_false composite
  third := false
  third_correct := by
    have composite : ¬ Nat.Prime (corner 13094 3) := by change ¬ Nat.Prime 685785157; exact PrattCertificates.composite_of_product (factor := 2473) (cofactor := 277309) rfl (by decide) (by decide)
    exact decide_eq_false composite
theorem row_count_13094 : row 13094 = 0 := (row_13094.checked).trans rfl
theorem model_count_13094 : modelRow 13094 = 0 := by decide
theorem modelrow_agrees_13094 : modelRow 13094 = row 13094 := model_count_13094.trans row_count_13094.symm
def row_13095 : RowCertificate 13095 where
  first := false
  first_correct := by
    have composite : ¬ Nat.Prime (corner 13095 1) := by change ¬ Nat.Prime 685942291; exact PrattCertificates.composite_of_product (factor := 1399) (cofactor := 490309) rfl (by decide) (by decide)
    exact decide_eq_false composite
  second := false
  second_correct := by
    have composite : ¬ Nat.Prime (corner 13095 2) := by change ¬ Nat.Prime 685916101; exact PrattCertificates.composite_of_product (factor := 13) (cofactor := 52762777) rfl (by decide) (by decide)
    exact decide_eq_false composite
  third := false
  third_correct := by
    have composite : ¬ Nat.Prime (corner 13095 3) := by change ¬ Nat.Prime 685889911; exact PrattCertificates.composite_of_product (factor := 7) (cofactor := 97984273) rfl (by decide) (by decide)
    exact decide_eq_false composite
theorem row_count_13095 : row 13095 = 0 := (row_13095.checked).trans rfl
theorem model_count_13095 : modelRow 13095 = 0 := by decide
theorem modelrow_agrees_13095 : modelRow 13095 = row 13095 := model_count_13095.trans row_count_13095.symm
def row_13096 : RowCertificate 13096 where
  first := false
  first_correct := by
    have composite : ¬ Nat.Prime (corner 13096 1) := by change ¬ Nat.Prime 686047057; exact PrattCertificates.composite_of_product (factor := 1327) (cofactor := 516991) rfl (by decide) (by decide)
    exact decide_eq_false composite
  second := false
  second_correct := by
    have composite : ¬ Nat.Prime (corner 13096 2) := by change ¬ Nat.Prime 686020865; exact PrattCertificates.composite_of_product (factor := 5) (cofactor := 137204173) rfl (by decide) (by decide)
    exact decide_eq_false composite
  third := false
  third_correct := by
    have composite : ¬ Nat.Prime (corner 13096 3) := by change ¬ Nat.Prime 685994673; exact PrattCertificates.composite_of_product (factor := 3) (cofactor := 228664891) rfl (by decide) (by decide)
    exact decide_eq_false composite
theorem row_count_13096 : row 13096 = 0 := (row_13096.checked).trans rfl
theorem model_count_13096 : modelRow 13096 = 0 := by decide
theorem modelrow_agrees_13096 : modelRow 13096 = row 13096 := model_count_13096.trans row_count_13096.symm
def row_13097 : RowCertificate 13097 where
  first := false
  first_correct := by
    have composite : ¬ Nat.Prime (corner 13097 1) := by change ¬ Nat.Prime 686151831; exact PrattCertificates.composite_of_product (factor := 3) (cofactor := 228717277) rfl (by decide) (by decide)
    exact decide_eq_false composite
  second := true
  second_correct := by
    have prime : Nat.Prime (corner 13097 2) := by change Nat.Prime 686125637; exact ProjectEulerSolutions.P58PrimeNodes.p_686125637.prime
    exact decide_eq_true prime
  third := false
  third_correct := by
    have composite : ¬ Nat.Prime (corner 13097 3) := by change ¬ Nat.Prime 686099443; exact PrattCertificates.composite_of_product (factor := 19) (cofactor := 36110497) rfl (by decide) (by decide)
    exact decide_eq_false composite
theorem row_count_13097 : row 13097 = 1 := (row_13097.checked).trans rfl
theorem model_count_13097 : modelRow 13097 = 1 := by decide
theorem modelrow_agrees_13097 : modelRow 13097 = row 13097 := model_count_13097.trans row_count_13097.symm
def row_13098 : RowCertificate 13098 where
  first := false
  first_correct := by
    have composite : ¬ Nat.Prime (corner 13098 1) := by change ¬ Nat.Prime 686256613; exact PrattCertificates.composite_of_product (factor := 7) (cofactor := 98036659) rfl (by decide) (by decide)
    exact decide_eq_false composite
  second := false
  second_correct := by
    have composite : ¬ Nat.Prime (corner 13098 2) := by change ¬ Nat.Prime 686230417; exact PrattCertificates.composite_of_product (factor := 24169) (cofactor := 28393) rfl (by decide) (by decide)
    exact decide_eq_false composite
  third := true
  third_correct := by
    have prime : Nat.Prime (corner 13098 3) := by change Nat.Prime 686204221; exact ProjectEulerSolutions.P58PrimeNodes.p_686204221.prime
    exact decide_eq_true prime
theorem row_count_13098 : row 13098 = 1 := (row_13098.checked).trans rfl
theorem model_count_13098 : modelRow 13098 = 1 := by decide
theorem modelrow_agrees_13098 : modelRow 13098 = row 13098 := model_count_13098.trans row_count_13098.symm
def row_13099 : RowCertificate 13099 where
  first := false
  first_correct := by
    have composite : ¬ Nat.Prime (corner 13099 1) := by change ¬ Nat.Prime 686361403; exact PrattCertificates.composite_of_product (factor := 7) (cofactor := 98051629) rfl (by decide) (by decide)
    exact decide_eq_false composite
  second := false
  second_correct := by
    have composite : ¬ Nat.Prime (corner 13099 2) := by change ¬ Nat.Prime 686335205; exact PrattCertificates.composite_of_product (factor := 5) (cofactor := 137267041) rfl (by decide) (by decide)
    exact decide_eq_false composite
  third := false
  third_correct := by
    have composite : ¬ Nat.Prime (corner 13099 3) := by change ¬ Nat.Prime 686309007; exact PrattCertificates.composite_of_product (factor := 3) (cofactor := 228769669) rfl (by decide) (by decide)
    exact decide_eq_false composite
theorem row_count_13099 : row 13099 = 0 := (row_13099.checked).trans rfl
theorem model_count_13099 : modelRow 13099 = 0 := by decide
theorem modelrow_agrees_13099 : modelRow 13099 = row 13099 := model_count_13099.trans row_count_13099.symm
def row_13100 : RowCertificate 13100 where
  first := false
  first_correct := by
    have composite : ¬ Nat.Prime (corner 13100 1) := by change ¬ Nat.Prime 686466201; exact PrattCertificates.composite_of_product (factor := 3) (cofactor := 228822067) rfl (by decide) (by decide)
    exact decide_eq_false composite
  second := false
  second_correct := by
    have composite : ¬ Nat.Prime (corner 13100 2) := by change ¬ Nat.Prime 686440001; exact PrattCertificates.composite_of_product (factor := 13) (cofactor := 52803077) rfl (by decide) (by decide)
    exact decide_eq_false composite
  third := false
  third_correct := by
    have composite : ¬ Nat.Prime (corner 13100 3) := by change ¬ Nat.Prime 686413801; exact PrattCertificates.composite_of_product (factor := 8311) (cofactor := 82591) rfl (by decide) (by decide)
    exact decide_eq_false composite
theorem row_count_13100 : row 13100 = 0 := (row_13100.checked).trans rfl
theorem model_count_13100 : modelRow 13100 = 0 := by decide
theorem modelrow_agrees_13100 : modelRow 13100 = row 13100 := model_count_13100.trans row_count_13100.symm
def row_13101 : RowCertificate 13101 where
  first := false
  first_correct := by
    have composite : ¬ Nat.Prime (corner 13101 1) := by change ¬ Nat.Prime 686571007; exact PrattCertificates.composite_of_product (factor := 379) (cofactor := 1811533) rfl (by decide) (by decide)
    exact decide_eq_false composite
  second := false
  second_correct := by
    have composite : ¬ Nat.Prime (corner 13101 2) := by change ¬ Nat.Prime 686544805; exact PrattCertificates.composite_of_product (factor := 5) (cofactor := 137308961) rfl (by decide) (by decide)
    exact decide_eq_false composite
  third := true
  third_correct := by
    have prime : Nat.Prime (corner 13101 3) := by change Nat.Prime 686518603; exact ProjectEulerSolutions.P58PrimeNodes.p_686518603.prime
    exact decide_eq_true prime
theorem row_count_13101 : row 13101 = 1 := (row_13101.checked).trans rfl
theorem model_count_13101 : modelRow 13101 = 1 := by decide
theorem modelrow_agrees_13101 : modelRow 13101 = row 13101 := model_count_13101.trans row_count_13101.symm
def row_13102 : RowCertificate 13102 where
  first := false
  first_correct := by
    have composite : ¬ Nat.Prime (corner 13102 1) := by change ¬ Nat.Prime 686675821; exact PrattCertificates.composite_of_product (factor := 13) (cofactor := 52821217) rfl (by decide) (by decide)
    exact decide_eq_false composite
  second := false
  second_correct := by
    have composite : ¬ Nat.Prime (corner 13102 2) := by change ¬ Nat.Prime 686649617; exact PrattCertificates.composite_of_product (factor := 29) (cofactor := 23677573) rfl (by decide) (by decide)
    exact decide_eq_false composite
  third := false
  third_correct := by
    have composite : ¬ Nat.Prime (corner 13102 3) := by change ¬ Nat.Prime 686623413; exact PrattCertificates.composite_of_product (factor := 3) (cofactor := 228874471) rfl (by decide) (by decide)
    exact decide_eq_false composite
theorem row_count_13102 : row 13102 = 0 := (row_13102.checked).trans rfl
theorem model_count_13102 : modelRow 13102 = 0 := by decide
theorem modelrow_agrees_13102 : modelRow 13102 = row 13102 := model_count_13102.trans row_count_13102.symm
def row_13103 : RowCertificate 13103 where
  first := false
  first_correct := by
    have composite : ¬ Nat.Prime (corner 13103 1) := by change ¬ Nat.Prime 686780643; exact PrattCertificates.composite_of_product (factor := 3) (cofactor := 228926881) rfl (by decide) (by decide)
    exact decide_eq_false composite
  second := false
  second_correct := by
    have composite : ¬ Nat.Prime (corner 13103 2) := by change ¬ Nat.Prime 686754437; exact PrattCertificates.composite_of_product (factor := 809) (cofactor := 848893) rfl (by decide) (by decide)
    exact decide_eq_false composite
  third := false
  third_correct := by
    have composite : ¬ Nat.Prime (corner 13103 3) := by change ¬ Nat.Prime 686728231; exact PrattCertificates.composite_of_product (factor := 7) (cofactor := 98104033) rfl (by decide) (by decide)
    exact decide_eq_false composite
theorem row_count_13103 : row 13103 = 0 := (row_13103.checked).trans rfl
theorem model_count_13103 : modelRow 13103 = 0 := by decide
theorem modelrow_agrees_13103 : modelRow 13103 = row 13103 := model_count_13103.trans row_count_13103.symm
def row_13104 : RowCertificate 13104 where
  first := false
  first_correct := by
    have composite : ¬ Nat.Prime (corner 13104 1) := by change ¬ Nat.Prime 686885473; exact PrattCertificates.composite_of_product (factor := 19) (cofactor := 36151867) rfl (by decide) (by decide)
    exact decide_eq_false composite
  second := false
  second_correct := by
    have composite : ¬ Nat.Prime (corner 13104 2) := by change ¬ Nat.Prime 686859265; exact PrattCertificates.composite_of_product (factor := 5) (cofactor := 137371853) rfl (by decide) (by decide)
    exact decide_eq_false composite
  third := true
  third_correct := by
    have prime : Nat.Prime (corner 13104 3) := by change Nat.Prime 686833057; exact ProjectEulerSolutions.P58PrimeNodes.p_686833057.prime
    exact decide_eq_true prime
theorem row_count_13104 : row 13104 = 1 := (row_13104.checked).trans rfl
theorem model_count_13104 : modelRow 13104 = 1 := by decide
theorem modelrow_agrees_13104 : modelRow 13104 = row 13104 := model_count_13104.trans row_count_13104.symm
def row_13105 : RowCertificate 13105 where
  first := false
  first_correct := by
    have composite : ¬ Nat.Prime (corner 13105 1) := by change ¬ Nat.Prime 686990311; exact PrattCertificates.composite_of_product (factor := 7) (cofactor := 98141473) rfl (by decide) (by decide)
    exact decide_eq_false composite
  second := false
  second_correct := by
    have composite : ¬ Nat.Prime (corner 13105 2) := by change ¬ Nat.Prime 686964101; exact PrattCertificates.composite_of_product (factor := 17) (cofactor := 40409653) rfl (by decide) (by decide)
    exact decide_eq_false composite
  third := false
  third_correct := by
    have composite : ¬ Nat.Prime (corner 13105 3) := by change ¬ Nat.Prime 686937891; exact PrattCertificates.composite_of_product (factor := 3) (cofactor := 228979297) rfl (by decide) (by decide)
    exact decide_eq_false composite
theorem row_count_13105 : row 13105 = 0 := (row_13105.checked).trans rfl
theorem model_count_13105 : modelRow 13105 = 0 := by decide
theorem modelrow_agrees_13105 : modelRow 13105 = row 13105 := model_count_13105.trans row_count_13105.symm
def row_13106 : RowCertificate 13106 where
  first := false
  first_correct := by
    have composite : ¬ Nat.Prime (corner 13106 1) := by change ¬ Nat.Prime 687095157; exact PrattCertificates.composite_of_product (factor := 3) (cofactor := 229031719) rfl (by decide) (by decide)
    exact decide_eq_false composite
  second := false
  second_correct := by
    have composite : ¬ Nat.Prime (corner 13106 2) := by change ¬ Nat.Prime 687068945; exact PrattCertificates.composite_of_product (factor := 5) (cofactor := 137413789) rfl (by decide) (by decide)
    exact decide_eq_false composite
  third := false
  third_correct := by
    have composite : ¬ Nat.Prime (corner 13106 3) := by change ¬ Nat.Prime 687042733; exact PrattCertificates.composite_of_product (factor := 13) (cofactor := 52849441) rfl (by decide) (by decide)
    exact decide_eq_false composite
theorem row_count_13106 : row 13106 = 0 := (row_13106.checked).trans rfl
theorem model_count_13106 : modelRow 13106 = 0 := by decide
theorem modelrow_agrees_13106 : modelRow 13106 = row 13106 := model_count_13106.trans row_count_13106.symm
def row_13107 : RowCertificate 13107 where
  first := false
  first_correct := by
    have composite : ¬ Nat.Prime (corner 13107 1) := by change ¬ Nat.Prime 687200011; exact PrattCertificates.composite_of_product (factor := 433) (cofactor := 1587067) rfl (by decide) (by decide)
    exact decide_eq_false composite
  second := false
  second_correct := by
    have composite : ¬ Nat.Prime (corner 13107 2) := by change ¬ Nat.Prime 687173797; exact PrattCertificates.composite_of_product (factor := 373) (cofactor := 1842289) rfl (by decide) (by decide)
    exact decide_eq_false composite
  third := false
  third_correct := by
    have composite : ¬ Nat.Prime (corner 13107 3) := by change ¬ Nat.Prime 687147583; exact PrattCertificates.composite_of_product (factor := 9949) (cofactor := 69067) rfl (by decide) (by decide)
    exact decide_eq_false composite
theorem row_count_13107 : row 13107 = 0 := (row_13107.checked).trans rfl
theorem model_count_13107 : modelRow 13107 = 0 := by decide
theorem modelrow_agrees_13107 : modelRow 13107 = row 13107 := model_count_13107.trans row_count_13107.symm
def row_13108 : RowCertificate 13108 where
  first := false
  first_correct := by
    have composite : ¬ Nat.Prime (corner 13108 1) := by change ¬ Nat.Prime 687304873; exact PrattCertificates.composite_of_product (factor := 61) (cofactor := 11267293) rfl (by decide) (by decide)
    exact decide_eq_false composite
  second := false
  second_correct := by
    have composite : ¬ Nat.Prime (corner 13108 2) := by change ¬ Nat.Prime 687278657; exact PrattCertificates.composite_of_product (factor := 13) (cofactor := 52867589) rfl (by decide) (by decide)
    exact decide_eq_false composite
  third := false
  third_correct := by
    have composite : ¬ Nat.Prime (corner 13108 3) := by change ¬ Nat.Prime 687252441; exact PrattCertificates.composite_of_product (factor := 3) (cofactor := 229084147) rfl (by decide) (by decide)
    exact decide_eq_false composite
theorem row_count_13108 : row 13108 = 0 := (row_13108.checked).trans rfl
theorem model_count_13108 : modelRow 13108 = 0 := by decide
theorem modelrow_agrees_13108 : modelRow 13108 = row 13108 := model_count_13108.trans row_count_13108.symm
def row_13109 : RowCertificate 13109 where
  first := false
  first_correct := by
    have composite : ¬ Nat.Prime (corner 13109 1) := by change ¬ Nat.Prime 687409743; exact PrattCertificates.composite_of_product (factor := 3) (cofactor := 229136581) rfl (by decide) (by decide)
    exact decide_eq_false composite
  second := false
  second_correct := by
    have composite : ¬ Nat.Prime (corner 13109 2) := by change ¬ Nat.Prime 687383525; exact PrattCertificates.composite_of_product (factor := 5) (cofactor := 137476705) rfl (by decide) (by decide)
    exact decide_eq_false composite
  third := false
  third_correct := by
    have composite : ¬ Nat.Prime (corner 13109 3) := by change ¬ Nat.Prime 687357307; exact PrattCertificates.composite_of_product (factor := 7) (cofactor := 98193901) rfl (by decide) (by decide)
    exact decide_eq_false composite
theorem row_count_13109 : row 13109 = 0 := (row_13109.checked).trans rfl
theorem model_count_13109 : modelRow 13109 = 0 := by decide
theorem modelrow_agrees_13109 : modelRow 13109 = row 13109 := model_count_13109.trans row_count_13109.symm
def row_13110 : RowCertificate 13110 where
  first := false
  first_correct := by
    have composite : ¬ Nat.Prime (corner 13110 1) := by change ¬ Nat.Prime 687514621; exact PrattCertificates.composite_of_product (factor := 31) (cofactor := 22177891) rfl (by decide) (by decide)
    exact decide_eq_false composite
  second := false
  second_correct := by
    have composite : ¬ Nat.Prime (corner 13110 2) := by change ¬ Nat.Prime 687488401; exact PrattCertificates.composite_of_product (factor := 2677) (cofactor := 256813) rfl (by decide) (by decide)
    exact decide_eq_false composite
  third := false
  third_correct := by
    have composite : ¬ Nat.Prime (corner 13110 3) := by change ¬ Nat.Prime 687462181; exact PrattCertificates.composite_of_product (factor := 7) (cofactor := 98208883) rfl (by decide) (by decide)
    exact decide_eq_false composite
theorem row_count_13110 : row 13110 = 0 := (row_13110.checked).trans rfl
theorem model_count_13110 : modelRow 13110 = 0 := by decide
theorem modelrow_agrees_13110 : modelRow 13110 = row 13110 := model_count_13110.trans row_count_13110.symm
def row_13111 : RowCertificate 13111 where
  first := false
  first_correct := by
    have composite : ¬ Nat.Prime (corner 13111 1) := by change ¬ Nat.Prime 687619507; exact PrattCertificates.composite_of_product (factor := 37) (cofactor := 18584311) rfl (by decide) (by decide)
    exact decide_eq_false composite
  second := false
  second_correct := by
    have composite : ¬ Nat.Prime (corner 13111 2) := by change ¬ Nat.Prime 687593285; exact PrattCertificates.composite_of_product (factor := 5) (cofactor := 137518657) rfl (by decide) (by decide)
    exact decide_eq_false composite
  third := false
  third_correct := by
    have composite : ¬ Nat.Prime (corner 13111 3) := by change ¬ Nat.Prime 687567063; exact PrattCertificates.composite_of_product (factor := 3) (cofactor := 229189021) rfl (by decide) (by decide)
    exact decide_eq_false composite
theorem row_count_13111 : row 13111 = 0 := (row_13111.checked).trans rfl
theorem model_count_13111 : modelRow 13111 = 0 := by decide
theorem modelrow_agrees_13111 : modelRow 13111 = row 13111 := model_count_13111.trans row_count_13111.symm
def row_13112 : RowCertificate 13112 where
  first := false
  first_correct := by
    have composite : ¬ Nat.Prime (corner 13112 1) := by change ¬ Nat.Prime 687724401; exact PrattCertificates.composite_of_product (factor := 3) (cofactor := 229241467) rfl (by decide) (by decide)
    exact decide_eq_false composite
  second := true
  second_correct := by
    have prime : Nat.Prime (corner 13112 2) := by change Nat.Prime 687698177; exact ProjectEulerSolutions.P58PrimeNodes.p_687698177.prime
    exact decide_eq_true prime
  third := false
  third_correct := by
    have composite : ¬ Nat.Prime (corner 13112 3) := by change ¬ Nat.Prime 687671953; exact PrattCertificates.composite_of_product (factor := 43) (cofactor := 15992371) rfl (by decide) (by decide)
    exact decide_eq_false composite
theorem row_count_13112 : row 13112 = 1 := (row_13112.checked).trans rfl
theorem model_count_13112 : modelRow 13112 = 1 := by decide
theorem modelrow_agrees_13112 : modelRow 13112 = row 13112 := model_count_13112.trans row_count_13112.symm
def row_13113 : RowCertificate 13113 where
  first := false
  first_correct := by
    have composite : ¬ Nat.Prime (corner 13113 1) := by change ¬ Nat.Prime 687829303; exact PrattCertificates.composite_of_product (factor := 7) (cofactor := 98261329) rfl (by decide) (by decide)
    exact decide_eq_false composite
  second := false
  second_correct := by
    have composite : ¬ Nat.Prime (corner 13113 2) := by change ¬ Nat.Prime 687803077; exact PrattCertificates.composite_of_product (factor := 13) (cofactor := 52907929) rfl (by decide) (by decide)
    exact decide_eq_false composite
  third := false
  third_correct := by
    have composite : ¬ Nat.Prime (corner 13113 3) := by change ¬ Nat.Prime 687776851; exact PrattCertificates.composite_of_product (factor := 97) (cofactor := 7090483) rfl (by decide) (by decide)
    exact decide_eq_false composite
theorem row_count_13113 : row 13113 = 0 := (row_13113.checked).trans rfl
theorem model_count_13113 : modelRow 13113 = 0 := by decide
theorem modelrow_agrees_13113 : modelRow 13113 = row 13113 := model_count_13113.trans row_count_13113.symm
def row_13114 : RowCertificate 13114 where
  first := false
  first_correct := by
    have composite : ¬ Nat.Prime (corner 13114 1) := by change ¬ Nat.Prime 687934213; exact PrattCertificates.composite_of_product (factor := 139) (cofactor := 4949167) rfl (by decide) (by decide)
    exact decide_eq_false composite
  second := false
  second_correct := by
    have composite : ¬ Nat.Prime (corner 13114 2) := by change ¬ Nat.Prime 687907985; exact PrattCertificates.composite_of_product (factor := 5) (cofactor := 137581597) rfl (by decide) (by decide)
    exact decide_eq_false composite
  third := false
  third_correct := by
    have composite : ¬ Nat.Prime (corner 13114 3) := by change ¬ Nat.Prime 687881757; exact PrattCertificates.composite_of_product (factor := 3) (cofactor := 229293919) rfl (by decide) (by decide)
    exact decide_eq_false composite
theorem row_count_13114 : row 13114 = 0 := (row_13114.checked).trans rfl
theorem model_count_13114 : modelRow 13114 = 0 := by decide
theorem modelrow_agrees_13114 : modelRow 13114 = row 13114 := model_count_13114.trans row_count_13114.symm
def row_13115 : RowCertificate 13115 where
  first := false
  first_correct := by
    have composite : ¬ Nat.Prime (corner 13115 1) := by change ¬ Nat.Prime 688039131; exact PrattCertificates.composite_of_product (factor := 3) (cofactor := 229346377) rfl (by decide) (by decide)
    exact decide_eq_false composite
  second := false
  second_correct := by
    have composite : ¬ Nat.Prime (corner 13115 2) := by change ¬ Nat.Prime 688012901; exact PrattCertificates.composite_of_product (factor := 257) (cofactor := 2677093) rfl (by decide) (by decide)
    exact decide_eq_false composite
  third := false
  third_correct := by
    have composite : ¬ Nat.Prime (corner 13115 3) := by change ¬ Nat.Prime 687986671; exact PrattCertificates.composite_of_product (factor := 409) (cofactor := 1682119) rfl (by decide) (by decide)
    exact decide_eq_false composite
theorem row_count_13115 : row 13115 = 0 := (row_13115.checked).trans rfl
theorem model_count_13115 : modelRow 13115 = 0 := by decide
theorem modelrow_agrees_13115 : modelRow 13115 = row 13115 := model_count_13115.trans row_count_13115.symm
def row_13116 : RowCertificate 13116 where
  first := true
  first_correct := by
    have prime : Nat.Prime (corner 13116 1) := by change Nat.Prime 688144057; exact ProjectEulerSolutions.P58PrimeNodes.p_688144057.prime
    exact decide_eq_true prime
  second := false
  second_correct := by
    have composite : ¬ Nat.Prime (corner 13116 2) := by change ¬ Nat.Prime 688117825; exact PrattCertificates.composite_of_product (factor := 5) (cofactor := 137623565) rfl (by decide) (by decide)
    exact decide_eq_false composite
  third := false
  third_correct := by
    have composite : ¬ Nat.Prime (corner 13116 3) := by change ¬ Nat.Prime 688091593; exact PrattCertificates.composite_of_product (factor := 7) (cofactor := 98298799) rfl (by decide) (by decide)
    exact decide_eq_false composite
theorem row_count_13116 : row 13116 = 1 := (row_13116.checked).trans rfl
theorem model_count_13116 : modelRow 13116 = 1 := by decide
theorem modelrow_agrees_13116 : modelRow 13116 = row 13116 := model_count_13116.trans row_count_13116.symm
def row_13117 : RowCertificate 13117 where
  first := false
  first_correct := by
    have composite : ¬ Nat.Prime (corner 13117 1) := by change ¬ Nat.Prime 688248991; exact PrattCertificates.composite_of_product (factor := 67) (cofactor := 10272373) rfl (by decide) (by decide)
    exact decide_eq_false composite
  second := false
  second_correct := by
    have composite : ¬ Nat.Prime (corner 13117 2) := by change ¬ Nat.Prime 688222757; exact PrattCertificates.composite_of_product (factor := 73) (cofactor := 9427709) rfl (by decide) (by decide)
    exact decide_eq_false composite
  third := false
  third_correct := by
    have composite : ¬ Nat.Prime (corner 13117 3) := by change ¬ Nat.Prime 688196523; exact PrattCertificates.composite_of_product (factor := 3) (cofactor := 229398841) rfl (by decide) (by decide)
    exact decide_eq_false composite
theorem row_count_13117 : row 13117 = 0 := (row_13117.checked).trans rfl
theorem model_count_13117 : modelRow 13117 = 0 := by decide
theorem modelrow_agrees_13117 : modelRow 13117 = row 13117 := model_count_13117.trans row_count_13117.symm
def row_13118 : RowCertificate 13118 where
  first := false
  first_correct := by
    have composite : ¬ Nat.Prime (corner 13118 1) := by change ¬ Nat.Prime 688353933; exact PrattCertificates.composite_of_product (factor := 3) (cofactor := 229451311) rfl (by decide) (by decide)
    exact decide_eq_false composite
  second := false
  second_correct := by
    have composite : ¬ Nat.Prime (corner 13118 2) := by change ¬ Nat.Prime 688327697; exact PrattCertificates.composite_of_product (factor := 109) (cofactor := 6314933) rfl (by decide) (by decide)
    exact decide_eq_false composite
  third := false
  third_correct := by
    have composite : ¬ Nat.Prime (corner 13118 3) := by change ¬ Nat.Prime 688301461; exact PrattCertificates.composite_of_product (factor := 7069) (cofactor := 97369) rfl (by decide) (by decide)
    exact decide_eq_false composite
theorem row_count_13118 : row 13118 = 0 := (row_13118.checked).trans rfl
theorem model_count_13118 : modelRow 13118 = 0 := by decide
theorem modelrow_agrees_13118 : modelRow 13118 = row 13118 := model_count_13118.trans row_count_13118.symm
def row_13119 : RowCertificate 13119 where
  first := false
  first_correct := by
    have composite : ¬ Nat.Prime (corner 13119 1) := by change ¬ Nat.Prime 688458883; exact PrattCertificates.composite_of_product (factor := 7) (cofactor := 98351269) rfl (by decide) (by decide)
    exact decide_eq_false composite
  second := false
  second_correct := by
    have composite : ¬ Nat.Prime (corner 13119 2) := by change ¬ Nat.Prime 688432645; exact PrattCertificates.composite_of_product (factor := 5) (cofactor := 137686529) rfl (by decide) (by decide)
    exact decide_eq_false composite
  third := false
  third_correct := by
    have composite : ¬ Nat.Prime (corner 13119 3) := by change ¬ Nat.Prime 688406407; exact PrattCertificates.composite_of_product (factor := 13) (cofactor := 52954339) rfl (by decide) (by decide)
    exact decide_eq_false composite
theorem row_count_13119 : row 13119 = 0 := (row_13119.checked).trans rfl
theorem model_count_13119 : modelRow 13119 = 0 := by decide
theorem modelrow_agrees_13119 : modelRow 13119 = row 13119 := model_count_13119.trans row_count_13119.symm
def row_13120 : RowCertificate 13120 where
  first := false
  first_correct := by
    have composite : ¬ Nat.Prime (corner 13120 1) := by change ¬ Nat.Prime 688563841; exact PrattCertificates.composite_of_product (factor := 7) (cofactor := 98366263) rfl (by decide) (by decide)
    exact decide_eq_false composite
  second := false
  second_correct := by
    have composite : ¬ Nat.Prime (corner 13120 2) := by change ¬ Nat.Prime 688537601; exact PrattCertificates.composite_of_product (factor := 233) (cofactor := 2955097) rfl (by decide) (by decide)
    exact decide_eq_false composite
  third := false
  third_correct := by
    have composite : ¬ Nat.Prime (corner 13120 3) := by change ¬ Nat.Prime 688511361; exact PrattCertificates.composite_of_product (factor := 3) (cofactor := 229503787) rfl (by decide) (by decide)
    exact decide_eq_false composite
theorem row_count_13120 : row 13120 = 0 := (row_13120.checked).trans rfl
theorem model_count_13120 : modelRow 13120 = 0 := by decide
theorem modelrow_agrees_13120 : modelRow 13120 = row 13120 := model_count_13120.trans row_count_13120.symm
private def atomic_13057 : CountSegment 1 10 13120 13056 13057 5226 5227 := CountSegment.single (first := 13056) (initial := 5226) row_count_13057 (by decide)
private theorem atomic_13057_agreement : RowAgreement 13056 13057 := RowAgreement.single (first := 13056) modelrow_agrees_13057
private def atomic_13058 : CountSegment 1 10 13120 13057 13058 5227 5227 := CountSegment.single (first := 13057) (initial := 5227) row_count_13058 (by decide)
private theorem atomic_13058_agreement : RowAgreement 13057 13058 := RowAgreement.single (first := 13057) modelrow_agrees_13058
private def atomic_13059 : CountSegment 1 10 13120 13058 13059 5227 5228 := CountSegment.single (first := 13058) (initial := 5227) row_count_13059 (by decide)
private theorem atomic_13059_agreement : RowAgreement 13058 13059 := RowAgreement.single (first := 13058) modelrow_agrees_13059
private def atomic_13060 : CountSegment 1 10 13120 13059 13060 5228 5229 := CountSegment.single (first := 13059) (initial := 5228) row_count_13060 (by decide)
private theorem atomic_13060_agreement : RowAgreement 13059 13060 := RowAgreement.single (first := 13059) modelrow_agrees_13060
private def atomic_13061 : CountSegment 1 10 13120 13060 13061 5229 5229 := CountSegment.single (first := 13060) (initial := 5229) row_count_13061 (by decide)
private theorem atomic_13061_agreement : RowAgreement 13060 13061 := RowAgreement.single (first := 13060) modelrow_agrees_13061
private def atomic_13062 : CountSegment 1 10 13120 13061 13062 5229 5231 := CountSegment.single (first := 13061) (initial := 5229) row_count_13062 (by decide)
private theorem atomic_13062_agreement : RowAgreement 13061 13062 := RowAgreement.single (first := 13061) modelrow_agrees_13062
private def atomic_13063 : CountSegment 1 10 13120 13062 13063 5231 5231 := CountSegment.single (first := 13062) (initial := 5231) row_count_13063 (by decide)
private theorem atomic_13063_agreement : RowAgreement 13062 13063 := RowAgreement.single (first := 13062) modelrow_agrees_13063
private def atomic_13064 : CountSegment 1 10 13120 13063 13064 5231 5231 := CountSegment.single (first := 13063) (initial := 5231) row_count_13064 (by decide)
private theorem atomic_13064_agreement : RowAgreement 13063 13064 := RowAgreement.single (first := 13063) modelrow_agrees_13064
private def atomic_13065 : CountSegment 1 10 13120 13064 13065 5231 5232 := CountSegment.single (first := 13064) (initial := 5231) row_count_13065 (by decide)
private theorem atomic_13065_agreement : RowAgreement 13064 13065 := RowAgreement.single (first := 13064) modelrow_agrees_13065
private def atomic_13066 : CountSegment 1 10 13120 13065 13066 5232 5232 := CountSegment.single (first := 13065) (initial := 5232) row_count_13066 (by decide)
private theorem atomic_13066_agreement : RowAgreement 13065 13066 := RowAgreement.single (first := 13065) modelrow_agrees_13066
private def atomic_13067 : CountSegment 1 10 13120 13066 13067 5232 5233 := CountSegment.single (first := 13066) (initial := 5232) row_count_13067 (by decide)
private theorem atomic_13067_agreement : RowAgreement 13066 13067 := RowAgreement.single (first := 13066) modelrow_agrees_13067
private def atomic_13068 : CountSegment 1 10 13120 13067 13068 5233 5233 := CountSegment.single (first := 13067) (initial := 5233) row_count_13068 (by decide)
private theorem atomic_13068_agreement : RowAgreement 13067 13068 := RowAgreement.single (first := 13067) modelrow_agrees_13068
private def atomic_13069 : CountSegment 1 10 13120 13068 13069 5233 5233 := CountSegment.single (first := 13068) (initial := 5233) row_count_13069 (by decide)
private theorem atomic_13069_agreement : RowAgreement 13068 13069 := RowAgreement.single (first := 13068) modelrow_agrees_13069
private def atomic_13070 : CountSegment 1 10 13120 13069 13070 5233 5234 := CountSegment.single (first := 13069) (initial := 5233) row_count_13070 (by decide)
private theorem atomic_13070_agreement : RowAgreement 13069 13070 := RowAgreement.single (first := 13069) modelrow_agrees_13070
private def atomic_13071 : CountSegment 1 10 13120 13070 13071 5234 5234 := CountSegment.single (first := 13070) (initial := 5234) row_count_13071 (by decide)
private theorem atomic_13071_agreement : RowAgreement 13070 13071 := RowAgreement.single (first := 13070) modelrow_agrees_13071
private def atomic_13072 : CountSegment 1 10 13120 13071 13072 5234 5235 := CountSegment.single (first := 13071) (initial := 5234) row_count_13072 (by decide)
private theorem atomic_13072_agreement : RowAgreement 13071 13072 := RowAgreement.single (first := 13071) modelrow_agrees_13072
private def atomic_13073 : CountSegment 1 10 13120 13072 13073 5235 5236 := CountSegment.single (first := 13072) (initial := 5235) row_count_13073 (by decide)
private theorem atomic_13073_agreement : RowAgreement 13072 13073 := RowAgreement.single (first := 13072) modelrow_agrees_13073
private def atomic_13074 : CountSegment 1 10 13120 13073 13074 5236 5236 := CountSegment.single (first := 13073) (initial := 5236) row_count_13074 (by decide)
private theorem atomic_13074_agreement : RowAgreement 13073 13074 := RowAgreement.single (first := 13073) modelrow_agrees_13074
private def atomic_13075 : CountSegment 1 10 13120 13074 13075 5236 5236 := CountSegment.single (first := 13074) (initial := 5236) row_count_13075 (by decide)
private theorem atomic_13075_agreement : RowAgreement 13074 13075 := RowAgreement.single (first := 13074) modelrow_agrees_13075
private def atomic_13076 : CountSegment 1 10 13120 13075 13076 5236 5236 := CountSegment.single (first := 13075) (initial := 5236) row_count_13076 (by decide)
private theorem atomic_13076_agreement : RowAgreement 13075 13076 := RowAgreement.single (first := 13075) modelrow_agrees_13076
private def atomic_13077 : CountSegment 1 10 13120 13076 13077 5236 5237 := CountSegment.single (first := 13076) (initial := 5236) row_count_13077 (by decide)
private theorem atomic_13077_agreement : RowAgreement 13076 13077 := RowAgreement.single (first := 13076) modelrow_agrees_13077
private def atomic_13078 : CountSegment 1 10 13120 13077 13078 5237 5237 := CountSegment.single (first := 13077) (initial := 5237) row_count_13078 (by decide)
private theorem atomic_13078_agreement : RowAgreement 13077 13078 := RowAgreement.single (first := 13077) modelrow_agrees_13078
private def atomic_13079 : CountSegment 1 10 13120 13078 13079 5237 5237 := CountSegment.single (first := 13078) (initial := 5237) row_count_13079 (by decide)
private theorem atomic_13079_agreement : RowAgreement 13078 13079 := RowAgreement.single (first := 13078) modelrow_agrees_13079
private def atomic_13080 : CountSegment 1 10 13120 13079 13080 5237 5238 := CountSegment.single (first := 13079) (initial := 5237) row_count_13080 (by decide)
private theorem atomic_13080_agreement : RowAgreement 13079 13080 := RowAgreement.single (first := 13079) modelrow_agrees_13080
private def atomic_13081 : CountSegment 1 10 13120 13080 13081 5238 5239 := CountSegment.single (first := 13080) (initial := 5238) row_count_13081 (by decide)
private theorem atomic_13081_agreement : RowAgreement 13080 13081 := RowAgreement.single (first := 13080) modelrow_agrees_13081
private def atomic_13082 : CountSegment 1 10 13120 13081 13082 5239 5239 := CountSegment.single (first := 13081) (initial := 5239) row_count_13082 (by decide)
private theorem atomic_13082_agreement : RowAgreement 13081 13082 := RowAgreement.single (first := 13081) modelrow_agrees_13082
private def atomic_13083 : CountSegment 1 10 13120 13082 13083 5239 5239 := CountSegment.single (first := 13082) (initial := 5239) row_count_13083 (by decide)
private theorem atomic_13083_agreement : RowAgreement 13082 13083 := RowAgreement.single (first := 13082) modelrow_agrees_13083
private def atomic_13084 : CountSegment 1 10 13120 13083 13084 5239 5239 := CountSegment.single (first := 13083) (initial := 5239) row_count_13084 (by decide)
private theorem atomic_13084_agreement : RowAgreement 13083 13084 := RowAgreement.single (first := 13083) modelrow_agrees_13084
private def atomic_13085 : CountSegment 1 10 13120 13084 13085 5239 5239 := CountSegment.single (first := 13084) (initial := 5239) row_count_13085 (by decide)
private theorem atomic_13085_agreement : RowAgreement 13084 13085 := RowAgreement.single (first := 13084) modelrow_agrees_13085
private def atomic_13086 : CountSegment 1 10 13120 13085 13086 5239 5240 := CountSegment.single (first := 13085) (initial := 5239) row_count_13086 (by decide)
private theorem atomic_13086_agreement : RowAgreement 13085 13086 := RowAgreement.single (first := 13085) modelrow_agrees_13086
private def atomic_13087 : CountSegment 1 10 13120 13086 13087 5240 5240 := CountSegment.single (first := 13086) (initial := 5240) row_count_13087 (by decide)
private theorem atomic_13087_agreement : RowAgreement 13086 13087 := RowAgreement.single (first := 13086) modelrow_agrees_13087
private def atomic_13088 : CountSegment 1 10 13120 13087 13088 5240 5240 := CountSegment.single (first := 13087) (initial := 5240) row_count_13088 (by decide)
private theorem atomic_13088_agreement : RowAgreement 13087 13088 := RowAgreement.single (first := 13087) modelrow_agrees_13088
private def atomic_13089 : CountSegment 1 10 13120 13088 13089 5240 5240 := CountSegment.single (first := 13088) (initial := 5240) row_count_13089 (by decide)
private theorem atomic_13089_agreement : RowAgreement 13088 13089 := RowAgreement.single (first := 13088) modelrow_agrees_13089
private def atomic_13090 : CountSegment 1 10 13120 13089 13090 5240 5240 := CountSegment.single (first := 13089) (initial := 5240) row_count_13090 (by decide)
private theorem atomic_13090_agreement : RowAgreement 13089 13090 := RowAgreement.single (first := 13089) modelrow_agrees_13090
private def atomic_13091 : CountSegment 1 10 13120 13090 13091 5240 5240 := CountSegment.single (first := 13090) (initial := 5240) row_count_13091 (by decide)
private theorem atomic_13091_agreement : RowAgreement 13090 13091 := RowAgreement.single (first := 13090) modelrow_agrees_13091
private def atomic_13092 : CountSegment 1 10 13120 13091 13092 5240 5241 := CountSegment.single (first := 13091) (initial := 5240) row_count_13092 (by decide)
private theorem atomic_13092_agreement : RowAgreement 13091 13092 := RowAgreement.single (first := 13091) modelrow_agrees_13092
private def atomic_13093 : CountSegment 1 10 13120 13092 13093 5241 5242 := CountSegment.single (first := 13092) (initial := 5241) row_count_13093 (by decide)
private theorem atomic_13093_agreement : RowAgreement 13092 13093 := RowAgreement.single (first := 13092) modelrow_agrees_13093
private def atomic_13094 : CountSegment 1 10 13120 13093 13094 5242 5242 := CountSegment.single (first := 13093) (initial := 5242) row_count_13094 (by decide)
private theorem atomic_13094_agreement : RowAgreement 13093 13094 := RowAgreement.single (first := 13093) modelrow_agrees_13094
private def atomic_13095 : CountSegment 1 10 13120 13094 13095 5242 5242 := CountSegment.single (first := 13094) (initial := 5242) row_count_13095 (by decide)
private theorem atomic_13095_agreement : RowAgreement 13094 13095 := RowAgreement.single (first := 13094) modelrow_agrees_13095
private def atomic_13096 : CountSegment 1 10 13120 13095 13096 5242 5242 := CountSegment.single (first := 13095) (initial := 5242) row_count_13096 (by decide)
private theorem atomic_13096_agreement : RowAgreement 13095 13096 := RowAgreement.single (first := 13095) modelrow_agrees_13096
private def atomic_13097 : CountSegment 1 10 13120 13096 13097 5242 5243 := CountSegment.single (first := 13096) (initial := 5242) row_count_13097 (by decide)
private theorem atomic_13097_agreement : RowAgreement 13096 13097 := RowAgreement.single (first := 13096) modelrow_agrees_13097
private def atomic_13098 : CountSegment 1 10 13120 13097 13098 5243 5244 := CountSegment.single (first := 13097) (initial := 5243) row_count_13098 (by decide)
private theorem atomic_13098_agreement : RowAgreement 13097 13098 := RowAgreement.single (first := 13097) modelrow_agrees_13098
private def atomic_13099 : CountSegment 1 10 13120 13098 13099 5244 5244 := CountSegment.single (first := 13098) (initial := 5244) row_count_13099 (by decide)
private theorem atomic_13099_agreement : RowAgreement 13098 13099 := RowAgreement.single (first := 13098) modelrow_agrees_13099
private def atomic_13100 : CountSegment 1 10 13120 13099 13100 5244 5244 := CountSegment.single (first := 13099) (initial := 5244) row_count_13100 (by decide)
private theorem atomic_13100_agreement : RowAgreement 13099 13100 := RowAgreement.single (first := 13099) modelrow_agrees_13100
private def atomic_13101 : CountSegment 1 10 13120 13100 13101 5244 5245 := CountSegment.single (first := 13100) (initial := 5244) row_count_13101 (by decide)
private theorem atomic_13101_agreement : RowAgreement 13100 13101 := RowAgreement.single (first := 13100) modelrow_agrees_13101
private def atomic_13102 : CountSegment 1 10 13120 13101 13102 5245 5245 := CountSegment.single (first := 13101) (initial := 5245) row_count_13102 (by decide)
private theorem atomic_13102_agreement : RowAgreement 13101 13102 := RowAgreement.single (first := 13101) modelrow_agrees_13102
private def atomic_13103 : CountSegment 1 10 13120 13102 13103 5245 5245 := CountSegment.single (first := 13102) (initial := 5245) row_count_13103 (by decide)
private theorem atomic_13103_agreement : RowAgreement 13102 13103 := RowAgreement.single (first := 13102) modelrow_agrees_13103
private def atomic_13104 : CountSegment 1 10 13120 13103 13104 5245 5246 := CountSegment.single (first := 13103) (initial := 5245) row_count_13104 (by decide)
private theorem atomic_13104_agreement : RowAgreement 13103 13104 := RowAgreement.single (first := 13103) modelrow_agrees_13104
private def atomic_13105 : CountSegment 1 10 13120 13104 13105 5246 5246 := CountSegment.single (first := 13104) (initial := 5246) row_count_13105 (by decide)
private theorem atomic_13105_agreement : RowAgreement 13104 13105 := RowAgreement.single (first := 13104) modelrow_agrees_13105
private def atomic_13106 : CountSegment 1 10 13120 13105 13106 5246 5246 := CountSegment.single (first := 13105) (initial := 5246) row_count_13106 (by decide)
private theorem atomic_13106_agreement : RowAgreement 13105 13106 := RowAgreement.single (first := 13105) modelrow_agrees_13106
private def atomic_13107 : CountSegment 1 10 13120 13106 13107 5246 5246 := CountSegment.single (first := 13106) (initial := 5246) row_count_13107 (by decide)
private theorem atomic_13107_agreement : RowAgreement 13106 13107 := RowAgreement.single (first := 13106) modelrow_agrees_13107
private def atomic_13108 : CountSegment 1 10 13120 13107 13108 5246 5246 := CountSegment.single (first := 13107) (initial := 5246) row_count_13108 (by decide)
private theorem atomic_13108_agreement : RowAgreement 13107 13108 := RowAgreement.single (first := 13107) modelrow_agrees_13108
private def atomic_13109 : CountSegment 1 10 13120 13108 13109 5246 5246 := CountSegment.single (first := 13108) (initial := 5246) row_count_13109 (by decide)
private theorem atomic_13109_agreement : RowAgreement 13108 13109 := RowAgreement.single (first := 13108) modelrow_agrees_13109
private def atomic_13110 : CountSegment 1 10 13120 13109 13110 5246 5246 := CountSegment.single (first := 13109) (initial := 5246) row_count_13110 (by decide)
private theorem atomic_13110_agreement : RowAgreement 13109 13110 := RowAgreement.single (first := 13109) modelrow_agrees_13110
private def atomic_13111 : CountSegment 1 10 13120 13110 13111 5246 5246 := CountSegment.single (first := 13110) (initial := 5246) row_count_13111 (by decide)
private theorem atomic_13111_agreement : RowAgreement 13110 13111 := RowAgreement.single (first := 13110) modelrow_agrees_13111
private def atomic_13112 : CountSegment 1 10 13120 13111 13112 5246 5247 := CountSegment.single (first := 13111) (initial := 5246) row_count_13112 (by decide)
private theorem atomic_13112_agreement : RowAgreement 13111 13112 := RowAgreement.single (first := 13111) modelrow_agrees_13112
private def atomic_13113 : CountSegment 1 10 13120 13112 13113 5247 5247 := CountSegment.single (first := 13112) (initial := 5247) row_count_13113 (by decide)
private theorem atomic_13113_agreement : RowAgreement 13112 13113 := RowAgreement.single (first := 13112) modelrow_agrees_13113
private def atomic_13114 : CountSegment 1 10 13120 13113 13114 5247 5247 := CountSegment.single (first := 13113) (initial := 5247) row_count_13114 (by decide)
private theorem atomic_13114_agreement : RowAgreement 13113 13114 := RowAgreement.single (first := 13113) modelrow_agrees_13114
private def atomic_13115 : CountSegment 1 10 13120 13114 13115 5247 5247 := CountSegment.single (first := 13114) (initial := 5247) row_count_13115 (by decide)
private theorem atomic_13115_agreement : RowAgreement 13114 13115 := RowAgreement.single (first := 13114) modelrow_agrees_13115
private def atomic_13116 : CountSegment 1 10 13120 13115 13116 5247 5248 := CountSegment.single (first := 13115) (initial := 5247) row_count_13116 (by decide)
private theorem atomic_13116_agreement : RowAgreement 13115 13116 := RowAgreement.single (first := 13115) modelrow_agrees_13116
private def atomic_13117 : CountSegment 1 10 13120 13116 13117 5248 5248 := CountSegment.single (first := 13116) (initial := 5248) row_count_13117 (by decide)
private theorem atomic_13117_agreement : RowAgreement 13116 13117 := RowAgreement.single (first := 13116) modelrow_agrees_13117
private def atomic_13118 : CountSegment 1 10 13120 13117 13118 5248 5248 := CountSegment.single (first := 13117) (initial := 5248) row_count_13118 (by decide)
private theorem atomic_13118_agreement : RowAgreement 13117 13118 := RowAgreement.single (first := 13117) modelrow_agrees_13118
private def atomic_13119 : CountSegment 1 10 13120 13118 13119 5248 5248 := CountSegment.single (first := 13118) (initial := 5248) row_count_13119 (by decide)
private theorem atomic_13119_agreement : RowAgreement 13118 13119 := RowAgreement.single (first := 13118) modelrow_agrees_13119
private def atomic_13120 : CountSegment 1 10 13120 13119 13120 5248 5248 := CountSegment.single (first := 13119) (initial := 5248) row_count_13120 (by decide)
private theorem atomic_13120_agreement : RowAgreement 13119 13120 := RowAgreement.single (first := 13119) modelrow_agrees_13120
private def within_102_000 : CountSegment 1 10 13120 13056 13058 5226 5227 := CountSegment.join atomic_13057 atomic_13058
private theorem within_102_000_agreement : RowAgreement 13056 13058 := RowAgreement.join atomic_13057_agreement atomic_13058_agreement
private def within_102_001 : CountSegment 1 10 13120 13058 13060 5227 5229 := CountSegment.join atomic_13059 atomic_13060
private theorem within_102_001_agreement : RowAgreement 13058 13060 := RowAgreement.join atomic_13059_agreement atomic_13060_agreement
private def within_102_002 : CountSegment 1 10 13120 13056 13060 5226 5229 := CountSegment.join within_102_000 within_102_001
private theorem within_102_002_agreement : RowAgreement 13056 13060 := RowAgreement.join within_102_000_agreement within_102_001_agreement
private def within_102_003 : CountSegment 1 10 13120 13060 13062 5229 5231 := CountSegment.join atomic_13061 atomic_13062
private theorem within_102_003_agreement : RowAgreement 13060 13062 := RowAgreement.join atomic_13061_agreement atomic_13062_agreement
private def within_102_004 : CountSegment 1 10 13120 13062 13064 5231 5231 := CountSegment.join atomic_13063 atomic_13064
private theorem within_102_004_agreement : RowAgreement 13062 13064 := RowAgreement.join atomic_13063_agreement atomic_13064_agreement
private def within_102_005 : CountSegment 1 10 13120 13060 13064 5229 5231 := CountSegment.join within_102_003 within_102_004
private theorem within_102_005_agreement : RowAgreement 13060 13064 := RowAgreement.join within_102_003_agreement within_102_004_agreement
private def within_102_006 : CountSegment 1 10 13120 13056 13064 5226 5231 := CountSegment.join within_102_002 within_102_005
private theorem within_102_006_agreement : RowAgreement 13056 13064 := RowAgreement.join within_102_002_agreement within_102_005_agreement
private def within_102_007 : CountSegment 1 10 13120 13064 13066 5231 5232 := CountSegment.join atomic_13065 atomic_13066
private theorem within_102_007_agreement : RowAgreement 13064 13066 := RowAgreement.join atomic_13065_agreement atomic_13066_agreement
private def within_102_008 : CountSegment 1 10 13120 13066 13068 5232 5233 := CountSegment.join atomic_13067 atomic_13068
private theorem within_102_008_agreement : RowAgreement 13066 13068 := RowAgreement.join atomic_13067_agreement atomic_13068_agreement
private def within_102_009 : CountSegment 1 10 13120 13064 13068 5231 5233 := CountSegment.join within_102_007 within_102_008
private theorem within_102_009_agreement : RowAgreement 13064 13068 := RowAgreement.join within_102_007_agreement within_102_008_agreement
private def within_102_010 : CountSegment 1 10 13120 13068 13070 5233 5234 := CountSegment.join atomic_13069 atomic_13070
private theorem within_102_010_agreement : RowAgreement 13068 13070 := RowAgreement.join atomic_13069_agreement atomic_13070_agreement
private def within_102_011 : CountSegment 1 10 13120 13070 13072 5234 5235 := CountSegment.join atomic_13071 atomic_13072
private theorem within_102_011_agreement : RowAgreement 13070 13072 := RowAgreement.join atomic_13071_agreement atomic_13072_agreement
private def within_102_012 : CountSegment 1 10 13120 13068 13072 5233 5235 := CountSegment.join within_102_010 within_102_011
private theorem within_102_012_agreement : RowAgreement 13068 13072 := RowAgreement.join within_102_010_agreement within_102_011_agreement
private def within_102_013 : CountSegment 1 10 13120 13064 13072 5231 5235 := CountSegment.join within_102_009 within_102_012
private theorem within_102_013_agreement : RowAgreement 13064 13072 := RowAgreement.join within_102_009_agreement within_102_012_agreement
private def within_102_014 : CountSegment 1 10 13120 13056 13072 5226 5235 := CountSegment.join within_102_006 within_102_013
private theorem within_102_014_agreement : RowAgreement 13056 13072 := RowAgreement.join within_102_006_agreement within_102_013_agreement
private def within_102_015 : CountSegment 1 10 13120 13072 13074 5235 5236 := CountSegment.join atomic_13073 atomic_13074
private theorem within_102_015_agreement : RowAgreement 13072 13074 := RowAgreement.join atomic_13073_agreement atomic_13074_agreement
private def within_102_016 : CountSegment 1 10 13120 13074 13076 5236 5236 := CountSegment.join atomic_13075 atomic_13076
private theorem within_102_016_agreement : RowAgreement 13074 13076 := RowAgreement.join atomic_13075_agreement atomic_13076_agreement
private def within_102_017 : CountSegment 1 10 13120 13072 13076 5235 5236 := CountSegment.join within_102_015 within_102_016
private theorem within_102_017_agreement : RowAgreement 13072 13076 := RowAgreement.join within_102_015_agreement within_102_016_agreement
private def within_102_018 : CountSegment 1 10 13120 13076 13078 5236 5237 := CountSegment.join atomic_13077 atomic_13078
private theorem within_102_018_agreement : RowAgreement 13076 13078 := RowAgreement.join atomic_13077_agreement atomic_13078_agreement
private def within_102_019 : CountSegment 1 10 13120 13078 13080 5237 5238 := CountSegment.join atomic_13079 atomic_13080
private theorem within_102_019_agreement : RowAgreement 13078 13080 := RowAgreement.join atomic_13079_agreement atomic_13080_agreement
private def within_102_020 : CountSegment 1 10 13120 13076 13080 5236 5238 := CountSegment.join within_102_018 within_102_019
private theorem within_102_020_agreement : RowAgreement 13076 13080 := RowAgreement.join within_102_018_agreement within_102_019_agreement
private def within_102_021 : CountSegment 1 10 13120 13072 13080 5235 5238 := CountSegment.join within_102_017 within_102_020
private theorem within_102_021_agreement : RowAgreement 13072 13080 := RowAgreement.join within_102_017_agreement within_102_020_agreement
private def within_102_022 : CountSegment 1 10 13120 13080 13082 5238 5239 := CountSegment.join atomic_13081 atomic_13082
private theorem within_102_022_agreement : RowAgreement 13080 13082 := RowAgreement.join atomic_13081_agreement atomic_13082_agreement
private def within_102_023 : CountSegment 1 10 13120 13082 13084 5239 5239 := CountSegment.join atomic_13083 atomic_13084
private theorem within_102_023_agreement : RowAgreement 13082 13084 := RowAgreement.join atomic_13083_agreement atomic_13084_agreement
private def within_102_024 : CountSegment 1 10 13120 13080 13084 5238 5239 := CountSegment.join within_102_022 within_102_023
private theorem within_102_024_agreement : RowAgreement 13080 13084 := RowAgreement.join within_102_022_agreement within_102_023_agreement
private def within_102_025 : CountSegment 1 10 13120 13084 13086 5239 5240 := CountSegment.join atomic_13085 atomic_13086
private theorem within_102_025_agreement : RowAgreement 13084 13086 := RowAgreement.join atomic_13085_agreement atomic_13086_agreement
private def within_102_026 : CountSegment 1 10 13120 13086 13088 5240 5240 := CountSegment.join atomic_13087 atomic_13088
private theorem within_102_026_agreement : RowAgreement 13086 13088 := RowAgreement.join atomic_13087_agreement atomic_13088_agreement
private def within_102_027 : CountSegment 1 10 13120 13084 13088 5239 5240 := CountSegment.join within_102_025 within_102_026
private theorem within_102_027_agreement : RowAgreement 13084 13088 := RowAgreement.join within_102_025_agreement within_102_026_agreement
private def within_102_028 : CountSegment 1 10 13120 13080 13088 5238 5240 := CountSegment.join within_102_024 within_102_027
private theorem within_102_028_agreement : RowAgreement 13080 13088 := RowAgreement.join within_102_024_agreement within_102_027_agreement
private def within_102_029 : CountSegment 1 10 13120 13072 13088 5235 5240 := CountSegment.join within_102_021 within_102_028
private theorem within_102_029_agreement : RowAgreement 13072 13088 := RowAgreement.join within_102_021_agreement within_102_028_agreement
private def within_102_030 : CountSegment 1 10 13120 13056 13088 5226 5240 := CountSegment.join within_102_014 within_102_029
private theorem within_102_030_agreement : RowAgreement 13056 13088 := RowAgreement.join within_102_014_agreement within_102_029_agreement
private def within_102_031 : CountSegment 1 10 13120 13088 13090 5240 5240 := CountSegment.join atomic_13089 atomic_13090
private theorem within_102_031_agreement : RowAgreement 13088 13090 := RowAgreement.join atomic_13089_agreement atomic_13090_agreement
private def within_102_032 : CountSegment 1 10 13120 13090 13092 5240 5241 := CountSegment.join atomic_13091 atomic_13092
private theorem within_102_032_agreement : RowAgreement 13090 13092 := RowAgreement.join atomic_13091_agreement atomic_13092_agreement
private def within_102_033 : CountSegment 1 10 13120 13088 13092 5240 5241 := CountSegment.join within_102_031 within_102_032
private theorem within_102_033_agreement : RowAgreement 13088 13092 := RowAgreement.join within_102_031_agreement within_102_032_agreement
private def within_102_034 : CountSegment 1 10 13120 13092 13094 5241 5242 := CountSegment.join atomic_13093 atomic_13094
private theorem within_102_034_agreement : RowAgreement 13092 13094 := RowAgreement.join atomic_13093_agreement atomic_13094_agreement
private def within_102_035 : CountSegment 1 10 13120 13094 13096 5242 5242 := CountSegment.join atomic_13095 atomic_13096
private theorem within_102_035_agreement : RowAgreement 13094 13096 := RowAgreement.join atomic_13095_agreement atomic_13096_agreement
private def within_102_036 : CountSegment 1 10 13120 13092 13096 5241 5242 := CountSegment.join within_102_034 within_102_035
private theorem within_102_036_agreement : RowAgreement 13092 13096 := RowAgreement.join within_102_034_agreement within_102_035_agreement
private def within_102_037 : CountSegment 1 10 13120 13088 13096 5240 5242 := CountSegment.join within_102_033 within_102_036
private theorem within_102_037_agreement : RowAgreement 13088 13096 := RowAgreement.join within_102_033_agreement within_102_036_agreement
private def within_102_038 : CountSegment 1 10 13120 13096 13098 5242 5244 := CountSegment.join atomic_13097 atomic_13098
private theorem within_102_038_agreement : RowAgreement 13096 13098 := RowAgreement.join atomic_13097_agreement atomic_13098_agreement
private def within_102_039 : CountSegment 1 10 13120 13098 13100 5244 5244 := CountSegment.join atomic_13099 atomic_13100
private theorem within_102_039_agreement : RowAgreement 13098 13100 := RowAgreement.join atomic_13099_agreement atomic_13100_agreement
private def within_102_040 : CountSegment 1 10 13120 13096 13100 5242 5244 := CountSegment.join within_102_038 within_102_039
private theorem within_102_040_agreement : RowAgreement 13096 13100 := RowAgreement.join within_102_038_agreement within_102_039_agreement
private def within_102_041 : CountSegment 1 10 13120 13100 13102 5244 5245 := CountSegment.join atomic_13101 atomic_13102
private theorem within_102_041_agreement : RowAgreement 13100 13102 := RowAgreement.join atomic_13101_agreement atomic_13102_agreement
private def within_102_042 : CountSegment 1 10 13120 13102 13104 5245 5246 := CountSegment.join atomic_13103 atomic_13104
private theorem within_102_042_agreement : RowAgreement 13102 13104 := RowAgreement.join atomic_13103_agreement atomic_13104_agreement
private def within_102_043 : CountSegment 1 10 13120 13100 13104 5244 5246 := CountSegment.join within_102_041 within_102_042
private theorem within_102_043_agreement : RowAgreement 13100 13104 := RowAgreement.join within_102_041_agreement within_102_042_agreement
private def within_102_044 : CountSegment 1 10 13120 13096 13104 5242 5246 := CountSegment.join within_102_040 within_102_043
private theorem within_102_044_agreement : RowAgreement 13096 13104 := RowAgreement.join within_102_040_agreement within_102_043_agreement
private def within_102_045 : CountSegment 1 10 13120 13088 13104 5240 5246 := CountSegment.join within_102_037 within_102_044
private theorem within_102_045_agreement : RowAgreement 13088 13104 := RowAgreement.join within_102_037_agreement within_102_044_agreement
private def within_102_046 : CountSegment 1 10 13120 13104 13106 5246 5246 := CountSegment.join atomic_13105 atomic_13106
private theorem within_102_046_agreement : RowAgreement 13104 13106 := RowAgreement.join atomic_13105_agreement atomic_13106_agreement
private def within_102_047 : CountSegment 1 10 13120 13106 13108 5246 5246 := CountSegment.join atomic_13107 atomic_13108
private theorem within_102_047_agreement : RowAgreement 13106 13108 := RowAgreement.join atomic_13107_agreement atomic_13108_agreement
private def within_102_048 : CountSegment 1 10 13120 13104 13108 5246 5246 := CountSegment.join within_102_046 within_102_047
private theorem within_102_048_agreement : RowAgreement 13104 13108 := RowAgreement.join within_102_046_agreement within_102_047_agreement
private def within_102_049 : CountSegment 1 10 13120 13108 13110 5246 5246 := CountSegment.join atomic_13109 atomic_13110
private theorem within_102_049_agreement : RowAgreement 13108 13110 := RowAgreement.join atomic_13109_agreement atomic_13110_agreement
private def within_102_050 : CountSegment 1 10 13120 13110 13112 5246 5247 := CountSegment.join atomic_13111 atomic_13112
private theorem within_102_050_agreement : RowAgreement 13110 13112 := RowAgreement.join atomic_13111_agreement atomic_13112_agreement
private def within_102_051 : CountSegment 1 10 13120 13108 13112 5246 5247 := CountSegment.join within_102_049 within_102_050
private theorem within_102_051_agreement : RowAgreement 13108 13112 := RowAgreement.join within_102_049_agreement within_102_050_agreement
private def within_102_052 : CountSegment 1 10 13120 13104 13112 5246 5247 := CountSegment.join within_102_048 within_102_051
private theorem within_102_052_agreement : RowAgreement 13104 13112 := RowAgreement.join within_102_048_agreement within_102_051_agreement
private def within_102_053 : CountSegment 1 10 13120 13112 13114 5247 5247 := CountSegment.join atomic_13113 atomic_13114
private theorem within_102_053_agreement : RowAgreement 13112 13114 := RowAgreement.join atomic_13113_agreement atomic_13114_agreement
private def within_102_054 : CountSegment 1 10 13120 13114 13116 5247 5248 := CountSegment.join atomic_13115 atomic_13116
private theorem within_102_054_agreement : RowAgreement 13114 13116 := RowAgreement.join atomic_13115_agreement atomic_13116_agreement
private def within_102_055 : CountSegment 1 10 13120 13112 13116 5247 5248 := CountSegment.join within_102_053 within_102_054
private theorem within_102_055_agreement : RowAgreement 13112 13116 := RowAgreement.join within_102_053_agreement within_102_054_agreement
private def within_102_056 : CountSegment 1 10 13120 13116 13118 5248 5248 := CountSegment.join atomic_13117 atomic_13118
private theorem within_102_056_agreement : RowAgreement 13116 13118 := RowAgreement.join atomic_13117_agreement atomic_13118_agreement
private def within_102_057 : CountSegment 1 10 13120 13118 13120 5248 5248 := CountSegment.join atomic_13119 atomic_13120
private theorem within_102_057_agreement : RowAgreement 13118 13120 := RowAgreement.join atomic_13119_agreement atomic_13120_agreement
private def within_102_058 : CountSegment 1 10 13120 13116 13120 5248 5248 := CountSegment.join within_102_056 within_102_057
private theorem within_102_058_agreement : RowAgreement 13116 13120 := RowAgreement.join within_102_056_agreement within_102_057_agreement
private def within_102_059 : CountSegment 1 10 13120 13112 13120 5247 5248 := CountSegment.join within_102_055 within_102_058
private theorem within_102_059_agreement : RowAgreement 13112 13120 := RowAgreement.join within_102_055_agreement within_102_058_agreement
private def within_102_060 : CountSegment 1 10 13120 13104 13120 5246 5248 := CountSegment.join within_102_052 within_102_059
private theorem within_102_060_agreement : RowAgreement 13104 13120 := RowAgreement.join within_102_052_agreement within_102_059_agreement
private def within_102_061 : CountSegment 1 10 13120 13088 13120 5240 5248 := CountSegment.join within_102_045 within_102_060
private theorem within_102_061_agreement : RowAgreement 13088 13120 := RowAgreement.join within_102_045_agreement within_102_060_agreement
private def within_102_062 : CountSegment 1 10 13120 13056 13120 5226 5248 := CountSegment.join within_102_030 within_102_061
private theorem within_102_062_agreement : RowAgreement 13056 13120 := RowAgreement.join within_102_030_agreement within_102_061_agreement
def segment_102 : CountSegment 1 10 13120 13056 13120 5226 5248 := within_102_062
theorem agreement_102 : RowAgreement 13056 13120 := within_102_062_agreement
end ProjectEulerSolutions.P58Numerical
