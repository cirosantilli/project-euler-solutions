import ProjectEulerSolutions.Lib.SpiralPrimeSegments
import ProjectEulerSolutions.Lib.SpiralPrimeCertificates.All
import ProjectEulerSolutions.Lib.MillerRabin
import ProjectEulerSolutions.Lib.SpiralMillerAgreement
/-! Finite spiral row and counter certificates. Python execution is checked separately. -/
namespace ProjectEulerSolutions.P58Numerical
open SpiralPrimes
set_option Elab.async false
def row_3713 : RowCertificate 3713 where
  first := false
  first_correct := by
    have composite : ¬ Nat.Prime (corner 3713 1) := by change ¬ Nat.Prime 55152903; exact PrattCertificates.composite_of_product (factor := 3) (cofactor := 18384301) rfl (by decide) (by decide)
    exact decide_eq_false composite
  second := false
  second_correct := by
    have composite : ¬ Nat.Prime (corner 3713 2) := by change ¬ Nat.Prime 55145477; exact PrattCertificates.composite_of_product (factor := 1553) (cofactor := 35509) rfl (by decide) (by decide)
    exact decide_eq_false composite
  third := true
  third_correct := by
    have prime : Nat.Prime (corner 3713 3) := by change Nat.Prime 55138051; exact ProjectEulerSolutions.P58PrimeNodes.p_55138051.prime
    exact decide_eq_true prime
theorem row_count_3713 : row 3713 = 1 := (row_3713.checked).trans rfl
theorem model_count_3713 : modelRow 3713 = 1 := by decide
theorem modelrow_agrees_3713 : modelRow 3713 = row 3713 := model_count_3713.trans row_count_3713.symm
def row_3714 : RowCertificate 3714 where
  first := false
  first_correct := by
    have composite : ¬ Nat.Prime (corner 3714 1) := by change ¬ Nat.Prime 55182613; exact PrattCertificates.composite_of_product (factor := 61) (cofactor := 904633) rfl (by decide) (by decide)
    exact decide_eq_false composite
  second := false
  second_correct := by
    have composite : ¬ Nat.Prime (corner 3714 2) := by change ¬ Nat.Prime 55175185; exact PrattCertificates.composite_of_product (factor := 5) (cofactor := 11035037) rfl (by decide) (by decide)
    exact decide_eq_false composite
  third := true
  third_correct := by
    have prime : Nat.Prime (corner 3714 3) := by change Nat.Prime 55167757; exact ProjectEulerSolutions.P58PrimeNodes.p_55167757.prime
    exact decide_eq_true prime
theorem row_count_3714 : row 3714 = 1 := (row_3714.checked).trans rfl
theorem model_count_3714 : modelRow 3714 = 1 := by decide
theorem modelrow_agrees_3714 : modelRow 3714 = row 3714 := model_count_3714.trans row_count_3714.symm
def row_3715 : RowCertificate 3715 where
  first := true
  first_correct := by
    have prime : Nat.Prime (corner 3715 1) := by change Nat.Prime 55212331; exact ProjectEulerSolutions.P58PrimeNodes.p_55212331.prime
    exact decide_eq_true prime
  second := false
  second_correct := by
    have composite : ¬ Nat.Prime (corner 3715 2) := by change ¬ Nat.Prime 55204901; exact PrattCertificates.composite_of_product (factor := 41) (cofactor := 1346461) rfl (by decide) (by decide)
    exact decide_eq_false composite
  third := false
  third_correct := by
    have composite : ¬ Nat.Prime (corner 3715 3) := by change ¬ Nat.Prime 55197471; exact PrattCertificates.composite_of_product (factor := 3) (cofactor := 18399157) rfl (by decide) (by decide)
    exact decide_eq_false composite
theorem row_count_3715 : row 3715 = 1 := (row_3715.checked).trans rfl
theorem model_count_3715 : modelRow 3715 = 1 := by decide
theorem modelrow_agrees_3715 : modelRow 3715 = row 3715 := model_count_3715.trans row_count_3715.symm
def row_3716 : RowCertificate 3716 where
  first := false
  first_correct := by
    have composite : ¬ Nat.Prime (corner 3716 1) := by change ¬ Nat.Prime 55242057; exact PrattCertificates.composite_of_product (factor := 3) (cofactor := 18414019) rfl (by decide) (by decide)
    exact decide_eq_false composite
  second := false
  second_correct := by
    have composite : ¬ Nat.Prime (corner 3716 2) := by change ¬ Nat.Prime 55234625; exact PrattCertificates.composite_of_product (factor := 5) (cofactor := 11046925) rfl (by decide) (by decide)
    exact decide_eq_false composite
  third := false
  third_correct := by
    have composite : ¬ Nat.Prime (corner 3716 3) := by change ¬ Nat.Prime 55227193; exact PrattCertificates.composite_of_product (factor := 7) (cofactor := 7889599) rfl (by decide) (by decide)
    exact decide_eq_false composite
theorem row_count_3716 : row 3716 = 0 := (row_3716.checked).trans rfl
theorem model_count_3716 : modelRow 3716 = 0 := by decide
theorem modelrow_agrees_3716 : modelRow 3716 = row 3716 := model_count_3716.trans row_count_3716.symm
def row_3717 : RowCertificate 3717 where
  first := false
  first_correct := by
    have composite : ¬ Nat.Prime (corner 3717 1) := by change ¬ Nat.Prime 55271791; exact PrattCertificates.composite_of_product (factor := 31) (cofactor := 1782961) rfl (by decide) (by decide)
    exact decide_eq_false composite
  second := true
  second_correct := by
    have prime : Nat.Prime (corner 3717 2) := by change Nat.Prime 55264357; exact ProjectEulerSolutions.P58PrimeNodes.p_55264357.prime
    exact decide_eq_true prime
  third := false
  third_correct := by
    have composite : ¬ Nat.Prime (corner 3717 3) := by change ¬ Nat.Prime 55256923; exact PrattCertificates.composite_of_product (factor := 97) (cofactor := 569659) rfl (by decide) (by decide)
    exact decide_eq_false composite
theorem row_count_3717 : row 3717 = 1 := (row_3717.checked).trans rfl
theorem model_count_3717 : modelRow 3717 = 1 := by decide
theorem modelrow_agrees_3717 : modelRow 3717 = row 3717 := model_count_3717.trans row_count_3717.symm
def row_3718 : RowCertificate 3718 where
  first := false
  first_correct := by
    have composite : ¬ Nat.Prime (corner 3718 1) := by change ¬ Nat.Prime 55301533; exact PrattCertificates.composite_of_product (factor := 7) (cofactor := 7900219) rfl (by decide) (by decide)
    exact decide_eq_false composite
  second := false
  second_correct := by
    have composite : ¬ Nat.Prime (corner 3718 2) := by change ¬ Nat.Prime 55294097; exact PrattCertificates.composite_of_product (factor := 29) (cofactor := 1906693) rfl (by decide) (by decide)
    exact decide_eq_false composite
  third := false
  third_correct := by
    have composite : ¬ Nat.Prime (corner 3718 3) := by change ¬ Nat.Prime 55286661; exact PrattCertificates.composite_of_product (factor := 3) (cofactor := 18428887) rfl (by decide) (by decide)
    exact decide_eq_false composite
theorem row_count_3718 : row 3718 = 0 := (row_3718.checked).trans rfl
theorem model_count_3718 : modelRow 3718 = 0 := by decide
theorem modelrow_agrees_3718 : modelRow 3718 = row 3718 := model_count_3718.trans row_count_3718.symm
def row_3719 : RowCertificate 3719 where
  first := false
  first_correct := by
    have composite : ¬ Nat.Prime (corner 3719 1) := by change ¬ Nat.Prime 55331283; exact PrattCertificates.composite_of_product (factor := 3) (cofactor := 18443761) rfl (by decide) (by decide)
    exact decide_eq_false composite
  second := false
  second_correct := by
    have composite : ¬ Nat.Prime (corner 3719 2) := by change ¬ Nat.Prime 55323845; exact PrattCertificates.composite_of_product (factor := 5) (cofactor := 11064769) rfl (by decide) (by decide)
    exact decide_eq_false composite
  third := false
  third_correct := by
    have composite : ¬ Nat.Prime (corner 3719 3) := by change ¬ Nat.Prime 55316407; exact PrattCertificates.composite_of_product (factor := 73) (cofactor := 757759) rfl (by decide) (by decide)
    exact decide_eq_false composite
theorem row_count_3719 : row 3719 = 0 := (row_3719.checked).trans rfl
theorem model_count_3719 : modelRow 3719 = 0 := by decide
theorem modelrow_agrees_3719 : modelRow 3719 = row 3719 := model_count_3719.trans row_count_3719.symm
def row_3720 : RowCertificate 3720 where
  first := false
  first_correct := by
    have composite : ¬ Nat.Prime (corner 3720 1) := by change ¬ Nat.Prime 55361041; exact PrattCertificates.composite_of_product (factor := 19) (cofactor := 2913739) rfl (by decide) (by decide)
    exact decide_eq_false composite
  second := false
  second_correct := by
    have composite : ¬ Nat.Prime (corner 3720 2) := by change ¬ Nat.Prime 55353601; exact PrattCertificates.composite_of_product (factor := 181) (cofactor := 305821) rfl (by decide) (by decide)
    exact decide_eq_false composite
  third := false
  third_correct := by
    have composite : ¬ Nat.Prime (corner 3720 3) := by change ¬ Nat.Prime 55346161; exact PrattCertificates.composite_of_product (factor := 13) (cofactor := 4257397) rfl (by decide) (by decide)
    exact decide_eq_false composite
theorem row_count_3720 : row 3720 = 0 := (row_3720.checked).trans rfl
theorem model_count_3720 : modelRow 3720 = 0 := by decide
theorem modelrow_agrees_3720 : modelRow 3720 = row 3720 := model_count_3720.trans row_count_3720.symm
def row_3721 : RowCertificate 3721 where
  first := false
  first_correct := by
    have composite : ¬ Nat.Prime (corner 3721 1) := by change ¬ Nat.Prime 55390807; exact PrattCertificates.composite_of_product (factor := 193) (cofactor := 286999) rfl (by decide) (by decide)
    exact decide_eq_false composite
  second := false
  second_correct := by
    have composite : ¬ Nat.Prime (corner 3721 2) := by change ¬ Nat.Prime 55383365; exact PrattCertificates.composite_of_product (factor := 5) (cofactor := 11076673) rfl (by decide) (by decide)
    exact decide_eq_false composite
  third := false
  third_correct := by
    have composite : ¬ Nat.Prime (corner 3721 3) := by change ¬ Nat.Prime 55375923; exact PrattCertificates.composite_of_product (factor := 3) (cofactor := 18458641) rfl (by decide) (by decide)
    exact decide_eq_false composite
theorem row_count_3721 : row 3721 = 0 := (row_3721.checked).trans rfl
theorem model_count_3721 : modelRow 3721 = 0 := by decide
theorem modelrow_agrees_3721 : modelRow 3721 = row 3721 := model_count_3721.trans row_count_3721.symm
def row_3722 : RowCertificate 3722 where
  first := false
  first_correct := by
    have composite : ¬ Nat.Prime (corner 3722 1) := by change ¬ Nat.Prime 55420581; exact PrattCertificates.composite_of_product (factor := 3) (cofactor := 18473527) rfl (by decide) (by decide)
    exact decide_eq_false composite
  second := false
  second_correct := by
    have composite : ¬ Nat.Prime (corner 3722 2) := by change ¬ Nat.Prime 55413137; exact PrattCertificates.composite_of_product (factor := 13) (cofactor := 4262549) rfl (by decide) (by decide)
    exact decide_eq_false composite
  third := false
  third_correct := by
    have composite : ¬ Nat.Prime (corner 3722 3) := by change ¬ Nat.Prime 55405693; exact PrattCertificates.composite_of_product (factor := 7) (cofactor := 7915099) rfl (by decide) (by decide)
    exact decide_eq_false composite
theorem row_count_3722 : row 3722 = 0 := (row_3722.checked).trans rfl
theorem model_count_3722 : modelRow 3722 = 0 := by decide
theorem modelrow_agrees_3722 : modelRow 3722 = row 3722 := model_count_3722.trans row_count_3722.symm
def row_3723 : RowCertificate 3723 where
  first := false
  first_correct := by
    have composite : ¬ Nat.Prime (corner 3723 1) := by change ¬ Nat.Prime 55450363; exact PrattCertificates.composite_of_product (factor := 1171) (cofactor := 47353) rfl (by decide) (by decide)
    exact decide_eq_false composite
  second := false
  second_correct := by
    have composite : ¬ Nat.Prime (corner 3723 2) := by change ¬ Nat.Prime 55442917; exact PrattCertificates.composite_of_product (factor := 193) (cofactor := 287269) rfl (by decide) (by decide)
    exact decide_eq_false composite
  third := false
  third_correct := by
    have composite : ¬ Nat.Prime (corner 3723 3) := by change ¬ Nat.Prime 55435471; exact PrattCertificates.composite_of_product (factor := 7) (cofactor := 7919353) rfl (by decide) (by decide)
    exact decide_eq_false composite
theorem row_count_3723 : row 3723 = 0 := (row_3723.checked).trans rfl
theorem model_count_3723 : modelRow 3723 = 0 := by decide
theorem modelrow_agrees_3723 : modelRow 3723 = row 3723 := model_count_3723.trans row_count_3723.symm
def row_3724 : RowCertificate 3724 where
  first := true
  first_correct := by
    have prime : Nat.Prime (corner 3724 1) := by change Nat.Prime 55480153; exact ProjectEulerSolutions.P58PrimeNodes.p_55480153.prime
    exact decide_eq_true prime
  second := false
  second_correct := by
    have composite : ¬ Nat.Prime (corner 3724 2) := by change ¬ Nat.Prime 55472705; exact PrattCertificates.composite_of_product (factor := 5) (cofactor := 11094541) rfl (by decide) (by decide)
    exact decide_eq_false composite
  third := false
  third_correct := by
    have composite : ¬ Nat.Prime (corner 3724 3) := by change ¬ Nat.Prime 55465257; exact PrattCertificates.composite_of_product (factor := 3) (cofactor := 18488419) rfl (by decide) (by decide)
    exact decide_eq_false composite
theorem row_count_3724 : row 3724 = 1 := (row_3724.checked).trans rfl
theorem model_count_3724 : modelRow 3724 = 1 := by decide
theorem modelrow_agrees_3724 : modelRow 3724 = row 3724 := model_count_3724.trans row_count_3724.symm
def row_3725 : RowCertificate 3725 where
  first := false
  first_correct := by
    have composite : ¬ Nat.Prime (corner 3725 1) := by change ¬ Nat.Prime 55509951; exact PrattCertificates.composite_of_product (factor := 3) (cofactor := 18503317) rfl (by decide) (by decide)
    exact decide_eq_false composite
  second := false
  second_correct := by
    have composite : ¬ Nat.Prime (corner 3725 2) := by change ¬ Nat.Prime 55502501; exact PrattCertificates.composite_of_product (factor := 17) (cofactor := 3264853) rfl (by decide) (by decide)
    exact decide_eq_false composite
  third := false
  third_correct := by
    have composite : ¬ Nat.Prime (corner 3725 3) := by change ¬ Nat.Prime 55495051; exact PrattCertificates.composite_of_product (factor := 79) (cofactor := 702469) rfl (by decide) (by decide)
    exact decide_eq_false composite
theorem row_count_3725 : row 3725 = 0 := (row_3725.checked).trans rfl
theorem model_count_3725 : modelRow 3725 = 0 := by decide
theorem modelrow_agrees_3725 : modelRow 3725 = row 3725 := model_count_3725.trans row_count_3725.symm
def row_3726 : RowCertificate 3726 where
  first := false
  first_correct := by
    have composite : ¬ Nat.Prime (corner 3726 1) := by change ¬ Nat.Prime 55539757; exact PrattCertificates.composite_of_product (factor := 7) (cofactor := 7934251) rfl (by decide) (by decide)
    exact decide_eq_false composite
  second := false
  second_correct := by
    have composite : ¬ Nat.Prime (corner 3726 2) := by change ¬ Nat.Prime 55532305; exact PrattCertificates.composite_of_product (factor := 5) (cofactor := 11106461) rfl (by decide) (by decide)
    exact decide_eq_false composite
  third := false
  third_correct := by
    have composite : ¬ Nat.Prime (corner 3726 3) := by change ¬ Nat.Prime 55524853; exact PrattCertificates.composite_of_product (factor := 349) (cofactor := 159097) rfl (by decide) (by decide)
    exact decide_eq_false composite
theorem row_count_3726 : row 3726 = 0 := (row_3726.checked).trans rfl
theorem model_count_3726 : modelRow 3726 = 0 := by decide
theorem modelrow_agrees_3726 : modelRow 3726 = row 3726 := model_count_3726.trans row_count_3726.symm
def row_3727 : RowCertificate 3727 where
  first := false
  first_correct := by
    have composite : ¬ Nat.Prime (corner 3727 1) := by change ¬ Nat.Prime 55569571; exact PrattCertificates.composite_of_product (factor := 73) (cofactor := 761227) rfl (by decide) (by decide)
    exact decide_eq_false composite
  second := false
  second_correct := by
    have composite : ¬ Nat.Prime (corner 3727 2) := by change ¬ Nat.Prime 55562117; exact PrattCertificates.composite_of_product (factor := 13) (cofactor := 4274009) rfl (by decide) (by decide)
    exact decide_eq_false composite
  third := false
  third_correct := by
    have composite : ¬ Nat.Prime (corner 3727 3) := by change ¬ Nat.Prime 55554663; exact PrattCertificates.composite_of_product (factor := 3) (cofactor := 18518221) rfl (by decide) (by decide)
    exact decide_eq_false composite
theorem row_count_3727 : row 3727 = 0 := (row_3727.checked).trans rfl
theorem model_count_3727 : modelRow 3727 = 0 := by decide
theorem modelrow_agrees_3727 : modelRow 3727 = row 3727 := model_count_3727.trans row_count_3727.symm
def row_3728 : RowCertificate 3728 where
  first := false
  first_correct := by
    have composite : ¬ Nat.Prime (corner 3728 1) := by change ¬ Nat.Prime 55599393; exact PrattCertificates.composite_of_product (factor := 3) (cofactor := 18533131) rfl (by decide) (by decide)
    exact decide_eq_false composite
  second := true
  second_correct := by
    have prime : Nat.Prime (corner 3728 2) := by change Nat.Prime 55591937; exact ProjectEulerSolutions.P58PrimeNodes.p_55591937.prime
    exact decide_eq_true prime
  third := false
  third_correct := by
    have composite : ¬ Nat.Prime (corner 3728 3) := by change ¬ Nat.Prime 55584481; exact PrattCertificates.composite_of_product (factor := 19) (cofactor := 2925499) rfl (by decide) (by decide)
    exact decide_eq_false composite
theorem row_count_3728 : row 3728 = 1 := (row_3728.checked).trans rfl
theorem model_count_3728 : modelRow 3728 = 1 := by decide
theorem modelrow_agrees_3728 : modelRow 3728 = row 3728 := model_count_3728.trans row_count_3728.symm
def row_3729 : RowCertificate 3729 where
  first := false
  first_correct := by
    have composite : ¬ Nat.Prime (corner 3729 1) := by change ¬ Nat.Prime 55629223; exact PrattCertificates.composite_of_product (factor := 13) (cofactor := 4279171) rfl (by decide) (by decide)
    exact decide_eq_false composite
  second := false
  second_correct := by
    have composite : ¬ Nat.Prime (corner 3729 2) := by change ¬ Nat.Prime 55621765; exact PrattCertificates.composite_of_product (factor := 5) (cofactor := 11124353) rfl (by decide) (by decide)
    exact decide_eq_false composite
  third := false
  third_correct := by
    have composite : ¬ Nat.Prime (corner 3729 3) := by change ¬ Nat.Prime 55614307; exact PrattCertificates.composite_of_product (factor := 7) (cofactor := 7944901) rfl (by decide) (by decide)
    exact decide_eq_false composite
theorem row_count_3729 : row 3729 = 0 := (row_3729.checked).trans rfl
theorem model_count_3729 : modelRow 3729 = 0 := by decide
theorem modelrow_agrees_3729 : modelRow 3729 = row 3729 := model_count_3729.trans row_count_3729.symm
def row_3730 : RowCertificate 3730 where
  first := true
  first_correct := by
    have prime : Nat.Prime (corner 3730 1) := by change Nat.Prime 55659061; exact ProjectEulerSolutions.P58PrimeNodes.p_55659061.prime
    exact decide_eq_true prime
  second := true
  second_correct := by
    have prime : Nat.Prime (corner 3730 2) := by change Nat.Prime 55651601; exact ProjectEulerSolutions.P58PrimeNodes.p_55651601.prime
    exact decide_eq_true prime
  third := false
  third_correct := by
    have composite : ¬ Nat.Prime (corner 3730 3) := by change ¬ Nat.Prime 55644141; exact PrattCertificates.composite_of_product (factor := 3) (cofactor := 18548047) rfl (by decide) (by decide)
    exact decide_eq_false composite
theorem row_count_3730 : row 3730 = 2 := (row_3730.checked).trans rfl
theorem model_count_3730 : modelRow 3730 = 2 := by decide
theorem modelrow_agrees_3730 : modelRow 3730 = row 3730 := model_count_3730.trans row_count_3730.symm
def row_3731 : RowCertificate 3731 where
  first := false
  first_correct := by
    have composite : ¬ Nat.Prime (corner 3731 1) := by change ¬ Nat.Prime 55688907; exact PrattCertificates.composite_of_product (factor := 3) (cofactor := 18562969) rfl (by decide) (by decide)
    exact decide_eq_false composite
  second := false
  second_correct := by
    have composite : ¬ Nat.Prime (corner 3731 2) := by change ¬ Nat.Prime 55681445; exact PrattCertificates.composite_of_product (factor := 5) (cofactor := 11136289) rfl (by decide) (by decide)
    exact decide_eq_false composite
  third := false
  third_correct := by
    have composite : ¬ Nat.Prime (corner 3731 3) := by change ¬ Nat.Prime 55673983; exact PrattCertificates.composite_of_product (factor := 2251) (cofactor := 24733) rfl (by decide) (by decide)
    exact decide_eq_false composite
theorem row_count_3731 : row 3731 = 0 := (row_3731.checked).trans rfl
theorem model_count_3731 : modelRow 3731 = 0 := by decide
theorem modelrow_agrees_3731 : modelRow 3731 = row 3731 := model_count_3731.trans row_count_3731.symm
def row_3732 : RowCertificate 3732 where
  first := false
  first_correct := by
    have composite : ¬ Nat.Prime (corner 3732 1) := by change ¬ Nat.Prime 55718761; exact PrattCertificates.composite_of_product (factor := 7) (cofactor := 7959823) rfl (by decide) (by decide)
    exact decide_eq_false composite
  second := false
  second_correct := by
    have composite : ¬ Nat.Prime (corner 3732 2) := by change ¬ Nat.Prime 55711297; exact PrattCertificates.composite_of_product (factor := 101) (cofactor := 551597) rfl (by decide) (by decide)
    exact decide_eq_false composite
  third := false
  third_correct := by
    have composite : ¬ Nat.Prime (corner 3732 3) := by change ¬ Nat.Prime 55703833; exact PrattCertificates.composite_of_product (factor := 37) (cofactor := 1505509) rfl (by decide) (by decide)
    exact decide_eq_false composite
theorem row_count_3732 : row 3732 = 0 := (row_3732.checked).trans rfl
theorem model_count_3732 : modelRow 3732 = 0 := by decide
theorem modelrow_agrees_3732 : modelRow 3732 = row 3732 := model_count_3732.trans row_count_3732.symm
def row_3733 : RowCertificate 3733 where
  first := false
  first_correct := by
    have composite : ¬ Nat.Prime (corner 3733 1) := by change ¬ Nat.Prime 55748623; exact PrattCertificates.composite_of_product (factor := 7) (cofactor := 7964089) rfl (by decide) (by decide)
    exact decide_eq_false composite
  second := true
  second_correct := by
    have prime : Nat.Prime (corner 3733 2) := by change Nat.Prime 55741157; exact ProjectEulerSolutions.P58PrimeNodes.p_55741157.prime
    exact decide_eq_true prime
  third := false
  third_correct := by
    have composite : ¬ Nat.Prime (corner 3733 3) := by change ¬ Nat.Prime 55733691; exact PrattCertificates.composite_of_product (factor := 3) (cofactor := 18577897) rfl (by decide) (by decide)
    exact decide_eq_false composite
theorem row_count_3733 : row 3733 = 1 := (row_3733.checked).trans rfl
theorem model_count_3733 : modelRow 3733 = 1 := by decide
theorem modelrow_agrees_3733 : modelRow 3733 = row 3733 := model_count_3733.trans row_count_3733.symm
def row_3734 : RowCertificate 3734 where
  first := false
  first_correct := by
    have composite : ¬ Nat.Prime (corner 3734 1) := by change ¬ Nat.Prime 55778493; exact PrattCertificates.composite_of_product (factor := 3) (cofactor := 18592831) rfl (by decide) (by decide)
    exact decide_eq_false composite
  second := false
  second_correct := by
    have composite : ¬ Nat.Prime (corner 3734 2) := by change ¬ Nat.Prime 55771025; exact PrattCertificates.composite_of_product (factor := 5) (cofactor := 11154205) rfl (by decide) (by decide)
    exact decide_eq_false composite
  third := true
  third_correct := by
    have prime : Nat.Prime (corner 3734 3) := by change Nat.Prime 55763557; exact ProjectEulerSolutions.P58PrimeNodes.p_55763557.prime
    exact decide_eq_true prime
theorem row_count_3734 : row 3734 = 1 := (row_3734.checked).trans rfl
theorem model_count_3734 : modelRow 3734 = 1 := by decide
theorem modelrow_agrees_3734 : modelRow 3734 = row 3734 := model_count_3734.trans row_count_3734.symm
def row_3735 : RowCertificate 3735 where
  first := false
  first_correct := by
    have composite : ¬ Nat.Prime (corner 3735 1) := by change ¬ Nat.Prime 55808371; exact PrattCertificates.composite_of_product (factor := 859) (cofactor := 64969) rfl (by decide) (by decide)
    exact decide_eq_false composite
  second := false
  second_correct := by
    have composite : ¬ Nat.Prime (corner 3735 2) := by change ¬ Nat.Prime 55800901; exact PrattCertificates.composite_of_product (factor := 13) (cofactor := 4292377) rfl (by decide) (by decide)
    exact decide_eq_false composite
  third := false
  third_correct := by
    have composite : ¬ Nat.Prime (corner 3735 3) := by change ¬ Nat.Prime 55793431; exact PrattCertificates.composite_of_product (factor := 181) (cofactor := 308251) rfl (by decide) (by decide)
    exact decide_eq_false composite
theorem row_count_3735 : row 3735 = 0 := (row_3735.checked).trans rfl
theorem model_count_3735 : modelRow 3735 = 0 := by decide
theorem modelrow_agrees_3735 : modelRow 3735 = row 3735 := model_count_3735.trans row_count_3735.symm
def row_3736 : RowCertificate 3736 where
  first := false
  first_correct := by
    have composite : ¬ Nat.Prime (corner 3736 1) := by change ¬ Nat.Prime 55838257; exact PrattCertificates.composite_of_product (factor := 103) (cofactor := 542119) rfl (by decide) (by decide)
    exact decide_eq_false composite
  second := false
  second_correct := by
    have composite : ¬ Nat.Prime (corner 3736 2) := by change ¬ Nat.Prime 55830785; exact PrattCertificates.composite_of_product (factor := 5) (cofactor := 11166157) rfl (by decide) (by decide)
    exact decide_eq_false composite
  third := false
  third_correct := by
    have composite : ¬ Nat.Prime (corner 3736 3) := by change ¬ Nat.Prime 55823313; exact PrattCertificates.composite_of_product (factor := 3) (cofactor := 18607771) rfl (by decide) (by decide)
    exact decide_eq_false composite
theorem row_count_3736 : row 3736 = 0 := (row_3736.checked).trans rfl
theorem model_count_3736 : modelRow 3736 = 0 := by decide
theorem modelrow_agrees_3736 : modelRow 3736 = row 3736 := model_count_3736.trans row_count_3736.symm
def row_3737 : RowCertificate 3737 where
  first := false
  first_correct := by
    have composite : ¬ Nat.Prime (corner 3737 1) := by change ¬ Nat.Prime 55868151; exact PrattCertificates.composite_of_product (factor := 3) (cofactor := 18622717) rfl (by decide) (by decide)
    exact decide_eq_false composite
  second := true
  second_correct := by
    have prime : Nat.Prime (corner 3737 2) := by change Nat.Prime 55860677; exact ProjectEulerSolutions.P58PrimeNodes.p_55860677.prime
    exact decide_eq_true prime
  third := false
  third_correct := by
    have composite : ¬ Nat.Prime (corner 3737 3) := by change ¬ Nat.Prime 55853203; exact PrattCertificates.composite_of_product (factor := 7) (cofactor := 7979029) rfl (by decide) (by decide)
    exact decide_eq_false composite
theorem row_count_3737 : row 3737 = 1 := (row_3737.checked).trans rfl
theorem model_count_3737 : modelRow 3737 = 1 := by decide
theorem modelrow_agrees_3737 : modelRow 3737 = row 3737 := model_count_3737.trans row_count_3737.symm
def row_3738 : RowCertificate 3738 where
  first := false
  first_correct := by
    have composite : ¬ Nat.Prime (corner 3738 1) := by change ¬ Nat.Prime 55898053; exact PrattCertificates.composite_of_product (factor := 31) (cofactor := 1803163) rfl (by decide) (by decide)
    exact decide_eq_false composite
  second := false
  second_correct := by
    have composite : ¬ Nat.Prime (corner 3738 2) := by change ¬ Nat.Prime 55890577; exact PrattCertificates.composite_of_product (factor := 17) (cofactor := 3287681) rfl (by decide) (by decide)
    exact decide_eq_false composite
  third := false
  third_correct := by
    have composite : ¬ Nat.Prime (corner 3738 3) := by change ¬ Nat.Prime 55883101; exact PrattCertificates.composite_of_product (factor := 43) (cofactor := 1299607) rfl (by decide) (by decide)
    exact decide_eq_false composite
theorem row_count_3738 : row 3738 = 0 := (row_3738.checked).trans rfl
theorem model_count_3738 : modelRow 3738 = 0 := by decide
theorem modelrow_agrees_3738 : modelRow 3738 = row 3738 := model_count_3738.trans row_count_3738.symm
def row_3739 : RowCertificate 3739 where
  first := false
  first_correct := by
    have composite : ¬ Nat.Prime (corner 3739 1) := by change ¬ Nat.Prime 55927963; exact PrattCertificates.composite_of_product (factor := 7) (cofactor := 7989709) rfl (by decide) (by decide)
    exact decide_eq_false composite
  second := false
  second_correct := by
    have composite : ¬ Nat.Prime (corner 3739 2) := by change ¬ Nat.Prime 55920485; exact PrattCertificates.composite_of_product (factor := 5) (cofactor := 11184097) rfl (by decide) (by decide)
    exact decide_eq_false composite
  third := false
  third_correct := by
    have composite : ¬ Nat.Prime (corner 3739 3) := by change ¬ Nat.Prime 55913007; exact PrattCertificates.composite_of_product (factor := 3) (cofactor := 18637669) rfl (by decide) (by decide)
    exact decide_eq_false composite
theorem row_count_3739 : row 3739 = 0 := (row_3739.checked).trans rfl
theorem model_count_3739 : modelRow 3739 = 0 := by decide
theorem modelrow_agrees_3739 : modelRow 3739 = row 3739 := model_count_3739.trans row_count_3739.symm
def row_3740 : RowCertificate 3740 where
  first := false
  first_correct := by
    have composite : ¬ Nat.Prime (corner 3740 1) := by change ¬ Nat.Prime 55957881; exact PrattCertificates.composite_of_product (factor := 3) (cofactor := 18652627) rfl (by decide) (by decide)
    exact decide_eq_false composite
  second := false
  second_correct := by
    have composite : ¬ Nat.Prime (corner 3740 2) := by change ¬ Nat.Prime 55950401; exact PrattCertificates.composite_of_product (factor := 13) (cofactor := 4303877) rfl (by decide) (by decide)
    exact decide_eq_false composite
  third := true
  third_correct := by
    have prime : Nat.Prime (corner 3740 3) := by change Nat.Prime 55942921; exact ProjectEulerSolutions.P58PrimeNodes.p_55942921.prime
    exact decide_eq_true prime
theorem row_count_3740 : row 3740 = 1 := (row_3740.checked).trans rfl
theorem model_count_3740 : modelRow 3740 = 1 := by decide
theorem modelrow_agrees_3740 : modelRow 3740 = row 3740 := model_count_3740.trans row_count_3740.symm
def row_3741 : RowCertificate 3741 where
  first := true
  first_correct := by
    have prime : Nat.Prime (corner 3741 1) := by change Nat.Prime 55987807; exact ProjectEulerSolutions.P58PrimeNodes.p_55987807.prime
    exact decide_eq_true prime
  second := false
  second_correct := by
    have composite : ¬ Nat.Prime (corner 3741 2) := by change ¬ Nat.Prime 55980325; exact PrattCertificates.composite_of_product (factor := 5) (cofactor := 11196065) rfl (by decide) (by decide)
    exact decide_eq_false composite
  third := false
  third_correct := by
    have composite : ¬ Nat.Prime (corner 3741 3) := by change ¬ Nat.Prime 55972843; exact PrattCertificates.composite_of_product (factor := 79) (cofactor := 708517) rfl (by decide) (by decide)
    exact decide_eq_false composite
theorem row_count_3741 : row 3741 = 1 := (row_3741.checked).trans rfl
theorem model_count_3741 : modelRow 3741 = 1 := by decide
theorem modelrow_agrees_3741 : modelRow 3741 = row 3741 := model_count_3741.trans row_count_3741.symm
def row_3742 : RowCertificate 3742 where
  first := false
  first_correct := by
    have composite : ¬ Nat.Prime (corner 3742 1) := by change ¬ Nat.Prime 56017741; exact PrattCertificates.composite_of_product (factor := 13) (cofactor := 4309057) rfl (by decide) (by decide)
    exact decide_eq_false composite
  second := false
  second_correct := by
    have composite : ¬ Nat.Prime (corner 3742 2) := by change ¬ Nat.Prime 56010257; exact PrattCertificates.composite_of_product (factor := 17) (cofactor := 3294721) rfl (by decide) (by decide)
    exact decide_eq_false composite
  third := false
  third_correct := by
    have composite : ¬ Nat.Prime (corner 3742 3) := by change ¬ Nat.Prime 56002773; exact PrattCertificates.composite_of_product (factor := 3) (cofactor := 18667591) rfl (by decide) (by decide)
    exact decide_eq_false composite
theorem row_count_3742 : row 3742 = 0 := (row_3742.checked).trans rfl
theorem model_count_3742 : modelRow 3742 = 0 := by decide
theorem modelrow_agrees_3742 : modelRow 3742 = row 3742 := model_count_3742.trans row_count_3742.symm
def row_3743 : RowCertificate 3743 where
  first := false
  first_correct := by
    have composite : ¬ Nat.Prime (corner 3743 1) := by change ¬ Nat.Prime 56047683; exact PrattCertificates.composite_of_product (factor := 3) (cofactor := 18682561) rfl (by decide) (by decide)
    exact decide_eq_false composite
  second := false
  second_correct := by
    have composite : ¬ Nat.Prime (corner 3743 2) := by change ¬ Nat.Prime 56040197; exact PrattCertificates.composite_of_product (factor := 5233) (cofactor := 10709) rfl (by decide) (by decide)
    exact decide_eq_false composite
  third := false
  third_correct := by
    have composite : ¬ Nat.Prime (corner 3743 3) := by change ¬ Nat.Prime 56032711; exact PrattCertificates.composite_of_product (factor := 7) (cofactor := 8004673) rfl (by decide) (by decide)
    exact decide_eq_false composite
theorem row_count_3743 : row 3743 = 0 := (row_3743.checked).trans rfl
theorem model_count_3743 : modelRow 3743 = 0 := by decide
theorem modelrow_agrees_3743 : modelRow 3743 = row 3743 := model_count_3743.trans row_count_3743.symm
def row_3744 : RowCertificate 3744 where
  first := false
  first_correct := by
    have composite : ¬ Nat.Prime (corner 3744 1) := by change ¬ Nat.Prime 56077633; exact PrattCertificates.composite_of_product (factor := 43) (cofactor := 1304131) rfl (by decide) (by decide)
    exact decide_eq_false composite
  second := false
  second_correct := by
    have composite : ¬ Nat.Prime (corner 3744 2) := by change ¬ Nat.Prime 56070145; exact PrattCertificates.composite_of_product (factor := 5) (cofactor := 11214029) rfl (by decide) (by decide)
    exact decide_eq_false composite
  third := false
  third_correct := by
    have composite : ¬ Nat.Prime (corner 3744 3) := by change ¬ Nat.Prime 56062657; exact PrattCertificates.composite_of_product (factor := 7) (cofactor := 8008951) rfl (by decide) (by decide)
    exact decide_eq_false composite
theorem row_count_3744 : row 3744 = 0 := (row_3744.checked).trans rfl
theorem model_count_3744 : modelRow 3744 = 0 := by decide
theorem modelrow_agrees_3744 : modelRow 3744 = row 3744 := model_count_3744.trans row_count_3744.symm
def row_3745 : RowCertificate 3745 where
  first := false
  first_correct := by
    have composite : ¬ Nat.Prime (corner 3745 1) := by change ¬ Nat.Prime 56107591; exact PrattCertificates.composite_of_product (factor := 787) (cofactor := 71293) rfl (by decide) (by decide)
    exact decide_eq_false composite
  second := true
  second_correct := by
    have prime : Nat.Prime (corner 3745 2) := by change Nat.Prime 56100101; exact ProjectEulerSolutions.P58PrimeNodes.p_56100101.prime
    exact decide_eq_true prime
  third := false
  third_correct := by
    have composite : ¬ Nat.Prime (corner 3745 3) := by change ¬ Nat.Prime 56092611; exact PrattCertificates.composite_of_product (factor := 3) (cofactor := 18697537) rfl (by decide) (by decide)
    exact decide_eq_false composite
theorem row_count_3745 : row 3745 = 1 := (row_3745.checked).trans rfl
theorem model_count_3745 : modelRow 3745 = 1 := by decide
theorem modelrow_agrees_3745 : modelRow 3745 = row 3745 := model_count_3745.trans row_count_3745.symm
def row_3746 : RowCertificate 3746 where
  first := false
  first_correct := by
    have composite : ¬ Nat.Prime (corner 3746 1) := by change ¬ Nat.Prime 56137557; exact PrattCertificates.composite_of_product (factor := 3) (cofactor := 18712519) rfl (by decide) (by decide)
    exact decide_eq_false composite
  second := false
  second_correct := by
    have composite : ¬ Nat.Prime (corner 3746 2) := by change ¬ Nat.Prime 56130065; exact PrattCertificates.composite_of_product (factor := 5) (cofactor := 11226013) rfl (by decide) (by decide)
    exact decide_eq_false composite
  third := false
  third_correct := by
    have composite : ¬ Nat.Prime (corner 3746 3) := by change ¬ Nat.Prime 56122573; exact PrattCertificates.composite_of_product (factor := 13) (cofactor := 4317121) rfl (by decide) (by decide)
    exact decide_eq_false composite
theorem row_count_3746 : row 3746 = 0 := (row_3746.checked).trans rfl
theorem model_count_3746 : modelRow 3746 = 0 := by decide
theorem modelrow_agrees_3746 : modelRow 3746 = row 3746 := model_count_3746.trans row_count_3746.symm
def row_3747 : RowCertificate 3747 where
  first := false
  first_correct := by
    have composite : ¬ Nat.Prime (corner 3747 1) := by change ¬ Nat.Prime 56167531; exact PrattCertificates.composite_of_product (factor := 7) (cofactor := 8023933) rfl (by decide) (by decide)
    exact decide_eq_false composite
  second := false
  second_correct := by
    have composite : ¬ Nat.Prime (corner 3747 2) := by change ¬ Nat.Prime 56160037; exact PrattCertificates.composite_of_product (factor := 29) (cofactor := 1936553) rfl (by decide) (by decide)
    exact decide_eq_false composite
  third := false
  third_correct := by
    have composite : ¬ Nat.Prime (corner 3747 3) := by change ¬ Nat.Prime 56152543; exact PrattCertificates.composite_of_product (factor := 19) (cofactor := 2955397) rfl (by decide) (by decide)
    exact decide_eq_false composite
theorem row_count_3747 : row 3747 = 0 := (row_3747.checked).trans rfl
theorem model_count_3747 : modelRow 3747 = 0 := by decide
theorem modelrow_agrees_3747 : modelRow 3747 = row 3747 := model_count_3747.trans row_count_3747.symm
def row_3748 : RowCertificate 3748 where
  first := false
  first_correct := by
    have composite : ¬ Nat.Prime (corner 3748 1) := by change ¬ Nat.Prime 56197513; exact PrattCertificates.composite_of_product (factor := 31) (cofactor := 1812823) rfl (by decide) (by decide)
    exact decide_eq_false composite
  second := false
  second_correct := by
    have composite : ¬ Nat.Prime (corner 3748 2) := by change ¬ Nat.Prime 56190017; exact PrattCertificates.composite_of_product (factor := 13) (cofactor := 4322309) rfl (by decide) (by decide)
    exact decide_eq_false composite
  third := false
  third_correct := by
    have composite : ¬ Nat.Prime (corner 3748 3) := by change ¬ Nat.Prime 56182521; exact PrattCertificates.composite_of_product (factor := 3) (cofactor := 18727507) rfl (by decide) (by decide)
    exact decide_eq_false composite
theorem row_count_3748 : row 3748 = 0 := (row_3748.checked).trans rfl
theorem model_count_3748 : modelRow 3748 = 0 := by decide
theorem modelrow_agrees_3748 : modelRow 3748 = row 3748 := model_count_3748.trans row_count_3748.symm
def row_3749 : RowCertificate 3749 where
  first := false
  first_correct := by
    have composite : ¬ Nat.Prime (corner 3749 1) := by change ¬ Nat.Prime 56227503; exact PrattCertificates.composite_of_product (factor := 3) (cofactor := 18742501) rfl (by decide) (by decide)
    exact decide_eq_false composite
  second := false
  second_correct := by
    have composite : ¬ Nat.Prime (corner 3749 2) := by change ¬ Nat.Prime 56220005; exact PrattCertificates.composite_of_product (factor := 5) (cofactor := 11244001) rfl (by decide) (by decide)
    exact decide_eq_false composite
  third := false
  third_correct := by
    have composite : ¬ Nat.Prime (corner 3749 3) := by change ¬ Nat.Prime 56212507; exact PrattCertificates.composite_of_product (factor := 13) (cofactor := 4324039) rfl (by decide) (by decide)
    exact decide_eq_false composite
theorem row_count_3749 : row 3749 = 0 := (row_3749.checked).trans rfl
theorem model_count_3749 : modelRow 3749 = 0 := by decide
theorem modelrow_agrees_3749 : modelRow 3749 = row 3749 := model_count_3749.trans row_count_3749.symm
def row_3750 : RowCertificate 3750 where
  first := false
  first_correct := by
    have composite : ¬ Nat.Prime (corner 3750 1) := by change ¬ Nat.Prime 56257501; exact PrattCertificates.composite_of_product (factor := 37) (cofactor := 1520473) rfl (by decide) (by decide)
    exact decide_eq_false composite
  second := false
  second_correct := by
    have composite : ¬ Nat.Prime (corner 3750 2) := by change ¬ Nat.Prime 56250001; exact PrattCertificates.composite_of_product (factor := 197) (cofactor := 285533) rfl (by decide) (by decide)
    exact decide_eq_false composite
  third := false
  third_correct := by
    have composite : ¬ Nat.Prime (corner 3750 3) := by change ¬ Nat.Prime 56242501; exact PrattCertificates.composite_of_product (factor := 7) (cofactor := 8034643) rfl (by decide) (by decide)
    exact decide_eq_false composite
theorem row_count_3750 : row 3750 = 0 := (row_3750.checked).trans rfl
theorem model_count_3750 : modelRow 3750 = 0 := by decide
theorem modelrow_agrees_3750 : modelRow 3750 = row 3750 := model_count_3750.trans row_count_3750.symm
def row_3751 : RowCertificate 3751 where
  first := true
  first_correct := by
    have prime : Nat.Prime (corner 3751 1) := by change Nat.Prime 56287507; exact ProjectEulerSolutions.P58PrimeNodes.p_56287507.prime
    exact decide_eq_true prime
  second := false
  second_correct := by
    have composite : ¬ Nat.Prime (corner 3751 2) := by change ¬ Nat.Prime 56280005; exact PrattCertificates.composite_of_product (factor := 5) (cofactor := 11256001) rfl (by decide) (by decide)
    exact decide_eq_false composite
  third := false
  third_correct := by
    have composite : ¬ Nat.Prime (corner 3751 3) := by change ¬ Nat.Prime 56272503; exact PrattCertificates.composite_of_product (factor := 3) (cofactor := 18757501) rfl (by decide) (by decide)
    exact decide_eq_false composite
theorem row_count_3751 : row 3751 = 1 := (row_3751.checked).trans rfl
theorem model_count_3751 : modelRow 3751 = 1 := by decide
theorem modelrow_agrees_3751 : modelRow 3751 = row 3751 := model_count_3751.trans row_count_3751.symm
def row_3752 : RowCertificate 3752 where
  first := false
  first_correct := by
    have composite : ¬ Nat.Prime (corner 3752 1) := by change ¬ Nat.Prime 56317521; exact PrattCertificates.composite_of_product (factor := 3) (cofactor := 18772507) rfl (by decide) (by decide)
    exact decide_eq_false composite
  second := true
  second_correct := by
    have prime : Nat.Prime (corner 3752 2) := by change Nat.Prime 56310017; exact ProjectEulerSolutions.P58PrimeNodes.p_56310017.prime
    exact decide_eq_true prime
  third := true
  third_correct := by
    have prime : Nat.Prime (corner 3752 3) := by change Nat.Prime 56302513; exact ProjectEulerSolutions.P58PrimeNodes.p_56302513.prime
    exact decide_eq_true prime
theorem row_count_3752 : row 3752 = 2 := (row_3752.checked).trans rfl
theorem model_count_3752 : modelRow 3752 = 2 := by decide
theorem modelrow_agrees_3752 : modelRow 3752 = row 3752 := model_count_3752.trans row_count_3752.symm
def row_3753 : RowCertificate 3753 where
  first := false
  first_correct := by
    have composite : ¬ Nat.Prime (corner 3753 1) := by change ¬ Nat.Prime 56347543; exact PrattCertificates.composite_of_product (factor := 7) (cofactor := 8049649) rfl (by decide) (by decide)
    exact decide_eq_false composite
  second := false
  second_correct := by
    have composite : ¬ Nat.Prime (corner 3753 2) := by change ¬ Nat.Prime 56340037; exact PrattCertificates.composite_of_product (factor := 13) (cofactor := 4333849) rfl (by decide) (by decide)
    exact decide_eq_false composite
  third := true
  third_correct := by
    have prime : Nat.Prime (corner 3753 3) := by change Nat.Prime 56332531; exact ProjectEulerSolutions.P58PrimeNodes.p_56332531.prime
    exact decide_eq_true prime
theorem row_count_3753 : row 3753 = 1 := (row_3753.checked).trans rfl
theorem model_count_3753 : modelRow 3753 = 1 := by decide
theorem modelrow_agrees_3753 : modelRow 3753 = row 3753 := model_count_3753.trans row_count_3753.symm
def row_3754 : RowCertificate 3754 where
  first := false
  first_correct := by
    have composite : ¬ Nat.Prime (corner 3754 1) := by change ¬ Nat.Prime 56377573; exact PrattCertificates.composite_of_product (factor := 7) (cofactor := 8053939) rfl (by decide) (by decide)
    exact decide_eq_false composite
  second := false
  second_correct := by
    have composite : ¬ Nat.Prime (corner 3754 2) := by change ¬ Nat.Prime 56370065; exact PrattCertificates.composite_of_product (factor := 5) (cofactor := 11274013) rfl (by decide) (by decide)
    exact decide_eq_false composite
  third := false
  third_correct := by
    have composite : ¬ Nat.Prime (corner 3754 3) := by change ¬ Nat.Prime 56362557; exact PrattCertificates.composite_of_product (factor := 3) (cofactor := 18787519) rfl (by decide) (by decide)
    exact decide_eq_false composite
theorem row_count_3754 : row 3754 = 0 := (row_3754.checked).trans rfl
theorem model_count_3754 : modelRow 3754 = 0 := by decide
theorem modelrow_agrees_3754 : modelRow 3754 = row 3754 := model_count_3754.trans row_count_3754.symm
def row_3755 : RowCertificate 3755 where
  first := false
  first_correct := by
    have composite : ¬ Nat.Prime (corner 3755 1) := by change ¬ Nat.Prime 56407611; exact PrattCertificates.composite_of_product (factor := 3) (cofactor := 18802537) rfl (by decide) (by decide)
    exact decide_eq_false composite
  second := false
  second_correct := by
    have composite : ¬ Nat.Prime (corner 3755 2) := by change ¬ Nat.Prime 56400101; exact PrattCertificates.composite_of_product (factor := 17) (cofactor := 3317653) rfl (by decide) (by decide)
    exact decide_eq_false composite
  third := true
  third_correct := by
    have prime : Nat.Prime (corner 3755 3) := by change Nat.Prime 56392591; exact ProjectEulerSolutions.P58PrimeNodes.p_56392591.prime
    exact decide_eq_true prime
theorem row_count_3755 : row 3755 = 1 := (row_3755.checked).trans rfl
theorem model_count_3755 : modelRow 3755 = 1 := by decide
theorem modelrow_agrees_3755 : modelRow 3755 = row 3755 := model_count_3755.trans row_count_3755.symm
def row_3756 : RowCertificate 3756 where
  first := false
  first_correct := by
    have composite : ¬ Nat.Prime (corner 3756 1) := by change ¬ Nat.Prime 56437657; exact PrattCertificates.composite_of_product (factor := 19) (cofactor := 2970403) rfl (by decide) (by decide)
    exact decide_eq_false composite
  second := false
  second_correct := by
    have composite : ¬ Nat.Prime (corner 3756 2) := by change ¬ Nat.Prime 56430145; exact PrattCertificates.composite_of_product (factor := 5) (cofactor := 11286029) rfl (by decide) (by decide)
    exact decide_eq_false composite
  third := false
  third_correct := by
    have composite : ¬ Nat.Prime (corner 3756 3) := by change ¬ Nat.Prime 56422633; exact PrattCertificates.composite_of_product (factor := 6427) (cofactor := 8779) rfl (by decide) (by decide)
    exact decide_eq_false composite
theorem row_count_3756 : row 3756 = 0 := (row_3756.checked).trans rfl
theorem model_count_3756 : modelRow 3756 = 0 := by decide
theorem modelrow_agrees_3756 : modelRow 3756 = row 3756 := model_count_3756.trans row_count_3756.symm
def row_3757 : RowCertificate 3757 where
  first := true
  first_correct := by
    have prime : Nat.Prime (corner 3757 1) := by change Nat.Prime 56467711; exact ProjectEulerSolutions.P58PrimeNodes.p_56467711.prime
    exact decide_eq_true prime
  second := false
  second_correct := by
    have composite : ¬ Nat.Prime (corner 3757 2) := by change ¬ Nat.Prime 56460197; exact PrattCertificates.composite_of_product (factor := 61) (cofactor := 925577) rfl (by decide) (by decide)
    exact decide_eq_false composite
  third := false
  third_correct := by
    have composite : ¬ Nat.Prime (corner 3757 3) := by change ¬ Nat.Prime 56452683; exact PrattCertificates.composite_of_product (factor := 3) (cofactor := 18817561) rfl (by decide) (by decide)
    exact decide_eq_false composite
theorem row_count_3757 : row 3757 = 1 := (row_3757.checked).trans rfl
theorem model_count_3757 : modelRow 3757 = 1 := by decide
theorem modelrow_agrees_3757 : modelRow 3757 = row 3757 := model_count_3757.trans row_count_3757.symm
def row_3758 : RowCertificate 3758 where
  first := false
  first_correct := by
    have composite : ¬ Nat.Prime (corner 3758 1) := by change ¬ Nat.Prime 56497773; exact PrattCertificates.composite_of_product (factor := 3) (cofactor := 18832591) rfl (by decide) (by decide)
    exact decide_eq_false composite
  second := true
  second_correct := by
    have prime : Nat.Prime (corner 3758 2) := by change Nat.Prime 56490257; exact ProjectEulerSolutions.P58PrimeNodes.p_56490257.prime
    exact decide_eq_true prime
  third := false
  third_correct := by
    have composite : ¬ Nat.Prime (corner 3758 3) := by change ¬ Nat.Prime 56482741; exact PrattCertificates.composite_of_product (factor := 7) (cofactor := 8068963) rfl (by decide) (by decide)
    exact decide_eq_false composite
theorem row_count_3758 : row 3758 = 1 := (row_3758.checked).trans rfl
theorem model_count_3758 : modelRow 3758 = 1 := by decide
theorem modelrow_agrees_3758 : modelRow 3758 = row 3758 := model_count_3758.trans row_count_3758.symm
def row_3759 : RowCertificate 3759 where
  first := false
  first_correct := by
    have composite : ¬ Nat.Prime (corner 3759 1) := by change ¬ Nat.Prime 56527843; exact PrattCertificates.composite_of_product (factor := 43) (cofactor := 1314601) rfl (by decide) (by decide)
    exact decide_eq_false composite
  second := false
  second_correct := by
    have composite : ¬ Nat.Prime (corner 3759 2) := by change ¬ Nat.Prime 56520325; exact PrattCertificates.composite_of_product (factor := 5) (cofactor := 11304065) rfl (by decide) (by decide)
    exact decide_eq_false composite
  third := false
  third_correct := by
    have composite : ¬ Nat.Prime (corner 3759 3) := by change ¬ Nat.Prime 56512807; exact PrattCertificates.composite_of_product (factor := 13) (cofactor := 4347139) rfl (by decide) (by decide)
    exact decide_eq_false composite
theorem row_count_3759 : row 3759 = 0 := (row_3759.checked).trans rfl
theorem model_count_3759 : modelRow 3759 = 0 := by decide
theorem modelrow_agrees_3759 : modelRow 3759 = row 3759 := model_count_3759.trans row_count_3759.symm
def row_3760 : RowCertificate 3760 where
  first := false
  first_correct := by
    have composite : ¬ Nat.Prime (corner 3760 1) := by change ¬ Nat.Prime 56557921; exact PrattCertificates.composite_of_product (factor := 7) (cofactor := 8079703) rfl (by decide) (by decide)
    exact decide_eq_false composite
  second := true
  second_correct := by
    have prime : Nat.Prime (corner 3760 2) := by change Nat.Prime 56550401; exact ProjectEulerSolutions.P58PrimeNodes.p_56550401.prime
    exact decide_eq_true prime
  third := false
  third_correct := by
    have composite : ¬ Nat.Prime (corner 3760 3) := by change ¬ Nat.Prime 56542881; exact PrattCertificates.composite_of_product (factor := 3) (cofactor := 18847627) rfl (by decide) (by decide)
    exact decide_eq_false composite
theorem row_count_3760 : row 3760 = 1 := (row_3760.checked).trans rfl
theorem model_count_3760 : modelRow 3760 = 1 := by decide
theorem modelrow_agrees_3760 : modelRow 3760 = row 3760 := model_count_3760.trans row_count_3760.symm
def row_3761 : RowCertificate 3761 where
  first := false
  first_correct := by
    have composite : ¬ Nat.Prime (corner 3761 1) := by change ¬ Nat.Prime 56588007; exact PrattCertificates.composite_of_product (factor := 3) (cofactor := 18862669) rfl (by decide) (by decide)
    exact decide_eq_false composite
  second := false
  second_correct := by
    have composite : ¬ Nat.Prime (corner 3761 2) := by change ¬ Nat.Prime 56580485; exact PrattCertificates.composite_of_product (factor := 5) (cofactor := 11316097) rfl (by decide) (by decide)
    exact decide_eq_false composite
  third := false
  third_correct := by
    have composite : ¬ Nat.Prime (corner 3761 3) := by change ¬ Nat.Prime 56572963; exact PrattCertificates.composite_of_product (factor := 37) (cofactor := 1528999) rfl (by decide) (by decide)
    exact decide_eq_false composite
theorem row_count_3761 : row 3761 = 0 := (row_3761.checked).trans rfl
theorem model_count_3761 : modelRow 3761 = 0 := by decide
theorem modelrow_agrees_3761 : modelRow 3761 = row 3761 := model_count_3761.trans row_count_3761.symm
def row_3762 : RowCertificate 3762 where
  first := true
  first_correct := by
    have prime : Nat.Prime (corner 3762 1) := by change Nat.Prime 56618101; exact ProjectEulerSolutions.P58PrimeNodes.p_56618101.prime
    exact decide_eq_true prime
  second := true
  second_correct := by
    have prime : Nat.Prime (corner 3762 2) := by change Nat.Prime 56610577; exact ProjectEulerSolutions.P58PrimeNodes.p_56610577.prime
    exact decide_eq_true prime
  third := false
  third_correct := by
    have composite : ¬ Nat.Prime (corner 3762 3) := by change ¬ Nat.Prime 56603053; exact PrattCertificates.composite_of_product (factor := 13) (cofactor := 4354081) rfl (by decide) (by decide)
    exact decide_eq_false composite
theorem row_count_3762 : row 3762 = 2 := (row_3762.checked).trans rfl
theorem model_count_3762 : modelRow 3762 = 2 := by decide
theorem modelrow_agrees_3762 : modelRow 3762 = row 3762 := model_count_3762.trans row_count_3762.symm
def row_3763 : RowCertificate 3763 where
  first := true
  first_correct := by
    have prime : Nat.Prime (corner 3763 1) := by change Nat.Prime 56648203; exact ProjectEulerSolutions.P58PrimeNodes.p_56648203.prime
    exact decide_eq_true prime
  second := false
  second_correct := by
    have composite : ¬ Nat.Prime (corner 3763 2) := by change ¬ Nat.Prime 56640677; exact PrattCertificates.composite_of_product (factor := 3581) (cofactor := 15817) rfl (by decide) (by decide)
    exact decide_eq_false composite
  third := false
  third_correct := by
    have composite : ¬ Nat.Prime (corner 3763 3) := by change ¬ Nat.Prime 56633151; exact PrattCertificates.composite_of_product (factor := 3) (cofactor := 18877717) rfl (by decide) (by decide)
    exact decide_eq_false composite
theorem row_count_3763 : row 3763 = 1 := (row_3763.checked).trans rfl
theorem model_count_3763 : modelRow 3763 = 1 := by decide
theorem modelrow_agrees_3763 : modelRow 3763 = row 3763 := model_count_3763.trans row_count_3763.symm
def row_3764 : RowCertificate 3764 where
  first := false
  first_correct := by
    have composite : ¬ Nat.Prime (corner 3764 1) := by change ¬ Nat.Prime 56678313; exact PrattCertificates.composite_of_product (factor := 3) (cofactor := 18892771) rfl (by decide) (by decide)
    exact decide_eq_false composite
  second := false
  second_correct := by
    have composite : ¬ Nat.Prime (corner 3764 2) := by change ¬ Nat.Prime 56670785; exact PrattCertificates.composite_of_product (factor := 5) (cofactor := 11334157) rfl (by decide) (by decide)
    exact decide_eq_false composite
  third := false
  third_correct := by
    have composite : ¬ Nat.Prime (corner 3764 3) := by change ¬ Nat.Prime 56663257; exact PrattCertificates.composite_of_product (factor := 7) (cofactor := 8094751) rfl (by decide) (by decide)
    exact decide_eq_false composite
theorem row_count_3764 : row 3764 = 0 := (row_3764.checked).trans rfl
theorem model_count_3764 : modelRow 3764 = 0 := by decide
theorem modelrow_agrees_3764 : modelRow 3764 = row 3764 := model_count_3764.trans row_count_3764.symm
def row_3765 : RowCertificate 3765 where
  first := false
  first_correct := by
    have composite : ¬ Nat.Prime (corner 3765 1) := by change ¬ Nat.Prime 56708431; exact PrattCertificates.composite_of_product (factor := 13) (cofactor := 4362187) rfl (by decide) (by decide)
    exact decide_eq_false composite
  second := false
  second_correct := by
    have composite : ¬ Nat.Prime (corner 3765 2) := by change ¬ Nat.Prime 56700901; exact PrattCertificates.composite_of_product (factor := 1061) (cofactor := 53441) rfl (by decide) (by decide)
    exact decide_eq_false composite
  third := false
  third_correct := by
    have composite : ¬ Nat.Prime (corner 3765 3) := by change ¬ Nat.Prime 56693371; exact PrattCertificates.composite_of_product (factor := 7) (cofactor := 8099053) rfl (by decide) (by decide)
    exact decide_eq_false composite
theorem row_count_3765 : row 3765 = 0 := (row_3765.checked).trans rfl
theorem model_count_3765 : modelRow 3765 = 0 := by decide
theorem modelrow_agrees_3765 : modelRow 3765 = row 3765 := model_count_3765.trans row_count_3765.symm
def row_3766 : RowCertificate 3766 where
  first := false
  first_correct := by
    have composite : ¬ Nat.Prime (corner 3766 1) := by change ¬ Nat.Prime 56738557; exact PrattCertificates.composite_of_product (factor := 571) (cofactor := 99367) rfl (by decide) (by decide)
    exact decide_eq_false composite
  second := false
  second_correct := by
    have composite : ¬ Nat.Prime (corner 3766 2) := by change ¬ Nat.Prime 56731025; exact PrattCertificates.composite_of_product (factor := 5) (cofactor := 11346205) rfl (by decide) (by decide)
    exact decide_eq_false composite
  third := false
  third_correct := by
    have composite : ¬ Nat.Prime (corner 3766 3) := by change ¬ Nat.Prime 56723493; exact PrattCertificates.composite_of_product (factor := 3) (cofactor := 18907831) rfl (by decide) (by decide)
    exact decide_eq_false composite
theorem row_count_3766 : row 3766 = 0 := (row_3766.checked).trans rfl
theorem model_count_3766 : modelRow 3766 = 0 := by decide
theorem modelrow_agrees_3766 : modelRow 3766 = row 3766 := model_count_3766.trans row_count_3766.symm
def row_3767 : RowCertificate 3767 where
  first := false
  first_correct := by
    have composite : ¬ Nat.Prime (corner 3767 1) := by change ¬ Nat.Prime 56768691; exact PrattCertificates.composite_of_product (factor := 3) (cofactor := 18922897) rfl (by decide) (by decide)
    exact decide_eq_false composite
  second := false
  second_correct := by
    have composite : ¬ Nat.Prime (corner 3767 2) := by change ¬ Nat.Prime 56761157; exact PrattCertificates.composite_of_product (factor := 281) (cofactor := 201997) rfl (by decide) (by decide)
    exact decide_eq_false composite
  third := false
  third_correct := by
    have composite : ¬ Nat.Prime (corner 3767 3) := by change ¬ Nat.Prime 56753623; exact PrattCertificates.composite_of_product (factor := 67) (cofactor := 847069) rfl (by decide) (by decide)
    exact decide_eq_false composite
theorem row_count_3767 : row 3767 = 0 := (row_3767.checked).trans rfl
theorem model_count_3767 : modelRow 3767 = 0 := by decide
theorem modelrow_agrees_3767 : modelRow 3767 = row 3767 := model_count_3767.trans row_count_3767.symm
def row_3768 : RowCertificate 3768 where
  first := false
  first_correct := by
    have composite : ¬ Nat.Prime (corner 3768 1) := by change ¬ Nat.Prime 56798833; exact PrattCertificates.composite_of_product (factor := 7) (cofactor := 8114119) rfl (by decide) (by decide)
    exact decide_eq_false composite
  second := true
  second_correct := by
    have prime : Nat.Prime (corner 3768 2) := by change Nat.Prime 56791297; exact ProjectEulerSolutions.P58PrimeNodes.p_56791297.prime
    exact decide_eq_true prime
  third := false
  third_correct := by
    have composite : ¬ Nat.Prime (corner 3768 3) := by change ¬ Nat.Prime 56783761; exact PrattCertificates.composite_of_product (factor := 19) (cofactor := 2988619) rfl (by decide) (by decide)
    exact decide_eq_false composite
theorem row_count_3768 : row 3768 = 1 := (row_3768.checked).trans rfl
theorem model_count_3768 : modelRow 3768 = 1 := by decide
theorem modelrow_agrees_3768 : modelRow 3768 = row 3768 := model_count_3768.trans row_count_3768.symm
def row_3769 : RowCertificate 3769 where
  first := false
  first_correct := by
    have composite : ¬ Nat.Prime (corner 3769 1) := by change ¬ Nat.Prime 56828983; exact PrattCertificates.composite_of_product (factor := 31) (cofactor := 1833193) rfl (by decide) (by decide)
    exact decide_eq_false composite
  second := false
  second_correct := by
    have composite : ¬ Nat.Prime (corner 3769 2) := by change ¬ Nat.Prime 56821445; exact PrattCertificates.composite_of_product (factor := 5) (cofactor := 11364289) rfl (by decide) (by decide)
    exact decide_eq_false composite
  third := false
  third_correct := by
    have composite : ¬ Nat.Prime (corner 3769 3) := by change ¬ Nat.Prime 56813907; exact PrattCertificates.composite_of_product (factor := 3) (cofactor := 18937969) rfl (by decide) (by decide)
    exact decide_eq_false composite
theorem row_count_3769 : row 3769 = 0 := (row_3769.checked).trans rfl
theorem model_count_3769 : modelRow 3769 = 0 := by decide
theorem modelrow_agrees_3769 : modelRow 3769 = row 3769 := model_count_3769.trans row_count_3769.symm
def row_3770 : RowCertificate 3770 where
  first := false
  first_correct := by
    have composite : ¬ Nat.Prime (corner 3770 1) := by change ¬ Nat.Prime 56859141; exact PrattCertificates.composite_of_product (factor := 3) (cofactor := 18953047) rfl (by decide) (by decide)
    exact decide_eq_false composite
  second := false
  second_correct := by
    have composite : ¬ Nat.Prime (corner 3770 2) := by change ¬ Nat.Prime 56851601; exact PrattCertificates.composite_of_product (factor := 433) (cofactor := 131297) rfl (by decide) (by decide)
    exact decide_eq_false composite
  third := true
  third_correct := by
    have prime : Nat.Prime (corner 3770 3) := by change Nat.Prime 56844061; exact ProjectEulerSolutions.P58PrimeNodes.p_56844061.prime
    exact decide_eq_true prime
theorem row_count_3770 : row 3770 = 1 := (row_3770.checked).trans rfl
theorem model_count_3770 : modelRow 3770 = 1 := by decide
theorem modelrow_agrees_3770 : modelRow 3770 = row 3770 := model_count_3770.trans row_count_3770.symm
def row_3771 : RowCertificate 3771 where
  first := false
  first_correct := by
    have composite : ¬ Nat.Prime (corner 3771 1) := by change ¬ Nat.Prime 56889307; exact PrattCertificates.composite_of_product (factor := 223) (cofactor := 255109) rfl (by decide) (by decide)
    exact decide_eq_false composite
  second := false
  second_correct := by
    have composite : ¬ Nat.Prime (corner 3771 2) := by change ¬ Nat.Prime 56881765; exact PrattCertificates.composite_of_product (factor := 5) (cofactor := 11376353) rfl (by decide) (by decide)
    exact decide_eq_false composite
  third := false
  third_correct := by
    have composite : ¬ Nat.Prime (corner 3771 3) := by change ¬ Nat.Prime 56874223; exact PrattCertificates.composite_of_product (factor := 7) (cofactor := 8124889) rfl (by decide) (by decide)
    exact decide_eq_false composite
theorem row_count_3771 : row 3771 = 0 := (row_3771.checked).trans rfl
theorem model_count_3771 : modelRow 3771 = 0 := by decide
theorem modelrow_agrees_3771 : modelRow 3771 = row 3771 := model_count_3771.trans row_count_3771.symm
def row_3772 : RowCertificate 3772 where
  first := false
  first_correct := by
    have composite : ¬ Nat.Prime (corner 3772 1) := by change ¬ Nat.Prime 56919481; exact PrattCertificates.composite_of_product (factor := 1759) (cofactor := 32359) rfl (by decide) (by decide)
    exact decide_eq_false composite
  second := false
  second_correct := by
    have composite : ¬ Nat.Prime (corner 3772 2) := by change ¬ Nat.Prime 56911937; exact PrattCertificates.composite_of_product (factor := 17) (cofactor := 3347761) rfl (by decide) (by decide)
    exact decide_eq_false composite
  third := false
  third_correct := by
    have composite : ¬ Nat.Prime (corner 3772 3) := by change ¬ Nat.Prime 56904393; exact PrattCertificates.composite_of_product (factor := 3) (cofactor := 18968131) rfl (by decide) (by decide)
    exact decide_eq_false composite
theorem row_count_3772 : row 3772 = 0 := (row_3772.checked).trans rfl
theorem model_count_3772 : modelRow 3772 = 0 := by decide
theorem modelrow_agrees_3772 : modelRow 3772 = row 3772 := model_count_3772.trans row_count_3772.symm
def row_3773 : RowCertificate 3773 where
  first := false
  first_correct := by
    have composite : ¬ Nat.Prime (corner 3773 1) := by change ¬ Nat.Prime 56949663; exact PrattCertificates.composite_of_product (factor := 3) (cofactor := 18983221) rfl (by decide) (by decide)
    exact decide_eq_false composite
  second := false
  second_correct := by
    have composite : ¬ Nat.Prime (corner 3773 2) := by change ¬ Nat.Prime 56942117; exact PrattCertificates.composite_of_product (factor := 73) (cofactor := 780029) rfl (by decide) (by decide)
    exact decide_eq_false composite
  third := true
  third_correct := by
    have prime : Nat.Prime (corner 3773 3) := by change Nat.Prime 56934571; exact ProjectEulerSolutions.P58PrimeNodes.p_56934571.prime
    exact decide_eq_true prime
theorem row_count_3773 : row 3773 = 1 := (row_3773.checked).trans rfl
theorem model_count_3773 : modelRow 3773 = 1 := by decide
theorem modelrow_agrees_3773 : modelRow 3773 = row 3773 := model_count_3773.trans row_count_3773.symm
def row_3774 : RowCertificate 3774 where
  first := false
  first_correct := by
    have composite : ¬ Nat.Prime (corner 3774 1) := by change ¬ Nat.Prime 56979853; exact PrattCertificates.composite_of_product (factor := 7) (cofactor := 8139979) rfl (by decide) (by decide)
    exact decide_eq_false composite
  second := false
  second_correct := by
    have composite : ¬ Nat.Prime (corner 3774 2) := by change ¬ Nat.Prime 56972305; exact PrattCertificates.composite_of_product (factor := 5) (cofactor := 11394461) rfl (by decide) (by decide)
    exact decide_eq_false composite
  third := true
  third_correct := by
    have prime : Nat.Prime (corner 3774 3) := by change Nat.Prime 56964757; exact ProjectEulerSolutions.P58PrimeNodes.p_56964757.prime
    exact decide_eq_true prime
theorem row_count_3774 : row 3774 = 1 := (row_3774.checked).trans rfl
theorem model_count_3774 : modelRow 3774 = 1 := by decide
theorem modelrow_agrees_3774 : modelRow 3774 = row 3774 := model_count_3774.trans row_count_3774.symm
def row_3775 : RowCertificate 3775 where
  first := false
  first_correct := by
    have composite : ¬ Nat.Prime (corner 3775 1) := by change ¬ Nat.Prime 57010051; exact PrattCertificates.composite_of_product (factor := 7) (cofactor := 8144293) rfl (by decide) (by decide)
    exact decide_eq_false composite
  second := true
  second_correct := by
    have prime : Nat.Prime (corner 3775 2) := by change Nat.Prime 57002501; exact ProjectEulerSolutions.P58PrimeNodes.p_57002501.prime
    exact decide_eq_true prime
  third := false
  third_correct := by
    have composite : ¬ Nat.Prime (corner 3775 3) := by change ¬ Nat.Prime 56994951; exact PrattCertificates.composite_of_product (factor := 3) (cofactor := 18998317) rfl (by decide) (by decide)
    exact decide_eq_false composite
theorem row_count_3775 : row 3775 = 1 := (row_3775.checked).trans rfl
theorem model_count_3775 : modelRow 3775 = 1 := by decide
theorem modelrow_agrees_3775 : modelRow 3775 = row 3775 := model_count_3775.trans row_count_3775.symm
def row_3776 : RowCertificate 3776 where
  first := false
  first_correct := by
    have composite : ¬ Nat.Prime (corner 3776 1) := by change ¬ Nat.Prime 57040257; exact PrattCertificates.composite_of_product (factor := 3) (cofactor := 19013419) rfl (by decide) (by decide)
    exact decide_eq_false composite
  second := false
  second_correct := by
    have composite : ¬ Nat.Prime (corner 3776 2) := by change ¬ Nat.Prime 57032705; exact PrattCertificates.composite_of_product (factor := 5) (cofactor := 11406541) rfl (by decide) (by decide)
    exact decide_eq_false composite
  third := false
  third_correct := by
    have composite : ¬ Nat.Prime (corner 3776 3) := by change ¬ Nat.Prime 57025153; exact PrattCertificates.composite_of_product (factor := 727) (cofactor := 78439) rfl (by decide) (by decide)
    exact decide_eq_false composite
theorem row_count_3776 : row 3776 = 0 := (row_3776.checked).trans rfl
theorem model_count_3776 : modelRow 3776 = 0 := by decide
theorem modelrow_agrees_3776 : modelRow 3776 = row 3776 := model_count_3776.trans row_count_3776.symm
def row_3777 : RowCertificate 3777 where
  first := false
  first_correct := by
    have composite : ¬ Nat.Prime (corner 3777 1) := by change ¬ Nat.Prime 57070471; exact PrattCertificates.composite_of_product (factor := 19) (cofactor := 3003709) rfl (by decide) (by decide)
    exact decide_eq_false composite
  second := false
  second_correct := by
    have composite : ¬ Nat.Prime (corner 3777 2) := by change ¬ Nat.Prime 57062917; exact PrattCertificates.composite_of_product (factor := 37) (cofactor := 1542241) rfl (by decide) (by decide)
    exact decide_eq_false composite
  third := false
  third_correct := by
    have composite : ¬ Nat.Prime (corner 3777 3) := by change ¬ Nat.Prime 57055363; exact PrattCertificates.composite_of_product (factor := 181) (cofactor := 315223) rfl (by decide) (by decide)
    exact decide_eq_false composite
theorem row_count_3777 : row 3777 = 0 := (row_3777.checked).trans rfl
theorem model_count_3777 : modelRow 3777 = 0 := by decide
theorem modelrow_agrees_3777 : modelRow 3777 = row 3777 := model_count_3777.trans row_count_3777.symm
def row_3778 : RowCertificate 3778 where
  first := false
  first_correct := by
    have composite : ¬ Nat.Prime (corner 3778 1) := by change ¬ Nat.Prime 57100693; exact PrattCertificates.composite_of_product (factor := 13) (cofactor := 4392361) rfl (by decide) (by decide)
    exact decide_eq_false composite
  second := false
  second_correct := by
    have composite : ¬ Nat.Prime (corner 3778 2) := by change ¬ Nat.Prime 57093137; exact PrattCertificates.composite_of_product (factor := 53) (cofactor := 1077229) rfl (by decide) (by decide)
    exact decide_eq_false composite
  third := false
  third_correct := by
    have composite : ¬ Nat.Prime (corner 3778 3) := by change ¬ Nat.Prime 57085581; exact PrattCertificates.composite_of_product (factor := 3) (cofactor := 19028527) rfl (by decide) (by decide)
    exact decide_eq_false composite
theorem row_count_3778 : row 3778 = 0 := (row_3778.checked).trans rfl
theorem model_count_3778 : modelRow 3778 = 0 := by decide
theorem modelrow_agrees_3778 : modelRow 3778 = row 3778 := model_count_3778.trans row_count_3778.symm
def row_3779 : RowCertificate 3779 where
  first := false
  first_correct := by
    have composite : ¬ Nat.Prime (corner 3779 1) := by change ¬ Nat.Prime 57130923; exact PrattCertificates.composite_of_product (factor := 3) (cofactor := 19043641) rfl (by decide) (by decide)
    exact decide_eq_false composite
  second := false
  second_correct := by
    have composite : ¬ Nat.Prime (corner 3779 2) := by change ¬ Nat.Prime 57123365; exact PrattCertificates.composite_of_product (factor := 5) (cofactor := 11424673) rfl (by decide) (by decide)
    exact decide_eq_false composite
  third := false
  third_correct := by
    have composite : ¬ Nat.Prime (corner 3779 3) := by change ¬ Nat.Prime 57115807; exact PrattCertificates.composite_of_product (factor := 7) (cofactor := 8159401) rfl (by decide) (by decide)
    exact decide_eq_false composite
theorem row_count_3779 : row 3779 = 0 := (row_3779.checked).trans rfl
theorem model_count_3779 : modelRow 3779 = 0 := by decide
theorem modelrow_agrees_3779 : modelRow 3779 = row 3779 := model_count_3779.trans row_count_3779.symm
def row_3780 : RowCertificate 3780 where
  first := false
  first_correct := by
    have composite : ¬ Nat.Prime (corner 3780 1) := by change ¬ Nat.Prime 57161161; exact PrattCertificates.composite_of_product (factor := 79) (cofactor := 723559) rfl (by decide) (by decide)
    exact decide_eq_false composite
  second := false
  second_correct := by
    have composite : ¬ Nat.Prime (corner 3780 2) := by change ¬ Nat.Prime 57153601; exact PrattCertificates.composite_of_product (factor := 4517) (cofactor := 12653) rfl (by decide) (by decide)
    exact decide_eq_false composite
  third := false
  third_correct := by
    have composite : ¬ Nat.Prime (corner 3780 3) := by change ¬ Nat.Prime 57146041; exact PrattCertificates.composite_of_product (factor := 271) (cofactor := 210871) rfl (by decide) (by decide)
    exact decide_eq_false composite
theorem row_count_3780 : row 3780 = 0 := (row_3780.checked).trans rfl
theorem model_count_3780 : modelRow 3780 = 0 := by decide
theorem modelrow_agrees_3780 : modelRow 3780 = row 3780 := model_count_3780.trans row_count_3780.symm
def row_3781 : RowCertificate 3781 where
  first := false
  first_correct := by
    have composite : ¬ Nat.Prime (corner 3781 1) := by change ¬ Nat.Prime 57191407; exact PrattCertificates.composite_of_product (factor := 7) (cofactor := 8170201) rfl (by decide) (by decide)
    exact decide_eq_false composite
  second := false
  second_correct := by
    have composite : ¬ Nat.Prime (corner 3781 2) := by change ¬ Nat.Prime 57183845; exact PrattCertificates.composite_of_product (factor := 5) (cofactor := 11436769) rfl (by decide) (by decide)
    exact decide_eq_false composite
  third := false
  third_correct := by
    have composite : ¬ Nat.Prime (corner 3781 3) := by change ¬ Nat.Prime 57176283; exact PrattCertificates.composite_of_product (factor := 3) (cofactor := 19058761) rfl (by decide) (by decide)
    exact decide_eq_false composite
theorem row_count_3781 : row 3781 = 0 := (row_3781.checked).trans rfl
theorem model_count_3781 : modelRow 3781 = 0 := by decide
theorem modelrow_agrees_3781 : modelRow 3781 = row 3781 := model_count_3781.trans row_count_3781.symm
def row_3782 : RowCertificate 3782 where
  first := false
  first_correct := by
    have composite : ¬ Nat.Prime (corner 3782 1) := by change ¬ Nat.Prime 57221661; exact PrattCertificates.composite_of_product (factor := 3) (cofactor := 19073887) rfl (by decide) (by decide)
    exact decide_eq_false composite
  second := false
  second_correct := by
    have composite : ¬ Nat.Prime (corner 3782 2) := by change ¬ Nat.Prime 57214097; exact PrattCertificates.composite_of_product (factor := 157) (cofactor := 364421) rfl (by decide) (by decide)
    exact decide_eq_false composite
  third := false
  third_correct := by
    have composite : ¬ Nat.Prime (corner 3782 3) := by change ¬ Nat.Prime 57206533; exact PrattCertificates.composite_of_product (factor := 6829) (cofactor := 8377) rfl (by decide) (by decide)
    exact decide_eq_false composite
theorem row_count_3782 : row 3782 = 0 := (row_3782.checked).trans rfl
theorem model_count_3782 : modelRow 3782 = 0 := by decide
theorem modelrow_agrees_3782 : modelRow 3782 = row 3782 := model_count_3782.trans row_count_3782.symm
def row_3783 : RowCertificate 3783 where
  first := false
  first_correct := by
    have composite : ¬ Nat.Prime (corner 3783 1) := by change ¬ Nat.Prime 57251923; exact PrattCertificates.composite_of_product (factor := 109) (cofactor := 525247) rfl (by decide) (by decide)
    exact decide_eq_false composite
  second := false
  second_correct := by
    have composite : ¬ Nat.Prime (corner 3783 2) := by change ¬ Nat.Prime 57244357; exact PrattCertificates.composite_of_product (factor := 449) (cofactor := 127493) rfl (by decide) (by decide)
    exact decide_eq_false composite
  third := false
  third_correct := by
    have composite : ¬ Nat.Prime (corner 3783 3) := by change ¬ Nat.Prime 57236791; exact PrattCertificates.composite_of_product (factor := 103) (cofactor := 555697) rfl (by decide) (by decide)
    exact decide_eq_false composite
theorem row_count_3783 : row 3783 = 0 := (row_3783.checked).trans rfl
theorem model_count_3783 : modelRow 3783 = 0 := by decide
theorem modelrow_agrees_3783 : modelRow 3783 = row 3783 := model_count_3783.trans row_count_3783.symm
def row_3784 : RowCertificate 3784 where
  first := false
  first_correct := by
    have composite : ¬ Nat.Prime (corner 3784 1) := by change ¬ Nat.Prime 57282193; exact PrattCertificates.composite_of_product (factor := 1153) (cofactor := 49681) rfl (by decide) (by decide)
    exact decide_eq_false composite
  second := false
  second_correct := by
    have composite : ¬ Nat.Prime (corner 3784 2) := by change ¬ Nat.Prime 57274625; exact PrattCertificates.composite_of_product (factor := 5) (cofactor := 11454925) rfl (by decide) (by decide)
    exact decide_eq_false composite
  third := false
  third_correct := by
    have composite : ¬ Nat.Prime (corner 3784 3) := by change ¬ Nat.Prime 57267057; exact PrattCertificates.composite_of_product (factor := 3) (cofactor := 19089019) rfl (by decide) (by decide)
    exact decide_eq_false composite
theorem row_count_3784 : row 3784 = 0 := (row_3784.checked).trans rfl
theorem model_count_3784 : modelRow 3784 = 0 := by decide
theorem modelrow_agrees_3784 : modelRow 3784 = row 3784 := model_count_3784.trans row_count_3784.symm
def row_3785 : RowCertificate 3785 where
  first := false
  first_correct := by
    have composite : ¬ Nat.Prime (corner 3785 1) := by change ¬ Nat.Prime 57312471; exact PrattCertificates.composite_of_product (factor := 3) (cofactor := 19104157) rfl (by decide) (by decide)
    exact decide_eq_false composite
  second := false
  second_correct := by
    have composite : ¬ Nat.Prime (corner 3785 2) := by change ¬ Nat.Prime 57304901; exact PrattCertificates.composite_of_product (factor := 1373) (cofactor := 41737) rfl (by decide) (by decide)
    exact decide_eq_false composite
  third := false
  third_correct := by
    have composite : ¬ Nat.Prime (corner 3785 3) := by change ¬ Nat.Prime 57297331; exact PrattCertificates.composite_of_product (factor := 7) (cofactor := 8185333) rfl (by decide) (by decide)
    exact decide_eq_false composite
theorem row_count_3785 : row 3785 = 0 := (row_3785.checked).trans rfl
theorem model_count_3785 : modelRow 3785 = 0 := by decide
theorem modelrow_agrees_3785 : modelRow 3785 = row 3785 := model_count_3785.trans row_count_3785.symm
def row_3786 : RowCertificate 3786 where
  first := true
  first_correct := by
    have prime : Nat.Prime (corner 3786 1) := by change Nat.Prime 57342757; exact ProjectEulerSolutions.P58PrimeNodes.p_57342757.prime
    exact decide_eq_true prime
  second := false
  second_correct := by
    have composite : ¬ Nat.Prime (corner 3786 2) := by change ¬ Nat.Prime 57335185; exact PrattCertificates.composite_of_product (factor := 5) (cofactor := 11467037) rfl (by decide) (by decide)
    exact decide_eq_false composite
  third := false
  third_correct := by
    have composite : ¬ Nat.Prime (corner 3786 3) := by change ¬ Nat.Prime 57327613; exact PrattCertificates.composite_of_product (factor := 7) (cofactor := 8189659) rfl (by decide) (by decide)
    exact decide_eq_false composite
theorem row_count_3786 : row 3786 = 1 := (row_3786.checked).trans rfl
theorem model_count_3786 : modelRow 3786 = 1 := by decide
theorem modelrow_agrees_3786 : modelRow 3786 = row 3786 := model_count_3786.trans row_count_3786.symm
def row_3787 : RowCertificate 3787 where
  first := false
  first_correct := by
    have composite : ¬ Nat.Prime (corner 3787 1) := by change ¬ Nat.Prime 57373051; exact PrattCertificates.composite_of_product (factor := 37) (cofactor := 1550623) rfl (by decide) (by decide)
    exact decide_eq_false composite
  second := false
  second_correct := by
    have composite : ¬ Nat.Prime (corner 3787 2) := by change ¬ Nat.Prime 57365477; exact PrattCertificates.composite_of_product (factor := 13) (cofactor := 4412729) rfl (by decide) (by decide)
    exact decide_eq_false composite
  third := false
  third_correct := by
    have composite : ¬ Nat.Prime (corner 3787 3) := by change ¬ Nat.Prime 57357903; exact PrattCertificates.composite_of_product (factor := 3) (cofactor := 19119301) rfl (by decide) (by decide)
    exact decide_eq_false composite
theorem row_count_3787 : row 3787 = 0 := (row_3787.checked).trans rfl
theorem model_count_3787 : modelRow 3787 = 0 := by decide
theorem modelrow_agrees_3787 : modelRow 3787 = row 3787 := model_count_3787.trans row_count_3787.symm
def row_3788 : RowCertificate 3788 where
  first := false
  first_correct := by
    have composite : ¬ Nat.Prime (corner 3788 1) := by change ¬ Nat.Prime 57403353; exact PrattCertificates.composite_of_product (factor := 3) (cofactor := 19134451) rfl (by decide) (by decide)
    exact decide_eq_false composite
  second := false
  second_correct := by
    have composite : ¬ Nat.Prime (corner 3788 2) := by change ¬ Nat.Prime 57395777; exact PrattCertificates.composite_of_product (factor := 41) (cofactor := 1399897) rfl (by decide) (by decide)
    exact decide_eq_false composite
  third := false
  third_correct := by
    have composite : ¬ Nat.Prime (corner 3788 3) := by change ¬ Nat.Prime 57388201; exact PrattCertificates.composite_of_product (factor := 13) (cofactor := 4414477) rfl (by decide) (by decide)
    exact decide_eq_false composite
theorem row_count_3788 : row 3788 = 0 := (row_3788.checked).trans rfl
theorem model_count_3788 : modelRow 3788 = 0 := by decide
theorem modelrow_agrees_3788 : modelRow 3788 = row 3788 := model_count_3788.trans row_count_3788.symm
def row_3789 : RowCertificate 3789 where
  first := false
  first_correct := by
    have composite : ¬ Nat.Prime (corner 3789 1) := by change ¬ Nat.Prime 57433663; exact PrattCertificates.composite_of_product (factor := 7) (cofactor := 8204809) rfl (by decide) (by decide)
    exact decide_eq_false composite
  second := false
  second_correct := by
    have composite : ¬ Nat.Prime (corner 3789 2) := by change ¬ Nat.Prime 57426085; exact PrattCertificates.composite_of_product (factor := 5) (cofactor := 11485217) rfl (by decide) (by decide)
    exact decide_eq_false composite
  third := false
  third_correct := by
    have composite : ¬ Nat.Prime (corner 3789 3) := by change ¬ Nat.Prime 57418507; exact PrattCertificates.composite_of_product (factor := 61) (cofactor := 941287) rfl (by decide) (by decide)
    exact decide_eq_false composite
theorem row_count_3789 : row 3789 = 0 := (row_3789.checked).trans rfl
theorem model_count_3789 : modelRow 3789 = 0 := by decide
theorem modelrow_agrees_3789 : modelRow 3789 = row 3789 := model_count_3789.trans row_count_3789.symm
def row_3790 : RowCertificate 3790 where
  first := true
  first_correct := by
    have prime : Nat.Prime (corner 3790 1) := by change Nat.Prime 57463981; exact ProjectEulerSolutions.P58PrimeNodes.p_57463981.prime
    exact decide_eq_true prime
  second := false
  second_correct := by
    have composite : ¬ Nat.Prime (corner 3790 2) := by change ¬ Nat.Prime 57456401; exact PrattCertificates.composite_of_product (factor := 521) (cofactor := 110281) rfl (by decide) (by decide)
    exact decide_eq_false composite
  third := false
  third_correct := by
    have composite : ¬ Nat.Prime (corner 3790 3) := by change ¬ Nat.Prime 57448821; exact PrattCertificates.composite_of_product (factor := 3) (cofactor := 19149607) rfl (by decide) (by decide)
    exact decide_eq_false composite
theorem row_count_3790 : row 3790 = 1 := (row_3790.checked).trans rfl
theorem model_count_3790 : modelRow 3790 = 1 := by decide
theorem modelrow_agrees_3790 : modelRow 3790 = row 3790 := model_count_3790.trans row_count_3790.symm
def row_3791 : RowCertificate 3791 where
  first := false
  first_correct := by
    have composite : ¬ Nat.Prime (corner 3791 1) := by change ¬ Nat.Prime 57494307; exact PrattCertificates.composite_of_product (factor := 3) (cofactor := 19164769) rfl (by decide) (by decide)
    exact decide_eq_false composite
  second := false
  second_correct := by
    have composite : ¬ Nat.Prime (corner 3791 2) := by change ¬ Nat.Prime 57486725; exact PrattCertificates.composite_of_product (factor := 5) (cofactor := 11497345) rfl (by decide) (by decide)
    exact decide_eq_false composite
  third := false
  third_correct := by
    have composite : ¬ Nat.Prime (corner 3791 3) := by change ¬ Nat.Prime 57479143; exact PrattCertificates.composite_of_product (factor := 211) (cofactor := 272413) rfl (by decide) (by decide)
    exact decide_eq_false composite
theorem row_count_3791 : row 3791 = 0 := (row_3791.checked).trans rfl
theorem model_count_3791 : modelRow 3791 = 0 := by decide
theorem modelrow_agrees_3791 : modelRow 3791 = row 3791 := model_count_3791.trans row_count_3791.symm
def row_3792 : RowCertificate 3792 where
  first := false
  first_correct := by
    have composite : ¬ Nat.Prime (corner 3792 1) := by change ¬ Nat.Prime 57524641; exact PrattCertificates.composite_of_product (factor := 109) (cofactor := 527749) rfl (by decide) (by decide)
    exact decide_eq_false composite
  second := false
  second_correct := by
    have composite : ¬ Nat.Prime (corner 3792 2) := by change ¬ Nat.Prime 57517057; exact PrattCertificates.composite_of_product (factor := 13) (cofactor := 4424389) rfl (by decide) (by decide)
    exact decide_eq_false composite
  third := false
  third_correct := by
    have composite : ¬ Nat.Prime (corner 3792 3) := by change ¬ Nat.Prime 57509473; exact PrattCertificates.composite_of_product (factor := 7) (cofactor := 8215639) rfl (by decide) (by decide)
    exact decide_eq_false composite
theorem row_count_3792 : row 3792 = 0 := (row_3792.checked).trans rfl
theorem model_count_3792 : modelRow 3792 = 0 := by decide
theorem modelrow_agrees_3792 : modelRow 3792 = row 3792 := model_count_3792.trans row_count_3792.symm
def row_3793 : RowCertificate 3793 where
  first := false
  first_correct := by
    have composite : ¬ Nat.Prime (corner 3793 1) := by change ¬ Nat.Prime 57554983; exact PrattCertificates.composite_of_product (factor := 829) (cofactor := 69427) rfl (by decide) (by decide)
    exact decide_eq_false composite
  second := false
  second_correct := by
    have composite : ¬ Nat.Prime (corner 3793 2) := by change ¬ Nat.Prime 57547397; exact PrattCertificates.composite_of_product (factor := 17) (cofactor := 3385141) rfl (by decide) (by decide)
    exact decide_eq_false composite
  third := false
  third_correct := by
    have composite : ¬ Nat.Prime (corner 3793 3) := by change ¬ Nat.Prime 57539811; exact PrattCertificates.composite_of_product (factor := 3) (cofactor := 19179937) rfl (by decide) (by decide)
    exact decide_eq_false composite
theorem row_count_3793 : row 3793 = 0 := (row_3793.checked).trans rfl
theorem model_count_3793 : modelRow 3793 = 0 := by decide
theorem modelrow_agrees_3793 : modelRow 3793 = row 3793 := model_count_3793.trans row_count_3793.symm
def row_3794 : RowCertificate 3794 where
  first := false
  first_correct := by
    have composite : ¬ Nat.Prime (corner 3794 1) := by change ¬ Nat.Prime 57585333; exact PrattCertificates.composite_of_product (factor := 3) (cofactor := 19195111) rfl (by decide) (by decide)
    exact decide_eq_false composite
  second := false
  second_correct := by
    have composite : ¬ Nat.Prime (corner 3794 2) := by change ¬ Nat.Prime 57577745; exact PrattCertificates.composite_of_product (factor := 5) (cofactor := 11515549) rfl (by decide) (by decide)
    exact decide_eq_false composite
  third := false
  third_correct := by
    have composite : ¬ Nat.Prime (corner 3794 3) := by change ¬ Nat.Prime 57570157; exact PrattCertificates.composite_of_product (factor := 1249) (cofactor := 46093) rfl (by decide) (by decide)
    exact decide_eq_false composite
theorem row_count_3794 : row 3794 = 0 := (row_3794.checked).trans rfl
theorem model_count_3794 : modelRow 3794 = 0 := by decide
theorem modelrow_agrees_3794 : modelRow 3794 = row 3794 := model_count_3794.trans row_count_3794.symm
def row_3795 : RowCertificate 3795 where
  first := false
  first_correct := by
    have composite : ¬ Nat.Prime (corner 3795 1) := by change ¬ Nat.Prime 57615691; exact PrattCertificates.composite_of_product (factor := 7) (cofactor := 8230813) rfl (by decide) (by decide)
    exact decide_eq_false composite
  second := false
  second_correct := by
    have composite : ¬ Nat.Prime (corner 3795 2) := by change ¬ Nat.Prime 57608101; exact PrattCertificates.composite_of_product (factor := 5573) (cofactor := 10337) rfl (by decide) (by decide)
    exact decide_eq_false composite
  third := false
  third_correct := by
    have composite : ¬ Nat.Prime (corner 3795 3) := by change ¬ Nat.Prime 57600511; exact PrattCertificates.composite_of_product (factor := 31) (cofactor := 1858081) rfl (by decide) (by decide)
    exact decide_eq_false composite
theorem row_count_3795 : row 3795 = 0 := (row_3795.checked).trans rfl
theorem model_count_3795 : modelRow 3795 = 0 := by decide
theorem modelrow_agrees_3795 : modelRow 3795 = row 3795 := model_count_3795.trans row_count_3795.symm
def row_3796 : RowCertificate 3796 where
  first := false
  first_correct := by
    have composite : ¬ Nat.Prime (corner 3796 1) := by change ¬ Nat.Prime 57646057; exact PrattCertificates.composite_of_product (factor := 7) (cofactor := 8235151) rfl (by decide) (by decide)
    exact decide_eq_false composite
  second := false
  second_correct := by
    have composite : ¬ Nat.Prime (corner 3796 2) := by change ¬ Nat.Prime 57638465; exact PrattCertificates.composite_of_product (factor := 5) (cofactor := 11527693) rfl (by decide) (by decide)
    exact decide_eq_false composite
  third := false
  third_correct := by
    have composite : ¬ Nat.Prime (corner 3796 3) := by change ¬ Nat.Prime 57630873; exact PrattCertificates.composite_of_product (factor := 3) (cofactor := 19210291) rfl (by decide) (by decide)
    exact decide_eq_false composite
theorem row_count_3796 : row 3796 = 0 := (row_3796.checked).trans rfl
theorem model_count_3796 : modelRow 3796 = 0 := by decide
theorem modelrow_agrees_3796 : modelRow 3796 = row 3796 := model_count_3796.trans row_count_3796.symm
def row_3797 : RowCertificate 3797 where
  first := false
  first_correct := by
    have composite : ¬ Nat.Prime (corner 3797 1) := by change ¬ Nat.Prime 57676431; exact PrattCertificates.composite_of_product (factor := 3) (cofactor := 19225477) rfl (by decide) (by decide)
    exact decide_eq_false composite
  second := false
  second_correct := by
    have composite : ¬ Nat.Prime (corner 3797 2) := by change ¬ Nat.Prime 57668837; exact PrattCertificates.composite_of_product (factor := 41) (cofactor := 1406557) rfl (by decide) (by decide)
    exact decide_eq_false composite
  third := false
  third_correct := by
    have composite : ¬ Nat.Prime (corner 3797 3) := by change ¬ Nat.Prime 57661243; exact PrattCertificates.composite_of_product (factor := 967) (cofactor := 59629) rfl (by decide) (by decide)
    exact decide_eq_false composite
theorem row_count_3797 : row 3797 = 0 := (row_3797.checked).trans rfl
theorem model_count_3797 : modelRow 3797 = 0 := by decide
theorem modelrow_agrees_3797 : modelRow 3797 = row 3797 := model_count_3797.trans row_count_3797.symm
def row_3798 : RowCertificate 3798 where
  first := false
  first_correct := by
    have composite : ¬ Nat.Prime (corner 3798 1) := by change ¬ Nat.Prime 57706813; exact PrattCertificates.composite_of_product (factor := 283) (cofactor := 203911) rfl (by decide) (by decide)
    exact decide_eq_false composite
  second := true
  second_correct := by
    have prime : Nat.Prime (corner 3798 2) := by change Nat.Prime 57699217; exact ProjectEulerSolutions.P58PrimeNodes.p_57699217.prime
    exact decide_eq_true prime
  third := false
  third_correct := by
    have composite : ¬ Nat.Prime (corner 3798 3) := by change ¬ Nat.Prime 57691621; exact PrattCertificates.composite_of_product (factor := 13) (cofactor := 4437817) rfl (by decide) (by decide)
    exact decide_eq_false composite
theorem row_count_3798 : row 3798 = 1 := (row_3798.checked).trans rfl
theorem model_count_3798 : modelRow 3798 = 1 := by decide
theorem modelrow_agrees_3798 : modelRow 3798 = row 3798 := model_count_3798.trans row_count_3798.symm
def row_3799 : RowCertificate 3799 where
  first := false
  first_correct := by
    have composite : ¬ Nat.Prime (corner 3799 1) := by change ¬ Nat.Prime 57737203; exact PrattCertificates.composite_of_product (factor := 421) (cofactor := 137143) rfl (by decide) (by decide)
    exact decide_eq_false composite
  second := false
  second_correct := by
    have composite : ¬ Nat.Prime (corner 3799 2) := by change ¬ Nat.Prime 57729605; exact PrattCertificates.composite_of_product (factor := 5) (cofactor := 11545921) rfl (by decide) (by decide)
    exact decide_eq_false composite
  third := false
  third_correct := by
    have composite : ¬ Nat.Prime (corner 3799 3) := by change ¬ Nat.Prime 57722007; exact PrattCertificates.composite_of_product (factor := 3) (cofactor := 19240669) rfl (by decide) (by decide)
    exact decide_eq_false composite
theorem row_count_3799 : row 3799 = 0 := (row_3799.checked).trans rfl
theorem model_count_3799 : modelRow 3799 = 0 := by decide
theorem modelrow_agrees_3799 : modelRow 3799 = row 3799 := model_count_3799.trans row_count_3799.symm
def row_3800 : RowCertificate 3800 where
  first := false
  first_correct := by
    have composite : ¬ Nat.Prime (corner 3800 1) := by change ¬ Nat.Prime 57767601; exact PrattCertificates.composite_of_product (factor := 3) (cofactor := 19255867) rfl (by decide) (by decide)
    exact decide_eq_false composite
  second := false
  second_correct := by
    have composite : ¬ Nat.Prime (corner 3800 2) := by change ¬ Nat.Prime 57760001; exact PrattCertificates.composite_of_product (factor := 13) (cofactor := 4443077) rfl (by decide) (by decide)
    exact decide_eq_false composite
  third := false
  third_correct := by
    have composite : ¬ Nat.Prime (corner 3800 3) := by change ¬ Nat.Prime 57752401; exact PrattCertificates.composite_of_product (factor := 7) (cofactor := 8250343) rfl (by decide) (by decide)
    exact decide_eq_false composite
theorem row_count_3800 : row 3800 = 0 := (row_3800.checked).trans rfl
theorem model_count_3800 : modelRow 3800 = 0 := by decide
theorem modelrow_agrees_3800 : modelRow 3800 = row 3800 := model_count_3800.trans row_count_3800.symm
def row_3801 : RowCertificate 3801 where
  first := false
  first_correct := by
    have composite : ¬ Nat.Prime (corner 3801 1) := by change ¬ Nat.Prime 57798007; exact PrattCertificates.composite_of_product (factor := 139) (cofactor := 415813) rfl (by decide) (by decide)
    exact decide_eq_false composite
  second := false
  second_correct := by
    have composite : ¬ Nat.Prime (corner 3801 2) := by change ¬ Nat.Prime 57790405; exact PrattCertificates.composite_of_product (factor := 5) (cofactor := 11558081) rfl (by decide) (by decide)
    exact decide_eq_false composite
  third := false
  third_correct := by
    have composite : ¬ Nat.Prime (corner 3801 3) := by change ¬ Nat.Prime 57782803; exact PrattCertificates.composite_of_product (factor := 13) (cofactor := 4444831) rfl (by decide) (by decide)
    exact decide_eq_false composite
theorem row_count_3801 : row 3801 = 0 := (row_3801.checked).trans rfl
theorem model_count_3801 : modelRow 3801 = 0 := by decide
theorem modelrow_agrees_3801 : modelRow 3801 = row 3801 := model_count_3801.trans row_count_3801.symm
def row_3802 : RowCertificate 3802 where
  first := false
  first_correct := by
    have composite : ¬ Nat.Prime (corner 3802 1) := by change ¬ Nat.Prime 57828421; exact PrattCertificates.composite_of_product (factor := 7) (cofactor := 8261203) rfl (by decide) (by decide)
    exact decide_eq_false composite
  second := true
  second_correct := by
    have prime : Nat.Prime (corner 3802 2) := by change Nat.Prime 57820817; exact ProjectEulerSolutions.P58PrimeNodes.p_57820817.prime
    exact decide_eq_true prime
  third := false
  third_correct := by
    have composite : ¬ Nat.Prime (corner 3802 3) := by change ¬ Nat.Prime 57813213; exact PrattCertificates.composite_of_product (factor := 3) (cofactor := 19271071) rfl (by decide) (by decide)
    exact decide_eq_false composite
theorem row_count_3802 : row 3802 = 1 := (row_3802.checked).trans rfl
theorem model_count_3802 : modelRow 3802 = 1 := by decide
theorem modelrow_agrees_3802 : modelRow 3802 = row 3802 := model_count_3802.trans row_count_3802.symm
def row_3803 : RowCertificate 3803 where
  first := false
  first_correct := by
    have composite : ¬ Nat.Prime (corner 3803 1) := by change ¬ Nat.Prime 57858843; exact PrattCertificates.composite_of_product (factor := 3) (cofactor := 19286281) rfl (by decide) (by decide)
    exact decide_eq_false composite
  second := false
  second_correct := by
    have composite : ¬ Nat.Prime (corner 3803 2) := by change ¬ Nat.Prime 57851237; exact PrattCertificates.composite_of_product (factor := 397) (cofactor := 145721) rfl (by decide) (by decide)
    exact decide_eq_false composite
  third := true
  third_correct := by
    have prime : Nat.Prime (corner 3803 3) := by change Nat.Prime 57843631; exact ProjectEulerSolutions.P58PrimeNodes.p_57843631.prime
    exact decide_eq_true prime
theorem row_count_3803 : row 3803 = 1 := (row_3803.checked).trans rfl
theorem model_count_3803 : modelRow 3803 = 1 := by decide
theorem modelrow_agrees_3803 : modelRow 3803 = row 3803 := model_count_3803.trans row_count_3803.symm
def row_3804 : RowCertificate 3804 where
  first := false
  first_correct := by
    have composite : ¬ Nat.Prime (corner 3804 1) := by change ¬ Nat.Prime 57889273; exact PrattCertificates.composite_of_product (factor := 13) (cofactor := 4453021) rfl (by decide) (by decide)
    exact decide_eq_false composite
  second := false
  second_correct := by
    have composite : ¬ Nat.Prime (corner 3804 2) := by change ¬ Nat.Prime 57881665; exact PrattCertificates.composite_of_product (factor := 5) (cofactor := 11576333) rfl (by decide) (by decide)
    exact decide_eq_false composite
  third := false
  third_correct := by
    have composite : ¬ Nat.Prime (corner 3804 3) := by change ¬ Nat.Prime 57874057; exact PrattCertificates.composite_of_product (factor := 19) (cofactor := 3046003) rfl (by decide) (by decide)
    exact decide_eq_false composite
theorem row_count_3804 : row 3804 = 0 := (row_3804.checked).trans rfl
theorem model_count_3804 : modelRow 3804 = 0 := by decide
theorem modelrow_agrees_3804 : modelRow 3804 = row 3804 := model_count_3804.trans row_count_3804.symm
def row_3805 : RowCertificate 3805 where
  first := false
  first_correct := by
    have composite : ¬ Nat.Prime (corner 3805 1) := by change ¬ Nat.Prime 57919711; exact PrattCertificates.composite_of_product (factor := 211) (cofactor := 274501) rfl (by decide) (by decide)
    exact decide_eq_false composite
  second := false
  second_correct := by
    have composite : ¬ Nat.Prime (corner 3805 2) := by change ¬ Nat.Prime 57912101; exact PrattCertificates.composite_of_product (factor := 13) (cofactor := 4454777) rfl (by decide) (by decide)
    exact decide_eq_false composite
  third := false
  third_correct := by
    have composite : ¬ Nat.Prime (corner 3805 3) := by change ¬ Nat.Prime 57904491; exact PrattCertificates.composite_of_product (factor := 3) (cofactor := 19301497) rfl (by decide) (by decide)
    exact decide_eq_false composite
theorem row_count_3805 : row 3805 = 0 := (row_3805.checked).trans rfl
theorem model_count_3805 : modelRow 3805 = 0 := by decide
theorem modelrow_agrees_3805 : modelRow 3805 = row 3805 := model_count_3805.trans row_count_3805.symm
def row_3806 : RowCertificate 3806 where
  first := false
  first_correct := by
    have composite : ¬ Nat.Prime (corner 3806 1) := by change ¬ Nat.Prime 57950157; exact PrattCertificates.composite_of_product (factor := 3) (cofactor := 19316719) rfl (by decide) (by decide)
    exact decide_eq_false composite
  second := false
  second_correct := by
    have composite : ¬ Nat.Prime (corner 3806 2) := by change ¬ Nat.Prime 57942545; exact PrattCertificates.composite_of_product (factor := 5) (cofactor := 11588509) rfl (by decide) (by decide)
    exact decide_eq_false composite
  third := false
  third_correct := by
    have composite : ¬ Nat.Prime (corner 3806 3) := by change ¬ Nat.Prime 57934933; exact PrattCertificates.composite_of_product (factor := 7) (cofactor := 8276419) rfl (by decide) (by decide)
    exact decide_eq_false composite
theorem row_count_3806 : row 3806 = 0 := (row_3806.checked).trans rfl
theorem model_count_3806 : modelRow 3806 = 0 := by decide
theorem modelrow_agrees_3806 : modelRow 3806 = row 3806 := model_count_3806.trans row_count_3806.symm
def row_3807 : RowCertificate 3807 where
  first := false
  first_correct := by
    have composite : ¬ Nat.Prime (corner 3807 1) := by change ¬ Nat.Prime 57980611; exact PrattCertificates.composite_of_product (factor := 13) (cofactor := 4460047) rfl (by decide) (by decide)
    exact decide_eq_false composite
  second := false
  second_correct := by
    have composite : ¬ Nat.Prime (corner 3807 2) := by change ¬ Nat.Prime 57972997; exact PrattCertificates.composite_of_product (factor := 61) (cofactor := 950377) rfl (by decide) (by decide)
    exact decide_eq_false composite
  third := false
  third_correct := by
    have composite : ¬ Nat.Prime (corner 3807 3) := by change ¬ Nat.Prime 57965383; exact PrattCertificates.composite_of_product (factor := 7) (cofactor := 8280769) rfl (by decide) (by decide)
    exact decide_eq_false composite
theorem row_count_3807 : row 3807 = 0 := (row_3807.checked).trans rfl
theorem model_count_3807 : modelRow 3807 = 0 := by decide
theorem modelrow_agrees_3807 : modelRow 3807 = row 3807 := model_count_3807.trans row_count_3807.symm
def row_3808 : RowCertificate 3808 where
  first := false
  first_correct := by
    have composite : ¬ Nat.Prime (corner 3808 1) := by change ¬ Nat.Prime 58011073; exact PrattCertificates.composite_of_product (factor := 271) (cofactor := 214063) rfl (by decide) (by decide)
    exact decide_eq_false composite
  second := false
  second_correct := by
    have composite : ¬ Nat.Prime (corner 3808 2) := by change ¬ Nat.Prime 58003457; exact PrattCertificates.composite_of_product (factor := 37) (cofactor := 1567661) rfl (by decide) (by decide)
    exact decide_eq_false composite
  third := false
  third_correct := by
    have composite : ¬ Nat.Prime (corner 3808 3) := by change ¬ Nat.Prime 57995841; exact PrattCertificates.composite_of_product (factor := 3) (cofactor := 19331947) rfl (by decide) (by decide)
    exact decide_eq_false composite
theorem row_count_3808 : row 3808 = 0 := (row_3808.checked).trans rfl
theorem model_count_3808 : modelRow 3808 = 0 := by decide
theorem modelrow_agrees_3808 : modelRow 3808 = row 3808 := model_count_3808.trans row_count_3808.symm
def row_3809 : RowCertificate 3809 where
  first := false
  first_correct := by
    have composite : ¬ Nat.Prime (corner 3809 1) := by change ¬ Nat.Prime 58041543; exact PrattCertificates.composite_of_product (factor := 3) (cofactor := 19347181) rfl (by decide) (by decide)
    exact decide_eq_false composite
  second := false
  second_correct := by
    have composite : ¬ Nat.Prime (corner 3809 2) := by change ¬ Nat.Prime 58033925; exact PrattCertificates.composite_of_product (factor := 5) (cofactor := 11606785) rfl (by decide) (by decide)
    exact decide_eq_false composite
  third := false
  third_correct := by
    have composite : ¬ Nat.Prime (corner 3809 3) := by change ¬ Nat.Prime 58026307; exact PrattCertificates.composite_of_product (factor := 43) (cofactor := 1349449) rfl (by decide) (by decide)
    exact decide_eq_false composite
theorem row_count_3809 : row 3809 = 0 := (row_3809.checked).trans rfl
theorem model_count_3809 : modelRow 3809 = 0 := by decide
theorem modelrow_agrees_3809 : modelRow 3809 = row 3809 := model_count_3809.trans row_count_3809.symm
def row_3810 : RowCertificate 3810 where
  first := false
  first_correct := by
    have composite : ¬ Nat.Prime (corner 3810 1) := by change ¬ Nat.Prime 58072021; exact PrattCertificates.composite_of_product (factor := 7) (cofactor := 8296003) rfl (by decide) (by decide)
    exact decide_eq_false composite
  second := false
  second_correct := by
    have composite : ¬ Nat.Prime (corner 3810 2) := by change ¬ Nat.Prime 58064401; exact PrattCertificates.composite_of_product (factor := 17) (cofactor := 3415553) rfl (by decide) (by decide)
    exact decide_eq_false composite
  third := true
  third_correct := by
    have prime : Nat.Prime (corner 3810 3) := by change Nat.Prime 58056781; exact ProjectEulerSolutions.P58PrimeNodes.p_58056781.prime
    exact decide_eq_true prime
theorem row_count_3810 : row 3810 = 1 := (row_3810.checked).trans rfl
theorem model_count_3810 : modelRow 3810 = 1 := by decide
theorem modelrow_agrees_3810 : modelRow 3810 = row 3810 := model_count_3810.trans row_count_3810.symm
def row_3811 : RowCertificate 3811 where
  first := false
  first_correct := by
    have composite : ¬ Nat.Prime (corner 3811 1) := by change ¬ Nat.Prime 58102507; exact PrattCertificates.composite_of_product (factor := 337) (cofactor := 172411) rfl (by decide) (by decide)
    exact decide_eq_false composite
  second := false
  second_correct := by
    have composite : ¬ Nat.Prime (corner 3811 2) := by change ¬ Nat.Prime 58094885; exact PrattCertificates.composite_of_product (factor := 5) (cofactor := 11618977) rfl (by decide) (by decide)
    exact decide_eq_false composite
  third := false
  third_correct := by
    have composite : ¬ Nat.Prime (corner 3811 3) := by change ¬ Nat.Prime 58087263; exact PrattCertificates.composite_of_product (factor := 3) (cofactor := 19362421) rfl (by decide) (by decide)
    exact decide_eq_false composite
theorem row_count_3811 : row 3811 = 0 := (row_3811.checked).trans rfl
theorem model_count_3811 : modelRow 3811 = 0 := by decide
theorem modelrow_agrees_3811 : modelRow 3811 = row 3811 := model_count_3811.trans row_count_3811.symm
def row_3812 : RowCertificate 3812 where
  first := false
  first_correct := by
    have composite : ¬ Nat.Prime (corner 3812 1) := by change ¬ Nat.Prime 58133001; exact PrattCertificates.composite_of_product (factor := 3) (cofactor := 19377667) rfl (by decide) (by decide)
    exact decide_eq_false composite
  second := true
  second_correct := by
    have prime : Nat.Prime (corner 3812 2) := by change Nat.Prime 58125377; exact ProjectEulerSolutions.P58PrimeNodes.p_58125377.prime
    exact decide_eq_true prime
  third := false
  third_correct := by
    have composite : ¬ Nat.Prime (corner 3812 3) := by change ¬ Nat.Prime 58117753; exact PrattCertificates.composite_of_product (factor := 367) (cofactor := 158359) rfl (by decide) (by decide)
    exact decide_eq_false composite
theorem row_count_3812 : row 3812 = 1 := (row_3812.checked).trans rfl
theorem model_count_3812 : modelRow 3812 = 1 := by decide
theorem modelrow_agrees_3812 : modelRow 3812 = row 3812 := model_count_3812.trans row_count_3812.symm
def row_3813 : RowCertificate 3813 where
  first := false
  first_correct := by
    have composite : ¬ Nat.Prime (corner 3813 1) := by change ¬ Nat.Prime 58163503; exact PrattCertificates.composite_of_product (factor := 19) (cofactor := 3061237) rfl (by decide) (by decide)
    exact decide_eq_false composite
  second := false
  second_correct := by
    have composite : ¬ Nat.Prime (corner 3813 2) := by change ¬ Nat.Prime 58155877; exact PrattCertificates.composite_of_product (factor := 13) (cofactor := 4473529) rfl (by decide) (by decide)
    exact decide_eq_false composite
  third := false
  third_correct := by
    have composite : ¬ Nat.Prime (corner 3813 3) := by change ¬ Nat.Prime 58148251; exact PrattCertificates.composite_of_product (factor := 7) (cofactor := 8306893) rfl (by decide) (by decide)
    exact decide_eq_false composite
theorem row_count_3813 : row 3813 = 0 := (row_3813.checked).trans rfl
theorem model_count_3813 : modelRow 3813 = 0 := by decide
theorem modelrow_agrees_3813 : modelRow 3813 = row 3813 := model_count_3813.trans row_count_3813.symm
def row_3814 : RowCertificate 3814 where
  first := false
  first_correct := by
    have composite : ¬ Nat.Prime (corner 3814 1) := by change ¬ Nat.Prime 58194013; exact PrattCertificates.composite_of_product (factor := 1321) (cofactor := 44053) rfl (by decide) (by decide)
    exact decide_eq_false composite
  second := false
  second_correct := by
    have composite : ¬ Nat.Prime (corner 3814 2) := by change ¬ Nat.Prime 58186385; exact PrattCertificates.composite_of_product (factor := 5) (cofactor := 11637277) rfl (by decide) (by decide)
    exact decide_eq_false composite
  third := false
  third_correct := by
    have composite : ¬ Nat.Prime (corner 3814 3) := by change ¬ Nat.Prime 58178757; exact PrattCertificates.composite_of_product (factor := 3) (cofactor := 19392919) rfl (by decide) (by decide)
    exact decide_eq_false composite
theorem row_count_3814 : row 3814 = 0 := (row_3814.checked).trans rfl
theorem model_count_3814 : modelRow 3814 = 0 := by decide
theorem modelrow_agrees_3814 : modelRow 3814 = row 3814 := model_count_3814.trans row_count_3814.symm
def row_3815 : RowCertificate 3815 where
  first := false
  first_correct := by
    have composite : ¬ Nat.Prime (corner 3815 1) := by change ¬ Nat.Prime 58224531; exact PrattCertificates.composite_of_product (factor := 3) (cofactor := 19408177) rfl (by decide) (by decide)
    exact decide_eq_false composite
  second := false
  second_correct := by
    have composite : ¬ Nat.Prime (corner 3815 2) := by change ¬ Nat.Prime 58216901; exact PrattCertificates.composite_of_product (factor := 3517) (cofactor := 16553) rfl (by decide) (by decide)
    exact decide_eq_false composite
  third := false
  third_correct := by
    have composite : ¬ Nat.Prime (corner 3815 3) := by change ¬ Nat.Prime 58209271; exact PrattCertificates.composite_of_product (factor := 7321) (cofactor := 7951) rfl (by decide) (by decide)
    exact decide_eq_false composite
theorem row_count_3815 : row 3815 = 0 := (row_3815.checked).trans rfl
theorem model_count_3815 : modelRow 3815 = 0 := by decide
theorem modelrow_agrees_3815 : modelRow 3815 = row 3815 := model_count_3815.trans row_count_3815.symm
def row_3816 : RowCertificate 3816 where
  first := false
  first_correct := by
    have composite : ¬ Nat.Prime (corner 3816 1) := by change ¬ Nat.Prime 58255057; exact PrattCertificates.composite_of_product (factor := 7) (cofactor := 8322151) rfl (by decide) (by decide)
    exact decide_eq_false composite
  second := false
  second_correct := by
    have composite : ¬ Nat.Prime (corner 3816 2) := by change ¬ Nat.Prime 58247425; exact PrattCertificates.composite_of_product (factor := 5) (cofactor := 11649485) rfl (by decide) (by decide)
    exact decide_eq_false composite
  third := false
  third_correct := by
    have composite : ¬ Nat.Prime (corner 3816 3) := by change ¬ Nat.Prime 58239793; exact PrattCertificates.composite_of_product (factor := 31) (cofactor := 1878703) rfl (by decide) (by decide)
    exact decide_eq_false composite
theorem row_count_3816 : row 3816 = 0 := (row_3816.checked).trans rfl
theorem model_count_3816 : modelRow 3816 = 0 := by decide
theorem modelrow_agrees_3816 : modelRow 3816 = row 3816 := model_count_3816.trans row_count_3816.symm
def row_3817 : RowCertificate 3817 where
  first := false
  first_correct := by
    have composite : ¬ Nat.Prime (corner 3817 1) := by change ¬ Nat.Prime 58285591; exact PrattCertificates.composite_of_product (factor := 7) (cofactor := 8326513) rfl (by decide) (by decide)
    exact decide_eq_false composite
  second := false
  second_correct := by
    have composite : ¬ Nat.Prime (corner 3817 2) := by change ¬ Nat.Prime 58277957; exact PrattCertificates.composite_of_product (factor := 2521) (cofactor := 23117) rfl (by decide) (by decide)
    exact decide_eq_false composite
  third := false
  third_correct := by
    have composite : ¬ Nat.Prime (corner 3817 3) := by change ¬ Nat.Prime 58270323; exact PrattCertificates.composite_of_product (factor := 3) (cofactor := 19423441) rfl (by decide) (by decide)
    exact decide_eq_false composite
theorem row_count_3817 : row 3817 = 0 := (row_3817.checked).trans rfl
theorem model_count_3817 : modelRow 3817 = 0 := by decide
theorem modelrow_agrees_3817 : modelRow 3817 = row 3817 := model_count_3817.trans row_count_3817.symm
def row_3818 : RowCertificate 3818 where
  first := false
  first_correct := by
    have composite : ¬ Nat.Prime (corner 3818 1) := by change ¬ Nat.Prime 58316133; exact PrattCertificates.composite_of_product (factor := 3) (cofactor := 19438711) rfl (by decide) (by decide)
    exact decide_eq_false composite
  second := false
  second_correct := by
    have composite : ¬ Nat.Prime (corner 3818 2) := by change ¬ Nat.Prime 58308497; exact PrattCertificates.composite_of_product (factor := 13) (cofactor := 4485269) rfl (by decide) (by decide)
    exact decide_eq_false composite
  third := false
  third_correct := by
    have composite : ¬ Nat.Prime (corner 3818 3) := by change ¬ Nat.Prime 58300861; exact PrattCertificates.composite_of_product (factor := 193) (cofactor := 302077) rfl (by decide) (by decide)
    exact decide_eq_false composite
theorem row_count_3818 : row 3818 = 0 := (row_3818.checked).trans rfl
theorem model_count_3818 : modelRow 3818 = 0 := by decide
theorem modelrow_agrees_3818 : modelRow 3818 = row 3818 := model_count_3818.trans row_count_3818.symm
def row_3819 : RowCertificate 3819 where
  first := false
  first_correct := by
    have composite : ¬ Nat.Prime (corner 3819 1) := by change ¬ Nat.Prime 58346683; exact PrattCertificates.composite_of_product (factor := 61) (cofactor := 956503) rfl (by decide) (by decide)
    exact decide_eq_false composite
  second := false
  second_correct := by
    have composite : ¬ Nat.Prime (corner 3819 2) := by change ¬ Nat.Prime 58339045; exact PrattCertificates.composite_of_product (factor := 5) (cofactor := 11667809) rfl (by decide) (by decide)
    exact decide_eq_false composite
  third := false
  third_correct := by
    have composite : ¬ Nat.Prime (corner 3819 3) := by change ¬ Nat.Prime 58331407; exact PrattCertificates.composite_of_product (factor := 733) (cofactor := 79579) rfl (by decide) (by decide)
    exact decide_eq_false composite
theorem row_count_3819 : row 3819 = 0 := (row_3819.checked).trans rfl
theorem model_count_3819 : modelRow 3819 = 0 := by decide
theorem modelrow_agrees_3819 : modelRow 3819 = row 3819 := model_count_3819.trans row_count_3819.symm
def row_3820 : RowCertificate 3820 where
  first := false
  first_correct := by
    have composite : ¬ Nat.Prime (corner 3820 1) := by change ¬ Nat.Prime 58377241; exact PrattCertificates.composite_of_product (factor := 13) (cofactor := 4490557) rfl (by decide) (by decide)
    exact decide_eq_false composite
  second := false
  second_correct := by
    have composite : ¬ Nat.Prime (corner 3820 2) := by change ¬ Nat.Prime 58369601; exact PrattCertificates.composite_of_product (factor := 281) (cofactor := 207721) rfl (by decide) (by decide)
    exact decide_eq_false composite
  third := false
  third_correct := by
    have composite : ¬ Nat.Prime (corner 3820 3) := by change ¬ Nat.Prime 58361961; exact PrattCertificates.composite_of_product (factor := 3) (cofactor := 19453987) rfl (by decide) (by decide)
    exact decide_eq_false composite
theorem row_count_3820 : row 3820 = 0 := (row_3820.checked).trans rfl
theorem model_count_3820 : modelRow 3820 = 0 := by decide
theorem modelrow_agrees_3820 : modelRow 3820 = row 3820 := model_count_3820.trans row_count_3820.symm
def row_3821 : RowCertificate 3821 where
  first := false
  first_correct := by
    have composite : ¬ Nat.Prime (corner 3821 1) := by change ¬ Nat.Prime 58407807; exact PrattCertificates.composite_of_product (factor := 3) (cofactor := 19469269) rfl (by decide) (by decide)
    exact decide_eq_false composite
  second := false
  second_correct := by
    have composite : ¬ Nat.Prime (corner 3821 2) := by change ¬ Nat.Prime 58400165; exact PrattCertificates.composite_of_product (factor := 5) (cofactor := 11680033) rfl (by decide) (by decide)
    exact decide_eq_false composite
  third := false
  third_correct := by
    have composite : ¬ Nat.Prime (corner 3821 3) := by change ¬ Nat.Prime 58392523; exact PrattCertificates.composite_of_product (factor := 7) (cofactor := 8341789) rfl (by decide) (by decide)
    exact decide_eq_false composite
theorem row_count_3821 : row 3821 = 0 := (row_3821.checked).trans rfl
theorem model_count_3821 : modelRow 3821 = 0 := by decide
theorem modelrow_agrees_3821 : modelRow 3821 = row 3821 := model_count_3821.trans row_count_3821.symm
def row_3822 : RowCertificate 3822 where
  first := false
  first_correct := by
    have composite : ¬ Nat.Prime (corner 3822 1) := by change ¬ Nat.Prime 58438381; exact PrattCertificates.composite_of_product (factor := 331) (cofactor := 176551) rfl (by decide) (by decide)
    exact decide_eq_false composite
  second := false
  second_correct := by
    have composite : ¬ Nat.Prime (corner 3822 2) := by change ¬ Nat.Prime 58430737; exact PrattCertificates.composite_of_product (factor := 29) (cofactor := 2014853) rfl (by decide) (by decide)
    exact decide_eq_false composite
  third := false
  third_correct := by
    have composite : ¬ Nat.Prime (corner 3822 3) := by change ¬ Nat.Prime 58423093; exact PrattCertificates.composite_of_product (factor := 4993) (cofactor := 11701) rfl (by decide) (by decide)
    exact decide_eq_false composite
theorem row_count_3822 : row 3822 = 0 := (row_3822.checked).trans rfl
theorem model_count_3822 : modelRow 3822 = 0 := by decide
theorem modelrow_agrees_3822 : modelRow 3822 = row 3822 := model_count_3822.trans row_count_3822.symm
def row_3823 : RowCertificate 3823 where
  first := false
  first_correct := by
    have composite : ¬ Nat.Prime (corner 3823 1) := by change ¬ Nat.Prime 58468963; exact PrattCertificates.composite_of_product (factor := 7) (cofactor := 8352709) rfl (by decide) (by decide)
    exact decide_eq_false composite
  second := false
  second_correct := by
    have composite : ¬ Nat.Prime (corner 3823 2) := by change ¬ Nat.Prime 58461317; exact PrattCertificates.composite_of_product (factor := 17) (cofactor := 3438901) rfl (by decide) (by decide)
    exact decide_eq_false composite
  third := false
  third_correct := by
    have composite : ¬ Nat.Prime (corner 3823 3) := by change ¬ Nat.Prime 58453671; exact PrattCertificates.composite_of_product (factor := 3) (cofactor := 19484557) rfl (by decide) (by decide)
    exact decide_eq_false composite
theorem row_count_3823 : row 3823 = 0 := (row_3823.checked).trans rfl
theorem model_count_3823 : modelRow 3823 = 0 := by decide
theorem modelrow_agrees_3823 : modelRow 3823 = row 3823 := model_count_3823.trans row_count_3823.symm
def row_3824 : RowCertificate 3824 where
  first := false
  first_correct := by
    have composite : ¬ Nat.Prime (corner 3824 1) := by change ¬ Nat.Prime 58499553; exact PrattCertificates.composite_of_product (factor := 3) (cofactor := 19499851) rfl (by decide) (by decide)
    exact decide_eq_false composite
  second := false
  second_correct := by
    have composite : ¬ Nat.Prime (corner 3824 2) := by change ¬ Nat.Prime 58491905; exact PrattCertificates.composite_of_product (factor := 5) (cofactor := 11698381) rfl (by decide) (by decide)
    exact decide_eq_false composite
  third := false
  third_correct := by
    have composite : ¬ Nat.Prime (corner 3824 3) := by change ¬ Nat.Prime 58484257; exact PrattCertificates.composite_of_product (factor := 13) (cofactor := 4498789) rfl (by decide) (by decide)
    exact decide_eq_false composite
theorem row_count_3824 : row 3824 = 0 := (row_3824.checked).trans rfl
theorem model_count_3824 : modelRow 3824 = 0 := by decide
theorem modelrow_agrees_3824 : modelRow 3824 = row 3824 := model_count_3824.trans row_count_3824.symm
def row_3825 : RowCertificate 3825 where
  first := false
  first_correct := by
    have composite : ¬ Nat.Prime (corner 3825 1) := by change ¬ Nat.Prime 58530151; exact PrattCertificates.composite_of_product (factor := 181) (cofactor := 323371) rfl (by decide) (by decide)
    exact decide_eq_false composite
  second := false
  second_correct := by
    have composite : ¬ Nat.Prime (corner 3825 2) := by change ¬ Nat.Prime 58522501; exact PrattCertificates.composite_of_product (factor := 3001) (cofactor := 19501) rfl (by decide) (by decide)
    exact decide_eq_false composite
  third := false
  third_correct := by
    have composite : ¬ Nat.Prime (corner 3825 3) := by change ¬ Nat.Prime 58514851; exact PrattCertificates.composite_of_product (factor := 19) (cofactor := 3079729) rfl (by decide) (by decide)
    exact decide_eq_false composite
theorem row_count_3825 : row 3825 = 0 := (row_3825.checked).trans rfl
theorem model_count_3825 : modelRow 3825 = 0 := by decide
theorem modelrow_agrees_3825 : modelRow 3825 = row 3825 := model_count_3825.trans row_count_3825.symm
def row_3826 : RowCertificate 3826 where
  first := true
  first_correct := by
    have prime : Nat.Prime (corner 3826 1) := by change Nat.Prime 58560757; exact ProjectEulerSolutions.P58PrimeNodes.p_58560757.prime
    exact decide_eq_true prime
  second := false
  second_correct := by
    have composite : ¬ Nat.Prime (corner 3826 2) := by change ¬ Nat.Prime 58553105; exact PrattCertificates.composite_of_product (factor := 5) (cofactor := 11710621) rfl (by decide) (by decide)
    exact decide_eq_false composite
  third := false
  third_correct := by
    have composite : ¬ Nat.Prime (corner 3826 3) := by change ¬ Nat.Prime 58545453; exact PrattCertificates.composite_of_product (factor := 3) (cofactor := 19515151) rfl (by decide) (by decide)
    exact decide_eq_false composite
theorem row_count_3826 : row 3826 = 1 := (row_3826.checked).trans rfl
theorem model_count_3826 : modelRow 3826 = 1 := by decide
theorem modelrow_agrees_3826 : modelRow 3826 = row 3826 := model_count_3826.trans row_count_3826.symm
def row_3827 : RowCertificate 3827 where
  first := false
  first_correct := by
    have composite : ¬ Nat.Prime (corner 3827 1) := by change ¬ Nat.Prime 58591371; exact PrattCertificates.composite_of_product (factor := 3) (cofactor := 19530457) rfl (by decide) (by decide)
    exact decide_eq_false composite
  second := false
  second_correct := by
    have composite : ¬ Nat.Prime (corner 3827 2) := by change ¬ Nat.Prime 58583717; exact PrattCertificates.composite_of_product (factor := 17) (cofactor := 3446101) rfl (by decide) (by decide)
    exact decide_eq_false composite
  third := false
  third_correct := by
    have composite : ¬ Nat.Prime (corner 3827 3) := by change ¬ Nat.Prime 58576063; exact PrattCertificates.composite_of_product (factor := 7) (cofactor := 8368009) rfl (by decide) (by decide)
    exact decide_eq_false composite
theorem row_count_3827 : row 3827 = 0 := (row_3827.checked).trans rfl
theorem model_count_3827 : modelRow 3827 = 0 := by decide
theorem modelrow_agrees_3827 : modelRow 3827 = row 3827 := model_count_3827.trans row_count_3827.symm
def row_3828 : RowCertificate 3828 where
  first := false
  first_correct := by
    have composite : ¬ Nat.Prime (corner 3828 1) := by change ¬ Nat.Prime 58621993; exact PrattCertificates.composite_of_product (factor := 73) (cofactor := 803041) rfl (by decide) (by decide)
    exact decide_eq_false composite
  second := true
  second_correct := by
    have prime : Nat.Prime (corner 3828 2) := by change Nat.Prime 58614337; exact ProjectEulerSolutions.P58PrimeNodes.p_58614337.prime
    exact decide_eq_true prime
  third := false
  third_correct := by
    have composite : ¬ Nat.Prime (corner 3828 3) := by change ¬ Nat.Prime 58606681; exact PrattCertificates.composite_of_product (factor := 7) (cofactor := 8372383) rfl (by decide) (by decide)
    exact decide_eq_false composite
theorem row_count_3828 : row 3828 = 1 := (row_3828.checked).trans rfl
theorem model_count_3828 : modelRow 3828 = 1 := by decide
theorem modelrow_agrees_3828 : modelRow 3828 = row 3828 := model_count_3828.trans row_count_3828.symm
def row_3829 : RowCertificate 3829 where
  first := false
  first_correct := by
    have composite : ¬ Nat.Prime (corner 3829 1) := by change ¬ Nat.Prime 58652623; exact PrattCertificates.composite_of_product (factor := 673) (cofactor := 87151) rfl (by decide) (by decide)
    exact decide_eq_false composite
  second := false
  second_correct := by
    have composite : ¬ Nat.Prime (corner 3829 2) := by change ¬ Nat.Prime 58644965; exact PrattCertificates.composite_of_product (factor := 5) (cofactor := 11728993) rfl (by decide) (by decide)
    exact decide_eq_false composite
  third := false
  third_correct := by
    have composite : ¬ Nat.Prime (corner 3829 3) := by change ¬ Nat.Prime 58637307; exact PrattCertificates.composite_of_product (factor := 3) (cofactor := 19545769) rfl (by decide) (by decide)
    exact decide_eq_false composite
theorem row_count_3829 : row 3829 = 0 := (row_3829.checked).trans rfl
theorem model_count_3829 : modelRow 3829 = 0 := by decide
theorem modelrow_agrees_3829 : modelRow 3829 = row 3829 := model_count_3829.trans row_count_3829.symm
def row_3830 : RowCertificate 3830 where
  first := false
  first_correct := by
    have composite : ¬ Nat.Prime (corner 3830 1) := by change ¬ Nat.Prime 58683261; exact PrattCertificates.composite_of_product (factor := 3) (cofactor := 19561087) rfl (by decide) (by decide)
    exact decide_eq_false composite
  second := false
  second_correct := by
    have composite : ¬ Nat.Prime (corner 3830 2) := by change ¬ Nat.Prime 58675601; exact PrattCertificates.composite_of_product (factor := 457) (cofactor := 128393) rfl (by decide) (by decide)
    exact decide_eq_false composite
  third := false
  third_correct := by
    have composite : ¬ Nat.Prime (corner 3830 3) := by change ¬ Nat.Prime 58667941; exact PrattCertificates.composite_of_product (factor := 1993) (cofactor := 29437) rfl (by decide) (by decide)
    exact decide_eq_false composite
theorem row_count_3830 : row 3830 = 0 := (row_3830.checked).trans rfl
theorem model_count_3830 : modelRow 3830 = 0 := by decide
theorem modelrow_agrees_3830 : modelRow 3830 = row 3830 := model_count_3830.trans row_count_3830.symm
def row_3831 : RowCertificate 3831 where
  first := false
  first_correct := by
    have composite : ¬ Nat.Prime (corner 3831 1) := by change ¬ Nat.Prime 58713907; exact PrattCertificates.composite_of_product (factor := 7) (cofactor := 8387701) rfl (by decide) (by decide)
    exact decide_eq_false composite
  second := false
  second_correct := by
    have composite : ¬ Nat.Prime (corner 3831 2) := by change ¬ Nat.Prime 58706245; exact PrattCertificates.composite_of_product (factor := 5) (cofactor := 11741249) rfl (by decide) (by decide)
    exact decide_eq_false composite
  third := true
  third_correct := by
    have prime : Nat.Prime (corner 3831 3) := by change Nat.Prime 58698583; exact ProjectEulerSolutions.P58PrimeNodes.p_58698583.prime
    exact decide_eq_true prime
theorem row_count_3831 : row 3831 = 1 := (row_3831.checked).trans rfl
theorem model_count_3831 : modelRow 3831 = 1 := by decide
theorem modelrow_agrees_3831 : modelRow 3831 = row 3831 := model_count_3831.trans row_count_3831.symm
def row_3832 : RowCertificate 3832 where
  first := false
  first_correct := by
    have composite : ¬ Nat.Prime (corner 3832 1) := by change ¬ Nat.Prime 58744561; exact PrattCertificates.composite_of_product (factor := 19) (cofactor := 3091819) rfl (by decide) (by decide)
    exact decide_eq_false composite
  second := false
  second_correct := by
    have composite : ¬ Nat.Prime (corner 3832 2) := by change ¬ Nat.Prime 58736897; exact PrattCertificates.composite_of_product (factor := 229) (cofactor := 256493) rfl (by decide) (by decide)
    exact decide_eq_false composite
  third := false
  third_correct := by
    have composite : ¬ Nat.Prime (corner 3832 3) := by change ¬ Nat.Prime 58729233; exact PrattCertificates.composite_of_product (factor := 3) (cofactor := 19576411) rfl (by decide) (by decide)
    exact decide_eq_false composite
theorem row_count_3832 : row 3832 = 0 := (row_3832.checked).trans rfl
theorem model_count_3832 : modelRow 3832 = 0 := by decide
theorem modelrow_agrees_3832 : modelRow 3832 = row 3832 := model_count_3832.trans row_count_3832.symm
def row_3833 : RowCertificate 3833 where
  first := false
  first_correct := by
    have composite : ¬ Nat.Prime (corner 3833 1) := by change ¬ Nat.Prime 58775223; exact PrattCertificates.composite_of_product (factor := 3) (cofactor := 19591741) rfl (by decide) (by decide)
    exact decide_eq_false composite
  second := false
  second_correct := by
    have composite : ¬ Nat.Prime (corner 3833 2) := by change ¬ Nat.Prime 58767557; exact PrattCertificates.composite_of_product (factor := 101) (cofactor := 581857) rfl (by decide) (by decide)
    exact decide_eq_false composite
  third := false
  third_correct := by
    have composite : ¬ Nat.Prime (corner 3833 3) := by change ¬ Nat.Prime 58759891; exact PrattCertificates.composite_of_product (factor := 2287) (cofactor := 25693) rfl (by decide) (by decide)
    exact decide_eq_false composite
theorem row_count_3833 : row 3833 = 0 := (row_3833.checked).trans rfl
theorem model_count_3833 : modelRow 3833 = 0 := by decide
theorem modelrow_agrees_3833 : modelRow 3833 = row 3833 := model_count_3833.trans row_count_3833.symm
def row_3834 : RowCertificate 3834 where
  first := false
  first_correct := by
    have composite : ¬ Nat.Prime (corner 3834 1) := by change ¬ Nat.Prime 58805893; exact PrattCertificates.composite_of_product (factor := 19) (cofactor := 3095047) rfl (by decide) (by decide)
    exact decide_eq_false composite
  second := false
  second_correct := by
    have composite : ¬ Nat.Prime (corner 3834 2) := by change ¬ Nat.Prime 58798225; exact PrattCertificates.composite_of_product (factor := 5) (cofactor := 11759645) rfl (by decide) (by decide)
    exact decide_eq_false composite
  third := false
  third_correct := by
    have composite : ¬ Nat.Prime (corner 3834 3) := by change ¬ Nat.Prime 58790557; exact PrattCertificates.composite_of_product (factor := 7) (cofactor := 8398651) rfl (by decide) (by decide)
    exact decide_eq_false composite
theorem row_count_3834 : row 3834 = 0 := (row_3834.checked).trans rfl
theorem model_count_3834 : modelRow 3834 = 0 := by decide
theorem modelrow_agrees_3834 : modelRow 3834 = row 3834 := model_count_3834.trans row_count_3834.symm
def row_3835 : RowCertificate 3835 where
  first := true
  first_correct := by
    have prime : Nat.Prime (corner 3835 1) := by change Nat.Prime 58836571; exact ProjectEulerSolutions.P58PrimeNodes.p_58836571.prime
    exact decide_eq_true prime
  second := false
  second_correct := by
    have composite : ¬ Nat.Prime (corner 3835 2) := by change ¬ Nat.Prime 58828901; exact PrattCertificates.composite_of_product (factor := 1489) (cofactor := 39509) rfl (by decide) (by decide)
    exact decide_eq_false composite
  third := false
  third_correct := by
    have composite : ¬ Nat.Prime (corner 3835 3) := by change ¬ Nat.Prime 58821231; exact PrattCertificates.composite_of_product (factor := 3) (cofactor := 19607077) rfl (by decide) (by decide)
    exact decide_eq_false composite
theorem row_count_3835 : row 3835 = 1 := (row_3835.checked).trans rfl
theorem model_count_3835 : modelRow 3835 = 1 := by decide
theorem modelrow_agrees_3835 : modelRow 3835 = row 3835 := model_count_3835.trans row_count_3835.symm
def row_3836 : RowCertificate 3836 where
  first := false
  first_correct := by
    have composite : ¬ Nat.Prime (corner 3836 1) := by change ¬ Nat.Prime 58867257; exact PrattCertificates.composite_of_product (factor := 3) (cofactor := 19622419) rfl (by decide) (by decide)
    exact decide_eq_false composite
  second := false
  second_correct := by
    have composite : ¬ Nat.Prime (corner 3836 2) := by change ¬ Nat.Prime 58859585; exact PrattCertificates.composite_of_product (factor := 5) (cofactor := 11771917) rfl (by decide) (by decide)
    exact decide_eq_false composite
  third := false
  third_correct := by
    have composite : ¬ Nat.Prime (corner 3836 3) := by change ¬ Nat.Prime 58851913; exact PrattCertificates.composite_of_product (factor := 997) (cofactor := 59029) rfl (by decide) (by decide)
    exact decide_eq_false composite
theorem row_count_3836 : row 3836 = 0 := (row_3836.checked).trans rfl
theorem model_count_3836 : modelRow 3836 = 0 := by decide
theorem modelrow_agrees_3836 : modelRow 3836 = row 3836 := model_count_3836.trans row_count_3836.symm
def row_3837 : RowCertificate 3837 where
  first := false
  first_correct := by
    have composite : ¬ Nat.Prime (corner 3837 1) := by change ¬ Nat.Prime 58897951; exact PrattCertificates.composite_of_product (factor := 7) (cofactor := 8413993) rfl (by decide) (by decide)
    exact decide_eq_false composite
  second := true
  second_correct := by
    have prime : Nat.Prime (corner 3837 2) := by change Nat.Prime 58890277; exact ProjectEulerSolutions.P58PrimeNodes.p_58890277.prime
    exact decide_eq_true prime
  third := false
  third_correct := by
    have composite : ¬ Nat.Prime (corner 3837 3) := by change ¬ Nat.Prime 58882603; exact PrattCertificates.composite_of_product (factor := 13) (cofactor := 4529431) rfl (by decide) (by decide)
    exact decide_eq_false composite
theorem row_count_3837 : row 3837 = 1 := (row_3837.checked).trans rfl
theorem model_count_3837 : modelRow 3837 = 1 := by decide
theorem modelrow_agrees_3837 : modelRow 3837 = row 3837 := model_count_3837.trans row_count_3837.symm
def row_3838 : RowCertificate 3838 where
  first := false
  first_correct := by
    have composite : ¬ Nat.Prime (corner 3838 1) := by change ¬ Nat.Prime 58928653; exact PrattCertificates.composite_of_product (factor := 7) (cofactor := 8418379) rfl (by decide) (by decide)
    exact decide_eq_false composite
  second := false
  second_correct := by
    have composite : ¬ Nat.Prime (corner 3838 2) := by change ¬ Nat.Prime 58920977; exact PrattCertificates.composite_of_product (factor := 41) (cofactor := 1437097) rfl (by decide) (by decide)
    exact decide_eq_false composite
  third := false
  third_correct := by
    have composite : ¬ Nat.Prime (corner 3838 3) := by change ¬ Nat.Prime 58913301; exact PrattCertificates.composite_of_product (factor := 3) (cofactor := 19637767) rfl (by decide) (by decide)
    exact decide_eq_false composite
theorem row_count_3838 : row 3838 = 0 := (row_3838.checked).trans rfl
theorem model_count_3838 : modelRow 3838 = 0 := by decide
theorem modelrow_agrees_3838 : modelRow 3838 = row 3838 := model_count_3838.trans row_count_3838.symm
def row_3839 : RowCertificate 3839 where
  first := false
  first_correct := by
    have composite : ¬ Nat.Prime (corner 3839 1) := by change ¬ Nat.Prime 58959363; exact PrattCertificates.composite_of_product (factor := 3) (cofactor := 19653121) rfl (by decide) (by decide)
    exact decide_eq_false composite
  second := false
  second_correct := by
    have composite : ¬ Nat.Prime (corner 3839 2) := by change ¬ Nat.Prime 58951685; exact PrattCertificates.composite_of_product (factor := 5) (cofactor := 11790337) rfl (by decide) (by decide)
    exact decide_eq_false composite
  third := true
  third_correct := by
    have prime : Nat.Prime (corner 3839 3) := by change Nat.Prime 58944007; exact ProjectEulerSolutions.P58PrimeNodes.p_58944007.prime
    exact decide_eq_true prime
theorem row_count_3839 : row 3839 = 1 := (row_3839.checked).trans rfl
theorem model_count_3839 : modelRow 3839 = 1 := by decide
theorem modelrow_agrees_3839 : modelRow 3839 = row 3839 := model_count_3839.trans row_count_3839.symm
def row_3840 : RowCertificate 3840 where
  first := false
  first_correct := by
    have composite : ¬ Nat.Prime (corner 3840 1) := by change ¬ Nat.Prime 58990081; exact PrattCertificates.composite_of_product (factor := 157) (cofactor := 375733) rfl (by decide) (by decide)
    exact decide_eq_false composite
  second := false
  second_correct := by
    have composite : ¬ Nat.Prime (corner 3840 2) := by change ¬ Nat.Prime 58982401; exact PrattCertificates.composite_of_product (factor := 17) (cofactor := 3469553) rfl (by decide) (by decide)
    exact decide_eq_false composite
  third := false
  third_correct := by
    have composite : ¬ Nat.Prime (corner 3840 3) := by change ¬ Nat.Prime 58974721; exact PrattCertificates.composite_of_product (factor := 13) (cofactor := 4536517) rfl (by decide) (by decide)
    exact decide_eq_false composite
theorem row_count_3840 : row 3840 = 0 := (row_3840.checked).trans rfl
theorem model_count_3840 : modelRow 3840 = 0 := by decide
theorem modelrow_agrees_3840 : modelRow 3840 = row 3840 := model_count_3840.trans row_count_3840.symm
private def atomic_3713 : CountSegment 1 10 13120 3712 3713 1740 1741 := CountSegment.single (first := 3712) (initial := 1740) row_count_3713 (by decide)
private theorem atomic_3713_agreement : RowAgreement 3712 3713 := RowAgreement.single (first := 3712) modelrow_agrees_3713
private def atomic_3714 : CountSegment 1 10 13120 3713 3714 1741 1742 := CountSegment.single (first := 3713) (initial := 1741) row_count_3714 (by decide)
private theorem atomic_3714_agreement : RowAgreement 3713 3714 := RowAgreement.single (first := 3713) modelrow_agrees_3714
private def atomic_3715 : CountSegment 1 10 13120 3714 3715 1742 1743 := CountSegment.single (first := 3714) (initial := 1742) row_count_3715 (by decide)
private theorem atomic_3715_agreement : RowAgreement 3714 3715 := RowAgreement.single (first := 3714) modelrow_agrees_3715
private def atomic_3716 : CountSegment 1 10 13120 3715 3716 1743 1743 := CountSegment.single (first := 3715) (initial := 1743) row_count_3716 (by decide)
private theorem atomic_3716_agreement : RowAgreement 3715 3716 := RowAgreement.single (first := 3715) modelrow_agrees_3716
private def atomic_3717 : CountSegment 1 10 13120 3716 3717 1743 1744 := CountSegment.single (first := 3716) (initial := 1743) row_count_3717 (by decide)
private theorem atomic_3717_agreement : RowAgreement 3716 3717 := RowAgreement.single (first := 3716) modelrow_agrees_3717
private def atomic_3718 : CountSegment 1 10 13120 3717 3718 1744 1744 := CountSegment.single (first := 3717) (initial := 1744) row_count_3718 (by decide)
private theorem atomic_3718_agreement : RowAgreement 3717 3718 := RowAgreement.single (first := 3717) modelrow_agrees_3718
private def atomic_3719 : CountSegment 1 10 13120 3718 3719 1744 1744 := CountSegment.single (first := 3718) (initial := 1744) row_count_3719 (by decide)
private theorem atomic_3719_agreement : RowAgreement 3718 3719 := RowAgreement.single (first := 3718) modelrow_agrees_3719
private def atomic_3720 : CountSegment 1 10 13120 3719 3720 1744 1744 := CountSegment.single (first := 3719) (initial := 1744) row_count_3720 (by decide)
private theorem atomic_3720_agreement : RowAgreement 3719 3720 := RowAgreement.single (first := 3719) modelrow_agrees_3720
private def atomic_3721 : CountSegment 1 10 13120 3720 3721 1744 1744 := CountSegment.single (first := 3720) (initial := 1744) row_count_3721 (by decide)
private theorem atomic_3721_agreement : RowAgreement 3720 3721 := RowAgreement.single (first := 3720) modelrow_agrees_3721
private def atomic_3722 : CountSegment 1 10 13120 3721 3722 1744 1744 := CountSegment.single (first := 3721) (initial := 1744) row_count_3722 (by decide)
private theorem atomic_3722_agreement : RowAgreement 3721 3722 := RowAgreement.single (first := 3721) modelrow_agrees_3722
private def atomic_3723 : CountSegment 1 10 13120 3722 3723 1744 1744 := CountSegment.single (first := 3722) (initial := 1744) row_count_3723 (by decide)
private theorem atomic_3723_agreement : RowAgreement 3722 3723 := RowAgreement.single (first := 3722) modelrow_agrees_3723
private def atomic_3724 : CountSegment 1 10 13120 3723 3724 1744 1745 := CountSegment.single (first := 3723) (initial := 1744) row_count_3724 (by decide)
private theorem atomic_3724_agreement : RowAgreement 3723 3724 := RowAgreement.single (first := 3723) modelrow_agrees_3724
private def atomic_3725 : CountSegment 1 10 13120 3724 3725 1745 1745 := CountSegment.single (first := 3724) (initial := 1745) row_count_3725 (by decide)
private theorem atomic_3725_agreement : RowAgreement 3724 3725 := RowAgreement.single (first := 3724) modelrow_agrees_3725
private def atomic_3726 : CountSegment 1 10 13120 3725 3726 1745 1745 := CountSegment.single (first := 3725) (initial := 1745) row_count_3726 (by decide)
private theorem atomic_3726_agreement : RowAgreement 3725 3726 := RowAgreement.single (first := 3725) modelrow_agrees_3726
private def atomic_3727 : CountSegment 1 10 13120 3726 3727 1745 1745 := CountSegment.single (first := 3726) (initial := 1745) row_count_3727 (by decide)
private theorem atomic_3727_agreement : RowAgreement 3726 3727 := RowAgreement.single (first := 3726) modelrow_agrees_3727
private def atomic_3728 : CountSegment 1 10 13120 3727 3728 1745 1746 := CountSegment.single (first := 3727) (initial := 1745) row_count_3728 (by decide)
private theorem atomic_3728_agreement : RowAgreement 3727 3728 := RowAgreement.single (first := 3727) modelrow_agrees_3728
private def atomic_3729 : CountSegment 1 10 13120 3728 3729 1746 1746 := CountSegment.single (first := 3728) (initial := 1746) row_count_3729 (by decide)
private theorem atomic_3729_agreement : RowAgreement 3728 3729 := RowAgreement.single (first := 3728) modelrow_agrees_3729
private def atomic_3730 : CountSegment 1 10 13120 3729 3730 1746 1748 := CountSegment.single (first := 3729) (initial := 1746) row_count_3730 (by decide)
private theorem atomic_3730_agreement : RowAgreement 3729 3730 := RowAgreement.single (first := 3729) modelrow_agrees_3730
private def atomic_3731 : CountSegment 1 10 13120 3730 3731 1748 1748 := CountSegment.single (first := 3730) (initial := 1748) row_count_3731 (by decide)
private theorem atomic_3731_agreement : RowAgreement 3730 3731 := RowAgreement.single (first := 3730) modelrow_agrees_3731
private def atomic_3732 : CountSegment 1 10 13120 3731 3732 1748 1748 := CountSegment.single (first := 3731) (initial := 1748) row_count_3732 (by decide)
private theorem atomic_3732_agreement : RowAgreement 3731 3732 := RowAgreement.single (first := 3731) modelrow_agrees_3732
private def atomic_3733 : CountSegment 1 10 13120 3732 3733 1748 1749 := CountSegment.single (first := 3732) (initial := 1748) row_count_3733 (by decide)
private theorem atomic_3733_agreement : RowAgreement 3732 3733 := RowAgreement.single (first := 3732) modelrow_agrees_3733
private def atomic_3734 : CountSegment 1 10 13120 3733 3734 1749 1750 := CountSegment.single (first := 3733) (initial := 1749) row_count_3734 (by decide)
private theorem atomic_3734_agreement : RowAgreement 3733 3734 := RowAgreement.single (first := 3733) modelrow_agrees_3734
private def atomic_3735 : CountSegment 1 10 13120 3734 3735 1750 1750 := CountSegment.single (first := 3734) (initial := 1750) row_count_3735 (by decide)
private theorem atomic_3735_agreement : RowAgreement 3734 3735 := RowAgreement.single (first := 3734) modelrow_agrees_3735
private def atomic_3736 : CountSegment 1 10 13120 3735 3736 1750 1750 := CountSegment.single (first := 3735) (initial := 1750) row_count_3736 (by decide)
private theorem atomic_3736_agreement : RowAgreement 3735 3736 := RowAgreement.single (first := 3735) modelrow_agrees_3736
private def atomic_3737 : CountSegment 1 10 13120 3736 3737 1750 1751 := CountSegment.single (first := 3736) (initial := 1750) row_count_3737 (by decide)
private theorem atomic_3737_agreement : RowAgreement 3736 3737 := RowAgreement.single (first := 3736) modelrow_agrees_3737
private def atomic_3738 : CountSegment 1 10 13120 3737 3738 1751 1751 := CountSegment.single (first := 3737) (initial := 1751) row_count_3738 (by decide)
private theorem atomic_3738_agreement : RowAgreement 3737 3738 := RowAgreement.single (first := 3737) modelrow_agrees_3738
private def atomic_3739 : CountSegment 1 10 13120 3738 3739 1751 1751 := CountSegment.single (first := 3738) (initial := 1751) row_count_3739 (by decide)
private theorem atomic_3739_agreement : RowAgreement 3738 3739 := RowAgreement.single (first := 3738) modelrow_agrees_3739
private def atomic_3740 : CountSegment 1 10 13120 3739 3740 1751 1752 := CountSegment.single (first := 3739) (initial := 1751) row_count_3740 (by decide)
private theorem atomic_3740_agreement : RowAgreement 3739 3740 := RowAgreement.single (first := 3739) modelrow_agrees_3740
private def atomic_3741 : CountSegment 1 10 13120 3740 3741 1752 1753 := CountSegment.single (first := 3740) (initial := 1752) row_count_3741 (by decide)
private theorem atomic_3741_agreement : RowAgreement 3740 3741 := RowAgreement.single (first := 3740) modelrow_agrees_3741
private def atomic_3742 : CountSegment 1 10 13120 3741 3742 1753 1753 := CountSegment.single (first := 3741) (initial := 1753) row_count_3742 (by decide)
private theorem atomic_3742_agreement : RowAgreement 3741 3742 := RowAgreement.single (first := 3741) modelrow_agrees_3742
private def atomic_3743 : CountSegment 1 10 13120 3742 3743 1753 1753 := CountSegment.single (first := 3742) (initial := 1753) row_count_3743 (by decide)
private theorem atomic_3743_agreement : RowAgreement 3742 3743 := RowAgreement.single (first := 3742) modelrow_agrees_3743
private def atomic_3744 : CountSegment 1 10 13120 3743 3744 1753 1753 := CountSegment.single (first := 3743) (initial := 1753) row_count_3744 (by decide)
private theorem atomic_3744_agreement : RowAgreement 3743 3744 := RowAgreement.single (first := 3743) modelrow_agrees_3744
private def atomic_3745 : CountSegment 1 10 13120 3744 3745 1753 1754 := CountSegment.single (first := 3744) (initial := 1753) row_count_3745 (by decide)
private theorem atomic_3745_agreement : RowAgreement 3744 3745 := RowAgreement.single (first := 3744) modelrow_agrees_3745
private def atomic_3746 : CountSegment 1 10 13120 3745 3746 1754 1754 := CountSegment.single (first := 3745) (initial := 1754) row_count_3746 (by decide)
private theorem atomic_3746_agreement : RowAgreement 3745 3746 := RowAgreement.single (first := 3745) modelrow_agrees_3746
private def atomic_3747 : CountSegment 1 10 13120 3746 3747 1754 1754 := CountSegment.single (first := 3746) (initial := 1754) row_count_3747 (by decide)
private theorem atomic_3747_agreement : RowAgreement 3746 3747 := RowAgreement.single (first := 3746) modelrow_agrees_3747
private def atomic_3748 : CountSegment 1 10 13120 3747 3748 1754 1754 := CountSegment.single (first := 3747) (initial := 1754) row_count_3748 (by decide)
private theorem atomic_3748_agreement : RowAgreement 3747 3748 := RowAgreement.single (first := 3747) modelrow_agrees_3748
private def atomic_3749 : CountSegment 1 10 13120 3748 3749 1754 1754 := CountSegment.single (first := 3748) (initial := 1754) row_count_3749 (by decide)
private theorem atomic_3749_agreement : RowAgreement 3748 3749 := RowAgreement.single (first := 3748) modelrow_agrees_3749
private def atomic_3750 : CountSegment 1 10 13120 3749 3750 1754 1754 := CountSegment.single (first := 3749) (initial := 1754) row_count_3750 (by decide)
private theorem atomic_3750_agreement : RowAgreement 3749 3750 := RowAgreement.single (first := 3749) modelrow_agrees_3750
private def atomic_3751 : CountSegment 1 10 13120 3750 3751 1754 1755 := CountSegment.single (first := 3750) (initial := 1754) row_count_3751 (by decide)
private theorem atomic_3751_agreement : RowAgreement 3750 3751 := RowAgreement.single (first := 3750) modelrow_agrees_3751
private def atomic_3752 : CountSegment 1 10 13120 3751 3752 1755 1757 := CountSegment.single (first := 3751) (initial := 1755) row_count_3752 (by decide)
private theorem atomic_3752_agreement : RowAgreement 3751 3752 := RowAgreement.single (first := 3751) modelrow_agrees_3752
private def atomic_3753 : CountSegment 1 10 13120 3752 3753 1757 1758 := CountSegment.single (first := 3752) (initial := 1757) row_count_3753 (by decide)
private theorem atomic_3753_agreement : RowAgreement 3752 3753 := RowAgreement.single (first := 3752) modelrow_agrees_3753
private def atomic_3754 : CountSegment 1 10 13120 3753 3754 1758 1758 := CountSegment.single (first := 3753) (initial := 1758) row_count_3754 (by decide)
private theorem atomic_3754_agreement : RowAgreement 3753 3754 := RowAgreement.single (first := 3753) modelrow_agrees_3754
private def atomic_3755 : CountSegment 1 10 13120 3754 3755 1758 1759 := CountSegment.single (first := 3754) (initial := 1758) row_count_3755 (by decide)
private theorem atomic_3755_agreement : RowAgreement 3754 3755 := RowAgreement.single (first := 3754) modelrow_agrees_3755
private def atomic_3756 : CountSegment 1 10 13120 3755 3756 1759 1759 := CountSegment.single (first := 3755) (initial := 1759) row_count_3756 (by decide)
private theorem atomic_3756_agreement : RowAgreement 3755 3756 := RowAgreement.single (first := 3755) modelrow_agrees_3756
private def atomic_3757 : CountSegment 1 10 13120 3756 3757 1759 1760 := CountSegment.single (first := 3756) (initial := 1759) row_count_3757 (by decide)
private theorem atomic_3757_agreement : RowAgreement 3756 3757 := RowAgreement.single (first := 3756) modelrow_agrees_3757
private def atomic_3758 : CountSegment 1 10 13120 3757 3758 1760 1761 := CountSegment.single (first := 3757) (initial := 1760) row_count_3758 (by decide)
private theorem atomic_3758_agreement : RowAgreement 3757 3758 := RowAgreement.single (first := 3757) modelrow_agrees_3758
private def atomic_3759 : CountSegment 1 10 13120 3758 3759 1761 1761 := CountSegment.single (first := 3758) (initial := 1761) row_count_3759 (by decide)
private theorem atomic_3759_agreement : RowAgreement 3758 3759 := RowAgreement.single (first := 3758) modelrow_agrees_3759
private def atomic_3760 : CountSegment 1 10 13120 3759 3760 1761 1762 := CountSegment.single (first := 3759) (initial := 1761) row_count_3760 (by decide)
private theorem atomic_3760_agreement : RowAgreement 3759 3760 := RowAgreement.single (first := 3759) modelrow_agrees_3760
private def atomic_3761 : CountSegment 1 10 13120 3760 3761 1762 1762 := CountSegment.single (first := 3760) (initial := 1762) row_count_3761 (by decide)
private theorem atomic_3761_agreement : RowAgreement 3760 3761 := RowAgreement.single (first := 3760) modelrow_agrees_3761
private def atomic_3762 : CountSegment 1 10 13120 3761 3762 1762 1764 := CountSegment.single (first := 3761) (initial := 1762) row_count_3762 (by decide)
private theorem atomic_3762_agreement : RowAgreement 3761 3762 := RowAgreement.single (first := 3761) modelrow_agrees_3762
private def atomic_3763 : CountSegment 1 10 13120 3762 3763 1764 1765 := CountSegment.single (first := 3762) (initial := 1764) row_count_3763 (by decide)
private theorem atomic_3763_agreement : RowAgreement 3762 3763 := RowAgreement.single (first := 3762) modelrow_agrees_3763
private def atomic_3764 : CountSegment 1 10 13120 3763 3764 1765 1765 := CountSegment.single (first := 3763) (initial := 1765) row_count_3764 (by decide)
private theorem atomic_3764_agreement : RowAgreement 3763 3764 := RowAgreement.single (first := 3763) modelrow_agrees_3764
private def atomic_3765 : CountSegment 1 10 13120 3764 3765 1765 1765 := CountSegment.single (first := 3764) (initial := 1765) row_count_3765 (by decide)
private theorem atomic_3765_agreement : RowAgreement 3764 3765 := RowAgreement.single (first := 3764) modelrow_agrees_3765
private def atomic_3766 : CountSegment 1 10 13120 3765 3766 1765 1765 := CountSegment.single (first := 3765) (initial := 1765) row_count_3766 (by decide)
private theorem atomic_3766_agreement : RowAgreement 3765 3766 := RowAgreement.single (first := 3765) modelrow_agrees_3766
private def atomic_3767 : CountSegment 1 10 13120 3766 3767 1765 1765 := CountSegment.single (first := 3766) (initial := 1765) row_count_3767 (by decide)
private theorem atomic_3767_agreement : RowAgreement 3766 3767 := RowAgreement.single (first := 3766) modelrow_agrees_3767
private def atomic_3768 : CountSegment 1 10 13120 3767 3768 1765 1766 := CountSegment.single (first := 3767) (initial := 1765) row_count_3768 (by decide)
private theorem atomic_3768_agreement : RowAgreement 3767 3768 := RowAgreement.single (first := 3767) modelrow_agrees_3768
private def atomic_3769 : CountSegment 1 10 13120 3768 3769 1766 1766 := CountSegment.single (first := 3768) (initial := 1766) row_count_3769 (by decide)
private theorem atomic_3769_agreement : RowAgreement 3768 3769 := RowAgreement.single (first := 3768) modelrow_agrees_3769
private def atomic_3770 : CountSegment 1 10 13120 3769 3770 1766 1767 := CountSegment.single (first := 3769) (initial := 1766) row_count_3770 (by decide)
private theorem atomic_3770_agreement : RowAgreement 3769 3770 := RowAgreement.single (first := 3769) modelrow_agrees_3770
private def atomic_3771 : CountSegment 1 10 13120 3770 3771 1767 1767 := CountSegment.single (first := 3770) (initial := 1767) row_count_3771 (by decide)
private theorem atomic_3771_agreement : RowAgreement 3770 3771 := RowAgreement.single (first := 3770) modelrow_agrees_3771
private def atomic_3772 : CountSegment 1 10 13120 3771 3772 1767 1767 := CountSegment.single (first := 3771) (initial := 1767) row_count_3772 (by decide)
private theorem atomic_3772_agreement : RowAgreement 3771 3772 := RowAgreement.single (first := 3771) modelrow_agrees_3772
private def atomic_3773 : CountSegment 1 10 13120 3772 3773 1767 1768 := CountSegment.single (first := 3772) (initial := 1767) row_count_3773 (by decide)
private theorem atomic_3773_agreement : RowAgreement 3772 3773 := RowAgreement.single (first := 3772) modelrow_agrees_3773
private def atomic_3774 : CountSegment 1 10 13120 3773 3774 1768 1769 := CountSegment.single (first := 3773) (initial := 1768) row_count_3774 (by decide)
private theorem atomic_3774_agreement : RowAgreement 3773 3774 := RowAgreement.single (first := 3773) modelrow_agrees_3774
private def atomic_3775 : CountSegment 1 10 13120 3774 3775 1769 1770 := CountSegment.single (first := 3774) (initial := 1769) row_count_3775 (by decide)
private theorem atomic_3775_agreement : RowAgreement 3774 3775 := RowAgreement.single (first := 3774) modelrow_agrees_3775
private def atomic_3776 : CountSegment 1 10 13120 3775 3776 1770 1770 := CountSegment.single (first := 3775) (initial := 1770) row_count_3776 (by decide)
private theorem atomic_3776_agreement : RowAgreement 3775 3776 := RowAgreement.single (first := 3775) modelrow_agrees_3776
private def atomic_3777 : CountSegment 1 10 13120 3776 3777 1770 1770 := CountSegment.single (first := 3776) (initial := 1770) row_count_3777 (by decide)
private theorem atomic_3777_agreement : RowAgreement 3776 3777 := RowAgreement.single (first := 3776) modelrow_agrees_3777
private def atomic_3778 : CountSegment 1 10 13120 3777 3778 1770 1770 := CountSegment.single (first := 3777) (initial := 1770) row_count_3778 (by decide)
private theorem atomic_3778_agreement : RowAgreement 3777 3778 := RowAgreement.single (first := 3777) modelrow_agrees_3778
private def atomic_3779 : CountSegment 1 10 13120 3778 3779 1770 1770 := CountSegment.single (first := 3778) (initial := 1770) row_count_3779 (by decide)
private theorem atomic_3779_agreement : RowAgreement 3778 3779 := RowAgreement.single (first := 3778) modelrow_agrees_3779
private def atomic_3780 : CountSegment 1 10 13120 3779 3780 1770 1770 := CountSegment.single (first := 3779) (initial := 1770) row_count_3780 (by decide)
private theorem atomic_3780_agreement : RowAgreement 3779 3780 := RowAgreement.single (first := 3779) modelrow_agrees_3780
private def atomic_3781 : CountSegment 1 10 13120 3780 3781 1770 1770 := CountSegment.single (first := 3780) (initial := 1770) row_count_3781 (by decide)
private theorem atomic_3781_agreement : RowAgreement 3780 3781 := RowAgreement.single (first := 3780) modelrow_agrees_3781
private def atomic_3782 : CountSegment 1 10 13120 3781 3782 1770 1770 := CountSegment.single (first := 3781) (initial := 1770) row_count_3782 (by decide)
private theorem atomic_3782_agreement : RowAgreement 3781 3782 := RowAgreement.single (first := 3781) modelrow_agrees_3782
private def atomic_3783 : CountSegment 1 10 13120 3782 3783 1770 1770 := CountSegment.single (first := 3782) (initial := 1770) row_count_3783 (by decide)
private theorem atomic_3783_agreement : RowAgreement 3782 3783 := RowAgreement.single (first := 3782) modelrow_agrees_3783
private def atomic_3784 : CountSegment 1 10 13120 3783 3784 1770 1770 := CountSegment.single (first := 3783) (initial := 1770) row_count_3784 (by decide)
private theorem atomic_3784_agreement : RowAgreement 3783 3784 := RowAgreement.single (first := 3783) modelrow_agrees_3784
private def atomic_3785 : CountSegment 1 10 13120 3784 3785 1770 1770 := CountSegment.single (first := 3784) (initial := 1770) row_count_3785 (by decide)
private theorem atomic_3785_agreement : RowAgreement 3784 3785 := RowAgreement.single (first := 3784) modelrow_agrees_3785
private def atomic_3786 : CountSegment 1 10 13120 3785 3786 1770 1771 := CountSegment.single (first := 3785) (initial := 1770) row_count_3786 (by decide)
private theorem atomic_3786_agreement : RowAgreement 3785 3786 := RowAgreement.single (first := 3785) modelrow_agrees_3786
private def atomic_3787 : CountSegment 1 10 13120 3786 3787 1771 1771 := CountSegment.single (first := 3786) (initial := 1771) row_count_3787 (by decide)
private theorem atomic_3787_agreement : RowAgreement 3786 3787 := RowAgreement.single (first := 3786) modelrow_agrees_3787
private def atomic_3788 : CountSegment 1 10 13120 3787 3788 1771 1771 := CountSegment.single (first := 3787) (initial := 1771) row_count_3788 (by decide)
private theorem atomic_3788_agreement : RowAgreement 3787 3788 := RowAgreement.single (first := 3787) modelrow_agrees_3788
private def atomic_3789 : CountSegment 1 10 13120 3788 3789 1771 1771 := CountSegment.single (first := 3788) (initial := 1771) row_count_3789 (by decide)
private theorem atomic_3789_agreement : RowAgreement 3788 3789 := RowAgreement.single (first := 3788) modelrow_agrees_3789
private def atomic_3790 : CountSegment 1 10 13120 3789 3790 1771 1772 := CountSegment.single (first := 3789) (initial := 1771) row_count_3790 (by decide)
private theorem atomic_3790_agreement : RowAgreement 3789 3790 := RowAgreement.single (first := 3789) modelrow_agrees_3790
private def atomic_3791 : CountSegment 1 10 13120 3790 3791 1772 1772 := CountSegment.single (first := 3790) (initial := 1772) row_count_3791 (by decide)
private theorem atomic_3791_agreement : RowAgreement 3790 3791 := RowAgreement.single (first := 3790) modelrow_agrees_3791
private def atomic_3792 : CountSegment 1 10 13120 3791 3792 1772 1772 := CountSegment.single (first := 3791) (initial := 1772) row_count_3792 (by decide)
private theorem atomic_3792_agreement : RowAgreement 3791 3792 := RowAgreement.single (first := 3791) modelrow_agrees_3792
private def atomic_3793 : CountSegment 1 10 13120 3792 3793 1772 1772 := CountSegment.single (first := 3792) (initial := 1772) row_count_3793 (by decide)
private theorem atomic_3793_agreement : RowAgreement 3792 3793 := RowAgreement.single (first := 3792) modelrow_agrees_3793
private def atomic_3794 : CountSegment 1 10 13120 3793 3794 1772 1772 := CountSegment.single (first := 3793) (initial := 1772) row_count_3794 (by decide)
private theorem atomic_3794_agreement : RowAgreement 3793 3794 := RowAgreement.single (first := 3793) modelrow_agrees_3794
private def atomic_3795 : CountSegment 1 10 13120 3794 3795 1772 1772 := CountSegment.single (first := 3794) (initial := 1772) row_count_3795 (by decide)
private theorem atomic_3795_agreement : RowAgreement 3794 3795 := RowAgreement.single (first := 3794) modelrow_agrees_3795
private def atomic_3796 : CountSegment 1 10 13120 3795 3796 1772 1772 := CountSegment.single (first := 3795) (initial := 1772) row_count_3796 (by decide)
private theorem atomic_3796_agreement : RowAgreement 3795 3796 := RowAgreement.single (first := 3795) modelrow_agrees_3796
private def atomic_3797 : CountSegment 1 10 13120 3796 3797 1772 1772 := CountSegment.single (first := 3796) (initial := 1772) row_count_3797 (by decide)
private theorem atomic_3797_agreement : RowAgreement 3796 3797 := RowAgreement.single (first := 3796) modelrow_agrees_3797
private def atomic_3798 : CountSegment 1 10 13120 3797 3798 1772 1773 := CountSegment.single (first := 3797) (initial := 1772) row_count_3798 (by decide)
private theorem atomic_3798_agreement : RowAgreement 3797 3798 := RowAgreement.single (first := 3797) modelrow_agrees_3798
private def atomic_3799 : CountSegment 1 10 13120 3798 3799 1773 1773 := CountSegment.single (first := 3798) (initial := 1773) row_count_3799 (by decide)
private theorem atomic_3799_agreement : RowAgreement 3798 3799 := RowAgreement.single (first := 3798) modelrow_agrees_3799
private def atomic_3800 : CountSegment 1 10 13120 3799 3800 1773 1773 := CountSegment.single (first := 3799) (initial := 1773) row_count_3800 (by decide)
private theorem atomic_3800_agreement : RowAgreement 3799 3800 := RowAgreement.single (first := 3799) modelrow_agrees_3800
private def atomic_3801 : CountSegment 1 10 13120 3800 3801 1773 1773 := CountSegment.single (first := 3800) (initial := 1773) row_count_3801 (by decide)
private theorem atomic_3801_agreement : RowAgreement 3800 3801 := RowAgreement.single (first := 3800) modelrow_agrees_3801
private def atomic_3802 : CountSegment 1 10 13120 3801 3802 1773 1774 := CountSegment.single (first := 3801) (initial := 1773) row_count_3802 (by decide)
private theorem atomic_3802_agreement : RowAgreement 3801 3802 := RowAgreement.single (first := 3801) modelrow_agrees_3802
private def atomic_3803 : CountSegment 1 10 13120 3802 3803 1774 1775 := CountSegment.single (first := 3802) (initial := 1774) row_count_3803 (by decide)
private theorem atomic_3803_agreement : RowAgreement 3802 3803 := RowAgreement.single (first := 3802) modelrow_agrees_3803
private def atomic_3804 : CountSegment 1 10 13120 3803 3804 1775 1775 := CountSegment.single (first := 3803) (initial := 1775) row_count_3804 (by decide)
private theorem atomic_3804_agreement : RowAgreement 3803 3804 := RowAgreement.single (first := 3803) modelrow_agrees_3804
private def atomic_3805 : CountSegment 1 10 13120 3804 3805 1775 1775 := CountSegment.single (first := 3804) (initial := 1775) row_count_3805 (by decide)
private theorem atomic_3805_agreement : RowAgreement 3804 3805 := RowAgreement.single (first := 3804) modelrow_agrees_3805
private def atomic_3806 : CountSegment 1 10 13120 3805 3806 1775 1775 := CountSegment.single (first := 3805) (initial := 1775) row_count_3806 (by decide)
private theorem atomic_3806_agreement : RowAgreement 3805 3806 := RowAgreement.single (first := 3805) modelrow_agrees_3806
private def atomic_3807 : CountSegment 1 10 13120 3806 3807 1775 1775 := CountSegment.single (first := 3806) (initial := 1775) row_count_3807 (by decide)
private theorem atomic_3807_agreement : RowAgreement 3806 3807 := RowAgreement.single (first := 3806) modelrow_agrees_3807
private def atomic_3808 : CountSegment 1 10 13120 3807 3808 1775 1775 := CountSegment.single (first := 3807) (initial := 1775) row_count_3808 (by decide)
private theorem atomic_3808_agreement : RowAgreement 3807 3808 := RowAgreement.single (first := 3807) modelrow_agrees_3808
private def atomic_3809 : CountSegment 1 10 13120 3808 3809 1775 1775 := CountSegment.single (first := 3808) (initial := 1775) row_count_3809 (by decide)
private theorem atomic_3809_agreement : RowAgreement 3808 3809 := RowAgreement.single (first := 3808) modelrow_agrees_3809
private def atomic_3810 : CountSegment 1 10 13120 3809 3810 1775 1776 := CountSegment.single (first := 3809) (initial := 1775) row_count_3810 (by decide)
private theorem atomic_3810_agreement : RowAgreement 3809 3810 := RowAgreement.single (first := 3809) modelrow_agrees_3810
private def atomic_3811 : CountSegment 1 10 13120 3810 3811 1776 1776 := CountSegment.single (first := 3810) (initial := 1776) row_count_3811 (by decide)
private theorem atomic_3811_agreement : RowAgreement 3810 3811 := RowAgreement.single (first := 3810) modelrow_agrees_3811
private def atomic_3812 : CountSegment 1 10 13120 3811 3812 1776 1777 := CountSegment.single (first := 3811) (initial := 1776) row_count_3812 (by decide)
private theorem atomic_3812_agreement : RowAgreement 3811 3812 := RowAgreement.single (first := 3811) modelrow_agrees_3812
private def atomic_3813 : CountSegment 1 10 13120 3812 3813 1777 1777 := CountSegment.single (first := 3812) (initial := 1777) row_count_3813 (by decide)
private theorem atomic_3813_agreement : RowAgreement 3812 3813 := RowAgreement.single (first := 3812) modelrow_agrees_3813
private def atomic_3814 : CountSegment 1 10 13120 3813 3814 1777 1777 := CountSegment.single (first := 3813) (initial := 1777) row_count_3814 (by decide)
private theorem atomic_3814_agreement : RowAgreement 3813 3814 := RowAgreement.single (first := 3813) modelrow_agrees_3814
private def atomic_3815 : CountSegment 1 10 13120 3814 3815 1777 1777 := CountSegment.single (first := 3814) (initial := 1777) row_count_3815 (by decide)
private theorem atomic_3815_agreement : RowAgreement 3814 3815 := RowAgreement.single (first := 3814) modelrow_agrees_3815
private def atomic_3816 : CountSegment 1 10 13120 3815 3816 1777 1777 := CountSegment.single (first := 3815) (initial := 1777) row_count_3816 (by decide)
private theorem atomic_3816_agreement : RowAgreement 3815 3816 := RowAgreement.single (first := 3815) modelrow_agrees_3816
private def atomic_3817 : CountSegment 1 10 13120 3816 3817 1777 1777 := CountSegment.single (first := 3816) (initial := 1777) row_count_3817 (by decide)
private theorem atomic_3817_agreement : RowAgreement 3816 3817 := RowAgreement.single (first := 3816) modelrow_agrees_3817
private def atomic_3818 : CountSegment 1 10 13120 3817 3818 1777 1777 := CountSegment.single (first := 3817) (initial := 1777) row_count_3818 (by decide)
private theorem atomic_3818_agreement : RowAgreement 3817 3818 := RowAgreement.single (first := 3817) modelrow_agrees_3818
private def atomic_3819 : CountSegment 1 10 13120 3818 3819 1777 1777 := CountSegment.single (first := 3818) (initial := 1777) row_count_3819 (by decide)
private theorem atomic_3819_agreement : RowAgreement 3818 3819 := RowAgreement.single (first := 3818) modelrow_agrees_3819
private def atomic_3820 : CountSegment 1 10 13120 3819 3820 1777 1777 := CountSegment.single (first := 3819) (initial := 1777) row_count_3820 (by decide)
private theorem atomic_3820_agreement : RowAgreement 3819 3820 := RowAgreement.single (first := 3819) modelrow_agrees_3820
private def atomic_3821 : CountSegment 1 10 13120 3820 3821 1777 1777 := CountSegment.single (first := 3820) (initial := 1777) row_count_3821 (by decide)
private theorem atomic_3821_agreement : RowAgreement 3820 3821 := RowAgreement.single (first := 3820) modelrow_agrees_3821
private def atomic_3822 : CountSegment 1 10 13120 3821 3822 1777 1777 := CountSegment.single (first := 3821) (initial := 1777) row_count_3822 (by decide)
private theorem atomic_3822_agreement : RowAgreement 3821 3822 := RowAgreement.single (first := 3821) modelrow_agrees_3822
private def atomic_3823 : CountSegment 1 10 13120 3822 3823 1777 1777 := CountSegment.single (first := 3822) (initial := 1777) row_count_3823 (by decide)
private theorem atomic_3823_agreement : RowAgreement 3822 3823 := RowAgreement.single (first := 3822) modelrow_agrees_3823
private def atomic_3824 : CountSegment 1 10 13120 3823 3824 1777 1777 := CountSegment.single (first := 3823) (initial := 1777) row_count_3824 (by decide)
private theorem atomic_3824_agreement : RowAgreement 3823 3824 := RowAgreement.single (first := 3823) modelrow_agrees_3824
private def atomic_3825 : CountSegment 1 10 13120 3824 3825 1777 1777 := CountSegment.single (first := 3824) (initial := 1777) row_count_3825 (by decide)
private theorem atomic_3825_agreement : RowAgreement 3824 3825 := RowAgreement.single (first := 3824) modelrow_agrees_3825
private def atomic_3826 : CountSegment 1 10 13120 3825 3826 1777 1778 := CountSegment.single (first := 3825) (initial := 1777) row_count_3826 (by decide)
private theorem atomic_3826_agreement : RowAgreement 3825 3826 := RowAgreement.single (first := 3825) modelrow_agrees_3826
private def atomic_3827 : CountSegment 1 10 13120 3826 3827 1778 1778 := CountSegment.single (first := 3826) (initial := 1778) row_count_3827 (by decide)
private theorem atomic_3827_agreement : RowAgreement 3826 3827 := RowAgreement.single (first := 3826) modelrow_agrees_3827
private def atomic_3828 : CountSegment 1 10 13120 3827 3828 1778 1779 := CountSegment.single (first := 3827) (initial := 1778) row_count_3828 (by decide)
private theorem atomic_3828_agreement : RowAgreement 3827 3828 := RowAgreement.single (first := 3827) modelrow_agrees_3828
private def atomic_3829 : CountSegment 1 10 13120 3828 3829 1779 1779 := CountSegment.single (first := 3828) (initial := 1779) row_count_3829 (by decide)
private theorem atomic_3829_agreement : RowAgreement 3828 3829 := RowAgreement.single (first := 3828) modelrow_agrees_3829
private def atomic_3830 : CountSegment 1 10 13120 3829 3830 1779 1779 := CountSegment.single (first := 3829) (initial := 1779) row_count_3830 (by decide)
private theorem atomic_3830_agreement : RowAgreement 3829 3830 := RowAgreement.single (first := 3829) modelrow_agrees_3830
private def atomic_3831 : CountSegment 1 10 13120 3830 3831 1779 1780 := CountSegment.single (first := 3830) (initial := 1779) row_count_3831 (by decide)
private theorem atomic_3831_agreement : RowAgreement 3830 3831 := RowAgreement.single (first := 3830) modelrow_agrees_3831
private def atomic_3832 : CountSegment 1 10 13120 3831 3832 1780 1780 := CountSegment.single (first := 3831) (initial := 1780) row_count_3832 (by decide)
private theorem atomic_3832_agreement : RowAgreement 3831 3832 := RowAgreement.single (first := 3831) modelrow_agrees_3832
private def atomic_3833 : CountSegment 1 10 13120 3832 3833 1780 1780 := CountSegment.single (first := 3832) (initial := 1780) row_count_3833 (by decide)
private theorem atomic_3833_agreement : RowAgreement 3832 3833 := RowAgreement.single (first := 3832) modelrow_agrees_3833
private def atomic_3834 : CountSegment 1 10 13120 3833 3834 1780 1780 := CountSegment.single (first := 3833) (initial := 1780) row_count_3834 (by decide)
private theorem atomic_3834_agreement : RowAgreement 3833 3834 := RowAgreement.single (first := 3833) modelrow_agrees_3834
private def atomic_3835 : CountSegment 1 10 13120 3834 3835 1780 1781 := CountSegment.single (first := 3834) (initial := 1780) row_count_3835 (by decide)
private theorem atomic_3835_agreement : RowAgreement 3834 3835 := RowAgreement.single (first := 3834) modelrow_agrees_3835
private def atomic_3836 : CountSegment 1 10 13120 3835 3836 1781 1781 := CountSegment.single (first := 3835) (initial := 1781) row_count_3836 (by decide)
private theorem atomic_3836_agreement : RowAgreement 3835 3836 := RowAgreement.single (first := 3835) modelrow_agrees_3836
private def atomic_3837 : CountSegment 1 10 13120 3836 3837 1781 1782 := CountSegment.single (first := 3836) (initial := 1781) row_count_3837 (by decide)
private theorem atomic_3837_agreement : RowAgreement 3836 3837 := RowAgreement.single (first := 3836) modelrow_agrees_3837
private def atomic_3838 : CountSegment 1 10 13120 3837 3838 1782 1782 := CountSegment.single (first := 3837) (initial := 1782) row_count_3838 (by decide)
private theorem atomic_3838_agreement : RowAgreement 3837 3838 := RowAgreement.single (first := 3837) modelrow_agrees_3838
private def atomic_3839 : CountSegment 1 10 13120 3838 3839 1782 1783 := CountSegment.single (first := 3838) (initial := 1782) row_count_3839 (by decide)
private theorem atomic_3839_agreement : RowAgreement 3838 3839 := RowAgreement.single (first := 3838) modelrow_agrees_3839
private def atomic_3840 : CountSegment 1 10 13120 3839 3840 1783 1783 := CountSegment.single (first := 3839) (initial := 1783) row_count_3840 (by decide)
private theorem atomic_3840_agreement : RowAgreement 3839 3840 := RowAgreement.single (first := 3839) modelrow_agrees_3840
private def within_029_000 : CountSegment 1 10 13120 3712 3714 1740 1742 := CountSegment.join atomic_3713 atomic_3714
private theorem within_029_000_agreement : RowAgreement 3712 3714 := RowAgreement.join atomic_3713_agreement atomic_3714_agreement
private def within_029_001 : CountSegment 1 10 13120 3714 3716 1742 1743 := CountSegment.join atomic_3715 atomic_3716
private theorem within_029_001_agreement : RowAgreement 3714 3716 := RowAgreement.join atomic_3715_agreement atomic_3716_agreement
private def within_029_002 : CountSegment 1 10 13120 3712 3716 1740 1743 := CountSegment.join within_029_000 within_029_001
private theorem within_029_002_agreement : RowAgreement 3712 3716 := RowAgreement.join within_029_000_agreement within_029_001_agreement
private def within_029_003 : CountSegment 1 10 13120 3716 3718 1743 1744 := CountSegment.join atomic_3717 atomic_3718
private theorem within_029_003_agreement : RowAgreement 3716 3718 := RowAgreement.join atomic_3717_agreement atomic_3718_agreement
private def within_029_004 : CountSegment 1 10 13120 3718 3720 1744 1744 := CountSegment.join atomic_3719 atomic_3720
private theorem within_029_004_agreement : RowAgreement 3718 3720 := RowAgreement.join atomic_3719_agreement atomic_3720_agreement
private def within_029_005 : CountSegment 1 10 13120 3716 3720 1743 1744 := CountSegment.join within_029_003 within_029_004
private theorem within_029_005_agreement : RowAgreement 3716 3720 := RowAgreement.join within_029_003_agreement within_029_004_agreement
private def within_029_006 : CountSegment 1 10 13120 3712 3720 1740 1744 := CountSegment.join within_029_002 within_029_005
private theorem within_029_006_agreement : RowAgreement 3712 3720 := RowAgreement.join within_029_002_agreement within_029_005_agreement
private def within_029_007 : CountSegment 1 10 13120 3720 3722 1744 1744 := CountSegment.join atomic_3721 atomic_3722
private theorem within_029_007_agreement : RowAgreement 3720 3722 := RowAgreement.join atomic_3721_agreement atomic_3722_agreement
private def within_029_008 : CountSegment 1 10 13120 3722 3724 1744 1745 := CountSegment.join atomic_3723 atomic_3724
private theorem within_029_008_agreement : RowAgreement 3722 3724 := RowAgreement.join atomic_3723_agreement atomic_3724_agreement
private def within_029_009 : CountSegment 1 10 13120 3720 3724 1744 1745 := CountSegment.join within_029_007 within_029_008
private theorem within_029_009_agreement : RowAgreement 3720 3724 := RowAgreement.join within_029_007_agreement within_029_008_agreement
private def within_029_010 : CountSegment 1 10 13120 3724 3726 1745 1745 := CountSegment.join atomic_3725 atomic_3726
private theorem within_029_010_agreement : RowAgreement 3724 3726 := RowAgreement.join atomic_3725_agreement atomic_3726_agreement
private def within_029_011 : CountSegment 1 10 13120 3726 3728 1745 1746 := CountSegment.join atomic_3727 atomic_3728
private theorem within_029_011_agreement : RowAgreement 3726 3728 := RowAgreement.join atomic_3727_agreement atomic_3728_agreement
private def within_029_012 : CountSegment 1 10 13120 3724 3728 1745 1746 := CountSegment.join within_029_010 within_029_011
private theorem within_029_012_agreement : RowAgreement 3724 3728 := RowAgreement.join within_029_010_agreement within_029_011_agreement
private def within_029_013 : CountSegment 1 10 13120 3720 3728 1744 1746 := CountSegment.join within_029_009 within_029_012
private theorem within_029_013_agreement : RowAgreement 3720 3728 := RowAgreement.join within_029_009_agreement within_029_012_agreement
private def within_029_014 : CountSegment 1 10 13120 3712 3728 1740 1746 := CountSegment.join within_029_006 within_029_013
private theorem within_029_014_agreement : RowAgreement 3712 3728 := RowAgreement.join within_029_006_agreement within_029_013_agreement
private def within_029_015 : CountSegment 1 10 13120 3728 3730 1746 1748 := CountSegment.join atomic_3729 atomic_3730
private theorem within_029_015_agreement : RowAgreement 3728 3730 := RowAgreement.join atomic_3729_agreement atomic_3730_agreement
private def within_029_016 : CountSegment 1 10 13120 3730 3732 1748 1748 := CountSegment.join atomic_3731 atomic_3732
private theorem within_029_016_agreement : RowAgreement 3730 3732 := RowAgreement.join atomic_3731_agreement atomic_3732_agreement
private def within_029_017 : CountSegment 1 10 13120 3728 3732 1746 1748 := CountSegment.join within_029_015 within_029_016
private theorem within_029_017_agreement : RowAgreement 3728 3732 := RowAgreement.join within_029_015_agreement within_029_016_agreement
private def within_029_018 : CountSegment 1 10 13120 3732 3734 1748 1750 := CountSegment.join atomic_3733 atomic_3734
private theorem within_029_018_agreement : RowAgreement 3732 3734 := RowAgreement.join atomic_3733_agreement atomic_3734_agreement
private def within_029_019 : CountSegment 1 10 13120 3734 3736 1750 1750 := CountSegment.join atomic_3735 atomic_3736
private theorem within_029_019_agreement : RowAgreement 3734 3736 := RowAgreement.join atomic_3735_agreement atomic_3736_agreement
private def within_029_020 : CountSegment 1 10 13120 3732 3736 1748 1750 := CountSegment.join within_029_018 within_029_019
private theorem within_029_020_agreement : RowAgreement 3732 3736 := RowAgreement.join within_029_018_agreement within_029_019_agreement
private def within_029_021 : CountSegment 1 10 13120 3728 3736 1746 1750 := CountSegment.join within_029_017 within_029_020
private theorem within_029_021_agreement : RowAgreement 3728 3736 := RowAgreement.join within_029_017_agreement within_029_020_agreement
private def within_029_022 : CountSegment 1 10 13120 3736 3738 1750 1751 := CountSegment.join atomic_3737 atomic_3738
private theorem within_029_022_agreement : RowAgreement 3736 3738 := RowAgreement.join atomic_3737_agreement atomic_3738_agreement
private def within_029_023 : CountSegment 1 10 13120 3738 3740 1751 1752 := CountSegment.join atomic_3739 atomic_3740
private theorem within_029_023_agreement : RowAgreement 3738 3740 := RowAgreement.join atomic_3739_agreement atomic_3740_agreement
private def within_029_024 : CountSegment 1 10 13120 3736 3740 1750 1752 := CountSegment.join within_029_022 within_029_023
private theorem within_029_024_agreement : RowAgreement 3736 3740 := RowAgreement.join within_029_022_agreement within_029_023_agreement
private def within_029_025 : CountSegment 1 10 13120 3740 3742 1752 1753 := CountSegment.join atomic_3741 atomic_3742
private theorem within_029_025_agreement : RowAgreement 3740 3742 := RowAgreement.join atomic_3741_agreement atomic_3742_agreement
private def within_029_026 : CountSegment 1 10 13120 3742 3744 1753 1753 := CountSegment.join atomic_3743 atomic_3744
private theorem within_029_026_agreement : RowAgreement 3742 3744 := RowAgreement.join atomic_3743_agreement atomic_3744_agreement
private def within_029_027 : CountSegment 1 10 13120 3740 3744 1752 1753 := CountSegment.join within_029_025 within_029_026
private theorem within_029_027_agreement : RowAgreement 3740 3744 := RowAgreement.join within_029_025_agreement within_029_026_agreement
private def within_029_028 : CountSegment 1 10 13120 3736 3744 1750 1753 := CountSegment.join within_029_024 within_029_027
private theorem within_029_028_agreement : RowAgreement 3736 3744 := RowAgreement.join within_029_024_agreement within_029_027_agreement
private def within_029_029 : CountSegment 1 10 13120 3728 3744 1746 1753 := CountSegment.join within_029_021 within_029_028
private theorem within_029_029_agreement : RowAgreement 3728 3744 := RowAgreement.join within_029_021_agreement within_029_028_agreement
private def within_029_030 : CountSegment 1 10 13120 3712 3744 1740 1753 := CountSegment.join within_029_014 within_029_029
private theorem within_029_030_agreement : RowAgreement 3712 3744 := RowAgreement.join within_029_014_agreement within_029_029_agreement
private def within_029_031 : CountSegment 1 10 13120 3744 3746 1753 1754 := CountSegment.join atomic_3745 atomic_3746
private theorem within_029_031_agreement : RowAgreement 3744 3746 := RowAgreement.join atomic_3745_agreement atomic_3746_agreement
private def within_029_032 : CountSegment 1 10 13120 3746 3748 1754 1754 := CountSegment.join atomic_3747 atomic_3748
private theorem within_029_032_agreement : RowAgreement 3746 3748 := RowAgreement.join atomic_3747_agreement atomic_3748_agreement
private def within_029_033 : CountSegment 1 10 13120 3744 3748 1753 1754 := CountSegment.join within_029_031 within_029_032
private theorem within_029_033_agreement : RowAgreement 3744 3748 := RowAgreement.join within_029_031_agreement within_029_032_agreement
private def within_029_034 : CountSegment 1 10 13120 3748 3750 1754 1754 := CountSegment.join atomic_3749 atomic_3750
private theorem within_029_034_agreement : RowAgreement 3748 3750 := RowAgreement.join atomic_3749_agreement atomic_3750_agreement
private def within_029_035 : CountSegment 1 10 13120 3750 3752 1754 1757 := CountSegment.join atomic_3751 atomic_3752
private theorem within_029_035_agreement : RowAgreement 3750 3752 := RowAgreement.join atomic_3751_agreement atomic_3752_agreement
private def within_029_036 : CountSegment 1 10 13120 3748 3752 1754 1757 := CountSegment.join within_029_034 within_029_035
private theorem within_029_036_agreement : RowAgreement 3748 3752 := RowAgreement.join within_029_034_agreement within_029_035_agreement
private def within_029_037 : CountSegment 1 10 13120 3744 3752 1753 1757 := CountSegment.join within_029_033 within_029_036
private theorem within_029_037_agreement : RowAgreement 3744 3752 := RowAgreement.join within_029_033_agreement within_029_036_agreement
private def within_029_038 : CountSegment 1 10 13120 3752 3754 1757 1758 := CountSegment.join atomic_3753 atomic_3754
private theorem within_029_038_agreement : RowAgreement 3752 3754 := RowAgreement.join atomic_3753_agreement atomic_3754_agreement
private def within_029_039 : CountSegment 1 10 13120 3754 3756 1758 1759 := CountSegment.join atomic_3755 atomic_3756
private theorem within_029_039_agreement : RowAgreement 3754 3756 := RowAgreement.join atomic_3755_agreement atomic_3756_agreement
private def within_029_040 : CountSegment 1 10 13120 3752 3756 1757 1759 := CountSegment.join within_029_038 within_029_039
private theorem within_029_040_agreement : RowAgreement 3752 3756 := RowAgreement.join within_029_038_agreement within_029_039_agreement
private def within_029_041 : CountSegment 1 10 13120 3756 3758 1759 1761 := CountSegment.join atomic_3757 atomic_3758
private theorem within_029_041_agreement : RowAgreement 3756 3758 := RowAgreement.join atomic_3757_agreement atomic_3758_agreement
private def within_029_042 : CountSegment 1 10 13120 3758 3760 1761 1762 := CountSegment.join atomic_3759 atomic_3760
private theorem within_029_042_agreement : RowAgreement 3758 3760 := RowAgreement.join atomic_3759_agreement atomic_3760_agreement
private def within_029_043 : CountSegment 1 10 13120 3756 3760 1759 1762 := CountSegment.join within_029_041 within_029_042
private theorem within_029_043_agreement : RowAgreement 3756 3760 := RowAgreement.join within_029_041_agreement within_029_042_agreement
private def within_029_044 : CountSegment 1 10 13120 3752 3760 1757 1762 := CountSegment.join within_029_040 within_029_043
private theorem within_029_044_agreement : RowAgreement 3752 3760 := RowAgreement.join within_029_040_agreement within_029_043_agreement
private def within_029_045 : CountSegment 1 10 13120 3744 3760 1753 1762 := CountSegment.join within_029_037 within_029_044
private theorem within_029_045_agreement : RowAgreement 3744 3760 := RowAgreement.join within_029_037_agreement within_029_044_agreement
private def within_029_046 : CountSegment 1 10 13120 3760 3762 1762 1764 := CountSegment.join atomic_3761 atomic_3762
private theorem within_029_046_agreement : RowAgreement 3760 3762 := RowAgreement.join atomic_3761_agreement atomic_3762_agreement
private def within_029_047 : CountSegment 1 10 13120 3762 3764 1764 1765 := CountSegment.join atomic_3763 atomic_3764
private theorem within_029_047_agreement : RowAgreement 3762 3764 := RowAgreement.join atomic_3763_agreement atomic_3764_agreement
private def within_029_048 : CountSegment 1 10 13120 3760 3764 1762 1765 := CountSegment.join within_029_046 within_029_047
private theorem within_029_048_agreement : RowAgreement 3760 3764 := RowAgreement.join within_029_046_agreement within_029_047_agreement
private def within_029_049 : CountSegment 1 10 13120 3764 3766 1765 1765 := CountSegment.join atomic_3765 atomic_3766
private theorem within_029_049_agreement : RowAgreement 3764 3766 := RowAgreement.join atomic_3765_agreement atomic_3766_agreement
private def within_029_050 : CountSegment 1 10 13120 3766 3768 1765 1766 := CountSegment.join atomic_3767 atomic_3768
private theorem within_029_050_agreement : RowAgreement 3766 3768 := RowAgreement.join atomic_3767_agreement atomic_3768_agreement
private def within_029_051 : CountSegment 1 10 13120 3764 3768 1765 1766 := CountSegment.join within_029_049 within_029_050
private theorem within_029_051_agreement : RowAgreement 3764 3768 := RowAgreement.join within_029_049_agreement within_029_050_agreement
private def within_029_052 : CountSegment 1 10 13120 3760 3768 1762 1766 := CountSegment.join within_029_048 within_029_051
private theorem within_029_052_agreement : RowAgreement 3760 3768 := RowAgreement.join within_029_048_agreement within_029_051_agreement
private def within_029_053 : CountSegment 1 10 13120 3768 3770 1766 1767 := CountSegment.join atomic_3769 atomic_3770
private theorem within_029_053_agreement : RowAgreement 3768 3770 := RowAgreement.join atomic_3769_agreement atomic_3770_agreement
private def within_029_054 : CountSegment 1 10 13120 3770 3772 1767 1767 := CountSegment.join atomic_3771 atomic_3772
private theorem within_029_054_agreement : RowAgreement 3770 3772 := RowAgreement.join atomic_3771_agreement atomic_3772_agreement
private def within_029_055 : CountSegment 1 10 13120 3768 3772 1766 1767 := CountSegment.join within_029_053 within_029_054
private theorem within_029_055_agreement : RowAgreement 3768 3772 := RowAgreement.join within_029_053_agreement within_029_054_agreement
private def within_029_056 : CountSegment 1 10 13120 3772 3774 1767 1769 := CountSegment.join atomic_3773 atomic_3774
private theorem within_029_056_agreement : RowAgreement 3772 3774 := RowAgreement.join atomic_3773_agreement atomic_3774_agreement
private def within_029_057 : CountSegment 1 10 13120 3774 3776 1769 1770 := CountSegment.join atomic_3775 atomic_3776
private theorem within_029_057_agreement : RowAgreement 3774 3776 := RowAgreement.join atomic_3775_agreement atomic_3776_agreement
private def within_029_058 : CountSegment 1 10 13120 3772 3776 1767 1770 := CountSegment.join within_029_056 within_029_057
private theorem within_029_058_agreement : RowAgreement 3772 3776 := RowAgreement.join within_029_056_agreement within_029_057_agreement
private def within_029_059 : CountSegment 1 10 13120 3768 3776 1766 1770 := CountSegment.join within_029_055 within_029_058
private theorem within_029_059_agreement : RowAgreement 3768 3776 := RowAgreement.join within_029_055_agreement within_029_058_agreement
private def within_029_060 : CountSegment 1 10 13120 3760 3776 1762 1770 := CountSegment.join within_029_052 within_029_059
private theorem within_029_060_agreement : RowAgreement 3760 3776 := RowAgreement.join within_029_052_agreement within_029_059_agreement
private def within_029_061 : CountSegment 1 10 13120 3744 3776 1753 1770 := CountSegment.join within_029_045 within_029_060
private theorem within_029_061_agreement : RowAgreement 3744 3776 := RowAgreement.join within_029_045_agreement within_029_060_agreement
private def within_029_062 : CountSegment 1 10 13120 3712 3776 1740 1770 := CountSegment.join within_029_030 within_029_061
private theorem within_029_062_agreement : RowAgreement 3712 3776 := RowAgreement.join within_029_030_agreement within_029_061_agreement
private def within_029_063 : CountSegment 1 10 13120 3776 3778 1770 1770 := CountSegment.join atomic_3777 atomic_3778
private theorem within_029_063_agreement : RowAgreement 3776 3778 := RowAgreement.join atomic_3777_agreement atomic_3778_agreement
private def within_029_064 : CountSegment 1 10 13120 3778 3780 1770 1770 := CountSegment.join atomic_3779 atomic_3780
private theorem within_029_064_agreement : RowAgreement 3778 3780 := RowAgreement.join atomic_3779_agreement atomic_3780_agreement
private def within_029_065 : CountSegment 1 10 13120 3776 3780 1770 1770 := CountSegment.join within_029_063 within_029_064
private theorem within_029_065_agreement : RowAgreement 3776 3780 := RowAgreement.join within_029_063_agreement within_029_064_agreement
private def within_029_066 : CountSegment 1 10 13120 3780 3782 1770 1770 := CountSegment.join atomic_3781 atomic_3782
private theorem within_029_066_agreement : RowAgreement 3780 3782 := RowAgreement.join atomic_3781_agreement atomic_3782_agreement
private def within_029_067 : CountSegment 1 10 13120 3782 3784 1770 1770 := CountSegment.join atomic_3783 atomic_3784
private theorem within_029_067_agreement : RowAgreement 3782 3784 := RowAgreement.join atomic_3783_agreement atomic_3784_agreement
private def within_029_068 : CountSegment 1 10 13120 3780 3784 1770 1770 := CountSegment.join within_029_066 within_029_067
private theorem within_029_068_agreement : RowAgreement 3780 3784 := RowAgreement.join within_029_066_agreement within_029_067_agreement
private def within_029_069 : CountSegment 1 10 13120 3776 3784 1770 1770 := CountSegment.join within_029_065 within_029_068
private theorem within_029_069_agreement : RowAgreement 3776 3784 := RowAgreement.join within_029_065_agreement within_029_068_agreement
private def within_029_070 : CountSegment 1 10 13120 3784 3786 1770 1771 := CountSegment.join atomic_3785 atomic_3786
private theorem within_029_070_agreement : RowAgreement 3784 3786 := RowAgreement.join atomic_3785_agreement atomic_3786_agreement
private def within_029_071 : CountSegment 1 10 13120 3786 3788 1771 1771 := CountSegment.join atomic_3787 atomic_3788
private theorem within_029_071_agreement : RowAgreement 3786 3788 := RowAgreement.join atomic_3787_agreement atomic_3788_agreement
private def within_029_072 : CountSegment 1 10 13120 3784 3788 1770 1771 := CountSegment.join within_029_070 within_029_071
private theorem within_029_072_agreement : RowAgreement 3784 3788 := RowAgreement.join within_029_070_agreement within_029_071_agreement
private def within_029_073 : CountSegment 1 10 13120 3788 3790 1771 1772 := CountSegment.join atomic_3789 atomic_3790
private theorem within_029_073_agreement : RowAgreement 3788 3790 := RowAgreement.join atomic_3789_agreement atomic_3790_agreement
private def within_029_074 : CountSegment 1 10 13120 3790 3792 1772 1772 := CountSegment.join atomic_3791 atomic_3792
private theorem within_029_074_agreement : RowAgreement 3790 3792 := RowAgreement.join atomic_3791_agreement atomic_3792_agreement
private def within_029_075 : CountSegment 1 10 13120 3788 3792 1771 1772 := CountSegment.join within_029_073 within_029_074
private theorem within_029_075_agreement : RowAgreement 3788 3792 := RowAgreement.join within_029_073_agreement within_029_074_agreement
private def within_029_076 : CountSegment 1 10 13120 3784 3792 1770 1772 := CountSegment.join within_029_072 within_029_075
private theorem within_029_076_agreement : RowAgreement 3784 3792 := RowAgreement.join within_029_072_agreement within_029_075_agreement
private def within_029_077 : CountSegment 1 10 13120 3776 3792 1770 1772 := CountSegment.join within_029_069 within_029_076
private theorem within_029_077_agreement : RowAgreement 3776 3792 := RowAgreement.join within_029_069_agreement within_029_076_agreement
private def within_029_078 : CountSegment 1 10 13120 3792 3794 1772 1772 := CountSegment.join atomic_3793 atomic_3794
private theorem within_029_078_agreement : RowAgreement 3792 3794 := RowAgreement.join atomic_3793_agreement atomic_3794_agreement
private def within_029_079 : CountSegment 1 10 13120 3794 3796 1772 1772 := CountSegment.join atomic_3795 atomic_3796
private theorem within_029_079_agreement : RowAgreement 3794 3796 := RowAgreement.join atomic_3795_agreement atomic_3796_agreement
private def within_029_080 : CountSegment 1 10 13120 3792 3796 1772 1772 := CountSegment.join within_029_078 within_029_079
private theorem within_029_080_agreement : RowAgreement 3792 3796 := RowAgreement.join within_029_078_agreement within_029_079_agreement
private def within_029_081 : CountSegment 1 10 13120 3796 3798 1772 1773 := CountSegment.join atomic_3797 atomic_3798
private theorem within_029_081_agreement : RowAgreement 3796 3798 := RowAgreement.join atomic_3797_agreement atomic_3798_agreement
private def within_029_082 : CountSegment 1 10 13120 3798 3800 1773 1773 := CountSegment.join atomic_3799 atomic_3800
private theorem within_029_082_agreement : RowAgreement 3798 3800 := RowAgreement.join atomic_3799_agreement atomic_3800_agreement
private def within_029_083 : CountSegment 1 10 13120 3796 3800 1772 1773 := CountSegment.join within_029_081 within_029_082
private theorem within_029_083_agreement : RowAgreement 3796 3800 := RowAgreement.join within_029_081_agreement within_029_082_agreement
private def within_029_084 : CountSegment 1 10 13120 3792 3800 1772 1773 := CountSegment.join within_029_080 within_029_083
private theorem within_029_084_agreement : RowAgreement 3792 3800 := RowAgreement.join within_029_080_agreement within_029_083_agreement
private def within_029_085 : CountSegment 1 10 13120 3800 3802 1773 1774 := CountSegment.join atomic_3801 atomic_3802
private theorem within_029_085_agreement : RowAgreement 3800 3802 := RowAgreement.join atomic_3801_agreement atomic_3802_agreement
private def within_029_086 : CountSegment 1 10 13120 3802 3804 1774 1775 := CountSegment.join atomic_3803 atomic_3804
private theorem within_029_086_agreement : RowAgreement 3802 3804 := RowAgreement.join atomic_3803_agreement atomic_3804_agreement
private def within_029_087 : CountSegment 1 10 13120 3800 3804 1773 1775 := CountSegment.join within_029_085 within_029_086
private theorem within_029_087_agreement : RowAgreement 3800 3804 := RowAgreement.join within_029_085_agreement within_029_086_agreement
private def within_029_088 : CountSegment 1 10 13120 3804 3806 1775 1775 := CountSegment.join atomic_3805 atomic_3806
private theorem within_029_088_agreement : RowAgreement 3804 3806 := RowAgreement.join atomic_3805_agreement atomic_3806_agreement
private def within_029_089 : CountSegment 1 10 13120 3806 3808 1775 1775 := CountSegment.join atomic_3807 atomic_3808
private theorem within_029_089_agreement : RowAgreement 3806 3808 := RowAgreement.join atomic_3807_agreement atomic_3808_agreement
private def within_029_090 : CountSegment 1 10 13120 3804 3808 1775 1775 := CountSegment.join within_029_088 within_029_089
private theorem within_029_090_agreement : RowAgreement 3804 3808 := RowAgreement.join within_029_088_agreement within_029_089_agreement
private def within_029_091 : CountSegment 1 10 13120 3800 3808 1773 1775 := CountSegment.join within_029_087 within_029_090
private theorem within_029_091_agreement : RowAgreement 3800 3808 := RowAgreement.join within_029_087_agreement within_029_090_agreement
private def within_029_092 : CountSegment 1 10 13120 3792 3808 1772 1775 := CountSegment.join within_029_084 within_029_091
private theorem within_029_092_agreement : RowAgreement 3792 3808 := RowAgreement.join within_029_084_agreement within_029_091_agreement
private def within_029_093 : CountSegment 1 10 13120 3776 3808 1770 1775 := CountSegment.join within_029_077 within_029_092
private theorem within_029_093_agreement : RowAgreement 3776 3808 := RowAgreement.join within_029_077_agreement within_029_092_agreement
private def within_029_094 : CountSegment 1 10 13120 3808 3810 1775 1776 := CountSegment.join atomic_3809 atomic_3810
private theorem within_029_094_agreement : RowAgreement 3808 3810 := RowAgreement.join atomic_3809_agreement atomic_3810_agreement
private def within_029_095 : CountSegment 1 10 13120 3810 3812 1776 1777 := CountSegment.join atomic_3811 atomic_3812
private theorem within_029_095_agreement : RowAgreement 3810 3812 := RowAgreement.join atomic_3811_agreement atomic_3812_agreement
private def within_029_096 : CountSegment 1 10 13120 3808 3812 1775 1777 := CountSegment.join within_029_094 within_029_095
private theorem within_029_096_agreement : RowAgreement 3808 3812 := RowAgreement.join within_029_094_agreement within_029_095_agreement
private def within_029_097 : CountSegment 1 10 13120 3812 3814 1777 1777 := CountSegment.join atomic_3813 atomic_3814
private theorem within_029_097_agreement : RowAgreement 3812 3814 := RowAgreement.join atomic_3813_agreement atomic_3814_agreement
private def within_029_098 : CountSegment 1 10 13120 3814 3816 1777 1777 := CountSegment.join atomic_3815 atomic_3816
private theorem within_029_098_agreement : RowAgreement 3814 3816 := RowAgreement.join atomic_3815_agreement atomic_3816_agreement
private def within_029_099 : CountSegment 1 10 13120 3812 3816 1777 1777 := CountSegment.join within_029_097 within_029_098
private theorem within_029_099_agreement : RowAgreement 3812 3816 := RowAgreement.join within_029_097_agreement within_029_098_agreement
private def within_029_100 : CountSegment 1 10 13120 3808 3816 1775 1777 := CountSegment.join within_029_096 within_029_099
private theorem within_029_100_agreement : RowAgreement 3808 3816 := RowAgreement.join within_029_096_agreement within_029_099_agreement
private def within_029_101 : CountSegment 1 10 13120 3816 3818 1777 1777 := CountSegment.join atomic_3817 atomic_3818
private theorem within_029_101_agreement : RowAgreement 3816 3818 := RowAgreement.join atomic_3817_agreement atomic_3818_agreement
private def within_029_102 : CountSegment 1 10 13120 3818 3820 1777 1777 := CountSegment.join atomic_3819 atomic_3820
private theorem within_029_102_agreement : RowAgreement 3818 3820 := RowAgreement.join atomic_3819_agreement atomic_3820_agreement
private def within_029_103 : CountSegment 1 10 13120 3816 3820 1777 1777 := CountSegment.join within_029_101 within_029_102
private theorem within_029_103_agreement : RowAgreement 3816 3820 := RowAgreement.join within_029_101_agreement within_029_102_agreement
private def within_029_104 : CountSegment 1 10 13120 3820 3822 1777 1777 := CountSegment.join atomic_3821 atomic_3822
private theorem within_029_104_agreement : RowAgreement 3820 3822 := RowAgreement.join atomic_3821_agreement atomic_3822_agreement
private def within_029_105 : CountSegment 1 10 13120 3822 3824 1777 1777 := CountSegment.join atomic_3823 atomic_3824
private theorem within_029_105_agreement : RowAgreement 3822 3824 := RowAgreement.join atomic_3823_agreement atomic_3824_agreement
private def within_029_106 : CountSegment 1 10 13120 3820 3824 1777 1777 := CountSegment.join within_029_104 within_029_105
private theorem within_029_106_agreement : RowAgreement 3820 3824 := RowAgreement.join within_029_104_agreement within_029_105_agreement
private def within_029_107 : CountSegment 1 10 13120 3816 3824 1777 1777 := CountSegment.join within_029_103 within_029_106
private theorem within_029_107_agreement : RowAgreement 3816 3824 := RowAgreement.join within_029_103_agreement within_029_106_agreement
private def within_029_108 : CountSegment 1 10 13120 3808 3824 1775 1777 := CountSegment.join within_029_100 within_029_107
private theorem within_029_108_agreement : RowAgreement 3808 3824 := RowAgreement.join within_029_100_agreement within_029_107_agreement
private def within_029_109 : CountSegment 1 10 13120 3824 3826 1777 1778 := CountSegment.join atomic_3825 atomic_3826
private theorem within_029_109_agreement : RowAgreement 3824 3826 := RowAgreement.join atomic_3825_agreement atomic_3826_agreement
private def within_029_110 : CountSegment 1 10 13120 3826 3828 1778 1779 := CountSegment.join atomic_3827 atomic_3828
private theorem within_029_110_agreement : RowAgreement 3826 3828 := RowAgreement.join atomic_3827_agreement atomic_3828_agreement
private def within_029_111 : CountSegment 1 10 13120 3824 3828 1777 1779 := CountSegment.join within_029_109 within_029_110
private theorem within_029_111_agreement : RowAgreement 3824 3828 := RowAgreement.join within_029_109_agreement within_029_110_agreement
private def within_029_112 : CountSegment 1 10 13120 3828 3830 1779 1779 := CountSegment.join atomic_3829 atomic_3830
private theorem within_029_112_agreement : RowAgreement 3828 3830 := RowAgreement.join atomic_3829_agreement atomic_3830_agreement
private def within_029_113 : CountSegment 1 10 13120 3830 3832 1779 1780 := CountSegment.join atomic_3831 atomic_3832
private theorem within_029_113_agreement : RowAgreement 3830 3832 := RowAgreement.join atomic_3831_agreement atomic_3832_agreement
private def within_029_114 : CountSegment 1 10 13120 3828 3832 1779 1780 := CountSegment.join within_029_112 within_029_113
private theorem within_029_114_agreement : RowAgreement 3828 3832 := RowAgreement.join within_029_112_agreement within_029_113_agreement
private def within_029_115 : CountSegment 1 10 13120 3824 3832 1777 1780 := CountSegment.join within_029_111 within_029_114
private theorem within_029_115_agreement : RowAgreement 3824 3832 := RowAgreement.join within_029_111_agreement within_029_114_agreement
private def within_029_116 : CountSegment 1 10 13120 3832 3834 1780 1780 := CountSegment.join atomic_3833 atomic_3834
private theorem within_029_116_agreement : RowAgreement 3832 3834 := RowAgreement.join atomic_3833_agreement atomic_3834_agreement
private def within_029_117 : CountSegment 1 10 13120 3834 3836 1780 1781 := CountSegment.join atomic_3835 atomic_3836
private theorem within_029_117_agreement : RowAgreement 3834 3836 := RowAgreement.join atomic_3835_agreement atomic_3836_agreement
private def within_029_118 : CountSegment 1 10 13120 3832 3836 1780 1781 := CountSegment.join within_029_116 within_029_117
private theorem within_029_118_agreement : RowAgreement 3832 3836 := RowAgreement.join within_029_116_agreement within_029_117_agreement
private def within_029_119 : CountSegment 1 10 13120 3836 3838 1781 1782 := CountSegment.join atomic_3837 atomic_3838
private theorem within_029_119_agreement : RowAgreement 3836 3838 := RowAgreement.join atomic_3837_agreement atomic_3838_agreement
private def within_029_120 : CountSegment 1 10 13120 3838 3840 1782 1783 := CountSegment.join atomic_3839 atomic_3840
private theorem within_029_120_agreement : RowAgreement 3838 3840 := RowAgreement.join atomic_3839_agreement atomic_3840_agreement
private def within_029_121 : CountSegment 1 10 13120 3836 3840 1781 1783 := CountSegment.join within_029_119 within_029_120
private theorem within_029_121_agreement : RowAgreement 3836 3840 := RowAgreement.join within_029_119_agreement within_029_120_agreement
private def within_029_122 : CountSegment 1 10 13120 3832 3840 1780 1783 := CountSegment.join within_029_118 within_029_121
private theorem within_029_122_agreement : RowAgreement 3832 3840 := RowAgreement.join within_029_118_agreement within_029_121_agreement
private def within_029_123 : CountSegment 1 10 13120 3824 3840 1777 1783 := CountSegment.join within_029_115 within_029_122
private theorem within_029_123_agreement : RowAgreement 3824 3840 := RowAgreement.join within_029_115_agreement within_029_122_agreement
private def within_029_124 : CountSegment 1 10 13120 3808 3840 1775 1783 := CountSegment.join within_029_108 within_029_123
private theorem within_029_124_agreement : RowAgreement 3808 3840 := RowAgreement.join within_029_108_agreement within_029_123_agreement
private def within_029_125 : CountSegment 1 10 13120 3776 3840 1770 1783 := CountSegment.join within_029_093 within_029_124
private theorem within_029_125_agreement : RowAgreement 3776 3840 := RowAgreement.join within_029_093_agreement within_029_124_agreement
private def within_029_126 : CountSegment 1 10 13120 3712 3840 1740 1783 := CountSegment.join within_029_062 within_029_125
private theorem within_029_126_agreement : RowAgreement 3712 3840 := RowAgreement.join within_029_062_agreement within_029_125_agreement
def segment_029 : CountSegment 1 10 13120 3712 3840 1740 1783 := within_029_126
theorem agreement_029 : RowAgreement 3712 3840 := within_029_126_agreement
end ProjectEulerSolutions.P58Numerical
