import ProjectEulerSolutions.Lib.SpiralPrimeSegments
import ProjectEulerSolutions.Lib.SpiralPrimeCertificates.All
import ProjectEulerSolutions.Lib.MillerRabin
import ProjectEulerSolutions.Lib.SpiralMillerAgreement
/-! Finite spiral row and counter certificates. Python execution is checked separately. -/
namespace ProjectEulerSolutions.P58Numerical
open SpiralPrimes
set_option Elab.async false
def row_3969 : RowCertificate 3969 where
  first := true
  first_correct := by
    have prime : Nat.Prime (corner 3969 1) := by change Nat.Prime 63019783; exact ProjectEulerSolutions.P58PrimeNodes.p_63019783.prime
    exact decide_eq_true prime
  second := false
  second_correct := by
    have composite : ¬ Nat.Prime (corner 3969 2) := by change ¬ Nat.Prime 63011845; exact PrattCertificates.composite_of_product (factor := 5) (cofactor := 12602369) rfl (by decide) (by decide)
    exact decide_eq_false composite
  third := false
  third_correct := by
    have composite : ¬ Nat.Prime (corner 3969 3) := by change ¬ Nat.Prime 63003907; exact PrattCertificates.composite_of_product (factor := 241) (cofactor := 261427) rfl (by decide) (by decide)
    exact decide_eq_false composite
theorem row_count_3969 : row 3969 = 1 := (row_3969.checked).trans rfl
theorem model_count_3969 : modelRow 3969 = 1 := by decide
theorem modelrow_agrees_3969 : modelRow 3969 = row 3969 := model_count_3969.trans row_count_3969.symm
def row_3970 : RowCertificate 3970 where
  first := false
  first_correct := by
    have composite : ¬ Nat.Prime (corner 3970 1) := by change ¬ Nat.Prime 63051541; exact PrattCertificates.composite_of_product (factor := 7) (cofactor := 9007363) rfl (by decide) (by decide)
    exact decide_eq_false composite
  second := false
  second_correct := by
    have composite : ¬ Nat.Prime (corner 3970 2) := by change ¬ Nat.Prime 63043601; exact PrattCertificates.composite_of_product (factor := 337) (cofactor := 187073) rfl (by decide) (by decide)
    exact decide_eq_false composite
  third := false
  third_correct := by
    have composite : ¬ Nat.Prime (corner 3970 3) := by change ¬ Nat.Prime 63035661; exact PrattCertificates.composite_of_product (factor := 3) (cofactor := 21011887) rfl (by decide) (by decide)
    exact decide_eq_false composite
theorem row_count_3970 : row 3970 = 0 := (row_3970.checked).trans rfl
theorem model_count_3970 : modelRow 3970 = 0 := by decide
theorem modelrow_agrees_3970 : modelRow 3970 = row 3970 := model_count_3970.trans row_count_3970.symm
def row_3971 : RowCertificate 3971 where
  first := false
  first_correct := by
    have composite : ¬ Nat.Prime (corner 3971 1) := by change ¬ Nat.Prime 63083307; exact PrattCertificates.composite_of_product (factor := 3) (cofactor := 21027769) rfl (by decide) (by decide)
    exact decide_eq_false composite
  second := false
  second_correct := by
    have composite : ¬ Nat.Prime (corner 3971 2) := by change ¬ Nat.Prime 63075365; exact PrattCertificates.composite_of_product (factor := 5) (cofactor := 12615073) rfl (by decide) (by decide)
    exact decide_eq_false composite
  third := false
  third_correct := by
    have composite : ¬ Nat.Prime (corner 3971 3) := by change ¬ Nat.Prime 63067423; exact PrattCertificates.composite_of_product (factor := 31) (cofactor := 2034433) rfl (by decide) (by decide)
    exact decide_eq_false composite
theorem row_count_3971 : row 3971 = 0 := (row_3971.checked).trans rfl
theorem model_count_3971 : modelRow 3971 = 0 := by decide
theorem modelrow_agrees_3971 : modelRow 3971 = row 3971 := model_count_3971.trans row_count_3971.symm
def row_3972 : RowCertificate 3972 where
  first := false
  first_correct := by
    have composite : ¬ Nat.Prime (corner 3972 1) := by change ¬ Nat.Prime 63115081; exact PrattCertificates.composite_of_product (factor := 37) (cofactor := 1705813) rfl (by decide) (by decide)
    exact decide_eq_false composite
  second := true
  second_correct := by
    have prime : Nat.Prime (corner 3972 2) := by change Nat.Prime 63107137; exact ProjectEulerSolutions.P58PrimeNodes.p_63107137.prime
    exact decide_eq_true prime
  third := false
  third_correct := by
    have composite : ¬ Nat.Prime (corner 3972 3) := by change ¬ Nat.Prime 63099193; exact PrattCertificates.composite_of_product (factor := 61) (cofactor := 1034413) rfl (by decide) (by decide)
    exact decide_eq_false composite
theorem row_count_3972 : row 3972 = 1 := (row_3972.checked).trans rfl
theorem model_count_3972 : modelRow 3972 = 1 := by decide
theorem modelrow_agrees_3972 : modelRow 3972 = row 3972 := model_count_3972.trans row_count_3972.symm
def row_3973 : RowCertificate 3973 where
  first := false
  first_correct := by
    have composite : ¬ Nat.Prime (corner 3973 1) := by change ¬ Nat.Prime 63146863; exact PrattCertificates.composite_of_product (factor := 13) (cofactor := 4857451) rfl (by decide) (by decide)
    exact decide_eq_false composite
  second := true
  second_correct := by
    have prime : Nat.Prime (corner 3973 2) := by change Nat.Prime 63138917; exact ProjectEulerSolutions.P58PrimeNodes.p_63138917.prime
    exact decide_eq_true prime
  third := false
  third_correct := by
    have composite : ¬ Nat.Prime (corner 3973 3) := by change ¬ Nat.Prime 63130971; exact PrattCertificates.composite_of_product (factor := 3) (cofactor := 21043657) rfl (by decide) (by decide)
    exact decide_eq_false composite
theorem row_count_3973 : row 3973 = 1 := (row_3973.checked).trans rfl
theorem model_count_3973 : modelRow 3973 = 1 := by decide
theorem modelrow_agrees_3973 : modelRow 3973 = row 3973 := model_count_3973.trans row_count_3973.symm
def row_3974 : RowCertificate 3974 where
  first := false
  first_correct := by
    have composite : ¬ Nat.Prime (corner 3974 1) := by change ¬ Nat.Prime 63178653; exact PrattCertificates.composite_of_product (factor := 3) (cofactor := 21059551) rfl (by decide) (by decide)
    exact decide_eq_false composite
  second := false
  second_correct := by
    have composite : ¬ Nat.Prime (corner 3974 2) := by change ¬ Nat.Prime 63170705; exact PrattCertificates.composite_of_product (factor := 5) (cofactor := 12634141) rfl (by decide) (by decide)
    exact decide_eq_false composite
  third := false
  third_correct := by
    have composite : ¬ Nat.Prime (corner 3974 3) := by change ¬ Nat.Prime 63162757; exact PrattCertificates.composite_of_product (factor := 7) (cofactor := 9023251) rfl (by decide) (by decide)
    exact decide_eq_false composite
theorem row_count_3974 : row 3974 = 0 := (row_3974.checked).trans rfl
theorem model_count_3974 : modelRow 3974 = 0 := by decide
theorem modelrow_agrees_3974 : modelRow 3974 = row 3974 := model_count_3974.trans row_count_3974.symm
def row_3975 : RowCertificate 3975 where
  first := true
  first_correct := by
    have prime : Nat.Prime (corner 3975 1) := by change Nat.Prime 63210451; exact ProjectEulerSolutions.P58PrimeNodes.p_63210451.prime
    exact decide_eq_true prime
  second := false
  second_correct := by
    have composite : ¬ Nat.Prime (corner 3975 2) := by change ¬ Nat.Prime 63202501; exact PrattCertificates.composite_of_product (factor := 7673) (cofactor := 8237) rfl (by decide) (by decide)
    exact decide_eq_false composite
  third := false
  third_correct := by
    have composite : ¬ Nat.Prime (corner 3975 3) := by change ¬ Nat.Prime 63194551; exact PrattCertificates.composite_of_product (factor := 7) (cofactor := 9027793) rfl (by decide) (by decide)
    exact decide_eq_false composite
theorem row_count_3975 : row 3975 = 1 := (row_3975.checked).trans rfl
theorem model_count_3975 : modelRow 3975 = 1 := by decide
theorem modelrow_agrees_3975 : modelRow 3975 = row 3975 := model_count_3975.trans row_count_3975.symm
def row_3976 : RowCertificate 3976 where
  first := false
  first_correct := by
    have composite : ¬ Nat.Prime (corner 3976 1) := by change ¬ Nat.Prime 63242257; exact PrattCertificates.composite_of_product (factor := 13) (cofactor := 4864789) rfl (by decide) (by decide)
    exact decide_eq_false composite
  second := false
  second_correct := by
    have composite : ¬ Nat.Prime (corner 3976 2) := by change ¬ Nat.Prime 63234305; exact PrattCertificates.composite_of_product (factor := 5) (cofactor := 12646861) rfl (by decide) (by decide)
    exact decide_eq_false composite
  third := false
  third_correct := by
    have composite : ¬ Nat.Prime (corner 3976 3) := by change ¬ Nat.Prime 63226353; exact PrattCertificates.composite_of_product (factor := 3) (cofactor := 21075451) rfl (by decide) (by decide)
    exact decide_eq_false composite
theorem row_count_3976 : row 3976 = 0 := (row_3976.checked).trans rfl
theorem model_count_3976 : modelRow 3976 = 0 := by decide
theorem modelrow_agrees_3976 : modelRow 3976 = row 3976 := model_count_3976.trans row_count_3976.symm
def row_3977 : RowCertificate 3977 where
  first := false
  first_correct := by
    have composite : ¬ Nat.Prime (corner 3977 1) := by change ¬ Nat.Prime 63274071; exact PrattCertificates.composite_of_product (factor := 3) (cofactor := 21091357) rfl (by decide) (by decide)
    exact decide_eq_false composite
  second := false
  second_correct := by
    have composite : ¬ Nat.Prime (corner 3977 2) := by change ¬ Nat.Prime 63266117; exact PrattCertificates.composite_of_product (factor := 1429) (cofactor := 44273) rfl (by decide) (by decide)
    exact decide_eq_false composite
  third := false
  third_correct := by
    have composite : ¬ Nat.Prime (corner 3977 3) := by change ¬ Nat.Prime 63258163; exact PrattCertificates.composite_of_product (factor := 19) (cofactor := 3329377) rfl (by decide) (by decide)
    exact decide_eq_false composite
theorem row_count_3977 : row 3977 = 0 := (row_3977.checked).trans rfl
theorem model_count_3977 : modelRow 3977 = 0 := by decide
theorem modelrow_agrees_3977 : modelRow 3977 = row 3977 := model_count_3977.trans row_count_3977.symm
def row_3978 : RowCertificate 3978 where
  first := false
  first_correct := by
    have composite : ¬ Nat.Prime (corner 3978 1) := by change ¬ Nat.Prime 63305893; exact PrattCertificates.composite_of_product (factor := 7) (cofactor := 9043699) rfl (by decide) (by decide)
    exact decide_eq_false composite
  second := true
  second_correct := by
    have prime : Nat.Prime (corner 3978 2) := by change Nat.Prime 63297937; exact ProjectEulerSolutions.P58PrimeNodes.p_63297937.prime
    exact decide_eq_true prime
  third := false
  third_correct := by
    have composite : ¬ Nat.Prime (corner 3978 3) := by change ¬ Nat.Prime 63289981; exact PrattCertificates.composite_of_product (factor := 79) (cofactor := 801139) rfl (by decide) (by decide)
    exact decide_eq_false composite
theorem row_count_3978 : row 3978 = 1 := (row_3978.checked).trans rfl
theorem model_count_3978 : modelRow 3978 = 1 := by decide
theorem modelrow_agrees_3978 : modelRow 3978 = row 3978 := model_count_3978.trans row_count_3978.symm
def row_3979 : RowCertificate 3979 where
  first := false
  first_correct := by
    have composite : ¬ Nat.Prime (corner 3979 1) := by change ¬ Nat.Prime 63337723; exact PrattCertificates.composite_of_product (factor := 1453) (cofactor := 43591) rfl (by decide) (by decide)
    exact decide_eq_false composite
  second := false
  second_correct := by
    have composite : ¬ Nat.Prime (corner 3979 2) := by change ¬ Nat.Prime 63329765; exact PrattCertificates.composite_of_product (factor := 5) (cofactor := 12665953) rfl (by decide) (by decide)
    exact decide_eq_false composite
  third := false
  third_correct := by
    have composite : ¬ Nat.Prime (corner 3979 3) := by change ¬ Nat.Prime 63321807; exact PrattCertificates.composite_of_product (factor := 3) (cofactor := 21107269) rfl (by decide) (by decide)
    exact decide_eq_false composite
theorem row_count_3979 : row 3979 = 0 := (row_3979.checked).trans rfl
theorem model_count_3979 : modelRow 3979 = 0 := by decide
theorem modelrow_agrees_3979 : modelRow 3979 = row 3979 := model_count_3979.trans row_count_3979.symm
def row_3980 : RowCertificate 3980 where
  first := false
  first_correct := by
    have composite : ¬ Nat.Prime (corner 3980 1) := by change ¬ Nat.Prime 63369561; exact PrattCertificates.composite_of_product (factor := 3) (cofactor := 21123187) rfl (by decide) (by decide)
    exact decide_eq_false composite
  second := false
  second_correct := by
    have composite : ¬ Nat.Prime (corner 3980 2) := by change ¬ Nat.Prime 63361601; exact PrattCertificates.composite_of_product (factor := 17) (cofactor := 3727153) rfl (by decide) (by decide)
    exact decide_eq_false composite
  third := false
  third_correct := by
    have composite : ¬ Nat.Prime (corner 3980 3) := by change ¬ Nat.Prime 63353641; exact PrattCertificates.composite_of_product (factor := 13) (cofactor := 4873357) rfl (by decide) (by decide)
    exact decide_eq_false composite
theorem row_count_3980 : row 3980 = 0 := (row_3980.checked).trans rfl
theorem model_count_3980 : modelRow 3980 = 0 := by decide
theorem modelrow_agrees_3980 : modelRow 3980 = row 3980 := model_count_3980.trans row_count_3980.symm
def row_3981 : RowCertificate 3981 where
  first := false
  first_correct := by
    have composite : ¬ Nat.Prime (corner 3981 1) := by change ¬ Nat.Prime 63401407; exact PrattCertificates.composite_of_product (factor := 2887) (cofactor := 21961) rfl (by decide) (by decide)
    exact decide_eq_false composite
  second := false
  second_correct := by
    have composite : ¬ Nat.Prime (corner 3981 2) := by change ¬ Nat.Prime 63393445; exact PrattCertificates.composite_of_product (factor := 5) (cofactor := 12678689) rfl (by decide) (by decide)
    exact decide_eq_false composite
  third := false
  third_correct := by
    have composite : ¬ Nat.Prime (corner 3981 3) := by change ¬ Nat.Prime 63385483; exact PrattCertificates.composite_of_product (factor := 7) (cofactor := 9055069) rfl (by decide) (by decide)
    exact decide_eq_false composite
theorem row_count_3981 : row 3981 = 0 := (row_3981.checked).trans rfl
theorem model_count_3981 : modelRow 3981 = 0 := by decide
theorem modelrow_agrees_3981 : modelRow 3981 = row 3981 := model_count_3981.trans row_count_3981.symm
def row_3982 : RowCertificate 3982 where
  first := false
  first_correct := by
    have composite : ¬ Nat.Prime (corner 3982 1) := by change ¬ Nat.Prime 63433261; exact PrattCertificates.composite_of_product (factor := 307) (cofactor := 206623) rfl (by decide) (by decide)
    exact decide_eq_false composite
  second := false
  second_correct := by
    have composite : ¬ Nat.Prime (corner 3982 2) := by change ¬ Nat.Prime 63425297; exact PrattCertificates.composite_of_product (factor := 13) (cofactor := 4878869) rfl (by decide) (by decide)
    exact decide_eq_false composite
  third := false
  third_correct := by
    have composite : ¬ Nat.Prime (corner 3982 3) := by change ¬ Nat.Prime 63417333; exact PrattCertificates.composite_of_product (factor := 3) (cofactor := 21139111) rfl (by decide) (by decide)
    exact decide_eq_false composite
theorem row_count_3982 : row 3982 = 0 := (row_3982.checked).trans rfl
theorem model_count_3982 : modelRow 3982 = 0 := by decide
theorem modelrow_agrees_3982 : modelRow 3982 = row 3982 := model_count_3982.trans row_count_3982.symm
def row_3983 : RowCertificate 3983 where
  first := false
  first_correct := by
    have composite : ¬ Nat.Prime (corner 3983 1) := by change ¬ Nat.Prime 63465123; exact PrattCertificates.composite_of_product (factor := 3) (cofactor := 21155041) rfl (by decide) (by decide)
    exact decide_eq_false composite
  second := false
  second_correct := by
    have composite : ¬ Nat.Prime (corner 3983 2) := by change ¬ Nat.Prime 63457157; exact PrattCertificates.composite_of_product (factor := 1049) (cofactor := 60493) rfl (by decide) (by decide)
    exact decide_eq_false composite
  third := false
  third_correct := by
    have composite : ¬ Nat.Prime (corner 3983 3) := by change ¬ Nat.Prime 63449191; exact PrattCertificates.composite_of_product (factor := 13) (cofactor := 4880707) rfl (by decide) (by decide)
    exact decide_eq_false composite
theorem row_count_3983 : row 3983 = 0 := (row_3983.checked).trans rfl
theorem model_count_3983 : modelRow 3983 = 0 := by decide
theorem modelrow_agrees_3983 : modelRow 3983 = row 3983 := model_count_3983.trans row_count_3983.symm
def row_3984 : RowCertificate 3984 where
  first := false
  first_correct := by
    have composite : ¬ Nat.Prime (corner 3984 1) := by change ¬ Nat.Prime 63496993; exact PrattCertificates.composite_of_product (factor := 7) (cofactor := 9070999) rfl (by decide) (by decide)
    exact decide_eq_false composite
  second := false
  second_correct := by
    have composite : ¬ Nat.Prime (corner 3984 2) := by change ¬ Nat.Prime 63489025; exact PrattCertificates.composite_of_product (factor := 5) (cofactor := 12697805) rfl (by decide) (by decide)
    exact decide_eq_false composite
  third := true
  third_correct := by
    have prime : Nat.Prime (corner 3984 3) := by change Nat.Prime 63481057; exact ProjectEulerSolutions.P58PrimeNodes.p_63481057.prime
    exact decide_eq_true prime
theorem row_count_3984 : row 3984 = 1 := (row_3984.checked).trans rfl
theorem model_count_3984 : modelRow 3984 = 1 := by decide
theorem modelrow_agrees_3984 : modelRow 3984 = row 3984 := model_count_3984.trans row_count_3984.symm
def row_3985 : RowCertificate 3985 where
  first := false
  first_correct := by
    have composite : ¬ Nat.Prime (corner 3985 1) := by change ¬ Nat.Prime 63528871; exact PrattCertificates.composite_of_product (factor := 7) (cofactor := 9075553) rfl (by decide) (by decide)
    exact decide_eq_false composite
  second := false
  second_correct := by
    have composite : ¬ Nat.Prime (corner 3985 2) := by change ¬ Nat.Prime 63520901; exact PrattCertificates.composite_of_product (factor := 421) (cofactor := 150881) rfl (by decide) (by decide)
    exact decide_eq_false composite
  third := false
  third_correct := by
    have composite : ¬ Nat.Prime (corner 3985 3) := by change ¬ Nat.Prime 63512931; exact PrattCertificates.composite_of_product (factor := 3) (cofactor := 21170977) rfl (by decide) (by decide)
    exact decide_eq_false composite
theorem row_count_3985 : row 3985 = 0 := (row_3985.checked).trans rfl
theorem model_count_3985 : modelRow 3985 = 0 := by decide
theorem modelrow_agrees_3985 : modelRow 3985 = row 3985 := model_count_3985.trans row_count_3985.symm
def row_3986 : RowCertificate 3986 where
  first := false
  first_correct := by
    have composite : ¬ Nat.Prime (corner 3986 1) := by change ¬ Nat.Prime 63560757; exact PrattCertificates.composite_of_product (factor := 3) (cofactor := 21186919) rfl (by decide) (by decide)
    exact decide_eq_false composite
  second := false
  second_correct := by
    have composite : ¬ Nat.Prime (corner 3986 2) := by change ¬ Nat.Prime 63552785; exact PrattCertificates.composite_of_product (factor := 5) (cofactor := 12710557) rfl (by decide) (by decide)
    exact decide_eq_false composite
  third := true
  third_correct := by
    have prime : Nat.Prime (corner 3986 3) := by change Nat.Prime 63544813; exact ProjectEulerSolutions.P58PrimeNodes.p_63544813.prime
    exact decide_eq_true prime
theorem row_count_3986 : row 3986 = 1 := (row_3986.checked).trans rfl
theorem model_count_3986 : modelRow 3986 = 1 := by decide
theorem modelrow_agrees_3986 : modelRow 3986 = row 3986 := model_count_3986.trans row_count_3986.symm
def row_3987 : RowCertificate 3987 where
  first := false
  first_correct := by
    have composite : ¬ Nat.Prime (corner 3987 1) := by change ¬ Nat.Prime 63592651; exact PrattCertificates.composite_of_product (factor := 397) (cofactor := 160183) rfl (by decide) (by decide)
    exact decide_eq_false composite
  second := false
  second_correct := by
    have composite : ¬ Nat.Prime (corner 3987 2) := by change ¬ Nat.Prime 63584677; exact PrattCertificates.composite_of_product (factor := 13) (cofactor := 4891129) rfl (by decide) (by decide)
    exact decide_eq_false composite
  third := true
  third_correct := by
    have prime : Nat.Prime (corner 3987 3) := by change Nat.Prime 63576703; exact ProjectEulerSolutions.P58PrimeNodes.p_63576703.prime
    exact decide_eq_true prime
theorem row_count_3987 : row 3987 = 1 := (row_3987.checked).trans rfl
theorem model_count_3987 : modelRow 3987 = 1 := by decide
theorem modelrow_agrees_3987 : modelRow 3987 = row 3987 := model_count_3987.trans row_count_3987.symm
def row_3988 : RowCertificate 3988 where
  first := true
  first_correct := by
    have prime : Nat.Prime (corner 3988 1) := by change Nat.Prime 63624553; exact ProjectEulerSolutions.P58PrimeNodes.p_63624553.prime
    exact decide_eq_true prime
  second := false
  second_correct := by
    have composite : ¬ Nat.Prime (corner 3988 2) := by change ¬ Nat.Prime 63616577; exact PrattCertificates.composite_of_product (factor := 89) (cofactor := 714793) rfl (by decide) (by decide)
    exact decide_eq_false composite
  third := false
  third_correct := by
    have composite : ¬ Nat.Prime (corner 3988 3) := by change ¬ Nat.Prime 63608601; exact PrattCertificates.composite_of_product (factor := 3) (cofactor := 21202867) rfl (by decide) (by decide)
    exact decide_eq_false composite
theorem row_count_3988 : row 3988 = 1 := (row_3988.checked).trans rfl
theorem model_count_3988 : modelRow 3988 = 1 := by decide
theorem modelrow_agrees_3988 : modelRow 3988 = row 3988 := model_count_3988.trans row_count_3988.symm
def row_3989 : RowCertificate 3989 where
  first := false
  first_correct := by
    have composite : ¬ Nat.Prime (corner 3989 1) := by change ¬ Nat.Prime 63656463; exact PrattCertificates.composite_of_product (factor := 3) (cofactor := 21218821) rfl (by decide) (by decide)
    exact decide_eq_false composite
  second := false
  second_correct := by
    have composite : ¬ Nat.Prime (corner 3989 2) := by change ¬ Nat.Prime 63648485; exact PrattCertificates.composite_of_product (factor := 5) (cofactor := 12729697) rfl (by decide) (by decide)
    exact decide_eq_false composite
  third := false
  third_correct := by
    have composite : ¬ Nat.Prime (corner 3989 3) := by change ¬ Nat.Prime 63640507; exact PrattCertificates.composite_of_product (factor := 7) (cofactor := 9091501) rfl (by decide) (by decide)
    exact decide_eq_false composite
theorem row_count_3989 : row 3989 = 0 := (row_3989.checked).trans rfl
theorem model_count_3989 : modelRow 3989 = 0 := by decide
theorem modelrow_agrees_3989 : modelRow 3989 = row 3989 := model_count_3989.trans row_count_3989.symm
def row_3990 : RowCertificate 3990 where
  first := false
  first_correct := by
    have composite : ¬ Nat.Prime (corner 3990 1) := by change ¬ Nat.Prime 63688381; exact PrattCertificates.composite_of_product (factor := 4441) (cofactor := 14341) rfl (by decide) (by decide)
    exact decide_eq_false composite
  second := false
  second_correct := by
    have composite : ¬ Nat.Prime (corner 3990 2) := by change ¬ Nat.Prime 63680401; exact PrattCertificates.composite_of_product (factor := 53) (cofactor := 1201517) rfl (by decide) (by decide)
    exact decide_eq_false composite
  third := false
  third_correct := by
    have composite : ¬ Nat.Prime (corner 3990 3) := by change ¬ Nat.Prime 63672421; exact PrattCertificates.composite_of_product (factor := 811) (cofactor := 78511) rfl (by decide) (by decide)
    exact decide_eq_false composite
theorem row_count_3990 : row 3990 = 0 := (row_3990.checked).trans rfl
theorem model_count_3990 : modelRow 3990 = 0 := by decide
theorem modelrow_agrees_3990 : modelRow 3990 = row 3990 := model_count_3990.trans row_count_3990.symm
def row_3991 : RowCertificate 3991 where
  first := false
  first_correct := by
    have composite : ¬ Nat.Prime (corner 3991 1) := by change ¬ Nat.Prime 63720307; exact PrattCertificates.composite_of_product (factor := 7) (cofactor := 9102901) rfl (by decide) (by decide)
    exact decide_eq_false composite
  second := false
  second_correct := by
    have composite : ¬ Nat.Prime (corner 3991 2) := by change ¬ Nat.Prime 63712325; exact PrattCertificates.composite_of_product (factor := 5) (cofactor := 12742465) rfl (by decide) (by decide)
    exact decide_eq_false composite
  third := false
  third_correct := by
    have composite : ¬ Nat.Prime (corner 3991 3) := by change ¬ Nat.Prime 63704343; exact PrattCertificates.composite_of_product (factor := 3) (cofactor := 21234781) rfl (by decide) (by decide)
    exact decide_eq_false composite
theorem row_count_3991 : row 3991 = 0 := (row_3991.checked).trans rfl
theorem model_count_3991 : modelRow 3991 = 0 := by decide
theorem modelrow_agrees_3991 : modelRow 3991 = row 3991 := model_count_3991.trans row_count_3991.symm
def row_3992 : RowCertificate 3992 where
  first := false
  first_correct := by
    have composite : ¬ Nat.Prime (corner 3992 1) := by change ¬ Nat.Prime 63752241; exact PrattCertificates.composite_of_product (factor := 3) (cofactor := 21250747) rfl (by decide) (by decide)
    exact decide_eq_false composite
  second := false
  second_correct := by
    have composite : ¬ Nat.Prime (corner 3992 2) := by change ¬ Nat.Prime 63744257; exact PrattCertificates.composite_of_product (factor := 73) (cofactor := 873209) rfl (by decide) (by decide)
    exact decide_eq_false composite
  third := true
  third_correct := by
    have prime : Nat.Prime (corner 3992 3) := by change Nat.Prime 63736273; exact ProjectEulerSolutions.P58PrimeNodes.p_63736273.prime
    exact decide_eq_true prime
theorem row_count_3992 : row 3992 = 1 := (row_3992.checked).trans rfl
theorem model_count_3992 : modelRow 3992 = 1 := by decide
theorem modelrow_agrees_3992 : modelRow 3992 = row 3992 := model_count_3992.trans row_count_3992.symm
def row_3993 : RowCertificate 3993 where
  first := true
  first_correct := by
    have prime : Nat.Prime (corner 3993 1) := by change Nat.Prime 63784183; exact ProjectEulerSolutions.P58PrimeNodes.p_63784183.prime
    exact decide_eq_true prime
  second := false
  second_correct := by
    have composite : ¬ Nat.Prime (corner 3993 2) := by change ¬ Nat.Prime 63776197; exact PrattCertificates.composite_of_product (factor := 17) (cofactor := 3751541) rfl (by decide) (by decide)
    exact decide_eq_false composite
  third := false
  third_correct := by
    have composite : ¬ Nat.Prime (corner 3993 3) := by change ¬ Nat.Prime 63768211; exact PrattCertificates.composite_of_product (factor := 13) (cofactor := 4905247) rfl (by decide) (by decide)
    exact decide_eq_false composite
theorem row_count_3993 : row 3993 = 1 := (row_3993.checked).trans rfl
theorem model_count_3993 : modelRow 3993 = 1 := by decide
theorem modelrow_agrees_3993 : modelRow 3993 = row 3993 := model_count_3993.trans row_count_3993.symm
def row_3994 : RowCertificate 3994 where
  first := false
  first_correct := by
    have composite : ¬ Nat.Prime (corner 3994 1) := by change ¬ Nat.Prime 63816133; exact PrattCertificates.composite_of_product (factor := 223) (cofactor := 286171) rfl (by decide) (by decide)
    exact decide_eq_false composite
  second := false
  second_correct := by
    have composite : ¬ Nat.Prime (corner 3994 2) := by change ¬ Nat.Prime 63808145; exact PrattCertificates.composite_of_product (factor := 5) (cofactor := 12761629) rfl (by decide) (by decide)
    exact decide_eq_false composite
  third := false
  third_correct := by
    have composite : ¬ Nat.Prime (corner 3994 3) := by change ¬ Nat.Prime 63800157; exact PrattCertificates.composite_of_product (factor := 3) (cofactor := 21266719) rfl (by decide) (by decide)
    exact decide_eq_false composite
theorem row_count_3994 : row 3994 = 0 := (row_3994.checked).trans rfl
theorem model_count_3994 : modelRow 3994 = 0 := by decide
theorem modelrow_agrees_3994 : modelRow 3994 = row 3994 := model_count_3994.trans row_count_3994.symm
def row_3995 : RowCertificate 3995 where
  first := false
  first_correct := by
    have composite : ¬ Nat.Prime (corner 3995 1) := by change ¬ Nat.Prime 63848091; exact PrattCertificates.composite_of_product (factor := 3) (cofactor := 21282697) rfl (by decide) (by decide)
    exact decide_eq_false composite
  second := false
  second_correct := by
    have composite : ¬ Nat.Prime (corner 3995 2) := by change ¬ Nat.Prime 63840101; exact PrattCertificates.composite_of_product (factor := 13) (cofactor := 4910777) rfl (by decide) (by decide)
    exact decide_eq_false composite
  third := false
  third_correct := by
    have composite : ¬ Nat.Prime (corner 3995 3) := by change ¬ Nat.Prime 63832111; exact PrattCertificates.composite_of_product (factor := 7) (cofactor := 9118873) rfl (by decide) (by decide)
    exact decide_eq_false composite
theorem row_count_3995 : row 3995 = 0 := (row_3995.checked).trans rfl
theorem model_count_3995 : modelRow 3995 = 0 := by decide
theorem modelrow_agrees_3995 : modelRow 3995 = row 3995 := model_count_3995.trans row_count_3995.symm
def row_3996 : RowCertificate 3996 where
  first := false
  first_correct := by
    have composite : ¬ Nat.Prime (corner 3996 1) := by change ¬ Nat.Prime 63880057; exact PrattCertificates.composite_of_product (factor := 31) (cofactor := 2060647) rfl (by decide) (by decide)
    exact decide_eq_false composite
  second := false
  second_correct := by
    have composite : ¬ Nat.Prime (corner 3996 2) := by change ¬ Nat.Prime 63872065; exact PrattCertificates.composite_of_product (factor := 5) (cofactor := 12774413) rfl (by decide) (by decide)
    exact decide_eq_false composite
  third := false
  third_correct := by
    have composite : ¬ Nat.Prime (corner 3996 3) := by change ¬ Nat.Prime 63864073; exact PrattCertificates.composite_of_product (factor := 7) (cofactor := 9123439) rfl (by decide) (by decide)
    exact decide_eq_false composite
theorem row_count_3996 : row 3996 = 0 := (row_3996.checked).trans rfl
theorem model_count_3996 : modelRow 3996 = 0 := by decide
theorem modelrow_agrees_3996 : modelRow 3996 = row 3996 := model_count_3996.trans row_count_3996.symm
def row_3997 : RowCertificate 3997 where
  first := false
  first_correct := by
    have composite : ¬ Nat.Prime (corner 3997 1) := by change ¬ Nat.Prime 63912031; exact PrattCertificates.composite_of_product (factor := 157) (cofactor := 407083) rfl (by decide) (by decide)
    exact decide_eq_false composite
  second := false
  second_correct := by
    have composite : ¬ Nat.Prime (corner 3997 2) := by change ¬ Nat.Prime 63904037; exact PrattCertificates.composite_of_product (factor := 17) (cofactor := 3759061) rfl (by decide) (by decide)
    exact decide_eq_false composite
  third := false
  third_correct := by
    have composite : ¬ Nat.Prime (corner 3997 3) := by change ¬ Nat.Prime 63896043; exact PrattCertificates.composite_of_product (factor := 3) (cofactor := 21298681) rfl (by decide) (by decide)
    exact decide_eq_false composite
theorem row_count_3997 : row 3997 = 0 := (row_3997.checked).trans rfl
theorem model_count_3997 : modelRow 3997 = 0 := by decide
theorem modelrow_agrees_3997 : modelRow 3997 = row 3997 := model_count_3997.trans row_count_3997.symm
def row_3998 : RowCertificate 3998 where
  first := false
  first_correct := by
    have composite : ¬ Nat.Prime (corner 3998 1) := by change ¬ Nat.Prime 63944013; exact PrattCertificates.composite_of_product (factor := 3) (cofactor := 21314671) rfl (by decide) (by decide)
    exact decide_eq_false composite
  second := false
  second_correct := by
    have composite : ¬ Nat.Prime (corner 3998 2) := by change ¬ Nat.Prime 63936017; exact PrattCertificates.composite_of_product (factor := 1213) (cofactor := 52709) rfl (by decide) (by decide)
    exact decide_eq_false composite
  third := false
  third_correct := by
    have composite : ¬ Nat.Prime (corner 3998 3) := by change ¬ Nat.Prime 63928021; exact PrattCertificates.composite_of_product (factor := 2803) (cofactor := 22807) rfl (by decide) (by decide)
    exact decide_eq_false composite
theorem row_count_3998 : row 3998 = 0 := (row_3998.checked).trans rfl
theorem model_count_3998 : modelRow 3998 = 0 := by decide
theorem modelrow_agrees_3998 : modelRow 3998 = row 3998 := model_count_3998.trans row_count_3998.symm
def row_3999 : RowCertificate 3999 where
  first := false
  first_correct := by
    have composite : ¬ Nat.Prime (corner 3999 1) := by change ¬ Nat.Prime 63976003; exact PrattCertificates.composite_of_product (factor := 7) (cofactor := 9139429) rfl (by decide) (by decide)
    exact decide_eq_false composite
  second := false
  second_correct := by
    have composite : ¬ Nat.Prime (corner 3999 2) := by change ¬ Nat.Prime 63968005; exact PrattCertificates.composite_of_product (factor := 5) (cofactor := 12793601) rfl (by decide) (by decide)
    exact decide_eq_false composite
  third := false
  third_correct := by
    have composite : ¬ Nat.Prime (corner 3999 3) := by change ¬ Nat.Prime 63960007; exact PrattCertificates.composite_of_product (factor := 193) (cofactor := 331399) rfl (by decide) (by decide)
    exact decide_eq_false composite
theorem row_count_3999 : row 3999 = 0 := (row_3999.checked).trans rfl
theorem model_count_3999 : modelRow 3999 = 0 := by decide
theorem modelrow_agrees_3999 : modelRow 3999 = row 3999 := model_count_3999.trans row_count_3999.symm
def row_4000 : RowCertificate 4000 where
  first := true
  first_correct := by
    have prime : Nat.Prime (corner 4000 1) := by change Nat.Prime 64008001; exact ProjectEulerSolutions.P58PrimeNodes.p_64008001.prime
    exact decide_eq_true prime
  second := false
  second_correct := by
    have composite : ¬ Nat.Prime (corner 4000 2) := by change ¬ Nat.Prime 64000001; exact PrattCertificates.composite_of_product (factor := 13) (cofactor := 4923077) rfl (by decide) (by decide)
    exact decide_eq_false composite
  third := false
  third_correct := by
    have composite : ¬ Nat.Prime (corner 4000 3) := by change ¬ Nat.Prime 63992001; exact PrattCertificates.composite_of_product (factor := 3) (cofactor := 21330667) rfl (by decide) (by decide)
    exact decide_eq_false composite
theorem row_count_4000 : row 4000 = 1 := (row_4000.checked).trans rfl
theorem model_count_4000 : modelRow 4000 = 1 := by decide
theorem modelrow_agrees_4000 : modelRow 4000 = row 4000 := model_count_4000.trans row_count_4000.symm
def row_4001 : RowCertificate 4001 where
  first := false
  first_correct := by
    have composite : ¬ Nat.Prime (corner 4001 1) := by change ¬ Nat.Prime 64040007; exact PrattCertificates.composite_of_product (factor := 3) (cofactor := 21346669) rfl (by decide) (by decide)
    exact decide_eq_false composite
  second := false
  second_correct := by
    have composite : ¬ Nat.Prime (corner 4001 2) := by change ¬ Nat.Prime 64032005; exact PrattCertificates.composite_of_product (factor := 5) (cofactor := 12806401) rfl (by decide) (by decide)
    exact decide_eq_false composite
  third := false
  third_correct := by
    have composite : ¬ Nat.Prime (corner 4001 3) := by change ¬ Nat.Prime 64024003; exact PrattCertificates.composite_of_product (factor := 3769) (cofactor := 16987) rfl (by decide) (by decide)
    exact decide_eq_false composite
theorem row_count_4001 : row 4001 = 0 := (row_4001.checked).trans rfl
theorem model_count_4001 : modelRow 4001 = 0 := by decide
theorem modelrow_agrees_4001 : modelRow 4001 = row 4001 := model_count_4001.trans row_count_4001.symm
def row_4002 : RowCertificate 4002 where
  first := false
  first_correct := by
    have composite : ¬ Nat.Prime (corner 4002 1) := by change ¬ Nat.Prime 64072021; exact PrattCertificates.composite_of_product (factor := 13) (cofactor := 4928617) rfl (by decide) (by decide)
    exact decide_eq_false composite
  second := false
  second_correct := by
    have composite : ¬ Nat.Prime (corner 4002 2) := by change ¬ Nat.Prime 64064017; exact PrattCertificates.composite_of_product (factor := 41) (cofactor := 1562537) rfl (by decide) (by decide)
    exact decide_eq_false composite
  third := false
  third_correct := by
    have composite : ¬ Nat.Prime (corner 4002 3) := by change ¬ Nat.Prime 64056013; exact PrattCertificates.composite_of_product (factor := 7) (cofactor := 9150859) rfl (by decide) (by decide)
    exact decide_eq_false composite
theorem row_count_4002 : row 4002 = 0 := (row_4002.checked).trans rfl
theorem model_count_4002 : modelRow 4002 = 0 := by decide
theorem modelrow_agrees_4002 : modelRow 4002 = row 4002 := model_count_4002.trans row_count_4002.symm
def row_4003 : RowCertificate 4003 where
  first := false
  first_correct := by
    have composite : ¬ Nat.Prime (corner 4003 1) := by change ¬ Nat.Prime 64104043; exact PrattCertificates.composite_of_product (factor := 19) (cofactor := 3373897) rfl (by decide) (by decide)
    exact decide_eq_false composite
  second := false
  second_correct := by
    have composite : ¬ Nat.Prime (corner 4003 2) := by change ¬ Nat.Prime 64096037; exact PrattCertificates.composite_of_product (factor := 1229) (cofactor := 52153) rfl (by decide) (by decide)
    exact decide_eq_false composite
  third := false
  third_correct := by
    have composite : ¬ Nat.Prime (corner 4003 3) := by change ¬ Nat.Prime 64088031; exact PrattCertificates.composite_of_product (factor := 3) (cofactor := 21362677) rfl (by decide) (by decide)
    exact decide_eq_false composite
theorem row_count_4003 : row 4003 = 0 := (row_4003.checked).trans rfl
theorem model_count_4003 : modelRow 4003 = 0 := by decide
theorem modelrow_agrees_4003 : modelRow 4003 = row 4003 := model_count_4003.trans row_count_4003.symm
def row_4004 : RowCertificate 4004 where
  first := false
  first_correct := by
    have composite : ¬ Nat.Prime (corner 4004 1) := by change ¬ Nat.Prime 64136073; exact PrattCertificates.composite_of_product (factor := 3) (cofactor := 21378691) rfl (by decide) (by decide)
    exact decide_eq_false composite
  second := false
  second_correct := by
    have composite : ¬ Nat.Prime (corner 4004 2) := by change ¬ Nat.Prime 64128065; exact PrattCertificates.composite_of_product (factor := 5) (cofactor := 12825613) rfl (by decide) (by decide)
    exact decide_eq_false composite
  third := false
  third_correct := by
    have composite : ¬ Nat.Prime (corner 4004 3) := by change ¬ Nat.Prime 64120057; exact PrattCertificates.composite_of_product (factor := 6637) (cofactor := 9661) rfl (by decide) (by decide)
    exact decide_eq_false composite
theorem row_count_4004 : row 4004 = 0 := (row_4004.checked).trans rfl
theorem model_count_4004 : modelRow 4004 = 0 := by decide
theorem modelrow_agrees_4004 : modelRow 4004 = row 4004 := model_count_4004.trans row_count_4004.symm
def row_4005 : RowCertificate 4005 where
  first := false
  first_correct := by
    have composite : ¬ Nat.Prime (corner 4005 1) := by change ¬ Nat.Prime 64168111; exact PrattCertificates.composite_of_product (factor := 7) (cofactor := 9166873) rfl (by decide) (by decide)
    exact decide_eq_false composite
  second := false
  second_correct := by
    have composite : ¬ Nat.Prime (corner 4005 2) := by change ¬ Nat.Prime 64160101; exact PrattCertificates.composite_of_product (factor := 853) (cofactor := 75217) rfl (by decide) (by decide)
    exact decide_eq_false composite
  third := false
  third_correct := by
    have composite : ¬ Nat.Prime (corner 4005 3) := by change ¬ Nat.Prime 64152091; exact PrattCertificates.composite_of_product (factor := 6661) (cofactor := 9631) rfl (by decide) (by decide)
    exact decide_eq_false composite
theorem row_count_4005 : row 4005 = 0 := (row_4005.checked).trans rfl
theorem model_count_4005 : modelRow 4005 = 0 := by decide
theorem modelrow_agrees_4005 : modelRow 4005 = row 4005 := model_count_4005.trans row_count_4005.symm
def row_4006 : RowCertificate 4006 where
  first := false
  first_correct := by
    have composite : ¬ Nat.Prime (corner 4006 1) := by change ¬ Nat.Prime 64200157; exact PrattCertificates.composite_of_product (factor := 7) (cofactor := 9171451) rfl (by decide) (by decide)
    exact decide_eq_false composite
  second := false
  second_correct := by
    have composite : ¬ Nat.Prime (corner 4006 2) := by change ¬ Nat.Prime 64192145; exact PrattCertificates.composite_of_product (factor := 5) (cofactor := 12838429) rfl (by decide) (by decide)
    exact decide_eq_false composite
  third := false
  third_correct := by
    have composite : ¬ Nat.Prime (corner 4006 3) := by change ¬ Nat.Prime 64184133; exact PrattCertificates.composite_of_product (factor := 3) (cofactor := 21394711) rfl (by decide) (by decide)
    exact decide_eq_false composite
theorem row_count_4006 : row 4006 = 0 := (row_4006.checked).trans rfl
theorem model_count_4006 : modelRow 4006 = 0 := by decide
theorem modelrow_agrees_4006 : modelRow 4006 = row 4006 := model_count_4006.trans row_count_4006.symm
def row_4007 : RowCertificate 4007 where
  first := false
  first_correct := by
    have composite : ¬ Nat.Prime (corner 4007 1) := by change ¬ Nat.Prime 64232211; exact PrattCertificates.composite_of_product (factor := 3) (cofactor := 21410737) rfl (by decide) (by decide)
    exact decide_eq_false composite
  second := true
  second_correct := by
    have prime : Nat.Prime (corner 4007 2) := by change Nat.Prime 64224197; exact ProjectEulerSolutions.P58PrimeNodes.p_64224197.prime
    exact decide_eq_true prime
  third := false
  third_correct := by
    have composite : ¬ Nat.Prime (corner 4007 3) := by change ¬ Nat.Prime 64216183; exact PrattCertificates.composite_of_product (factor := 2113) (cofactor := 30391) rfl (by decide) (by decide)
    exact decide_eq_false composite
theorem row_count_4007 : row 4007 = 1 := (row_4007.checked).trans rfl
theorem model_count_4007 : modelRow 4007 = 1 := by decide
theorem modelrow_agrees_4007 : modelRow 4007 = row 4007 := model_count_4007.trans row_count_4007.symm
def row_4008 : RowCertificate 4008 where
  first := true
  first_correct := by
    have prime : Nat.Prime (corner 4008 1) := by change Nat.Prime 64264273; exact ProjectEulerSolutions.P58PrimeNodes.p_64264273.prime
    exact decide_eq_true prime
  second := false
  second_correct := by
    have composite : ¬ Nat.Prime (corner 4008 2) := by change ¬ Nat.Prime 64256257; exact PrattCertificates.composite_of_product (factor := 13) (cofactor := 4942789) rfl (by decide) (by decide)
    exact decide_eq_false composite
  third := false
  third_correct := by
    have composite : ¬ Nat.Prime (corner 4008 3) := by change ¬ Nat.Prime 64248241; exact PrattCertificates.composite_of_product (factor := 97) (cofactor := 662353) rfl (by decide) (by decide)
    exact decide_eq_false composite
theorem row_count_4008 : row 4008 = 1 := (row_4008.checked).trans rfl
theorem model_count_4008 : modelRow 4008 = 1 := by decide
theorem modelrow_agrees_4008 : modelRow 4008 = row 4008 := model_count_4008.trans row_count_4008.symm
def row_4009 : RowCertificate 4009 where
  first := false
  first_correct := by
    have composite : ¬ Nat.Prime (corner 4009 1) := by change ¬ Nat.Prime 64296343; exact PrattCertificates.composite_of_product (factor := 37) (cofactor := 1737739) rfl (by decide) (by decide)
    exact decide_eq_false composite
  second := false
  second_correct := by
    have composite : ¬ Nat.Prime (corner 4009 2) := by change ¬ Nat.Prime 64288325; exact PrattCertificates.composite_of_product (factor := 5) (cofactor := 12857665) rfl (by decide) (by decide)
    exact decide_eq_false composite
  third := false
  third_correct := by
    have composite : ¬ Nat.Prime (corner 4009 3) := by change ¬ Nat.Prime 64280307; exact PrattCertificates.composite_of_product (factor := 3) (cofactor := 21426769) rfl (by decide) (by decide)
    exact decide_eq_false composite
theorem row_count_4009 : row 4009 = 0 := (row_4009.checked).trans rfl
theorem model_count_4009 : modelRow 4009 = 0 := by decide
theorem modelrow_agrees_4009 : modelRow 4009 = row 4009 := model_count_4009.trans row_count_4009.symm
def row_4010 : RowCertificate 4010 where
  first := false
  first_correct := by
    have composite : ¬ Nat.Prime (corner 4010 1) := by change ¬ Nat.Prime 64328421; exact PrattCertificates.composite_of_product (factor := 3) (cofactor := 21442807) rfl (by decide) (by decide)
    exact decide_eq_false composite
  second := false
  second_correct := by
    have composite : ¬ Nat.Prime (corner 4010 2) := by change ¬ Nat.Prime 64320401; exact PrattCertificates.composite_of_product (factor := 17) (cofactor := 3783553) rfl (by decide) (by decide)
    exact decide_eq_false composite
  third := false
  third_correct := by
    have composite : ¬ Nat.Prime (corner 4010 3) := by change ¬ Nat.Prime 64312381; exact PrattCertificates.composite_of_product (factor := 7) (cofactor := 9187483) rfl (by decide) (by decide)
    exact decide_eq_false composite
theorem row_count_4010 : row 4010 = 0 := (row_4010.checked).trans rfl
theorem model_count_4010 : modelRow 4010 = 0 := by decide
theorem modelrow_agrees_4010 : modelRow 4010 = row 4010 := model_count_4010.trans row_count_4010.symm
def row_4011 : RowCertificate 4011 where
  first := false
  first_correct := by
    have composite : ¬ Nat.Prime (corner 4011 1) := by change ¬ Nat.Prime 64360507; exact PrattCertificates.composite_of_product (factor := 7789) (cofactor := 8263) rfl (by decide) (by decide)
    exact decide_eq_false composite
  second := false
  second_correct := by
    have composite : ¬ Nat.Prime (corner 4011 2) := by change ¬ Nat.Prime 64352485; exact PrattCertificates.composite_of_product (factor := 5) (cofactor := 12870497) rfl (by decide) (by decide)
    exact decide_eq_false composite
  third := false
  third_correct := by
    have composite : ¬ Nat.Prime (corner 4011 3) := by change ¬ Nat.Prime 64344463; exact PrattCertificates.composite_of_product (factor := 73) (cofactor := 881431) rfl (by decide) (by decide)
    exact decide_eq_false composite
theorem row_count_4011 : row 4011 = 0 := (row_4011.checked).trans rfl
theorem model_count_4011 : modelRow 4011 = 0 := by decide
theorem modelrow_agrees_4011 : modelRow 4011 = row 4011 := model_count_4011.trans row_count_4011.symm
def row_4012 : RowCertificate 4012 where
  first := false
  first_correct := by
    have composite : ¬ Nat.Prime (corner 4012 1) := by change ¬ Nat.Prime 64392601; exact PrattCertificates.composite_of_product (factor := 7) (cofactor := 9198943) rfl (by decide) (by decide)
    exact decide_eq_false composite
  second := false
  second_correct := by
    have composite : ¬ Nat.Prime (corner 4012 2) := by change ¬ Nat.Prime 64384577; exact PrattCertificates.composite_of_product (factor := 1181) (cofactor := 54517) rfl (by decide) (by decide)
    exact decide_eq_false composite
  third := false
  third_correct := by
    have composite : ¬ Nat.Prime (corner 4012 3) := by change ¬ Nat.Prime 64376553; exact PrattCertificates.composite_of_product (factor := 3) (cofactor := 21458851) rfl (by decide) (by decide)
    exact decide_eq_false composite
theorem row_count_4012 : row 4012 = 0 := (row_4012.checked).trans rfl
theorem model_count_4012 : modelRow 4012 = 0 := by decide
theorem modelrow_agrees_4012 : modelRow 4012 = row 4012 := model_count_4012.trans row_count_4012.symm
def row_4013 : RowCertificate 4013 where
  first := false
  first_correct := by
    have composite : ¬ Nat.Prime (corner 4013 1) := by change ¬ Nat.Prime 64424703; exact PrattCertificates.composite_of_product (factor := 3) (cofactor := 21474901) rfl (by decide) (by decide)
    exact decide_eq_false composite
  second := false
  second_correct := by
    have composite : ¬ Nat.Prime (corner 4013 2) := by change ¬ Nat.Prime 64416677; exact PrattCertificates.composite_of_product (factor := 13) (cofactor := 4955129) rfl (by decide) (by decide)
    exact decide_eq_false composite
  third := false
  third_correct := by
    have composite : ¬ Nat.Prime (corner 4013 3) := by change ¬ Nat.Prime 64408651; exact PrattCertificates.composite_of_product (factor := 19) (cofactor := 3389929) rfl (by decide) (by decide)
    exact decide_eq_false composite
theorem row_count_4013 : row 4013 = 0 := (row_4013.checked).trans rfl
theorem model_count_4013 : modelRow 4013 = 0 := by decide
theorem modelrow_agrees_4013 : modelRow 4013 = row 4013 := model_count_4013.trans row_count_4013.symm
def row_4014 : RowCertificate 4014 where
  first := false
  first_correct := by
    have composite : ¬ Nat.Prime (corner 4014 1) := by change ¬ Nat.Prime 64456813; exact PrattCertificates.composite_of_product (factor := 433) (cofactor := 148861) rfl (by decide) (by decide)
    exact decide_eq_false composite
  second := false
  second_correct := by
    have composite : ¬ Nat.Prime (corner 4014 2) := by change ¬ Nat.Prime 64448785; exact PrattCertificates.composite_of_product (factor := 5) (cofactor := 12889757) rfl (by decide) (by decide)
    exact decide_eq_false composite
  third := true
  third_correct := by
    have prime : Nat.Prime (corner 4014 3) := by change Nat.Prime 64440757; exact ProjectEulerSolutions.P58PrimeNodes.p_64440757.prime
    exact decide_eq_true prime
theorem row_count_4014 : row 4014 = 1 := (row_4014.checked).trans rfl
theorem model_count_4014 : modelRow 4014 = 1 := by decide
theorem modelrow_agrees_4014 : modelRow 4014 = row 4014 := model_count_4014.trans row_count_4014.symm
def row_4015 : RowCertificate 4015 where
  first := false
  first_correct := by
    have composite : ¬ Nat.Prime (corner 4015 1) := by change ¬ Nat.Prime 64488931; exact PrattCertificates.composite_of_product (factor := 13) (cofactor := 4960687) rfl (by decide) (by decide)
    exact decide_eq_false composite
  second := true
  second_correct := by
    have prime : Nat.Prime (corner 4015 2) := by change Nat.Prime 64480901; exact ProjectEulerSolutions.P58PrimeNodes.p_64480901.prime
    exact decide_eq_true prime
  third := false
  third_correct := by
    have composite : ¬ Nat.Prime (corner 4015 3) := by change ¬ Nat.Prime 64472871; exact PrattCertificates.composite_of_product (factor := 3) (cofactor := 21490957) rfl (by decide) (by decide)
    exact decide_eq_false composite
theorem row_count_4015 : row 4015 = 1 := (row_4015.checked).trans rfl
theorem model_count_4015 : modelRow 4015 = 1 := by decide
theorem modelrow_agrees_4015 : modelRow 4015 = row 4015 := model_count_4015.trans row_count_4015.symm
def row_4016 : RowCertificate 4016 where
  first := false
  first_correct := by
    have composite : ¬ Nat.Prime (corner 4016 1) := by change ¬ Nat.Prime 64521057; exact PrattCertificates.composite_of_product (factor := 3) (cofactor := 21507019) rfl (by decide) (by decide)
    exact decide_eq_false composite
  second := false
  second_correct := by
    have composite : ¬ Nat.Prime (corner 4016 2) := by change ¬ Nat.Prime 64513025; exact PrattCertificates.composite_of_product (factor := 5) (cofactor := 12902605) rfl (by decide) (by decide)
    exact decide_eq_false composite
  third := false
  third_correct := by
    have composite : ¬ Nat.Prime (corner 4016 3) := by change ¬ Nat.Prime 64504993; exact PrattCertificates.composite_of_product (factor := 7) (cofactor := 9214999) rfl (by decide) (by decide)
    exact decide_eq_false composite
theorem row_count_4016 : row 4016 = 0 := (row_4016.checked).trans rfl
theorem model_count_4016 : modelRow 4016 = 0 := by decide
theorem modelrow_agrees_4016 : modelRow 4016 = row 4016 := model_count_4016.trans row_count_4016.symm
def row_4017 : RowCertificate 4017 where
  first := false
  first_correct := by
    have composite : ¬ Nat.Prime (corner 4017 1) := by change ¬ Nat.Prime 64553191; exact PrattCertificates.composite_of_product (factor := 31) (cofactor := 2082361) rfl (by decide) (by decide)
    exact decide_eq_false composite
  second := true
  second_correct := by
    have prime : Nat.Prime (corner 4017 2) := by change Nat.Prime 64545157; exact ProjectEulerSolutions.P58PrimeNodes.p_64545157.prime
    exact decide_eq_true prime
  third := false
  third_correct := by
    have composite : ¬ Nat.Prime (corner 4017 3) := by change ¬ Nat.Prime 64537123; exact PrattCertificates.composite_of_product (factor := 7) (cofactor := 9219589) rfl (by decide) (by decide)
    exact decide_eq_false composite
theorem row_count_4017 : row 4017 = 1 := (row_4017.checked).trans rfl
theorem model_count_4017 : modelRow 4017 = 1 := by decide
theorem modelrow_agrees_4017 : modelRow 4017 = row 4017 := model_count_4017.trans row_count_4017.symm
def row_4018 : RowCertificate 4018 where
  first := true
  first_correct := by
    have prime : Nat.Prime (corner 4018 1) := by change Nat.Prime 64585333; exact ProjectEulerSolutions.P58PrimeNodes.p_64585333.prime
    exact decide_eq_true prime
  second := false
  second_correct := by
    have composite : ¬ Nat.Prime (corner 4018 2) := by change ¬ Nat.Prime 64577297; exact PrattCertificates.composite_of_product (factor := 5333) (cofactor := 12109) rfl (by decide) (by decide)
    exact decide_eq_false composite
  third := false
  third_correct := by
    have composite : ¬ Nat.Prime (corner 4018 3) := by change ¬ Nat.Prime 64569261; exact PrattCertificates.composite_of_product (factor := 3) (cofactor := 21523087) rfl (by decide) (by decide)
    exact decide_eq_false composite
theorem row_count_4018 : row 4018 = 1 := (row_4018.checked).trans rfl
theorem model_count_4018 : modelRow 4018 = 1 := by decide
theorem modelrow_agrees_4018 : modelRow 4018 = row 4018 := model_count_4018.trans row_count_4018.symm
def row_4019 : RowCertificate 4019 where
  first := false
  first_correct := by
    have composite : ¬ Nat.Prime (corner 4019 1) := by change ¬ Nat.Prime 64617483; exact PrattCertificates.composite_of_product (factor := 3) (cofactor := 21539161) rfl (by decide) (by decide)
    exact decide_eq_false composite
  second := false
  second_correct := by
    have composite : ¬ Nat.Prime (corner 4019 2) := by change ¬ Nat.Prime 64609445; exact PrattCertificates.composite_of_product (factor := 5) (cofactor := 12921889) rfl (by decide) (by decide)
    exact decide_eq_false composite
  third := false
  third_correct := by
    have composite : ¬ Nat.Prime (corner 4019 3) := by change ¬ Nat.Prime 64601407; exact PrattCertificates.composite_of_product (factor := 13) (cofactor := 4969339) rfl (by decide) (by decide)
    exact decide_eq_false composite
theorem row_count_4019 : row 4019 = 0 := (row_4019.checked).trans rfl
theorem model_count_4019 : modelRow 4019 = 0 := by decide
theorem modelrow_agrees_4019 : modelRow 4019 = row 4019 := model_count_4019.trans row_count_4019.symm
def row_4020 : RowCertificate 4020 where
  first := false
  first_correct := by
    have composite : ¬ Nat.Prime (corner 4020 1) := by change ¬ Nat.Prime 64649641; exact PrattCertificates.composite_of_product (factor := 7) (cofactor := 9235663) rfl (by decide) (by decide)
    exact decide_eq_false composite
  second := false
  second_correct := by
    have composite : ¬ Nat.Prime (corner 4020 2) := by change ¬ Nat.Prime 64641601; exact PrattCertificates.composite_of_product (factor := 401) (cofactor := 161201) rfl (by decide) (by decide)
    exact decide_eq_false composite
  third := false
  third_correct := by
    have composite : ¬ Nat.Prime (corner 4020 3) := by change ¬ Nat.Prime 64633561; exact PrattCertificates.composite_of_product (factor := 37) (cofactor := 1746853) rfl (by decide) (by decide)
    exact decide_eq_false composite
theorem row_count_4020 : row 4020 = 0 := (row_4020.checked).trans rfl
theorem model_count_4020 : modelRow 4020 = 0 := by decide
theorem modelrow_agrees_4020 : modelRow 4020 = row 4020 := model_count_4020.trans row_count_4020.symm
def row_4021 : RowCertificate 4021 where
  first := true
  first_correct := by
    have prime : Nat.Prime (corner 4021 1) := by change Nat.Prime 64681807; exact ProjectEulerSolutions.P58PrimeNodes.p_64681807.prime
    exact decide_eq_true prime
  second := false
  second_correct := by
    have composite : ¬ Nat.Prime (corner 4021 2) := by change ¬ Nat.Prime 64673765; exact PrattCertificates.composite_of_product (factor := 5) (cofactor := 12934753) rfl (by decide) (by decide)
    exact decide_eq_false composite
  third := false
  third_correct := by
    have composite : ¬ Nat.Prime (corner 4021 3) := by change ¬ Nat.Prime 64665723; exact PrattCertificates.composite_of_product (factor := 3) (cofactor := 21555241) rfl (by decide) (by decide)
    exact decide_eq_false composite
theorem row_count_4021 : row 4021 = 1 := (row_4021.checked).trans rfl
theorem model_count_4021 : modelRow 4021 = 1 := by decide
theorem modelrow_agrees_4021 : modelRow 4021 = row 4021 := model_count_4021.trans row_count_4021.symm
def row_4022 : RowCertificate 4022 where
  first := false
  first_correct := by
    have composite : ¬ Nat.Prime (corner 4022 1) := by change ¬ Nat.Prime 64713981; exact PrattCertificates.composite_of_product (factor := 3) (cofactor := 21571327) rfl (by decide) (by decide)
    exact decide_eq_false composite
  second := false
  second_correct := by
    have composite : ¬ Nat.Prime (corner 4022 2) := by change ¬ Nat.Prime 64705937; exact PrattCertificates.composite_of_product (factor := 89) (cofactor := 727033) rfl (by decide) (by decide)
    exact decide_eq_false composite
  third := false
  third_correct := by
    have composite : ¬ Nat.Prime (corner 4022 3) := by change ¬ Nat.Prime 64697893; exact PrattCertificates.composite_of_product (factor := 13) (cofactor := 4976761) rfl (by decide) (by decide)
    exact decide_eq_false composite
theorem row_count_4022 : row 4022 = 0 := (row_4022.checked).trans rfl
theorem model_count_4022 : modelRow 4022 = 0 := by decide
theorem modelrow_agrees_4022 : modelRow 4022 = row 4022 := model_count_4022.trans row_count_4022.symm
def row_4023 : RowCertificate 4023 where
  first := false
  first_correct := by
    have composite : ¬ Nat.Prime (corner 4023 1) := by change ¬ Nat.Prime 64746163; exact PrattCertificates.composite_of_product (factor := 487) (cofactor := 132949) rfl (by decide) (by decide)
    exact decide_eq_false composite
  second := false
  second_correct := by
    have composite : ¬ Nat.Prime (corner 4023 2) := by change ¬ Nat.Prime 64738117; exact PrattCertificates.composite_of_product (factor := 137) (cofactor := 472541) rfl (by decide) (by decide)
    exact decide_eq_false composite
  third := false
  third_correct := by
    have composite : ¬ Nat.Prime (corner 4023 3) := by change ¬ Nat.Prime 64730071; exact PrattCertificates.composite_of_product (factor := 7) (cofactor := 9247153) rfl (by decide) (by decide)
    exact decide_eq_false composite
theorem row_count_4023 : row 4023 = 0 := (row_4023.checked).trans rfl
theorem model_count_4023 : modelRow 4023 = 0 := by decide
theorem modelrow_agrees_4023 : modelRow 4023 = row 4023 := model_count_4023.trans row_count_4023.symm
def row_4024 : RowCertificate 4024 where
  first := false
  first_correct := by
    have composite : ¬ Nat.Prime (corner 4024 1) := by change ¬ Nat.Prime 64778353; exact PrattCertificates.composite_of_product (factor := 19) (cofactor := 3409387) rfl (by decide) (by decide)
    exact decide_eq_false composite
  second := false
  second_correct := by
    have composite : ¬ Nat.Prime (corner 4024 2) := by change ¬ Nat.Prime 64770305; exact PrattCertificates.composite_of_product (factor := 5) (cofactor := 12954061) rfl (by decide) (by decide)
    exact decide_eq_false composite
  third := false
  third_correct := by
    have composite : ¬ Nat.Prime (corner 4024 3) := by change ¬ Nat.Prime 64762257; exact PrattCertificates.composite_of_product (factor := 3) (cofactor := 21587419) rfl (by decide) (by decide)
    exact decide_eq_false composite
theorem row_count_4024 : row 4024 = 0 := (row_4024.checked).trans rfl
theorem model_count_4024 : modelRow 4024 = 0 := by decide
theorem modelrow_agrees_4024 : modelRow 4024 = row 4024 := model_count_4024.trans row_count_4024.symm
def row_4025 : RowCertificate 4025 where
  first := false
  first_correct := by
    have composite : ¬ Nat.Prime (corner 4025 1) := by change ¬ Nat.Prime 64810551; exact PrattCertificates.composite_of_product (factor := 3) (cofactor := 21603517) rfl (by decide) (by decide)
    exact decide_eq_false composite
  second := false
  second_correct := by
    have composite : ¬ Nat.Prime (corner 4025 2) := by change ¬ Nat.Prime 64802501; exact PrattCertificates.composite_of_product (factor := 29) (cofactor := 2234569) rfl (by decide) (by decide)
    exact decide_eq_false composite
  third := true
  third_correct := by
    have prime : Nat.Prime (corner 4025 3) := by change Nat.Prime 64794451; exact ProjectEulerSolutions.P58PrimeNodes.p_64794451.prime
    exact decide_eq_true prime
theorem row_count_4025 : row 4025 = 1 := (row_4025.checked).trans rfl
theorem model_count_4025 : modelRow 4025 = 1 := by decide
theorem modelrow_agrees_4025 : modelRow 4025 = row 4025 := model_count_4025.trans row_count_4025.symm
def row_4026 : RowCertificate 4026 where
  first := false
  first_correct := by
    have composite : ¬ Nat.Prime (corner 4026 1) := by change ¬ Nat.Prime 64842757; exact PrattCertificates.composite_of_product (factor := 7) (cofactor := 9263251) rfl (by decide) (by decide)
    exact decide_eq_false composite
  second := false
  second_correct := by
    have composite : ¬ Nat.Prime (corner 4026 2) := by change ¬ Nat.Prime 64834705; exact PrattCertificates.composite_of_product (factor := 5) (cofactor := 12966941) rfl (by decide) (by decide)
    exact decide_eq_false composite
  third := true
  third_correct := by
    have prime : Nat.Prime (corner 4026 3) := by change Nat.Prime 64826653; exact ProjectEulerSolutions.P58PrimeNodes.p_64826653.prime
    exact decide_eq_true prime
theorem row_count_4026 : row 4026 = 1 := (row_4026.checked).trans rfl
theorem model_count_4026 : modelRow 4026 = 1 := by decide
theorem modelrow_agrees_4026 : modelRow 4026 = row 4026 := model_count_4026.trans row_count_4026.symm
def row_4027 : RowCertificate 4027 where
  first := false
  first_correct := by
    have composite : ¬ Nat.Prime (corner 4027 1) := by change ¬ Nat.Prime 64874971; exact PrattCertificates.composite_of_product (factor := 7) (cofactor := 9267853) rfl (by decide) (by decide)
    exact decide_eq_false composite
  second := false
  second_correct := by
    have composite : ¬ Nat.Prime (corner 4027 2) := by change ¬ Nat.Prime 64866917; exact PrattCertificates.composite_of_product (factor := 17) (cofactor := 3815701) rfl (by decide) (by decide)
    exact decide_eq_false composite
  third := false
  third_correct := by
    have composite : ¬ Nat.Prime (corner 4027 3) := by change ¬ Nat.Prime 64858863; exact PrattCertificates.composite_of_product (factor := 3) (cofactor := 21619621) rfl (by decide) (by decide)
    exact decide_eq_false composite
theorem row_count_4027 : row 4027 = 0 := (row_4027.checked).trans rfl
theorem model_count_4027 : modelRow 4027 = 0 := by decide
theorem modelrow_agrees_4027 : modelRow 4027 = row 4027 := model_count_4027.trans row_count_4027.symm
def row_4028 : RowCertificate 4028 where
  first := false
  first_correct := by
    have composite : ¬ Nat.Prime (corner 4028 1) := by change ¬ Nat.Prime 64907193; exact PrattCertificates.composite_of_product (factor := 3) (cofactor := 21635731) rfl (by decide) (by decide)
    exact decide_eq_false composite
  second := false
  second_correct := by
    have composite : ¬ Nat.Prime (corner 4028 2) := by change ¬ Nat.Prime 64899137; exact PrattCertificates.composite_of_product (factor := 1433) (cofactor := 45289) rfl (by decide) (by decide)
    exact decide_eq_false composite
  third := false
  third_correct := by
    have composite : ¬ Nat.Prime (corner 4028 3) := by change ¬ Nat.Prime 64891081; exact PrattCertificates.composite_of_product (factor := 37) (cofactor := 1753813) rfl (by decide) (by decide)
    exact decide_eq_false composite
theorem row_count_4028 : row 4028 = 0 := (row_4028.checked).trans rfl
theorem model_count_4028 : modelRow 4028 = 0 := by decide
theorem modelrow_agrees_4028 : modelRow 4028 = row 4028 := model_count_4028.trans row_count_4028.symm
def row_4029 : RowCertificate 4029 where
  first := false
  first_correct := by
    have composite : ¬ Nat.Prime (corner 4029 1) := by change ¬ Nat.Prime 64939423; exact PrattCertificates.composite_of_product (factor := 2953) (cofactor := 21991) rfl (by decide) (by decide)
    exact decide_eq_false composite
  second := false
  second_correct := by
    have composite : ¬ Nat.Prime (corner 4029 2) := by change ¬ Nat.Prime 64931365; exact PrattCertificates.composite_of_product (factor := 5) (cofactor := 12986273) rfl (by decide) (by decide)
    exact decide_eq_false composite
  third := true
  third_correct := by
    have prime : Nat.Prime (corner 4029 3) := by change Nat.Prime 64923307; exact ProjectEulerSolutions.P58PrimeNodes.p_64923307.prime
    exact decide_eq_true prime
theorem row_count_4029 : row 4029 = 1 := (row_4029.checked).trans rfl
theorem model_count_4029 : modelRow 4029 = 1 := by decide
theorem modelrow_agrees_4029 : modelRow 4029 = row 4029 := model_count_4029.trans row_count_4029.symm
def row_4030 : RowCertificate 4030 where
  first := true
  first_correct := by
    have prime : Nat.Prime (corner 4030 1) := by change Nat.Prime 64971661; exact ProjectEulerSolutions.P58PrimeNodes.p_64971661.prime
    exact decide_eq_true prime
  second := false
  second_correct := by
    have composite : ¬ Nat.Prime (corner 4030 2) := by change ¬ Nat.Prime 64963601; exact PrattCertificates.composite_of_product (factor := 37) (cofactor := 1755773) rfl (by decide) (by decide)
    exact decide_eq_false composite
  third := false
  third_correct := by
    have composite : ¬ Nat.Prime (corner 4030 3) := by change ¬ Nat.Prime 64955541; exact PrattCertificates.composite_of_product (factor := 3) (cofactor := 21651847) rfl (by decide) (by decide)
    exact decide_eq_false composite
theorem row_count_4030 : row 4030 = 1 := (row_4030.checked).trans rfl
theorem model_count_4030 : modelRow 4030 = 1 := by decide
theorem modelrow_agrees_4030 : modelRow 4030 = row 4030 := model_count_4030.trans row_count_4030.symm
def row_4031 : RowCertificate 4031 where
  first := false
  first_correct := by
    have composite : ¬ Nat.Prime (corner 4031 1) := by change ¬ Nat.Prime 65003907; exact PrattCertificates.composite_of_product (factor := 3) (cofactor := 21667969) rfl (by decide) (by decide)
    exact decide_eq_false composite
  second := false
  second_correct := by
    have composite : ¬ Nat.Prime (corner 4031 2) := by change ¬ Nat.Prime 64995845; exact PrattCertificates.composite_of_product (factor := 5) (cofactor := 12999169) rfl (by decide) (by decide)
    exact decide_eq_false composite
  third := false
  third_correct := by
    have composite : ¬ Nat.Prime (corner 4031 3) := by change ¬ Nat.Prime 64987783; exact PrattCertificates.composite_of_product (factor := 7) (cofactor := 9283969) rfl (by decide) (by decide)
    exact decide_eq_false composite
theorem row_count_4031 : row 4031 = 0 := (row_4031.checked).trans rfl
theorem model_count_4031 : modelRow 4031 = 0 := by decide
theorem modelrow_agrees_4031 : modelRow 4031 = row 4031 := model_count_4031.trans row_count_4031.symm
def row_4032 : RowCertificate 4032 where
  first := true
  first_correct := by
    have prime : Nat.Prime (corner 4032 1) := by change Nat.Prime 65036161; exact ProjectEulerSolutions.P58PrimeNodes.p_65036161.prime
    exact decide_eq_true prime
  second := true
  second_correct := by
    have prime : Nat.Prime (corner 4032 2) := by change Nat.Prime 65028097; exact ProjectEulerSolutions.P58PrimeNodes.p_65028097.prime
    exact decide_eq_true prime
  third := false
  third_correct := by
    have composite : ¬ Nat.Prime (corner 4032 3) := by change ¬ Nat.Prime 65020033; exact PrattCertificates.composite_of_product (factor := 13) (cofactor := 5001541) rfl (by decide) (by decide)
    exact decide_eq_false composite
theorem row_count_4032 : row 4032 = 2 := (row_4032.checked).trans rfl
theorem model_count_4032 : modelRow 4032 = 2 := by decide
theorem modelrow_agrees_4032 : modelRow 4032 = row 4032 := model_count_4032.trans row_count_4032.symm
def row_4033 : RowCertificate 4033 where
  first := false
  first_correct := by
    have composite : ¬ Nat.Prime (corner 4033 1) := by change ¬ Nat.Prime 65068423; exact PrattCertificates.composite_of_product (factor := 7) (cofactor := 9295489) rfl (by decide) (by decide)
    exact decide_eq_false composite
  second := false
  second_correct := by
    have composite : ¬ Nat.Prime (corner 4033 2) := by change ¬ Nat.Prime 65060357; exact PrattCertificates.composite_of_product (factor := 233) (cofactor := 279229) rfl (by decide) (by decide)
    exact decide_eq_false composite
  third := false
  third_correct := by
    have composite : ¬ Nat.Prime (corner 4033 3) := by change ¬ Nat.Prime 65052291; exact PrattCertificates.composite_of_product (factor := 3) (cofactor := 21684097) rfl (by decide) (by decide)
    exact decide_eq_false composite
theorem row_count_4033 : row 4033 = 0 := (row_4033.checked).trans rfl
theorem model_count_4033 : modelRow 4033 = 0 := by decide
theorem modelrow_agrees_4033 : modelRow 4033 = row 4033 := model_count_4033.trans row_count_4033.symm
def row_4034 : RowCertificate 4034 where
  first := false
  first_correct := by
    have composite : ¬ Nat.Prime (corner 4034 1) := by change ¬ Nat.Prime 65100693; exact PrattCertificates.composite_of_product (factor := 3) (cofactor := 21700231) rfl (by decide) (by decide)
    exact decide_eq_false composite
  second := false
  second_correct := by
    have composite : ¬ Nat.Prime (corner 4034 2) := by change ¬ Nat.Prime 65092625; exact PrattCertificates.composite_of_product (factor := 5) (cofactor := 13018525) rfl (by decide) (by decide)
    exact decide_eq_false composite
  third := false
  third_correct := by
    have composite : ¬ Nat.Prime (corner 4034 3) := by change ¬ Nat.Prime 65084557; exact PrattCertificates.composite_of_product (factor := 19) (cofactor := 3425503) rfl (by decide) (by decide)
    exact decide_eq_false composite
theorem row_count_4034 : row 4034 = 0 := (row_4034.checked).trans rfl
theorem model_count_4034 : modelRow 4034 = 0 := by decide
theorem modelrow_agrees_4034 : modelRow 4034 = row 4034 := model_count_4034.trans row_count_4034.symm
def row_4035 : RowCertificate 4035 where
  first := true
  first_correct := by
    have prime : Nat.Prime (corner 4035 1) := by change Nat.Prime 65132971; exact ProjectEulerSolutions.P58PrimeNodes.p_65132971.prime
    exact decide_eq_true prime
  second := false
  second_correct := by
    have composite : ¬ Nat.Prime (corner 4035 2) := by change ¬ Nat.Prime 65124901; exact PrattCertificates.composite_of_product (factor := 101) (cofactor := 644801) rfl (by decide) (by decide)
    exact decide_eq_false composite
  third := false
  third_correct := by
    have composite : ¬ Nat.Prime (corner 4035 3) := by change ¬ Nat.Prime 65116831; exact PrattCertificates.composite_of_product (factor := 13) (cofactor := 5008987) rfl (by decide) (by decide)
    exact decide_eq_false composite
theorem row_count_4035 : row 4035 = 1 := (row_4035.checked).trans rfl
theorem model_count_4035 : modelRow 4035 = 1 := by decide
theorem modelrow_agrees_4035 : modelRow 4035 = row 4035 := model_count_4035.trans row_count_4035.symm
def row_4036 : RowCertificate 4036 where
  first := true
  first_correct := by
    have prime : Nat.Prime (corner 4036 1) := by change Nat.Prime 65165257; exact ProjectEulerSolutions.P58PrimeNodes.p_65165257.prime
    exact decide_eq_true prime
  second := false
  second_correct := by
    have composite : ¬ Nat.Prime (corner 4036 2) := by change ¬ Nat.Prime 65157185; exact PrattCertificates.composite_of_product (factor := 5) (cofactor := 13031437) rfl (by decide) (by decide)
    exact decide_eq_false composite
  third := false
  third_correct := by
    have composite : ¬ Nat.Prime (corner 4036 3) := by change ¬ Nat.Prime 65149113; exact PrattCertificates.composite_of_product (factor := 3) (cofactor := 21716371) rfl (by decide) (by decide)
    exact decide_eq_false composite
theorem row_count_4036 : row 4036 = 1 := (row_4036.checked).trans rfl
theorem model_count_4036 : modelRow 4036 = 1 := by decide
theorem modelrow_agrees_4036 : modelRow 4036 = row 4036 := model_count_4036.trans row_count_4036.symm
def row_4037 : RowCertificate 4037 where
  first := false
  first_correct := by
    have composite : ¬ Nat.Prime (corner 4037 1) := by change ¬ Nat.Prime 65197551; exact PrattCertificates.composite_of_product (factor := 3) (cofactor := 21732517) rfl (by decide) (by decide)
    exact decide_eq_false composite
  second := false
  second_correct := by
    have composite : ¬ Nat.Prime (corner 4037 2) := by change ¬ Nat.Prime 65189477; exact PrattCertificates.composite_of_product (factor := 29) (cofactor := 2247913) rfl (by decide) (by decide)
    exact decide_eq_false composite
  third := false
  third_correct := by
    have composite : ¬ Nat.Prime (corner 4037 3) := by change ¬ Nat.Prime 65181403; exact PrattCertificates.composite_of_product (factor := 7) (cofactor := 9311629) rfl (by decide) (by decide)
    exact decide_eq_false composite
theorem row_count_4037 : row 4037 = 0 := (row_4037.checked).trans rfl
theorem model_count_4037 : modelRow 4037 = 0 := by decide
theorem modelrow_agrees_4037 : modelRow 4037 = row 4037 := model_count_4037.trans row_count_4037.symm
def row_4038 : RowCertificate 4038 where
  first := false
  first_correct := by
    have composite : ¬ Nat.Prime (corner 4038 1) := by change ¬ Nat.Prime 65229853; exact PrattCertificates.composite_of_product (factor := 13) (cofactor := 5017681) rfl (by decide) (by decide)
    exact decide_eq_false composite
  second := false
  second_correct := by
    have composite : ¬ Nat.Prime (corner 4038 2) := by change ¬ Nat.Prime 65221777; exact PrattCertificates.composite_of_product (factor := 73) (cofactor := 893449) rfl (by decide) (by decide)
    exact decide_eq_false composite
  third := false
  third_correct := by
    have composite : ¬ Nat.Prime (corner 4038 3) := by change ¬ Nat.Prime 65213701; exact PrattCertificates.composite_of_product (factor := 7) (cofactor := 9316243) rfl (by decide) (by decide)
    exact decide_eq_false composite
theorem row_count_4038 : row 4038 = 0 := (row_4038.checked).trans rfl
theorem model_count_4038 : modelRow 4038 = 0 := by decide
theorem modelrow_agrees_4038 : modelRow 4038 = row 4038 := model_count_4038.trans row_count_4038.symm
def row_4039 : RowCertificate 4039 where
  first := true
  first_correct := by
    have prime : Nat.Prime (corner 4039 1) := by change Nat.Prime 65262163; exact ProjectEulerSolutions.P58PrimeNodes.p_65262163.prime
    exact decide_eq_true prime
  second := false
  second_correct := by
    have composite : ¬ Nat.Prime (corner 4039 2) := by change ¬ Nat.Prime 65254085; exact PrattCertificates.composite_of_product (factor := 5) (cofactor := 13050817) rfl (by decide) (by decide)
    exact decide_eq_false composite
  third := false
  third_correct := by
    have composite : ¬ Nat.Prime (corner 4039 3) := by change ¬ Nat.Prime 65246007; exact PrattCertificates.composite_of_product (factor := 3) (cofactor := 21748669) rfl (by decide) (by decide)
    exact decide_eq_false composite
theorem row_count_4039 : row 4039 = 1 := (row_4039.checked).trans rfl
theorem model_count_4039 : modelRow 4039 = 1 := by decide
theorem modelrow_agrees_4039 : modelRow 4039 = row 4039 := model_count_4039.trans row_count_4039.symm
def row_4040 : RowCertificate 4040 where
  first := false
  first_correct := by
    have composite : ¬ Nat.Prime (corner 4040 1) := by change ¬ Nat.Prime 65294481; exact PrattCertificates.composite_of_product (factor := 3) (cofactor := 21764827) rfl (by decide) (by decide)
    exact decide_eq_false composite
  second := true
  second_correct := by
    have prime : Nat.Prime (corner 4040 2) := by change Nat.Prime 65286401; exact ProjectEulerSolutions.P58PrimeNodes.p_65286401.prime
    exact decide_eq_true prime
  third := true
  third_correct := by
    have prime : Nat.Prime (corner 4040 3) := by change Nat.Prime 65278321; exact ProjectEulerSolutions.P58PrimeNodes.p_65278321.prime
    exact decide_eq_true prime
theorem row_count_4040 : row 4040 = 2 := (row_4040.checked).trans rfl
theorem model_count_4040 : modelRow 4040 = 2 := by decide
theorem modelrow_agrees_4040 : modelRow 4040 = row 4040 := model_count_4040.trans row_count_4040.symm
def row_4041 : RowCertificate 4041 where
  first := false
  first_correct := by
    have composite : ¬ Nat.Prime (corner 4041 1) := by change ¬ Nat.Prime 65326807; exact PrattCertificates.composite_of_product (factor := 7) (cofactor := 9332401) rfl (by decide) (by decide)
    exact decide_eq_false composite
  second := false
  second_correct := by
    have composite : ¬ Nat.Prime (corner 4041 2) := by change ¬ Nat.Prime 65318725; exact PrattCertificates.composite_of_product (factor := 5) (cofactor := 13063745) rfl (by decide) (by decide)
    exact decide_eq_false composite
  third := false
  third_correct := by
    have composite : ¬ Nat.Prime (corner 4041 3) := by change ¬ Nat.Prime 65310643; exact PrattCertificates.composite_of_product (factor := 79) (cofactor := 826717) rfl (by decide) (by decide)
    exact decide_eq_false composite
theorem row_count_4041 : row 4041 = 0 := (row_4041.checked).trans rfl
theorem model_count_4041 : modelRow 4041 = 0 := by decide
theorem modelrow_agrees_4041 : modelRow 4041 = row 4041 := model_count_4041.trans row_count_4041.symm
def row_4042 : RowCertificate 4042 where
  first := true
  first_correct := by
    have prime : Nat.Prime (corner 4042 1) := by change Nat.Prime 65359141; exact ProjectEulerSolutions.P58PrimeNodes.p_65359141.prime
    exact decide_eq_true prime
  second := false
  second_correct := by
    have composite : ¬ Nat.Prime (corner 4042 2) := by change ¬ Nat.Prime 65351057; exact PrattCertificates.composite_of_product (factor := 3929) (cofactor := 16633) rfl (by decide) (by decide)
    exact decide_eq_false composite
  third := false
  third_correct := by
    have composite : ¬ Nat.Prime (corner 4042 3) := by change ¬ Nat.Prime 65342973; exact PrattCertificates.composite_of_product (factor := 3) (cofactor := 21780991) rfl (by decide) (by decide)
    exact decide_eq_false composite
theorem row_count_4042 : row 4042 = 1 := (row_4042.checked).trans rfl
theorem model_count_4042 : modelRow 4042 = 1 := by decide
theorem modelrow_agrees_4042 : modelRow 4042 = row 4042 := model_count_4042.trans row_count_4042.symm
def row_4043 : RowCertificate 4043 where
  first := false
  first_correct := by
    have composite : ¬ Nat.Prime (corner 4043 1) := by change ¬ Nat.Prime 65391483; exact PrattCertificates.composite_of_product (factor := 3) (cofactor := 21797161) rfl (by decide) (by decide)
    exact decide_eq_false composite
  second := false
  second_correct := by
    have composite : ¬ Nat.Prime (corner 4043 2) := by change ¬ Nat.Prime 65383397; exact PrattCertificates.composite_of_product (factor := 41) (cofactor := 1594717) rfl (by decide) (by decide)
    exact decide_eq_false composite
  third := false
  third_correct := by
    have composite : ¬ Nat.Prime (corner 4043 3) := by change ¬ Nat.Prime 65375311; exact PrattCertificates.composite_of_product (factor := 31) (cofactor := 2108881) rfl (by decide) (by decide)
    exact decide_eq_false composite
theorem row_count_4043 : row 4043 = 0 := (row_4043.checked).trans rfl
theorem model_count_4043 : modelRow 4043 = 0 := by decide
theorem modelrow_agrees_4043 : modelRow 4043 = row 4043 := model_count_4043.trans row_count_4043.symm
def row_4044 : RowCertificate 4044 where
  first := true
  first_correct := by
    have prime : Nat.Prime (corner 4044 1) := by change Nat.Prime 65423833; exact ProjectEulerSolutions.P58PrimeNodes.p_65423833.prime
    exact decide_eq_true prime
  second := false
  second_correct := by
    have composite : ¬ Nat.Prime (corner 4044 2) := by change ¬ Nat.Prime 65415745; exact PrattCertificates.composite_of_product (factor := 5) (cofactor := 13083149) rfl (by decide) (by decide)
    exact decide_eq_false composite
  third := false
  third_correct := by
    have composite : ¬ Nat.Prime (corner 4044 3) := by change ¬ Nat.Prime 65407657; exact PrattCertificates.composite_of_product (factor := 7) (cofactor := 9343951) rfl (by decide) (by decide)
    exact decide_eq_false composite
theorem row_count_4044 : row 4044 = 1 := (row_4044.checked).trans rfl
theorem model_count_4044 : modelRow 4044 = 1 := by decide
theorem modelrow_agrees_4044 : modelRow 4044 = row 4044 := model_count_4044.trans row_count_4044.symm
def row_4045 : RowCertificate 4045 where
  first := false
  first_correct := by
    have composite : ¬ Nat.Prime (corner 4045 1) := by change ¬ Nat.Prime 65456191; exact PrattCertificates.composite_of_product (factor := 43) (cofactor := 1522237) rfl (by decide) (by decide)
    exact decide_eq_false composite
  second := false
  second_correct := by
    have composite : ¬ Nat.Prime (corner 4045 2) := by change ¬ Nat.Prime 65448101; exact PrattCertificates.composite_of_product (factor := 101) (cofactor := 648001) rfl (by decide) (by decide)
    exact decide_eq_false composite
  third := false
  third_correct := by
    have composite : ¬ Nat.Prime (corner 4045 3) := by change ¬ Nat.Prime 65440011; exact PrattCertificates.composite_of_product (factor := 3) (cofactor := 21813337) rfl (by decide) (by decide)
    exact decide_eq_false composite
theorem row_count_4045 : row 4045 = 0 := (row_4045.checked).trans rfl
theorem model_count_4045 : modelRow 4045 = 0 := by decide
theorem modelrow_agrees_4045 : modelRow 4045 = row 4045 := model_count_4045.trans row_count_4045.symm
def row_4046 : RowCertificate 4046 where
  first := false
  first_correct := by
    have composite : ¬ Nat.Prime (corner 4046 1) := by change ¬ Nat.Prime 65488557; exact PrattCertificates.composite_of_product (factor := 3) (cofactor := 21829519) rfl (by decide) (by decide)
    exact decide_eq_false composite
  second := false
  second_correct := by
    have composite : ¬ Nat.Prime (corner 4046 2) := by change ¬ Nat.Prime 65480465; exact PrattCertificates.composite_of_product (factor := 5) (cofactor := 13096093) rfl (by decide) (by decide)
    exact decide_eq_false composite
  third := false
  third_correct := by
    have composite : ¬ Nat.Prime (corner 4046 3) := by change ¬ Nat.Prime 65472373; exact PrattCertificates.composite_of_product (factor := 163) (cofactor := 401671) rfl (by decide) (by decide)
    exact decide_eq_false composite
theorem row_count_4046 : row 4046 = 0 := (row_4046.checked).trans rfl
theorem model_count_4046 : modelRow 4046 = 0 := by decide
theorem modelrow_agrees_4046 : modelRow 4046 = row 4046 := model_count_4046.trans row_count_4046.symm
def row_4047 : RowCertificate 4047 where
  first := false
  first_correct := by
    have composite : ¬ Nat.Prime (corner 4047 1) := by change ¬ Nat.Prime 65520931; exact PrattCertificates.composite_of_product (factor := 7) (cofactor := 9360133) rfl (by decide) (by decide)
    exact decide_eq_false composite
  second := false
  second_correct := by
    have composite : ¬ Nat.Prime (corner 4047 2) := by change ¬ Nat.Prime 65512837; exact PrattCertificates.composite_of_product (factor := 13) (cofactor := 5039449) rfl (by decide) (by decide)
    exact decide_eq_false composite
  third := true
  third_correct := by
    have prime : Nat.Prime (corner 4047 3) := by change Nat.Prime 65504743; exact ProjectEulerSolutions.P58PrimeNodes.p_65504743.prime
    exact decide_eq_true prime
theorem row_count_4047 : row 4047 = 1 := (row_4047.checked).trans rfl
theorem model_count_4047 : modelRow 4047 = 1 := by decide
theorem modelrow_agrees_4047 : modelRow 4047 = row 4047 := model_count_4047.trans row_count_4047.symm
def row_4048 : RowCertificate 4048 where
  first := false
  first_correct := by
    have composite : ¬ Nat.Prime (corner 4048 1) := by change ¬ Nat.Prime 65553313; exact PrattCertificates.composite_of_product (factor := 7) (cofactor := 9364759) rfl (by decide) (by decide)
    exact decide_eq_false composite
  second := false
  second_correct := by
    have composite : ¬ Nat.Prime (corner 4048 2) := by change ¬ Nat.Prime 65545217; exact PrattCertificates.composite_of_product (factor := 17) (cofactor := 3855601) rfl (by decide) (by decide)
    exact decide_eq_false composite
  third := false
  third_correct := by
    have composite : ¬ Nat.Prime (corner 4048 3) := by change ¬ Nat.Prime 65537121; exact PrattCertificates.composite_of_product (factor := 3) (cofactor := 21845707) rfl (by decide) (by decide)
    exact decide_eq_false composite
theorem row_count_4048 : row 4048 = 0 := (row_4048.checked).trans rfl
theorem model_count_4048 : modelRow 4048 = 0 := by decide
theorem modelrow_agrees_4048 : modelRow 4048 = row 4048 := model_count_4048.trans row_count_4048.symm
def row_4049 : RowCertificate 4049 where
  first := false
  first_correct := by
    have composite : ¬ Nat.Prime (corner 4049 1) := by change ¬ Nat.Prime 65585703; exact PrattCertificates.composite_of_product (factor := 3) (cofactor := 21861901) rfl (by decide) (by decide)
    exact decide_eq_false composite
  second := false
  second_correct := by
    have composite : ¬ Nat.Prime (corner 4049 2) := by change ¬ Nat.Prime 65577605; exact PrattCertificates.composite_of_product (factor := 5) (cofactor := 13115521) rfl (by decide) (by decide)
    exact decide_eq_false composite
  third := false
  third_correct := by
    have composite : ¬ Nat.Prime (corner 4049 3) := by change ¬ Nat.Prime 65569507; exact PrattCertificates.composite_of_product (factor := 1699) (cofactor := 38593) rfl (by decide) (by decide)
    exact decide_eq_false composite
theorem row_count_4049 : row 4049 = 0 := (row_4049.checked).trans rfl
theorem model_count_4049 : modelRow 4049 = 0 := by decide
theorem modelrow_agrees_4049 : modelRow 4049 = row 4049 := model_count_4049.trans row_count_4049.symm
def row_4050 : RowCertificate 4050 where
  first := false
  first_correct := by
    have composite : ¬ Nat.Prime (corner 4050 1) := by change ¬ Nat.Prime 65618101; exact PrattCertificates.composite_of_product (factor := 8011) (cofactor := 8191) rfl (by decide) (by decide)
    exact decide_eq_false composite
  second := true
  second_correct := by
    have prime : Nat.Prime (corner 4050 2) := by change Nat.Prime 65610001; exact ProjectEulerSolutions.P58PrimeNodes.p_65610001.prime
    exact decide_eq_true prime
  third := false
  third_correct := by
    have composite : ¬ Nat.Prime (corner 4050 3) := by change ¬ Nat.Prime 65601901; exact PrattCertificates.composite_of_product (factor := 61) (cofactor := 1075441) rfl (by decide) (by decide)
    exact decide_eq_false composite
theorem row_count_4050 : row 4050 = 1 := (row_4050.checked).trans rfl
theorem model_count_4050 : modelRow 4050 = 1 := by decide
theorem modelrow_agrees_4050 : modelRow 4050 = row 4050 := model_count_4050.trans row_count_4050.symm
def row_4051 : RowCertificate 4051 where
  first := false
  first_correct := by
    have composite : ¬ Nat.Prime (corner 4051 1) := by change ¬ Nat.Prime 65650507; exact PrattCertificates.composite_of_product (factor := 13) (cofactor := 5050039) rfl (by decide) (by decide)
    exact decide_eq_false composite
  second := false
  second_correct := by
    have composite : ¬ Nat.Prime (corner 4051 2) := by change ¬ Nat.Prime 65642405; exact PrattCertificates.composite_of_product (factor := 5) (cofactor := 13128481) rfl (by decide) (by decide)
    exact decide_eq_false composite
  third := false
  third_correct := by
    have composite : ¬ Nat.Prime (corner 4051 3) := by change ¬ Nat.Prime 65634303; exact PrattCertificates.composite_of_product (factor := 3) (cofactor := 21878101) rfl (by decide) (by decide)
    exact decide_eq_false composite
theorem row_count_4051 : row 4051 = 0 := (row_4051.checked).trans rfl
theorem model_count_4051 : modelRow 4051 = 0 := by decide
theorem modelrow_agrees_4051 : modelRow 4051 = row 4051 := model_count_4051.trans row_count_4051.symm
def row_4052 : RowCertificate 4052 where
  first := false
  first_correct := by
    have composite : ¬ Nat.Prime (corner 4052 1) := by change ¬ Nat.Prime 65682921; exact PrattCertificates.composite_of_product (factor := 3) (cofactor := 21894307) rfl (by decide) (by decide)
    exact decide_eq_false composite
  second := false
  second_correct := by
    have composite : ¬ Nat.Prime (corner 4052 2) := by change ¬ Nat.Prime 65674817; exact PrattCertificates.composite_of_product (factor := 13) (cofactor := 5051909) rfl (by decide) (by decide)
    exact decide_eq_false composite
  third := false
  third_correct := by
    have composite : ¬ Nat.Prime (corner 4052 3) := by change ¬ Nat.Prime 65666713; exact PrattCertificates.composite_of_product (factor := 7) (cofactor := 9380959) rfl (by decide) (by decide)
    exact decide_eq_false composite
theorem row_count_4052 : row 4052 = 0 := (row_4052.checked).trans rfl
theorem model_count_4052 : modelRow 4052 = 0 := by decide
theorem modelrow_agrees_4052 : modelRow 4052 = row 4052 := model_count_4052.trans row_count_4052.symm
def row_4053 : RowCertificate 4053 where
  first := true
  first_correct := by
    have prime : Nat.Prime (corner 4053 1) := by change Nat.Prime 65715343; exact ProjectEulerSolutions.P58PrimeNodes.p_65715343.prime
    exact decide_eq_true prime
  second := false
  second_correct := by
    have composite : ¬ Nat.Prime (corner 4053 2) := by change ¬ Nat.Prime 65707237; exact PrattCertificates.composite_of_product (factor := 4793) (cofactor := 13709) rfl (by decide) (by decide)
    exact decide_eq_false composite
  third := false
  third_correct := by
    have composite : ¬ Nat.Prime (corner 4053 3) := by change ¬ Nat.Prime 65699131; exact PrattCertificates.composite_of_product (factor := 19) (cofactor := 3457849) rfl (by decide) (by decide)
    exact decide_eq_false composite
theorem row_count_4053 : row 4053 = 1 := (row_4053.checked).trans rfl
theorem model_count_4053 : modelRow 4053 = 1 := by decide
theorem modelrow_agrees_4053 : modelRow 4053 = row 4053 := model_count_4053.trans row_count_4053.symm
def row_4054 : RowCertificate 4054 where
  first := false
  first_correct := by
    have composite : ¬ Nat.Prime (corner 4054 1) := by change ¬ Nat.Prime 65747773; exact PrattCertificates.composite_of_product (factor := 7) (cofactor := 9392539) rfl (by decide) (by decide)
    exact decide_eq_false composite
  second := false
  second_correct := by
    have composite : ¬ Nat.Prime (corner 4054 2) := by change ¬ Nat.Prime 65739665; exact PrattCertificates.composite_of_product (factor := 5) (cofactor := 13147933) rfl (by decide) (by decide)
    exact decide_eq_false composite
  third := false
  third_correct := by
    have composite : ¬ Nat.Prime (corner 4054 3) := by change ¬ Nat.Prime 65731557; exact PrattCertificates.composite_of_product (factor := 3) (cofactor := 21910519) rfl (by decide) (by decide)
    exact decide_eq_false composite
theorem row_count_4054 : row 4054 = 0 := (row_4054.checked).trans rfl
theorem model_count_4054 : modelRow 4054 = 0 := by decide
theorem modelrow_agrees_4054 : modelRow 4054 = row 4054 := model_count_4054.trans row_count_4054.symm
def row_4055 : RowCertificate 4055 where
  first := false
  first_correct := by
    have composite : ¬ Nat.Prime (corner 4055 1) := by change ¬ Nat.Prime 65780211; exact PrattCertificates.composite_of_product (factor := 3) (cofactor := 21926737) rfl (by decide) (by decide)
    exact decide_eq_false composite
  second := false
  second_correct := by
    have composite : ¬ Nat.Prime (corner 4055 2) := by change ¬ Nat.Prime 65772101; exact PrattCertificates.composite_of_product (factor := 1777) (cofactor := 37013) rfl (by decide) (by decide)
    exact decide_eq_false composite
  third := false
  third_correct := by
    have composite : ¬ Nat.Prime (corner 4055 3) := by change ¬ Nat.Prime 65763991; exact PrattCertificates.composite_of_product (factor := 229) (cofactor := 287179) rfl (by decide) (by decide)
    exact decide_eq_false composite
theorem row_count_4055 : row 4055 = 0 := (row_4055.checked).trans rfl
theorem model_count_4055 : modelRow 4055 = 0 := by decide
theorem modelrow_agrees_4055 : modelRow 4055 = row 4055 := model_count_4055.trans row_count_4055.symm
def row_4056 : RowCertificate 4056 where
  first := false
  first_correct := by
    have composite : ¬ Nat.Prime (corner 4056 1) := by change ¬ Nat.Prime 65812657; exact PrattCertificates.composite_of_product (factor := 97) (cofactor := 678481) rfl (by decide) (by decide)
    exact decide_eq_false composite
  second := false
  second_correct := by
    have composite : ¬ Nat.Prime (corner 4056 2) := by change ¬ Nat.Prime 65804545; exact PrattCertificates.composite_of_product (factor := 5) (cofactor := 13160909) rfl (by decide) (by decide)
    exact decide_eq_false composite
  third := false
  third_correct := by
    have composite : ¬ Nat.Prime (corner 4056 3) := by change ¬ Nat.Prime 65796433; exact PrattCertificates.composite_of_product (factor := 73) (cofactor := 901321) rfl (by decide) (by decide)
    exact decide_eq_false composite
theorem row_count_4056 : row 4056 = 0 := (row_4056.checked).trans rfl
theorem model_count_4056 : modelRow 4056 = 0 := by decide
theorem modelrow_agrees_4056 : modelRow 4056 = row 4056 := model_count_4056.trans row_count_4056.symm
def row_4057 : RowCertificate 4057 where
  first := true
  first_correct := by
    have prime : Nat.Prime (corner 4057 1) := by change Nat.Prime 65845111; exact ProjectEulerSolutions.P58PrimeNodes.p_65845111.prime
    exact decide_eq_true prime
  second := true
  second_correct := by
    have prime : Nat.Prime (corner 4057 2) := by change Nat.Prime 65836997; exact ProjectEulerSolutions.P58PrimeNodes.p_65836997.prime
    exact decide_eq_true prime
  third := false
  third_correct := by
    have composite : ¬ Nat.Prime (corner 4057 3) := by change ¬ Nat.Prime 65828883; exact PrattCertificates.composite_of_product (factor := 3) (cofactor := 21942961) rfl (by decide) (by decide)
    exact decide_eq_false composite
theorem row_count_4057 : row 4057 = 2 := (row_4057.checked).trans rfl
theorem model_count_4057 : modelRow 4057 = 2 := by decide
theorem modelrow_agrees_4057 : modelRow 4057 = row 4057 := model_count_4057.trans row_count_4057.symm
def row_4058 : RowCertificate 4058 where
  first := false
  first_correct := by
    have composite : ¬ Nat.Prime (corner 4058 1) := by change ¬ Nat.Prime 65877573; exact PrattCertificates.composite_of_product (factor := 3) (cofactor := 21959191) rfl (by decide) (by decide)
    exact decide_eq_false composite
  second := true
  second_correct := by
    have prime : Nat.Prime (corner 4058 2) := by change Nat.Prime 65869457; exact ProjectEulerSolutions.P58PrimeNodes.p_65869457.prime
    exact decide_eq_true prime
  third := false
  third_correct := by
    have composite : ¬ Nat.Prime (corner 4058 3) := by change ¬ Nat.Prime 65861341; exact PrattCertificates.composite_of_product (factor := 7) (cofactor := 9408763) rfl (by decide) (by decide)
    exact decide_eq_false composite
theorem row_count_4058 : row 4058 = 1 := (row_4058.checked).trans rfl
theorem model_count_4058 : modelRow 4058 = 1 := by decide
theorem modelrow_agrees_4058 : modelRow 4058 = row 4058 := model_count_4058.trans row_count_4058.symm
def row_4059 : RowCertificate 4059 where
  first := true
  first_correct := by
    have prime : Nat.Prime (corner 4059 1) := by change Nat.Prime 65910043; exact ProjectEulerSolutions.P58PrimeNodes.p_65910043.prime
    exact decide_eq_true prime
  second := false
  second_correct := by
    have composite : ¬ Nat.Prime (corner 4059 2) := by change ¬ Nat.Prime 65901925; exact PrattCertificates.composite_of_product (factor := 5) (cofactor := 13180385) rfl (by decide) (by decide)
    exact decide_eq_false composite
  third := false
  third_correct := by
    have composite : ¬ Nat.Prime (corner 4059 3) := by change ¬ Nat.Prime 65893807; exact PrattCertificates.composite_of_product (factor := 7) (cofactor := 9413401) rfl (by decide) (by decide)
    exact decide_eq_false composite
theorem row_count_4059 : row 4059 = 1 := (row_4059.checked).trans rfl
theorem model_count_4059 : modelRow 4059 = 1 := by decide
theorem modelrow_agrees_4059 : modelRow 4059 = row 4059 := model_count_4059.trans row_count_4059.symm
def row_4060 : RowCertificate 4060 where
  first := false
  first_correct := by
    have composite : ¬ Nat.Prime (corner 4060 1) := by change ¬ Nat.Prime 65942521; exact PrattCertificates.composite_of_product (factor := 19) (cofactor := 3470659) rfl (by decide) (by decide)
    exact decide_eq_false composite
  second := false
  second_correct := by
    have composite : ¬ Nat.Prime (corner 4060 2) := by change ¬ Nat.Prime 65934401; exact PrattCertificates.composite_of_product (factor := 13) (cofactor := 5071877) rfl (by decide) (by decide)
    exact decide_eq_false composite
  third := false
  third_correct := by
    have composite : ¬ Nat.Prime (corner 4060 3) := by change ¬ Nat.Prime 65926281; exact PrattCertificates.composite_of_product (factor := 3) (cofactor := 21975427) rfl (by decide) (by decide)
    exact decide_eq_false composite
theorem row_count_4060 : row 4060 = 0 := (row_4060.checked).trans rfl
theorem model_count_4060 : modelRow 4060 = 0 := by decide
theorem modelrow_agrees_4060 : modelRow 4060 = row 4060 := model_count_4060.trans row_count_4060.symm
def row_4061 : RowCertificate 4061 where
  first := false
  first_correct := by
    have composite : ¬ Nat.Prime (corner 4061 1) := by change ¬ Nat.Prime 65975007; exact PrattCertificates.composite_of_product (factor := 3) (cofactor := 21991669) rfl (by decide) (by decide)
    exact decide_eq_false composite
  second := false
  second_correct := by
    have composite : ¬ Nat.Prime (corner 4061 2) := by change ¬ Nat.Prime 65966885; exact PrattCertificates.composite_of_product (factor := 5) (cofactor := 13193377) rfl (by decide) (by decide)
    exact decide_eq_false composite
  third := false
  third_correct := by
    have composite : ¬ Nat.Prime (corner 4061 3) := by change ¬ Nat.Prime 65958763; exact PrattCertificates.composite_of_product (factor := 13) (cofactor := 5073751) rfl (by decide) (by decide)
    exact decide_eq_false composite
theorem row_count_4061 : row 4061 = 0 := (row_4061.checked).trans rfl
theorem model_count_4061 : modelRow 4061 = 0 := by decide
theorem modelrow_agrees_4061 : modelRow 4061 = row 4061 := model_count_4061.trans row_count_4061.symm
def row_4062 : RowCertificate 4062 where
  first := false
  first_correct := by
    have composite : ¬ Nat.Prime (corner 4062 1) := by change ¬ Nat.Prime 66007501; exact PrattCertificates.composite_of_product (factor := 7) (cofactor := 9429643) rfl (by decide) (by decide)
    exact decide_eq_false composite
  second := false
  second_correct := by
    have composite : ¬ Nat.Prime (corner 4062 2) := by change ¬ Nat.Prime 65999377; exact PrattCertificates.composite_of_product (factor := 61) (cofactor := 1081957) rfl (by decide) (by decide)
    exact decide_eq_false composite
  third := false
  third_correct := by
    have composite : ¬ Nat.Prime (corner 4062 3) := by change ¬ Nat.Prime 65991253; exact PrattCertificates.composite_of_product (factor := 499) (cofactor := 132247) rfl (by decide) (by decide)
    exact decide_eq_false composite
theorem row_count_4062 : row 4062 = 0 := (row_4062.checked).trans rfl
theorem model_count_4062 : modelRow 4062 = 0 := by decide
theorem modelrow_agrees_4062 : modelRow 4062 = row 4062 := model_count_4062.trans row_count_4062.symm
def row_4063 : RowCertificate 4063 where
  first := false
  first_correct := by
    have composite : ¬ Nat.Prime (corner 4063 1) := by change ¬ Nat.Prime 66040003; exact PrattCertificates.composite_of_product (factor := 61) (cofactor := 1082623) rfl (by decide) (by decide)
    exact decide_eq_false composite
  second := false
  second_correct := by
    have composite : ¬ Nat.Prime (corner 4063 2) := by change ¬ Nat.Prime 66031877; exact PrattCertificates.composite_of_product (factor := 97) (cofactor := 680741) rfl (by decide) (by decide)
    exact decide_eq_false composite
  third := false
  third_correct := by
    have composite : ¬ Nat.Prime (corner 4063 3) := by change ¬ Nat.Prime 66023751; exact PrattCertificates.composite_of_product (factor := 3) (cofactor := 22007917) rfl (by decide) (by decide)
    exact decide_eq_false composite
theorem row_count_4063 : row 4063 = 0 := (row_4063.checked).trans rfl
theorem model_count_4063 : modelRow 4063 = 0 := by decide
theorem modelrow_agrees_4063 : modelRow 4063 = row 4063 := model_count_4063.trans row_count_4063.symm
def row_4064 : RowCertificate 4064 where
  first := false
  first_correct := by
    have composite : ¬ Nat.Prime (corner 4064 1) := by change ¬ Nat.Prime 66072513; exact PrattCertificates.composite_of_product (factor := 3) (cofactor := 22024171) rfl (by decide) (by decide)
    exact decide_eq_false composite
  second := false
  second_correct := by
    have composite : ¬ Nat.Prime (corner 4064 2) := by change ¬ Nat.Prime 66064385; exact PrattCertificates.composite_of_product (factor := 5) (cofactor := 13212877) rfl (by decide) (by decide)
    exact decide_eq_false composite
  third := false
  third_correct := by
    have composite : ¬ Nat.Prime (corner 4064 3) := by change ¬ Nat.Prime 66056257; exact PrattCertificates.composite_of_product (factor := 31) (cofactor := 2130847) rfl (by decide) (by decide)
    exact decide_eq_false composite
theorem row_count_4064 : row 4064 = 0 := (row_4064.checked).trans rfl
theorem model_count_4064 : modelRow 4064 = 0 := by decide
theorem modelrow_agrees_4064 : modelRow 4064 = row 4064 := model_count_4064.trans row_count_4064.symm
def row_4065 : RowCertificate 4065 where
  first := true
  first_correct := by
    have prime : Nat.Prime (corner 4065 1) := by change Nat.Prime 66105031; exact ProjectEulerSolutions.P58PrimeNodes.p_66105031.prime
    exact decide_eq_true prime
  second := false
  second_correct := by
    have composite : ¬ Nat.Prime (corner 4065 2) := by change ¬ Nat.Prime 66096901; exact PrattCertificates.composite_of_product (factor := 13) (cofactor := 5084377) rfl (by decide) (by decide)
    exact decide_eq_false composite
  third := false
  third_correct := by
    have composite : ¬ Nat.Prime (corner 4065 3) := by change ¬ Nat.Prime 66088771; exact PrattCertificates.composite_of_product (factor := 7) (cofactor := 9441253) rfl (by decide) (by decide)
    exact decide_eq_false composite
theorem row_count_4065 : row 4065 = 1 := (row_4065.checked).trans rfl
theorem model_count_4065 : modelRow 4065 = 1 := by decide
theorem modelrow_agrees_4065 : modelRow 4065 = row 4065 := model_count_4065.trans row_count_4065.symm
def row_4066 : RowCertificate 4066 where
  first := true
  first_correct := by
    have prime : Nat.Prime (corner 4066 1) := by change Nat.Prime 66137557; exact ProjectEulerSolutions.P58PrimeNodes.p_66137557.prime
    exact decide_eq_true prime
  second := false
  second_correct := by
    have composite : ¬ Nat.Prime (corner 4066 2) := by change ¬ Nat.Prime 66129425; exact PrattCertificates.composite_of_product (factor := 5) (cofactor := 13225885) rfl (by decide) (by decide)
    exact decide_eq_false composite
  third := false
  third_correct := by
    have composite : ¬ Nat.Prime (corner 4066 3) := by change ¬ Nat.Prime 66121293; exact PrattCertificates.composite_of_product (factor := 3) (cofactor := 22040431) rfl (by decide) (by decide)
    exact decide_eq_false composite
theorem row_count_4066 : row 4066 = 1 := (row_4066.checked).trans rfl
theorem model_count_4066 : modelRow 4066 = 1 := by decide
theorem modelrow_agrees_4066 : modelRow 4066 = row 4066 := model_count_4066.trans row_count_4066.symm
def row_4067 : RowCertificate 4067 where
  first := false
  first_correct := by
    have composite : ¬ Nat.Prime (corner 4067 1) := by change ¬ Nat.Prime 66170091; exact PrattCertificates.composite_of_product (factor := 3) (cofactor := 22056697) rfl (by decide) (by decide)
    exact decide_eq_false composite
  second := false
  second_correct := by
    have composite : ¬ Nat.Prime (corner 4067 2) := by change ¬ Nat.Prime 66161957; exact PrattCertificates.composite_of_product (factor := 37) (cofactor := 1788161) rfl (by decide) (by decide)
    exact decide_eq_false composite
  third := false
  third_correct := by
    have composite : ¬ Nat.Prime (corner 4067 3) := by change ¬ Nat.Prime 66153823; exact PrattCertificates.composite_of_product (factor := 43) (cofactor := 1538461) rfl (by decide) (by decide)
    exact decide_eq_false composite
theorem row_count_4067 : row 4067 = 0 := (row_4067.checked).trans rfl
theorem model_count_4067 : modelRow 4067 = 0 := by decide
theorem modelrow_agrees_4067 : modelRow 4067 = row 4067 := model_count_4067.trans row_count_4067.symm
def row_4068 : RowCertificate 4068 where
  first := false
  first_correct := by
    have composite : ¬ Nat.Prime (corner 4068 1) := by change ¬ Nat.Prime 66202633; exact PrattCertificates.composite_of_product (factor := 7) (cofactor := 9457519) rfl (by decide) (by decide)
    exact decide_eq_false composite
  second := false
  second_correct := by
    have composite : ¬ Nat.Prime (corner 4068 2) := by change ¬ Nat.Prime 66194497; exact PrattCertificates.composite_of_product (factor := 157) (cofactor := 421621) rfl (by decide) (by decide)
    exact decide_eq_false composite
  third := false
  third_correct := by
    have composite : ¬ Nat.Prime (corner 4068 3) := by change ¬ Nat.Prime 66186361; exact PrattCertificates.composite_of_product (factor := 4153) (cofactor := 15937) rfl (by decide) (by decide)
    exact decide_eq_false composite
theorem row_count_4068 : row 4068 = 0 := (row_4068.checked).trans rfl
theorem model_count_4068 : modelRow 4068 = 0 := by decide
theorem modelrow_agrees_4068 : modelRow 4068 = row 4068 := model_count_4068.trans row_count_4068.symm
def row_4069 : RowCertificate 4069 where
  first := false
  first_correct := by
    have composite : ¬ Nat.Prime (corner 4069 1) := by change ¬ Nat.Prime 66235183; exact PrattCertificates.composite_of_product (factor := 7) (cofactor := 9462169) rfl (by decide) (by decide)
    exact decide_eq_false composite
  second := false
  second_correct := by
    have composite : ¬ Nat.Prime (corner 4069 2) := by change ¬ Nat.Prime 66227045; exact PrattCertificates.composite_of_product (factor := 5) (cofactor := 13245409) rfl (by decide) (by decide)
    exact decide_eq_false composite
  third := false
  third_correct := by
    have composite : ¬ Nat.Prime (corner 4069 3) := by change ¬ Nat.Prime 66218907; exact PrattCertificates.composite_of_product (factor := 3) (cofactor := 22072969) rfl (by decide) (by decide)
    exact decide_eq_false composite
theorem row_count_4069 : row 4069 = 0 := (row_4069.checked).trans rfl
theorem model_count_4069 : modelRow 4069 = 0 := by decide
theorem modelrow_agrees_4069 : modelRow 4069 = row 4069 := model_count_4069.trans row_count_4069.symm
def row_4070 : RowCertificate 4070 where
  first := false
  first_correct := by
    have composite : ¬ Nat.Prime (corner 4070 1) := by change ¬ Nat.Prime 66267741; exact PrattCertificates.composite_of_product (factor := 3) (cofactor := 22089247) rfl (by decide) (by decide)
    exact decide_eq_false composite
  second := false
  second_correct := by
    have composite : ¬ Nat.Prime (corner 4070 2) := by change ¬ Nat.Prime 66259601; exact PrattCertificates.composite_of_product (factor := 6833) (cofactor := 9697) rfl (by decide) (by decide)
    exact decide_eq_false composite
  third := false
  third_correct := by
    have composite : ¬ Nat.Prime (corner 4070 3) := by change ¬ Nat.Prime 66251461; exact PrattCertificates.composite_of_product (factor := 19) (cofactor := 3486919) rfl (by decide) (by decide)
    exact decide_eq_false composite
theorem row_count_4070 : row 4070 = 0 := (row_4070.checked).trans rfl
theorem model_count_4070 : modelRow 4070 = 0 := by decide
theorem modelrow_agrees_4070 : modelRow 4070 = row 4070 := model_count_4070.trans row_count_4070.symm
def row_4071 : RowCertificate 4071 where
  first := true
  first_correct := by
    have prime : Nat.Prime (corner 4071 1) := by change Nat.Prime 66300307; exact ProjectEulerSolutions.P58PrimeNodes.p_66300307.prime
    exact decide_eq_true prime
  second := false
  second_correct := by
    have composite : ¬ Nat.Prime (corner 4071 2) := by change ¬ Nat.Prime 66292165; exact PrattCertificates.composite_of_product (factor := 5) (cofactor := 13258433) rfl (by decide) (by decide)
    exact decide_eq_false composite
  third := false
  third_correct := by
    have composite : ¬ Nat.Prime (corner 4071 3) := by change ¬ Nat.Prime 66284023; exact PrattCertificates.composite_of_product (factor := 13) (cofactor := 5098771) rfl (by decide) (by decide)
    exact decide_eq_false composite
theorem row_count_4071 : row 4071 = 1 := (row_4071.checked).trans rfl
theorem model_count_4071 : modelRow 4071 = 1 := by decide
theorem modelrow_agrees_4071 : modelRow 4071 = row 4071 := model_count_4071.trans row_count_4071.symm
def row_4072 : RowCertificate 4072 where
  first := false
  first_correct := by
    have composite : ¬ Nat.Prime (corner 4072 1) := by change ¬ Nat.Prime 66332881; exact PrattCertificates.composite_of_product (factor := 67) (cofactor := 990043) rfl (by decide) (by decide)
    exact decide_eq_false composite
  second := false
  second_correct := by
    have composite : ¬ Nat.Prime (corner 4072 2) := by change ¬ Nat.Prime 66324737; exact PrattCertificates.composite_of_product (factor := 2129) (cofactor := 31153) rfl (by decide) (by decide)
    exact decide_eq_false composite
  third := false
  third_correct := by
    have composite : ¬ Nat.Prime (corner 4072 3) := by change ¬ Nat.Prime 66316593; exact PrattCertificates.composite_of_product (factor := 3) (cofactor := 22105531) rfl (by decide) (by decide)
    exact decide_eq_false composite
theorem row_count_4072 : row 4072 = 0 := (row_4072.checked).trans rfl
theorem model_count_4072 : modelRow 4072 = 0 := by decide
theorem modelrow_agrees_4072 : modelRow 4072 = row 4072 := model_count_4072.trans row_count_4072.symm
def row_4073 : RowCertificate 4073 where
  first := false
  first_correct := by
    have composite : ¬ Nat.Prime (corner 4073 1) := by change ¬ Nat.Prime 66365463; exact PrattCertificates.composite_of_product (factor := 3) (cofactor := 22121821) rfl (by decide) (by decide)
    exact decide_eq_false composite
  second := false
  second_correct := by
    have composite : ¬ Nat.Prime (corner 4073 2) := by change ¬ Nat.Prime 66357317; exact PrattCertificates.composite_of_product (factor := 13) (cofactor := 5104409) rfl (by decide) (by decide)
    exact decide_eq_false composite
  third := false
  third_correct := by
    have composite : ¬ Nat.Prime (corner 4073 3) := by change ¬ Nat.Prime 66349171; exact PrattCertificates.composite_of_product (factor := 7) (cofactor := 9478453) rfl (by decide) (by decide)
    exact decide_eq_false composite
theorem row_count_4073 : row 4073 = 0 := (row_4073.checked).trans rfl
theorem model_count_4073 : modelRow 4073 = 0 := by decide
theorem modelrow_agrees_4073 : modelRow 4073 = row 4073 := model_count_4073.trans row_count_4073.symm
def row_4074 : RowCertificate 4074 where
  first := true
  first_correct := by
    have prime : Nat.Prime (corner 4074 1) := by change Nat.Prime 66398053; exact ProjectEulerSolutions.P58PrimeNodes.p_66398053.prime
    exact decide_eq_true prime
  second := false
  second_correct := by
    have composite : ¬ Nat.Prime (corner 4074 2) := by change ¬ Nat.Prime 66389905; exact PrattCertificates.composite_of_product (factor := 5) (cofactor := 13277981) rfl (by decide) (by decide)
    exact decide_eq_false composite
  third := false
  third_correct := by
    have composite : ¬ Nat.Prime (corner 4074 3) := by change ¬ Nat.Prime 66381757; exact PrattCertificates.composite_of_product (factor := 13) (cofactor := 5106289) rfl (by decide) (by decide)
    exact decide_eq_false composite
theorem row_count_4074 : row 4074 = 1 := (row_4074.checked).trans rfl
theorem model_count_4074 : modelRow 4074 = 1 := by decide
theorem modelrow_agrees_4074 : modelRow 4074 = row 4074 := model_count_4074.trans row_count_4074.symm
def row_4075 : RowCertificate 4075 where
  first := false
  first_correct := by
    have composite : ¬ Nat.Prime (corner 4075 1) := by change ¬ Nat.Prime 66430651; exact PrattCertificates.composite_of_product (factor := 7) (cofactor := 9490093) rfl (by decide) (by decide)
    exact decide_eq_false composite
  second := false
  second_correct := by
    have composite : ¬ Nat.Prime (corner 4075 2) := by change ¬ Nat.Prime 66422501; exact PrattCertificates.composite_of_product (factor := 41) (cofactor := 1620061) rfl (by decide) (by decide)
    exact decide_eq_false composite
  third := false
  third_correct := by
    have composite : ¬ Nat.Prime (corner 4075 3) := by change ¬ Nat.Prime 66414351; exact PrattCertificates.composite_of_product (factor := 3) (cofactor := 22138117) rfl (by decide) (by decide)
    exact decide_eq_false composite
theorem row_count_4075 : row 4075 = 0 := (row_4075.checked).trans rfl
theorem model_count_4075 : modelRow 4075 = 0 := by decide
theorem modelrow_agrees_4075 : modelRow 4075 = row 4075 := model_count_4075.trans row_count_4075.symm
def row_4076 : RowCertificate 4076 where
  first := false
  first_correct := by
    have composite : ¬ Nat.Prime (corner 4076 1) := by change ¬ Nat.Prime 66463257; exact PrattCertificates.composite_of_product (factor := 3) (cofactor := 22154419) rfl (by decide) (by decide)
    exact decide_eq_false composite
  second := false
  second_correct := by
    have composite : ¬ Nat.Prime (corner 4076 2) := by change ¬ Nat.Prime 66455105; exact PrattCertificates.composite_of_product (factor := 5) (cofactor := 13291021) rfl (by decide) (by decide)
    exact decide_eq_false composite
  third := false
  third_correct := by
    have composite : ¬ Nat.Prime (corner 4076 3) := by change ¬ Nat.Prime 66446953; exact PrattCertificates.composite_of_product (factor := 157) (cofactor := 423229) rfl (by decide) (by decide)
    exact decide_eq_false composite
theorem row_count_4076 : row 4076 = 0 := (row_4076.checked).trans rfl
theorem model_count_4076 : modelRow 4076 = 0 := by decide
theorem modelrow_agrees_4076 : modelRow 4076 = row 4076 := model_count_4076.trans row_count_4076.symm
def row_4077 : RowCertificate 4077 where
  first := false
  first_correct := by
    have composite : ¬ Nat.Prime (corner 4077 1) := by change ¬ Nat.Prime 66495871; exact PrattCertificates.composite_of_product (factor := 13) (cofactor := 5115067) rfl (by decide) (by decide)
    exact decide_eq_false composite
  second := false
  second_correct := by
    have composite : ¬ Nat.Prime (corner 4077 2) := by change ¬ Nat.Prime 66487717; exact PrattCertificates.composite_of_product (factor := 89) (cofactor := 747053) rfl (by decide) (by decide)
    exact decide_eq_false composite
  third := true
  third_correct := by
    have prime : Nat.Prime (corner 4077 3) := by change Nat.Prime 66479563; exact ProjectEulerSolutions.P58PrimeNodes.p_66479563.prime
    exact decide_eq_true prime
theorem row_count_4077 : row 4077 = 1 := (row_4077.checked).trans rfl
theorem model_count_4077 : modelRow 4077 = 1 := by decide
theorem modelrow_agrees_4077 : modelRow 4077 = row 4077 := model_count_4077.trans row_count_4077.symm
def row_4078 : RowCertificate 4078 where
  first := false
  first_correct := by
    have composite : ¬ Nat.Prime (corner 4078 1) := by change ¬ Nat.Prime 66528493; exact PrattCertificates.composite_of_product (factor := 7753) (cofactor := 8581) rfl (by decide) (by decide)
    exact decide_eq_false composite
  second := false
  second_correct := by
    have composite : ¬ Nat.Prime (corner 4078 2) := by change ¬ Nat.Prime 66520337; exact PrattCertificates.composite_of_product (factor := 13) (cofactor := 5116949) rfl (by decide) (by decide)
    exact decide_eq_false composite
  third := false
  third_correct := by
    have composite : ¬ Nat.Prime (corner 4078 3) := by change ¬ Nat.Prime 66512181; exact PrattCertificates.composite_of_product (factor := 3) (cofactor := 22170727) rfl (by decide) (by decide)
    exact decide_eq_false composite
theorem row_count_4078 : row 4078 = 0 := (row_4078.checked).trans rfl
theorem model_count_4078 : modelRow 4078 = 0 := by decide
theorem modelrow_agrees_4078 : modelRow 4078 = row 4078 := model_count_4078.trans row_count_4078.symm
def row_4079 : RowCertificate 4079 where
  first := false
  first_correct := by
    have composite : ¬ Nat.Prime (corner 4079 1) := by change ¬ Nat.Prime 66561123; exact PrattCertificates.composite_of_product (factor := 3) (cofactor := 22187041) rfl (by decide) (by decide)
    exact decide_eq_false composite
  second := false
  second_correct := by
    have composite : ¬ Nat.Prime (corner 4079 2) := by change ¬ Nat.Prime 66552965; exact PrattCertificates.composite_of_product (factor := 5) (cofactor := 13310593) rfl (by decide) (by decide)
    exact decide_eq_false composite
  third := false
  third_correct := by
    have composite : ¬ Nat.Prime (corner 4079 3) := by change ¬ Nat.Prime 66544807; exact PrattCertificates.composite_of_product (factor := 7) (cofactor := 9506401) rfl (by decide) (by decide)
    exact decide_eq_false composite
theorem row_count_4079 : row 4079 = 0 := (row_4079.checked).trans rfl
theorem model_count_4079 : modelRow 4079 = 0 := by decide
theorem modelrow_agrees_4079 : modelRow 4079 = row 4079 := model_count_4079.trans row_count_4079.symm
def row_4080 : RowCertificate 4080 where
  first := false
  first_correct := by
    have composite : ¬ Nat.Prime (corner 4080 1) := by change ¬ Nat.Prime 66593761; exact PrattCertificates.composite_of_product (factor := 13) (cofactor := 5122597) rfl (by decide) (by decide)
    exact decide_eq_false composite
  second := false
  second_correct := by
    have composite : ¬ Nat.Prime (corner 4080 2) := by change ¬ Nat.Prime 66585601; exact PrattCertificates.composite_of_product (factor := 7193) (cofactor := 9257) rfl (by decide) (by decide)
    exact decide_eq_false composite
  third := false
  third_correct := by
    have composite : ¬ Nat.Prime (corner 4080 3) := by change ¬ Nat.Prime 66577441; exact PrattCertificates.composite_of_product (factor := 7) (cofactor := 9511063) rfl (by decide) (by decide)
    exact decide_eq_false composite
theorem row_count_4080 : row 4080 = 0 := (row_4080.checked).trans rfl
theorem model_count_4080 : modelRow 4080 = 0 := by decide
theorem modelrow_agrees_4080 : modelRow 4080 = row 4080 := model_count_4080.trans row_count_4080.symm
def row_4081 : RowCertificate 4081 where
  first := false
  first_correct := by
    have composite : ¬ Nat.Prime (corner 4081 1) := by change ¬ Nat.Prime 66626407; exact PrattCertificates.composite_of_product (factor := 19) (cofactor := 3506653) rfl (by decide) (by decide)
    exact decide_eq_false composite
  second := false
  second_correct := by
    have composite : ¬ Nat.Prime (corner 4081 2) := by change ¬ Nat.Prime 66618245; exact PrattCertificates.composite_of_product (factor := 5) (cofactor := 13323649) rfl (by decide) (by decide)
    exact decide_eq_false composite
  third := false
  third_correct := by
    have composite : ¬ Nat.Prime (corner 4081 3) := by change ¬ Nat.Prime 66610083; exact PrattCertificates.composite_of_product (factor := 3) (cofactor := 22203361) rfl (by decide) (by decide)
    exact decide_eq_false composite
theorem row_count_4081 : row 4081 = 0 := (row_4081.checked).trans rfl
theorem model_count_4081 : modelRow 4081 = 0 := by decide
theorem modelrow_agrees_4081 : modelRow 4081 = row 4081 := model_count_4081.trans row_count_4081.symm
def row_4082 : RowCertificate 4082 where
  first := false
  first_correct := by
    have composite : ¬ Nat.Prime (corner 4082 1) := by change ¬ Nat.Prime 66659061; exact PrattCertificates.composite_of_product (factor := 3) (cofactor := 22219687) rfl (by decide) (by decide)
    exact decide_eq_false composite
  second := false
  second_correct := by
    have composite : ¬ Nat.Prime (corner 4082 2) := by change ¬ Nat.Prime 66650897; exact PrattCertificates.composite_of_product (factor := 17) (cofactor := 3920641) rfl (by decide) (by decide)
    exact decide_eq_false composite
  third := false
  third_correct := by
    have composite : ¬ Nat.Prime (corner 4082 3) := by change ¬ Nat.Prime 66642733; exact PrattCertificates.composite_of_product (factor := 43) (cofactor := 1549831) rfl (by decide) (by decide)
    exact decide_eq_false composite
theorem row_count_4082 : row 4082 = 0 := (row_4082.checked).trans rfl
theorem model_count_4082 : modelRow 4082 = 0 := by decide
theorem modelrow_agrees_4082 : modelRow 4082 = row 4082 := model_count_4082.trans row_count_4082.symm
def row_4083 : RowCertificate 4083 where
  first := false
  first_correct := by
    have composite : ¬ Nat.Prime (corner 4083 1) := by change ¬ Nat.Prime 66691723; exact PrattCertificates.composite_of_product (factor := 7) (cofactor := 9527389) rfl (by decide) (by decide)
    exact decide_eq_false composite
  second := false
  second_correct := by
    have composite : ¬ Nat.Prime (corner 4083 2) := by change ¬ Nat.Prime 66683557; exact PrattCertificates.composite_of_product (factor := 29) (cofactor := 2299433) rfl (by decide) (by decide)
    exact decide_eq_false composite
  third := false
  third_correct := by
    have composite : ¬ Nat.Prime (corner 4083 3) := by change ¬ Nat.Prime 66675391; exact PrattCertificates.composite_of_product (factor := 3319) (cofactor := 20089) rfl (by decide) (by decide)
    exact decide_eq_false composite
theorem row_count_4083 : row 4083 = 0 := (row_4083.checked).trans rfl
theorem model_count_4083 : modelRow 4083 = 0 := by decide
theorem modelrow_agrees_4083 : modelRow 4083 = row 4083 := model_count_4083.trans row_count_4083.symm
def row_4084 : RowCertificate 4084 where
  first := true
  first_correct := by
    have prime : Nat.Prime (corner 4084 1) := by change Nat.Prime 66724393; exact ProjectEulerSolutions.P58PrimeNodes.p_66724393.prime
    exact decide_eq_true prime
  second := false
  second_correct := by
    have composite : ¬ Nat.Prime (corner 4084 2) := by change ¬ Nat.Prime 66716225; exact PrattCertificates.composite_of_product (factor := 5) (cofactor := 13343245) rfl (by decide) (by decide)
    exact decide_eq_false composite
  third := false
  third_correct := by
    have composite : ¬ Nat.Prime (corner 4084 3) := by change ¬ Nat.Prime 66708057; exact PrattCertificates.composite_of_product (factor := 3) (cofactor := 22236019) rfl (by decide) (by decide)
    exact decide_eq_false composite
theorem row_count_4084 : row 4084 = 1 := (row_4084.checked).trans rfl
theorem model_count_4084 : modelRow 4084 = 1 := by decide
theorem modelrow_agrees_4084 : modelRow 4084 = row 4084 := model_count_4084.trans row_count_4084.symm
def row_4085 : RowCertificate 4085 where
  first := false
  first_correct := by
    have composite : ¬ Nat.Prime (corner 4085 1) := by change ¬ Nat.Prime 66757071; exact PrattCertificates.composite_of_product (factor := 3) (cofactor := 22252357) rfl (by decide) (by decide)
    exact decide_eq_false composite
  second := false
  second_correct := by
    have composite : ¬ Nat.Prime (corner 4085 2) := by change ¬ Nat.Prime 66748901; exact PrattCertificates.composite_of_product (factor := 97) (cofactor := 688133) rfl (by decide) (by decide)
    exact decide_eq_false composite
  third := false
  third_correct := by
    have composite : ¬ Nat.Prime (corner 4085 3) := by change ¬ Nat.Prime 66740731; exact PrattCertificates.composite_of_product (factor := 439) (cofactor := 152029) rfl (by decide) (by decide)
    exact decide_eq_false composite
theorem row_count_4085 : row 4085 = 0 := (row_4085.checked).trans rfl
theorem model_count_4085 : modelRow 4085 = 0 := by decide
theorem modelrow_agrees_4085 : modelRow 4085 = row 4085 := model_count_4085.trans row_count_4085.symm
def row_4086 : RowCertificate 4086 where
  first := false
  first_correct := by
    have composite : ¬ Nat.Prime (corner 4086 1) := by change ¬ Nat.Prime 66789757; exact PrattCertificates.composite_of_product (factor := 3739) (cofactor := 17863) rfl (by decide) (by decide)
    exact decide_eq_false composite
  second := false
  second_correct := by
    have composite : ¬ Nat.Prime (corner 4086 2) := by change ¬ Nat.Prime 66781585; exact PrattCertificates.composite_of_product (factor := 5) (cofactor := 13356317) rfl (by decide) (by decide)
    exact decide_eq_false composite
  third := false
  third_correct := by
    have composite : ¬ Nat.Prime (corner 4086 3) := by change ¬ Nat.Prime 66773413; exact PrattCertificates.composite_of_product (factor := 7) (cofactor := 9539059) rfl (by decide) (by decide)
    exact decide_eq_false composite
theorem row_count_4086 : row 4086 = 0 := (row_4086.checked).trans rfl
theorem model_count_4086 : modelRow 4086 = 0 := by decide
theorem modelrow_agrees_4086 : modelRow 4086 = row 4086 := model_count_4086.trans row_count_4086.symm
def row_4087 : RowCertificate 4087 where
  first := true
  first_correct := by
    have prime : Nat.Prime (corner 4087 1) := by change Nat.Prime 66822451; exact ProjectEulerSolutions.P58PrimeNodes.p_66822451.prime
    exact decide_eq_true prime
  second := true
  second_correct := by
    have prime : Nat.Prime (corner 4087 2) := by change Nat.Prime 66814277; exact ProjectEulerSolutions.P58PrimeNodes.p_66814277.prime
    exact decide_eq_true prime
  third := false
  third_correct := by
    have composite : ¬ Nat.Prime (corner 4087 3) := by change ¬ Nat.Prime 66806103; exact PrattCertificates.composite_of_product (factor := 3) (cofactor := 22268701) rfl (by decide) (by decide)
    exact decide_eq_false composite
theorem row_count_4087 : row 4087 = 2 := (row_4087.checked).trans rfl
theorem model_count_4087 : modelRow 4087 = 2 := by decide
theorem modelrow_agrees_4087 : modelRow 4087 = row 4087 := model_count_4087.trans row_count_4087.symm
def row_4088 : RowCertificate 4088 where
  first := false
  first_correct := by
    have composite : ¬ Nat.Prime (corner 4088 1) := by change ¬ Nat.Prime 66855153; exact PrattCertificates.composite_of_product (factor := 3) (cofactor := 22285051) rfl (by decide) (by decide)
    exact decide_eq_false composite
  second := true
  second_correct := by
    have prime : Nat.Prime (corner 4088 2) := by change Nat.Prime 66846977; exact ProjectEulerSolutions.P58PrimeNodes.p_66846977.prime
    exact decide_eq_true prime
  third := false
  third_correct := by
    have composite : ¬ Nat.Prime (corner 4088 3) := by change ¬ Nat.Prime 66838801; exact PrattCertificates.composite_of_product (factor := 877) (cofactor := 76213) rfl (by decide) (by decide)
    exact decide_eq_false composite
theorem row_count_4088 : row 4088 = 1 := (row_4088.checked).trans rfl
theorem model_count_4088 : modelRow 4088 = 1 := by decide
theorem modelrow_agrees_4088 : modelRow 4088 = row 4088 := model_count_4088.trans row_count_4088.symm
def row_4089 : RowCertificate 4089 where
  first := false
  first_correct := by
    have composite : ¬ Nat.Prime (corner 4089 1) := by change ¬ Nat.Prime 66887863; exact PrattCertificates.composite_of_product (factor := 7) (cofactor := 9555409) rfl (by decide) (by decide)
    exact decide_eq_false composite
  second := false
  second_correct := by
    have composite : ¬ Nat.Prime (corner 4089 2) := by change ¬ Nat.Prime 66879685; exact PrattCertificates.composite_of_product (factor := 5) (cofactor := 13375937) rfl (by decide) (by decide)
    exact decide_eq_false composite
  third := false
  third_correct := by
    have composite : ¬ Nat.Prime (corner 4089 3) := by change ¬ Nat.Prime 66871507; exact PrattCertificates.composite_of_product (factor := 19) (cofactor := 3519553) rfl (by decide) (by decide)
    exact decide_eq_false composite
theorem row_count_4089 : row 4089 = 0 := (row_4089.checked).trans rfl
theorem model_count_4089 : modelRow 4089 = 0 := by decide
theorem modelrow_agrees_4089 : modelRow 4089 = row 4089 := model_count_4089.trans row_count_4089.symm
def row_4090 : RowCertificate 4090 where
  first := false
  first_correct := by
    have composite : ¬ Nat.Prime (corner 4090 1) := by change ¬ Nat.Prime 66920581; exact PrattCertificates.composite_of_product (factor := 7) (cofactor := 9560083) rfl (by decide) (by decide)
    exact decide_eq_false composite
  second := true
  second_correct := by
    have prime : Nat.Prime (corner 4090 2) := by change Nat.Prime 66912401; exact ProjectEulerSolutions.P58PrimeNodes.p_66912401.prime
    exact decide_eq_true prime
  third := false
  third_correct := by
    have composite : ¬ Nat.Prime (corner 4090 3) := by change ¬ Nat.Prime 66904221; exact PrattCertificates.composite_of_product (factor := 3) (cofactor := 22301407) rfl (by decide) (by decide)
    exact decide_eq_false composite
theorem row_count_4090 : row 4090 = 1 := (row_4090.checked).trans rfl
theorem model_count_4090 : modelRow 4090 = 1 := by decide
theorem modelrow_agrees_4090 : modelRow 4090 = row 4090 := model_count_4090.trans row_count_4090.symm
def row_4091 : RowCertificate 4091 where
  first := false
  first_correct := by
    have composite : ¬ Nat.Prime (corner 4091 1) := by change ¬ Nat.Prime 66953307; exact PrattCertificates.composite_of_product (factor := 3) (cofactor := 22317769) rfl (by decide) (by decide)
    exact decide_eq_false composite
  second := false
  second_correct := by
    have composite : ¬ Nat.Prime (corner 4091 2) := by change ¬ Nat.Prime 66945125; exact PrattCertificates.composite_of_product (factor := 5) (cofactor := 13389025) rfl (by decide) (by decide)
    exact decide_eq_false composite
  third := false
  third_correct := by
    have composite : ¬ Nat.Prime (corner 4091 3) := by change ¬ Nat.Prime 66936943; exact PrattCertificates.composite_of_product (factor := 19) (cofactor := 3522997) rfl (by decide) (by decide)
    exact decide_eq_false composite
theorem row_count_4091 : row 4091 = 0 := (row_4091.checked).trans rfl
theorem model_count_4091 : modelRow 4091 = 0 := by decide
theorem modelrow_agrees_4091 : modelRow 4091 = row 4091 := model_count_4091.trans row_count_4091.symm
def row_4092 : RowCertificate 4092 where
  first := false
  first_correct := by
    have composite : ¬ Nat.Prime (corner 4092 1) := by change ¬ Nat.Prime 66986041; exact PrattCertificates.composite_of_product (factor := 73) (cofactor := 917617) rfl (by decide) (by decide)
    exact decide_eq_false composite
  second := true
  second_correct := by
    have prime : Nat.Prime (corner 4092 2) := by change Nat.Prime 66977857; exact ProjectEulerSolutions.P58PrimeNodes.p_66977857.prime
    exact decide_eq_true prime
  third := false
  third_correct := by
    have composite : ¬ Nat.Prime (corner 4092 3) := by change ¬ Nat.Prime 66969673; exact PrattCertificates.composite_of_product (factor := 97) (cofactor := 690409) rfl (by decide) (by decide)
    exact decide_eq_false composite
theorem row_count_4092 : row 4092 = 1 := (row_4092.checked).trans rfl
theorem model_count_4092 : modelRow 4092 = 1 := by decide
theorem modelrow_agrees_4092 : modelRow 4092 = row 4092 := model_count_4092.trans row_count_4092.symm
def row_4093 : RowCertificate 4093 where
  first := false
  first_correct := by
    have composite : ¬ Nat.Prime (corner 4093 1) := by change ¬ Nat.Prime 67018783; exact PrattCertificates.composite_of_product (factor := 13) (cofactor := 5155291) rfl (by decide) (by decide)
    exact decide_eq_false composite
  second := false
  second_correct := by
    have composite : ¬ Nat.Prime (corner 4093 2) := by change ¬ Nat.Prime 67010597; exact PrattCertificates.composite_of_product (factor := 773) (cofactor := 86689) rfl (by decide) (by decide)
    exact decide_eq_false composite
  third := false
  third_correct := by
    have composite : ¬ Nat.Prime (corner 4093 3) := by change ¬ Nat.Prime 67002411; exact PrattCertificates.composite_of_product (factor := 3) (cofactor := 22334137) rfl (by decide) (by decide)
    exact decide_eq_false composite
theorem row_count_4093 : row 4093 = 0 := (row_4093.checked).trans rfl
theorem model_count_4093 : modelRow 4093 = 0 := by decide
theorem modelrow_agrees_4093 : modelRow 4093 = row 4093 := model_count_4093.trans row_count_4093.symm
def row_4094 : RowCertificate 4094 where
  first := false
  first_correct := by
    have composite : ¬ Nat.Prime (corner 4094 1) := by change ¬ Nat.Prime 67051533; exact PrattCertificates.composite_of_product (factor := 3) (cofactor := 22350511) rfl (by decide) (by decide)
    exact decide_eq_false composite
  second := false
  second_correct := by
    have composite : ¬ Nat.Prime (corner 4094 2) := by change ¬ Nat.Prime 67043345; exact PrattCertificates.composite_of_product (factor := 5) (cofactor := 13408669) rfl (by decide) (by decide)
    exact decide_eq_false composite
  third := false
  third_correct := by
    have composite : ¬ Nat.Prime (corner 4094 3) := by change ¬ Nat.Prime 67035157; exact PrattCertificates.composite_of_product (factor := 7) (cofactor := 9576451) rfl (by decide) (by decide)
    exact decide_eq_false composite
theorem row_count_4094 : row 4094 = 0 := (row_4094.checked).trans rfl
theorem model_count_4094 : modelRow 4094 = 0 := by decide
theorem modelrow_agrees_4094 : modelRow 4094 = row 4094 := model_count_4094.trans row_count_4094.symm
def row_4095 : RowCertificate 4095 where
  first := false
  first_correct := by
    have composite : ¬ Nat.Prime (corner 4095 1) := by change ¬ Nat.Prime 67084291; exact PrattCertificates.composite_of_product (factor := 193) (cofactor := 347587) rfl (by decide) (by decide)
    exact decide_eq_false composite
  second := false
  second_correct := by
    have composite : ¬ Nat.Prime (corner 4095 2) := by change ¬ Nat.Prime 67076101; exact PrattCertificates.composite_of_product (factor := 17) (cofactor := 3945653) rfl (by decide) (by decide)
    exact decide_eq_false composite
  third := false
  third_correct := by
    have composite : ¬ Nat.Prime (corner 4095 3) := by change ¬ Nat.Prime 67067911; exact PrattCertificates.composite_of_product (factor := 31) (cofactor := 2163481) rfl (by decide) (by decide)
    exact decide_eq_false composite
theorem row_count_4095 : row 4095 = 0 := (row_4095.checked).trans rfl
theorem model_count_4095 : modelRow 4095 = 0 := by decide
theorem modelrow_agrees_4095 : modelRow 4095 = row 4095 := model_count_4095.trans row_count_4095.symm
def row_4096 : RowCertificate 4096 where
  first := false
  first_correct := by
    have composite : ¬ Nat.Prime (corner 4096 1) := by change ¬ Nat.Prime 67117057; exact PrattCertificates.composite_of_product (factor := 7) (cofactor := 9588151) rfl (by decide) (by decide)
    exact decide_eq_false composite
  second := false
  second_correct := by
    have composite : ¬ Nat.Prime (corner 4096 2) := by change ¬ Nat.Prime 67108865; exact PrattCertificates.composite_of_product (factor := 5) (cofactor := 13421773) rfl (by decide) (by decide)
    exact decide_eq_false composite
  third := false
  third_correct := by
    have composite : ¬ Nat.Prime (corner 4096 3) := by change ¬ Nat.Prime 67100673; exact PrattCertificates.composite_of_product (factor := 3) (cofactor := 22366891) rfl (by decide) (by decide)
    exact decide_eq_false composite
theorem row_count_4096 : row 4096 = 0 := (row_4096.checked).trans rfl
theorem model_count_4096 : modelRow 4096 = 0 := by decide
theorem modelrow_agrees_4096 : modelRow 4096 = row 4096 := model_count_4096.trans row_count_4096.symm
private def atomic_3969 : CountSegment 1 10 13120 3968 3969 1836 1837 := CountSegment.single (first := 3968) (initial := 1836) row_count_3969 (by decide)
private theorem atomic_3969_agreement : RowAgreement 3968 3969 := RowAgreement.single (first := 3968) modelrow_agrees_3969
private def atomic_3970 : CountSegment 1 10 13120 3969 3970 1837 1837 := CountSegment.single (first := 3969) (initial := 1837) row_count_3970 (by decide)
private theorem atomic_3970_agreement : RowAgreement 3969 3970 := RowAgreement.single (first := 3969) modelrow_agrees_3970
private def atomic_3971 : CountSegment 1 10 13120 3970 3971 1837 1837 := CountSegment.single (first := 3970) (initial := 1837) row_count_3971 (by decide)
private theorem atomic_3971_agreement : RowAgreement 3970 3971 := RowAgreement.single (first := 3970) modelrow_agrees_3971
private def atomic_3972 : CountSegment 1 10 13120 3971 3972 1837 1838 := CountSegment.single (first := 3971) (initial := 1837) row_count_3972 (by decide)
private theorem atomic_3972_agreement : RowAgreement 3971 3972 := RowAgreement.single (first := 3971) modelrow_agrees_3972
private def atomic_3973 : CountSegment 1 10 13120 3972 3973 1838 1839 := CountSegment.single (first := 3972) (initial := 1838) row_count_3973 (by decide)
private theorem atomic_3973_agreement : RowAgreement 3972 3973 := RowAgreement.single (first := 3972) modelrow_agrees_3973
private def atomic_3974 : CountSegment 1 10 13120 3973 3974 1839 1839 := CountSegment.single (first := 3973) (initial := 1839) row_count_3974 (by decide)
private theorem atomic_3974_agreement : RowAgreement 3973 3974 := RowAgreement.single (first := 3973) modelrow_agrees_3974
private def atomic_3975 : CountSegment 1 10 13120 3974 3975 1839 1840 := CountSegment.single (first := 3974) (initial := 1839) row_count_3975 (by decide)
private theorem atomic_3975_agreement : RowAgreement 3974 3975 := RowAgreement.single (first := 3974) modelrow_agrees_3975
private def atomic_3976 : CountSegment 1 10 13120 3975 3976 1840 1840 := CountSegment.single (first := 3975) (initial := 1840) row_count_3976 (by decide)
private theorem atomic_3976_agreement : RowAgreement 3975 3976 := RowAgreement.single (first := 3975) modelrow_agrees_3976
private def atomic_3977 : CountSegment 1 10 13120 3976 3977 1840 1840 := CountSegment.single (first := 3976) (initial := 1840) row_count_3977 (by decide)
private theorem atomic_3977_agreement : RowAgreement 3976 3977 := RowAgreement.single (first := 3976) modelrow_agrees_3977
private def atomic_3978 : CountSegment 1 10 13120 3977 3978 1840 1841 := CountSegment.single (first := 3977) (initial := 1840) row_count_3978 (by decide)
private theorem atomic_3978_agreement : RowAgreement 3977 3978 := RowAgreement.single (first := 3977) modelrow_agrees_3978
private def atomic_3979 : CountSegment 1 10 13120 3978 3979 1841 1841 := CountSegment.single (first := 3978) (initial := 1841) row_count_3979 (by decide)
private theorem atomic_3979_agreement : RowAgreement 3978 3979 := RowAgreement.single (first := 3978) modelrow_agrees_3979
private def atomic_3980 : CountSegment 1 10 13120 3979 3980 1841 1841 := CountSegment.single (first := 3979) (initial := 1841) row_count_3980 (by decide)
private theorem atomic_3980_agreement : RowAgreement 3979 3980 := RowAgreement.single (first := 3979) modelrow_agrees_3980
private def atomic_3981 : CountSegment 1 10 13120 3980 3981 1841 1841 := CountSegment.single (first := 3980) (initial := 1841) row_count_3981 (by decide)
private theorem atomic_3981_agreement : RowAgreement 3980 3981 := RowAgreement.single (first := 3980) modelrow_agrees_3981
private def atomic_3982 : CountSegment 1 10 13120 3981 3982 1841 1841 := CountSegment.single (first := 3981) (initial := 1841) row_count_3982 (by decide)
private theorem atomic_3982_agreement : RowAgreement 3981 3982 := RowAgreement.single (first := 3981) modelrow_agrees_3982
private def atomic_3983 : CountSegment 1 10 13120 3982 3983 1841 1841 := CountSegment.single (first := 3982) (initial := 1841) row_count_3983 (by decide)
private theorem atomic_3983_agreement : RowAgreement 3982 3983 := RowAgreement.single (first := 3982) modelrow_agrees_3983
private def atomic_3984 : CountSegment 1 10 13120 3983 3984 1841 1842 := CountSegment.single (first := 3983) (initial := 1841) row_count_3984 (by decide)
private theorem atomic_3984_agreement : RowAgreement 3983 3984 := RowAgreement.single (first := 3983) modelrow_agrees_3984
private def atomic_3985 : CountSegment 1 10 13120 3984 3985 1842 1842 := CountSegment.single (first := 3984) (initial := 1842) row_count_3985 (by decide)
private theorem atomic_3985_agreement : RowAgreement 3984 3985 := RowAgreement.single (first := 3984) modelrow_agrees_3985
private def atomic_3986 : CountSegment 1 10 13120 3985 3986 1842 1843 := CountSegment.single (first := 3985) (initial := 1842) row_count_3986 (by decide)
private theorem atomic_3986_agreement : RowAgreement 3985 3986 := RowAgreement.single (first := 3985) modelrow_agrees_3986
private def atomic_3987 : CountSegment 1 10 13120 3986 3987 1843 1844 := CountSegment.single (first := 3986) (initial := 1843) row_count_3987 (by decide)
private theorem atomic_3987_agreement : RowAgreement 3986 3987 := RowAgreement.single (first := 3986) modelrow_agrees_3987
private def atomic_3988 : CountSegment 1 10 13120 3987 3988 1844 1845 := CountSegment.single (first := 3987) (initial := 1844) row_count_3988 (by decide)
private theorem atomic_3988_agreement : RowAgreement 3987 3988 := RowAgreement.single (first := 3987) modelrow_agrees_3988
private def atomic_3989 : CountSegment 1 10 13120 3988 3989 1845 1845 := CountSegment.single (first := 3988) (initial := 1845) row_count_3989 (by decide)
private theorem atomic_3989_agreement : RowAgreement 3988 3989 := RowAgreement.single (first := 3988) modelrow_agrees_3989
private def atomic_3990 : CountSegment 1 10 13120 3989 3990 1845 1845 := CountSegment.single (first := 3989) (initial := 1845) row_count_3990 (by decide)
private theorem atomic_3990_agreement : RowAgreement 3989 3990 := RowAgreement.single (first := 3989) modelrow_agrees_3990
private def atomic_3991 : CountSegment 1 10 13120 3990 3991 1845 1845 := CountSegment.single (first := 3990) (initial := 1845) row_count_3991 (by decide)
private theorem atomic_3991_agreement : RowAgreement 3990 3991 := RowAgreement.single (first := 3990) modelrow_agrees_3991
private def atomic_3992 : CountSegment 1 10 13120 3991 3992 1845 1846 := CountSegment.single (first := 3991) (initial := 1845) row_count_3992 (by decide)
private theorem atomic_3992_agreement : RowAgreement 3991 3992 := RowAgreement.single (first := 3991) modelrow_agrees_3992
private def atomic_3993 : CountSegment 1 10 13120 3992 3993 1846 1847 := CountSegment.single (first := 3992) (initial := 1846) row_count_3993 (by decide)
private theorem atomic_3993_agreement : RowAgreement 3992 3993 := RowAgreement.single (first := 3992) modelrow_agrees_3993
private def atomic_3994 : CountSegment 1 10 13120 3993 3994 1847 1847 := CountSegment.single (first := 3993) (initial := 1847) row_count_3994 (by decide)
private theorem atomic_3994_agreement : RowAgreement 3993 3994 := RowAgreement.single (first := 3993) modelrow_agrees_3994
private def atomic_3995 : CountSegment 1 10 13120 3994 3995 1847 1847 := CountSegment.single (first := 3994) (initial := 1847) row_count_3995 (by decide)
private theorem atomic_3995_agreement : RowAgreement 3994 3995 := RowAgreement.single (first := 3994) modelrow_agrees_3995
private def atomic_3996 : CountSegment 1 10 13120 3995 3996 1847 1847 := CountSegment.single (first := 3995) (initial := 1847) row_count_3996 (by decide)
private theorem atomic_3996_agreement : RowAgreement 3995 3996 := RowAgreement.single (first := 3995) modelrow_agrees_3996
private def atomic_3997 : CountSegment 1 10 13120 3996 3997 1847 1847 := CountSegment.single (first := 3996) (initial := 1847) row_count_3997 (by decide)
private theorem atomic_3997_agreement : RowAgreement 3996 3997 := RowAgreement.single (first := 3996) modelrow_agrees_3997
private def atomic_3998 : CountSegment 1 10 13120 3997 3998 1847 1847 := CountSegment.single (first := 3997) (initial := 1847) row_count_3998 (by decide)
private theorem atomic_3998_agreement : RowAgreement 3997 3998 := RowAgreement.single (first := 3997) modelrow_agrees_3998
private def atomic_3999 : CountSegment 1 10 13120 3998 3999 1847 1847 := CountSegment.single (first := 3998) (initial := 1847) row_count_3999 (by decide)
private theorem atomic_3999_agreement : RowAgreement 3998 3999 := RowAgreement.single (first := 3998) modelrow_agrees_3999
private def atomic_4000 : CountSegment 1 10 13120 3999 4000 1847 1848 := CountSegment.single (first := 3999) (initial := 1847) row_count_4000 (by decide)
private theorem atomic_4000_agreement : RowAgreement 3999 4000 := RowAgreement.single (first := 3999) modelrow_agrees_4000
private def atomic_4001 : CountSegment 1 10 13120 4000 4001 1848 1848 := CountSegment.single (first := 4000) (initial := 1848) row_count_4001 (by decide)
private theorem atomic_4001_agreement : RowAgreement 4000 4001 := RowAgreement.single (first := 4000) modelrow_agrees_4001
private def atomic_4002 : CountSegment 1 10 13120 4001 4002 1848 1848 := CountSegment.single (first := 4001) (initial := 1848) row_count_4002 (by decide)
private theorem atomic_4002_agreement : RowAgreement 4001 4002 := RowAgreement.single (first := 4001) modelrow_agrees_4002
private def atomic_4003 : CountSegment 1 10 13120 4002 4003 1848 1848 := CountSegment.single (first := 4002) (initial := 1848) row_count_4003 (by decide)
private theorem atomic_4003_agreement : RowAgreement 4002 4003 := RowAgreement.single (first := 4002) modelrow_agrees_4003
private def atomic_4004 : CountSegment 1 10 13120 4003 4004 1848 1848 := CountSegment.single (first := 4003) (initial := 1848) row_count_4004 (by decide)
private theorem atomic_4004_agreement : RowAgreement 4003 4004 := RowAgreement.single (first := 4003) modelrow_agrees_4004
private def atomic_4005 : CountSegment 1 10 13120 4004 4005 1848 1848 := CountSegment.single (first := 4004) (initial := 1848) row_count_4005 (by decide)
private theorem atomic_4005_agreement : RowAgreement 4004 4005 := RowAgreement.single (first := 4004) modelrow_agrees_4005
private def atomic_4006 : CountSegment 1 10 13120 4005 4006 1848 1848 := CountSegment.single (first := 4005) (initial := 1848) row_count_4006 (by decide)
private theorem atomic_4006_agreement : RowAgreement 4005 4006 := RowAgreement.single (first := 4005) modelrow_agrees_4006
private def atomic_4007 : CountSegment 1 10 13120 4006 4007 1848 1849 := CountSegment.single (first := 4006) (initial := 1848) row_count_4007 (by decide)
private theorem atomic_4007_agreement : RowAgreement 4006 4007 := RowAgreement.single (first := 4006) modelrow_agrees_4007
private def atomic_4008 : CountSegment 1 10 13120 4007 4008 1849 1850 := CountSegment.single (first := 4007) (initial := 1849) row_count_4008 (by decide)
private theorem atomic_4008_agreement : RowAgreement 4007 4008 := RowAgreement.single (first := 4007) modelrow_agrees_4008
private def atomic_4009 : CountSegment 1 10 13120 4008 4009 1850 1850 := CountSegment.single (first := 4008) (initial := 1850) row_count_4009 (by decide)
private theorem atomic_4009_agreement : RowAgreement 4008 4009 := RowAgreement.single (first := 4008) modelrow_agrees_4009
private def atomic_4010 : CountSegment 1 10 13120 4009 4010 1850 1850 := CountSegment.single (first := 4009) (initial := 1850) row_count_4010 (by decide)
private theorem atomic_4010_agreement : RowAgreement 4009 4010 := RowAgreement.single (first := 4009) modelrow_agrees_4010
private def atomic_4011 : CountSegment 1 10 13120 4010 4011 1850 1850 := CountSegment.single (first := 4010) (initial := 1850) row_count_4011 (by decide)
private theorem atomic_4011_agreement : RowAgreement 4010 4011 := RowAgreement.single (first := 4010) modelrow_agrees_4011
private def atomic_4012 : CountSegment 1 10 13120 4011 4012 1850 1850 := CountSegment.single (first := 4011) (initial := 1850) row_count_4012 (by decide)
private theorem atomic_4012_agreement : RowAgreement 4011 4012 := RowAgreement.single (first := 4011) modelrow_agrees_4012
private def atomic_4013 : CountSegment 1 10 13120 4012 4013 1850 1850 := CountSegment.single (first := 4012) (initial := 1850) row_count_4013 (by decide)
private theorem atomic_4013_agreement : RowAgreement 4012 4013 := RowAgreement.single (first := 4012) modelrow_agrees_4013
private def atomic_4014 : CountSegment 1 10 13120 4013 4014 1850 1851 := CountSegment.single (first := 4013) (initial := 1850) row_count_4014 (by decide)
private theorem atomic_4014_agreement : RowAgreement 4013 4014 := RowAgreement.single (first := 4013) modelrow_agrees_4014
private def atomic_4015 : CountSegment 1 10 13120 4014 4015 1851 1852 := CountSegment.single (first := 4014) (initial := 1851) row_count_4015 (by decide)
private theorem atomic_4015_agreement : RowAgreement 4014 4015 := RowAgreement.single (first := 4014) modelrow_agrees_4015
private def atomic_4016 : CountSegment 1 10 13120 4015 4016 1852 1852 := CountSegment.single (first := 4015) (initial := 1852) row_count_4016 (by decide)
private theorem atomic_4016_agreement : RowAgreement 4015 4016 := RowAgreement.single (first := 4015) modelrow_agrees_4016
private def atomic_4017 : CountSegment 1 10 13120 4016 4017 1852 1853 := CountSegment.single (first := 4016) (initial := 1852) row_count_4017 (by decide)
private theorem atomic_4017_agreement : RowAgreement 4016 4017 := RowAgreement.single (first := 4016) modelrow_agrees_4017
private def atomic_4018 : CountSegment 1 10 13120 4017 4018 1853 1854 := CountSegment.single (first := 4017) (initial := 1853) row_count_4018 (by decide)
private theorem atomic_4018_agreement : RowAgreement 4017 4018 := RowAgreement.single (first := 4017) modelrow_agrees_4018
private def atomic_4019 : CountSegment 1 10 13120 4018 4019 1854 1854 := CountSegment.single (first := 4018) (initial := 1854) row_count_4019 (by decide)
private theorem atomic_4019_agreement : RowAgreement 4018 4019 := RowAgreement.single (first := 4018) modelrow_agrees_4019
private def atomic_4020 : CountSegment 1 10 13120 4019 4020 1854 1854 := CountSegment.single (first := 4019) (initial := 1854) row_count_4020 (by decide)
private theorem atomic_4020_agreement : RowAgreement 4019 4020 := RowAgreement.single (first := 4019) modelrow_agrees_4020
private def atomic_4021 : CountSegment 1 10 13120 4020 4021 1854 1855 := CountSegment.single (first := 4020) (initial := 1854) row_count_4021 (by decide)
private theorem atomic_4021_agreement : RowAgreement 4020 4021 := RowAgreement.single (first := 4020) modelrow_agrees_4021
private def atomic_4022 : CountSegment 1 10 13120 4021 4022 1855 1855 := CountSegment.single (first := 4021) (initial := 1855) row_count_4022 (by decide)
private theorem atomic_4022_agreement : RowAgreement 4021 4022 := RowAgreement.single (first := 4021) modelrow_agrees_4022
private def atomic_4023 : CountSegment 1 10 13120 4022 4023 1855 1855 := CountSegment.single (first := 4022) (initial := 1855) row_count_4023 (by decide)
private theorem atomic_4023_agreement : RowAgreement 4022 4023 := RowAgreement.single (first := 4022) modelrow_agrees_4023
private def atomic_4024 : CountSegment 1 10 13120 4023 4024 1855 1855 := CountSegment.single (first := 4023) (initial := 1855) row_count_4024 (by decide)
private theorem atomic_4024_agreement : RowAgreement 4023 4024 := RowAgreement.single (first := 4023) modelrow_agrees_4024
private def atomic_4025 : CountSegment 1 10 13120 4024 4025 1855 1856 := CountSegment.single (first := 4024) (initial := 1855) row_count_4025 (by decide)
private theorem atomic_4025_agreement : RowAgreement 4024 4025 := RowAgreement.single (first := 4024) modelrow_agrees_4025
private def atomic_4026 : CountSegment 1 10 13120 4025 4026 1856 1857 := CountSegment.single (first := 4025) (initial := 1856) row_count_4026 (by decide)
private theorem atomic_4026_agreement : RowAgreement 4025 4026 := RowAgreement.single (first := 4025) modelrow_agrees_4026
private def atomic_4027 : CountSegment 1 10 13120 4026 4027 1857 1857 := CountSegment.single (first := 4026) (initial := 1857) row_count_4027 (by decide)
private theorem atomic_4027_agreement : RowAgreement 4026 4027 := RowAgreement.single (first := 4026) modelrow_agrees_4027
private def atomic_4028 : CountSegment 1 10 13120 4027 4028 1857 1857 := CountSegment.single (first := 4027) (initial := 1857) row_count_4028 (by decide)
private theorem atomic_4028_agreement : RowAgreement 4027 4028 := RowAgreement.single (first := 4027) modelrow_agrees_4028
private def atomic_4029 : CountSegment 1 10 13120 4028 4029 1857 1858 := CountSegment.single (first := 4028) (initial := 1857) row_count_4029 (by decide)
private theorem atomic_4029_agreement : RowAgreement 4028 4029 := RowAgreement.single (first := 4028) modelrow_agrees_4029
private def atomic_4030 : CountSegment 1 10 13120 4029 4030 1858 1859 := CountSegment.single (first := 4029) (initial := 1858) row_count_4030 (by decide)
private theorem atomic_4030_agreement : RowAgreement 4029 4030 := RowAgreement.single (first := 4029) modelrow_agrees_4030
private def atomic_4031 : CountSegment 1 10 13120 4030 4031 1859 1859 := CountSegment.single (first := 4030) (initial := 1859) row_count_4031 (by decide)
private theorem atomic_4031_agreement : RowAgreement 4030 4031 := RowAgreement.single (first := 4030) modelrow_agrees_4031
private def atomic_4032 : CountSegment 1 10 13120 4031 4032 1859 1861 := CountSegment.single (first := 4031) (initial := 1859) row_count_4032 (by decide)
private theorem atomic_4032_agreement : RowAgreement 4031 4032 := RowAgreement.single (first := 4031) modelrow_agrees_4032
private def atomic_4033 : CountSegment 1 10 13120 4032 4033 1861 1861 := CountSegment.single (first := 4032) (initial := 1861) row_count_4033 (by decide)
private theorem atomic_4033_agreement : RowAgreement 4032 4033 := RowAgreement.single (first := 4032) modelrow_agrees_4033
private def atomic_4034 : CountSegment 1 10 13120 4033 4034 1861 1861 := CountSegment.single (first := 4033) (initial := 1861) row_count_4034 (by decide)
private theorem atomic_4034_agreement : RowAgreement 4033 4034 := RowAgreement.single (first := 4033) modelrow_agrees_4034
private def atomic_4035 : CountSegment 1 10 13120 4034 4035 1861 1862 := CountSegment.single (first := 4034) (initial := 1861) row_count_4035 (by decide)
private theorem atomic_4035_agreement : RowAgreement 4034 4035 := RowAgreement.single (first := 4034) modelrow_agrees_4035
private def atomic_4036 : CountSegment 1 10 13120 4035 4036 1862 1863 := CountSegment.single (first := 4035) (initial := 1862) row_count_4036 (by decide)
private theorem atomic_4036_agreement : RowAgreement 4035 4036 := RowAgreement.single (first := 4035) modelrow_agrees_4036
private def atomic_4037 : CountSegment 1 10 13120 4036 4037 1863 1863 := CountSegment.single (first := 4036) (initial := 1863) row_count_4037 (by decide)
private theorem atomic_4037_agreement : RowAgreement 4036 4037 := RowAgreement.single (first := 4036) modelrow_agrees_4037
private def atomic_4038 : CountSegment 1 10 13120 4037 4038 1863 1863 := CountSegment.single (first := 4037) (initial := 1863) row_count_4038 (by decide)
private theorem atomic_4038_agreement : RowAgreement 4037 4038 := RowAgreement.single (first := 4037) modelrow_agrees_4038
private def atomic_4039 : CountSegment 1 10 13120 4038 4039 1863 1864 := CountSegment.single (first := 4038) (initial := 1863) row_count_4039 (by decide)
private theorem atomic_4039_agreement : RowAgreement 4038 4039 := RowAgreement.single (first := 4038) modelrow_agrees_4039
private def atomic_4040 : CountSegment 1 10 13120 4039 4040 1864 1866 := CountSegment.single (first := 4039) (initial := 1864) row_count_4040 (by decide)
private theorem atomic_4040_agreement : RowAgreement 4039 4040 := RowAgreement.single (first := 4039) modelrow_agrees_4040
private def atomic_4041 : CountSegment 1 10 13120 4040 4041 1866 1866 := CountSegment.single (first := 4040) (initial := 1866) row_count_4041 (by decide)
private theorem atomic_4041_agreement : RowAgreement 4040 4041 := RowAgreement.single (first := 4040) modelrow_agrees_4041
private def atomic_4042 : CountSegment 1 10 13120 4041 4042 1866 1867 := CountSegment.single (first := 4041) (initial := 1866) row_count_4042 (by decide)
private theorem atomic_4042_agreement : RowAgreement 4041 4042 := RowAgreement.single (first := 4041) modelrow_agrees_4042
private def atomic_4043 : CountSegment 1 10 13120 4042 4043 1867 1867 := CountSegment.single (first := 4042) (initial := 1867) row_count_4043 (by decide)
private theorem atomic_4043_agreement : RowAgreement 4042 4043 := RowAgreement.single (first := 4042) modelrow_agrees_4043
private def atomic_4044 : CountSegment 1 10 13120 4043 4044 1867 1868 := CountSegment.single (first := 4043) (initial := 1867) row_count_4044 (by decide)
private theorem atomic_4044_agreement : RowAgreement 4043 4044 := RowAgreement.single (first := 4043) modelrow_agrees_4044
private def atomic_4045 : CountSegment 1 10 13120 4044 4045 1868 1868 := CountSegment.single (first := 4044) (initial := 1868) row_count_4045 (by decide)
private theorem atomic_4045_agreement : RowAgreement 4044 4045 := RowAgreement.single (first := 4044) modelrow_agrees_4045
private def atomic_4046 : CountSegment 1 10 13120 4045 4046 1868 1868 := CountSegment.single (first := 4045) (initial := 1868) row_count_4046 (by decide)
private theorem atomic_4046_agreement : RowAgreement 4045 4046 := RowAgreement.single (first := 4045) modelrow_agrees_4046
private def atomic_4047 : CountSegment 1 10 13120 4046 4047 1868 1869 := CountSegment.single (first := 4046) (initial := 1868) row_count_4047 (by decide)
private theorem atomic_4047_agreement : RowAgreement 4046 4047 := RowAgreement.single (first := 4046) modelrow_agrees_4047
private def atomic_4048 : CountSegment 1 10 13120 4047 4048 1869 1869 := CountSegment.single (first := 4047) (initial := 1869) row_count_4048 (by decide)
private theorem atomic_4048_agreement : RowAgreement 4047 4048 := RowAgreement.single (first := 4047) modelrow_agrees_4048
private def atomic_4049 : CountSegment 1 10 13120 4048 4049 1869 1869 := CountSegment.single (first := 4048) (initial := 1869) row_count_4049 (by decide)
private theorem atomic_4049_agreement : RowAgreement 4048 4049 := RowAgreement.single (first := 4048) modelrow_agrees_4049
private def atomic_4050 : CountSegment 1 10 13120 4049 4050 1869 1870 := CountSegment.single (first := 4049) (initial := 1869) row_count_4050 (by decide)
private theorem atomic_4050_agreement : RowAgreement 4049 4050 := RowAgreement.single (first := 4049) modelrow_agrees_4050
private def atomic_4051 : CountSegment 1 10 13120 4050 4051 1870 1870 := CountSegment.single (first := 4050) (initial := 1870) row_count_4051 (by decide)
private theorem atomic_4051_agreement : RowAgreement 4050 4051 := RowAgreement.single (first := 4050) modelrow_agrees_4051
private def atomic_4052 : CountSegment 1 10 13120 4051 4052 1870 1870 := CountSegment.single (first := 4051) (initial := 1870) row_count_4052 (by decide)
private theorem atomic_4052_agreement : RowAgreement 4051 4052 := RowAgreement.single (first := 4051) modelrow_agrees_4052
private def atomic_4053 : CountSegment 1 10 13120 4052 4053 1870 1871 := CountSegment.single (first := 4052) (initial := 1870) row_count_4053 (by decide)
private theorem atomic_4053_agreement : RowAgreement 4052 4053 := RowAgreement.single (first := 4052) modelrow_agrees_4053
private def atomic_4054 : CountSegment 1 10 13120 4053 4054 1871 1871 := CountSegment.single (first := 4053) (initial := 1871) row_count_4054 (by decide)
private theorem atomic_4054_agreement : RowAgreement 4053 4054 := RowAgreement.single (first := 4053) modelrow_agrees_4054
private def atomic_4055 : CountSegment 1 10 13120 4054 4055 1871 1871 := CountSegment.single (first := 4054) (initial := 1871) row_count_4055 (by decide)
private theorem atomic_4055_agreement : RowAgreement 4054 4055 := RowAgreement.single (first := 4054) modelrow_agrees_4055
private def atomic_4056 : CountSegment 1 10 13120 4055 4056 1871 1871 := CountSegment.single (first := 4055) (initial := 1871) row_count_4056 (by decide)
private theorem atomic_4056_agreement : RowAgreement 4055 4056 := RowAgreement.single (first := 4055) modelrow_agrees_4056
private def atomic_4057 : CountSegment 1 10 13120 4056 4057 1871 1873 := CountSegment.single (first := 4056) (initial := 1871) row_count_4057 (by decide)
private theorem atomic_4057_agreement : RowAgreement 4056 4057 := RowAgreement.single (first := 4056) modelrow_agrees_4057
private def atomic_4058 : CountSegment 1 10 13120 4057 4058 1873 1874 := CountSegment.single (first := 4057) (initial := 1873) row_count_4058 (by decide)
private theorem atomic_4058_agreement : RowAgreement 4057 4058 := RowAgreement.single (first := 4057) modelrow_agrees_4058
private def atomic_4059 : CountSegment 1 10 13120 4058 4059 1874 1875 := CountSegment.single (first := 4058) (initial := 1874) row_count_4059 (by decide)
private theorem atomic_4059_agreement : RowAgreement 4058 4059 := RowAgreement.single (first := 4058) modelrow_agrees_4059
private def atomic_4060 : CountSegment 1 10 13120 4059 4060 1875 1875 := CountSegment.single (first := 4059) (initial := 1875) row_count_4060 (by decide)
private theorem atomic_4060_agreement : RowAgreement 4059 4060 := RowAgreement.single (first := 4059) modelrow_agrees_4060
private def atomic_4061 : CountSegment 1 10 13120 4060 4061 1875 1875 := CountSegment.single (first := 4060) (initial := 1875) row_count_4061 (by decide)
private theorem atomic_4061_agreement : RowAgreement 4060 4061 := RowAgreement.single (first := 4060) modelrow_agrees_4061
private def atomic_4062 : CountSegment 1 10 13120 4061 4062 1875 1875 := CountSegment.single (first := 4061) (initial := 1875) row_count_4062 (by decide)
private theorem atomic_4062_agreement : RowAgreement 4061 4062 := RowAgreement.single (first := 4061) modelrow_agrees_4062
private def atomic_4063 : CountSegment 1 10 13120 4062 4063 1875 1875 := CountSegment.single (first := 4062) (initial := 1875) row_count_4063 (by decide)
private theorem atomic_4063_agreement : RowAgreement 4062 4063 := RowAgreement.single (first := 4062) modelrow_agrees_4063
private def atomic_4064 : CountSegment 1 10 13120 4063 4064 1875 1875 := CountSegment.single (first := 4063) (initial := 1875) row_count_4064 (by decide)
private theorem atomic_4064_agreement : RowAgreement 4063 4064 := RowAgreement.single (first := 4063) modelrow_agrees_4064
private def atomic_4065 : CountSegment 1 10 13120 4064 4065 1875 1876 := CountSegment.single (first := 4064) (initial := 1875) row_count_4065 (by decide)
private theorem atomic_4065_agreement : RowAgreement 4064 4065 := RowAgreement.single (first := 4064) modelrow_agrees_4065
private def atomic_4066 : CountSegment 1 10 13120 4065 4066 1876 1877 := CountSegment.single (first := 4065) (initial := 1876) row_count_4066 (by decide)
private theorem atomic_4066_agreement : RowAgreement 4065 4066 := RowAgreement.single (first := 4065) modelrow_agrees_4066
private def atomic_4067 : CountSegment 1 10 13120 4066 4067 1877 1877 := CountSegment.single (first := 4066) (initial := 1877) row_count_4067 (by decide)
private theorem atomic_4067_agreement : RowAgreement 4066 4067 := RowAgreement.single (first := 4066) modelrow_agrees_4067
private def atomic_4068 : CountSegment 1 10 13120 4067 4068 1877 1877 := CountSegment.single (first := 4067) (initial := 1877) row_count_4068 (by decide)
private theorem atomic_4068_agreement : RowAgreement 4067 4068 := RowAgreement.single (first := 4067) modelrow_agrees_4068
private def atomic_4069 : CountSegment 1 10 13120 4068 4069 1877 1877 := CountSegment.single (first := 4068) (initial := 1877) row_count_4069 (by decide)
private theorem atomic_4069_agreement : RowAgreement 4068 4069 := RowAgreement.single (first := 4068) modelrow_agrees_4069
private def atomic_4070 : CountSegment 1 10 13120 4069 4070 1877 1877 := CountSegment.single (first := 4069) (initial := 1877) row_count_4070 (by decide)
private theorem atomic_4070_agreement : RowAgreement 4069 4070 := RowAgreement.single (first := 4069) modelrow_agrees_4070
private def atomic_4071 : CountSegment 1 10 13120 4070 4071 1877 1878 := CountSegment.single (first := 4070) (initial := 1877) row_count_4071 (by decide)
private theorem atomic_4071_agreement : RowAgreement 4070 4071 := RowAgreement.single (first := 4070) modelrow_agrees_4071
private def atomic_4072 : CountSegment 1 10 13120 4071 4072 1878 1878 := CountSegment.single (first := 4071) (initial := 1878) row_count_4072 (by decide)
private theorem atomic_4072_agreement : RowAgreement 4071 4072 := RowAgreement.single (first := 4071) modelrow_agrees_4072
private def atomic_4073 : CountSegment 1 10 13120 4072 4073 1878 1878 := CountSegment.single (first := 4072) (initial := 1878) row_count_4073 (by decide)
private theorem atomic_4073_agreement : RowAgreement 4072 4073 := RowAgreement.single (first := 4072) modelrow_agrees_4073
private def atomic_4074 : CountSegment 1 10 13120 4073 4074 1878 1879 := CountSegment.single (first := 4073) (initial := 1878) row_count_4074 (by decide)
private theorem atomic_4074_agreement : RowAgreement 4073 4074 := RowAgreement.single (first := 4073) modelrow_agrees_4074
private def atomic_4075 : CountSegment 1 10 13120 4074 4075 1879 1879 := CountSegment.single (first := 4074) (initial := 1879) row_count_4075 (by decide)
private theorem atomic_4075_agreement : RowAgreement 4074 4075 := RowAgreement.single (first := 4074) modelrow_agrees_4075
private def atomic_4076 : CountSegment 1 10 13120 4075 4076 1879 1879 := CountSegment.single (first := 4075) (initial := 1879) row_count_4076 (by decide)
private theorem atomic_4076_agreement : RowAgreement 4075 4076 := RowAgreement.single (first := 4075) modelrow_agrees_4076
private def atomic_4077 : CountSegment 1 10 13120 4076 4077 1879 1880 := CountSegment.single (first := 4076) (initial := 1879) row_count_4077 (by decide)
private theorem atomic_4077_agreement : RowAgreement 4076 4077 := RowAgreement.single (first := 4076) modelrow_agrees_4077
private def atomic_4078 : CountSegment 1 10 13120 4077 4078 1880 1880 := CountSegment.single (first := 4077) (initial := 1880) row_count_4078 (by decide)
private theorem atomic_4078_agreement : RowAgreement 4077 4078 := RowAgreement.single (first := 4077) modelrow_agrees_4078
private def atomic_4079 : CountSegment 1 10 13120 4078 4079 1880 1880 := CountSegment.single (first := 4078) (initial := 1880) row_count_4079 (by decide)
private theorem atomic_4079_agreement : RowAgreement 4078 4079 := RowAgreement.single (first := 4078) modelrow_agrees_4079
private def atomic_4080 : CountSegment 1 10 13120 4079 4080 1880 1880 := CountSegment.single (first := 4079) (initial := 1880) row_count_4080 (by decide)
private theorem atomic_4080_agreement : RowAgreement 4079 4080 := RowAgreement.single (first := 4079) modelrow_agrees_4080
private def atomic_4081 : CountSegment 1 10 13120 4080 4081 1880 1880 := CountSegment.single (first := 4080) (initial := 1880) row_count_4081 (by decide)
private theorem atomic_4081_agreement : RowAgreement 4080 4081 := RowAgreement.single (first := 4080) modelrow_agrees_4081
private def atomic_4082 : CountSegment 1 10 13120 4081 4082 1880 1880 := CountSegment.single (first := 4081) (initial := 1880) row_count_4082 (by decide)
private theorem atomic_4082_agreement : RowAgreement 4081 4082 := RowAgreement.single (first := 4081) modelrow_agrees_4082
private def atomic_4083 : CountSegment 1 10 13120 4082 4083 1880 1880 := CountSegment.single (first := 4082) (initial := 1880) row_count_4083 (by decide)
private theorem atomic_4083_agreement : RowAgreement 4082 4083 := RowAgreement.single (first := 4082) modelrow_agrees_4083
private def atomic_4084 : CountSegment 1 10 13120 4083 4084 1880 1881 := CountSegment.single (first := 4083) (initial := 1880) row_count_4084 (by decide)
private theorem atomic_4084_agreement : RowAgreement 4083 4084 := RowAgreement.single (first := 4083) modelrow_agrees_4084
private def atomic_4085 : CountSegment 1 10 13120 4084 4085 1881 1881 := CountSegment.single (first := 4084) (initial := 1881) row_count_4085 (by decide)
private theorem atomic_4085_agreement : RowAgreement 4084 4085 := RowAgreement.single (first := 4084) modelrow_agrees_4085
private def atomic_4086 : CountSegment 1 10 13120 4085 4086 1881 1881 := CountSegment.single (first := 4085) (initial := 1881) row_count_4086 (by decide)
private theorem atomic_4086_agreement : RowAgreement 4085 4086 := RowAgreement.single (first := 4085) modelrow_agrees_4086
private def atomic_4087 : CountSegment 1 10 13120 4086 4087 1881 1883 := CountSegment.single (first := 4086) (initial := 1881) row_count_4087 (by decide)
private theorem atomic_4087_agreement : RowAgreement 4086 4087 := RowAgreement.single (first := 4086) modelrow_agrees_4087
private def atomic_4088 : CountSegment 1 10 13120 4087 4088 1883 1884 := CountSegment.single (first := 4087) (initial := 1883) row_count_4088 (by decide)
private theorem atomic_4088_agreement : RowAgreement 4087 4088 := RowAgreement.single (first := 4087) modelrow_agrees_4088
private def atomic_4089 : CountSegment 1 10 13120 4088 4089 1884 1884 := CountSegment.single (first := 4088) (initial := 1884) row_count_4089 (by decide)
private theorem atomic_4089_agreement : RowAgreement 4088 4089 := RowAgreement.single (first := 4088) modelrow_agrees_4089
private def atomic_4090 : CountSegment 1 10 13120 4089 4090 1884 1885 := CountSegment.single (first := 4089) (initial := 1884) row_count_4090 (by decide)
private theorem atomic_4090_agreement : RowAgreement 4089 4090 := RowAgreement.single (first := 4089) modelrow_agrees_4090
private def atomic_4091 : CountSegment 1 10 13120 4090 4091 1885 1885 := CountSegment.single (first := 4090) (initial := 1885) row_count_4091 (by decide)
private theorem atomic_4091_agreement : RowAgreement 4090 4091 := RowAgreement.single (first := 4090) modelrow_agrees_4091
private def atomic_4092 : CountSegment 1 10 13120 4091 4092 1885 1886 := CountSegment.single (first := 4091) (initial := 1885) row_count_4092 (by decide)
private theorem atomic_4092_agreement : RowAgreement 4091 4092 := RowAgreement.single (first := 4091) modelrow_agrees_4092
private def atomic_4093 : CountSegment 1 10 13120 4092 4093 1886 1886 := CountSegment.single (first := 4092) (initial := 1886) row_count_4093 (by decide)
private theorem atomic_4093_agreement : RowAgreement 4092 4093 := RowAgreement.single (first := 4092) modelrow_agrees_4093
private def atomic_4094 : CountSegment 1 10 13120 4093 4094 1886 1886 := CountSegment.single (first := 4093) (initial := 1886) row_count_4094 (by decide)
private theorem atomic_4094_agreement : RowAgreement 4093 4094 := RowAgreement.single (first := 4093) modelrow_agrees_4094
private def atomic_4095 : CountSegment 1 10 13120 4094 4095 1886 1886 := CountSegment.single (first := 4094) (initial := 1886) row_count_4095 (by decide)
private theorem atomic_4095_agreement : RowAgreement 4094 4095 := RowAgreement.single (first := 4094) modelrow_agrees_4095
private def atomic_4096 : CountSegment 1 10 13120 4095 4096 1886 1886 := CountSegment.single (first := 4095) (initial := 1886) row_count_4096 (by decide)
private theorem atomic_4096_agreement : RowAgreement 4095 4096 := RowAgreement.single (first := 4095) modelrow_agrees_4096
private def within_031_000 : CountSegment 1 10 13120 3968 3970 1836 1837 := CountSegment.join atomic_3969 atomic_3970
private theorem within_031_000_agreement : RowAgreement 3968 3970 := RowAgreement.join atomic_3969_agreement atomic_3970_agreement
private def within_031_001 : CountSegment 1 10 13120 3970 3972 1837 1838 := CountSegment.join atomic_3971 atomic_3972
private theorem within_031_001_agreement : RowAgreement 3970 3972 := RowAgreement.join atomic_3971_agreement atomic_3972_agreement
private def within_031_002 : CountSegment 1 10 13120 3968 3972 1836 1838 := CountSegment.join within_031_000 within_031_001
private theorem within_031_002_agreement : RowAgreement 3968 3972 := RowAgreement.join within_031_000_agreement within_031_001_agreement
private def within_031_003 : CountSegment 1 10 13120 3972 3974 1838 1839 := CountSegment.join atomic_3973 atomic_3974
private theorem within_031_003_agreement : RowAgreement 3972 3974 := RowAgreement.join atomic_3973_agreement atomic_3974_agreement
private def within_031_004 : CountSegment 1 10 13120 3974 3976 1839 1840 := CountSegment.join atomic_3975 atomic_3976
private theorem within_031_004_agreement : RowAgreement 3974 3976 := RowAgreement.join atomic_3975_agreement atomic_3976_agreement
private def within_031_005 : CountSegment 1 10 13120 3972 3976 1838 1840 := CountSegment.join within_031_003 within_031_004
private theorem within_031_005_agreement : RowAgreement 3972 3976 := RowAgreement.join within_031_003_agreement within_031_004_agreement
private def within_031_006 : CountSegment 1 10 13120 3968 3976 1836 1840 := CountSegment.join within_031_002 within_031_005
private theorem within_031_006_agreement : RowAgreement 3968 3976 := RowAgreement.join within_031_002_agreement within_031_005_agreement
private def within_031_007 : CountSegment 1 10 13120 3976 3978 1840 1841 := CountSegment.join atomic_3977 atomic_3978
private theorem within_031_007_agreement : RowAgreement 3976 3978 := RowAgreement.join atomic_3977_agreement atomic_3978_agreement
private def within_031_008 : CountSegment 1 10 13120 3978 3980 1841 1841 := CountSegment.join atomic_3979 atomic_3980
private theorem within_031_008_agreement : RowAgreement 3978 3980 := RowAgreement.join atomic_3979_agreement atomic_3980_agreement
private def within_031_009 : CountSegment 1 10 13120 3976 3980 1840 1841 := CountSegment.join within_031_007 within_031_008
private theorem within_031_009_agreement : RowAgreement 3976 3980 := RowAgreement.join within_031_007_agreement within_031_008_agreement
private def within_031_010 : CountSegment 1 10 13120 3980 3982 1841 1841 := CountSegment.join atomic_3981 atomic_3982
private theorem within_031_010_agreement : RowAgreement 3980 3982 := RowAgreement.join atomic_3981_agreement atomic_3982_agreement
private def within_031_011 : CountSegment 1 10 13120 3982 3984 1841 1842 := CountSegment.join atomic_3983 atomic_3984
private theorem within_031_011_agreement : RowAgreement 3982 3984 := RowAgreement.join atomic_3983_agreement atomic_3984_agreement
private def within_031_012 : CountSegment 1 10 13120 3980 3984 1841 1842 := CountSegment.join within_031_010 within_031_011
private theorem within_031_012_agreement : RowAgreement 3980 3984 := RowAgreement.join within_031_010_agreement within_031_011_agreement
private def within_031_013 : CountSegment 1 10 13120 3976 3984 1840 1842 := CountSegment.join within_031_009 within_031_012
private theorem within_031_013_agreement : RowAgreement 3976 3984 := RowAgreement.join within_031_009_agreement within_031_012_agreement
private def within_031_014 : CountSegment 1 10 13120 3968 3984 1836 1842 := CountSegment.join within_031_006 within_031_013
private theorem within_031_014_agreement : RowAgreement 3968 3984 := RowAgreement.join within_031_006_agreement within_031_013_agreement
private def within_031_015 : CountSegment 1 10 13120 3984 3986 1842 1843 := CountSegment.join atomic_3985 atomic_3986
private theorem within_031_015_agreement : RowAgreement 3984 3986 := RowAgreement.join atomic_3985_agreement atomic_3986_agreement
private def within_031_016 : CountSegment 1 10 13120 3986 3988 1843 1845 := CountSegment.join atomic_3987 atomic_3988
private theorem within_031_016_agreement : RowAgreement 3986 3988 := RowAgreement.join atomic_3987_agreement atomic_3988_agreement
private def within_031_017 : CountSegment 1 10 13120 3984 3988 1842 1845 := CountSegment.join within_031_015 within_031_016
private theorem within_031_017_agreement : RowAgreement 3984 3988 := RowAgreement.join within_031_015_agreement within_031_016_agreement
private def within_031_018 : CountSegment 1 10 13120 3988 3990 1845 1845 := CountSegment.join atomic_3989 atomic_3990
private theorem within_031_018_agreement : RowAgreement 3988 3990 := RowAgreement.join atomic_3989_agreement atomic_3990_agreement
private def within_031_019 : CountSegment 1 10 13120 3990 3992 1845 1846 := CountSegment.join atomic_3991 atomic_3992
private theorem within_031_019_agreement : RowAgreement 3990 3992 := RowAgreement.join atomic_3991_agreement atomic_3992_agreement
private def within_031_020 : CountSegment 1 10 13120 3988 3992 1845 1846 := CountSegment.join within_031_018 within_031_019
private theorem within_031_020_agreement : RowAgreement 3988 3992 := RowAgreement.join within_031_018_agreement within_031_019_agreement
private def within_031_021 : CountSegment 1 10 13120 3984 3992 1842 1846 := CountSegment.join within_031_017 within_031_020
private theorem within_031_021_agreement : RowAgreement 3984 3992 := RowAgreement.join within_031_017_agreement within_031_020_agreement
private def within_031_022 : CountSegment 1 10 13120 3992 3994 1846 1847 := CountSegment.join atomic_3993 atomic_3994
private theorem within_031_022_agreement : RowAgreement 3992 3994 := RowAgreement.join atomic_3993_agreement atomic_3994_agreement
private def within_031_023 : CountSegment 1 10 13120 3994 3996 1847 1847 := CountSegment.join atomic_3995 atomic_3996
private theorem within_031_023_agreement : RowAgreement 3994 3996 := RowAgreement.join atomic_3995_agreement atomic_3996_agreement
private def within_031_024 : CountSegment 1 10 13120 3992 3996 1846 1847 := CountSegment.join within_031_022 within_031_023
private theorem within_031_024_agreement : RowAgreement 3992 3996 := RowAgreement.join within_031_022_agreement within_031_023_agreement
private def within_031_025 : CountSegment 1 10 13120 3996 3998 1847 1847 := CountSegment.join atomic_3997 atomic_3998
private theorem within_031_025_agreement : RowAgreement 3996 3998 := RowAgreement.join atomic_3997_agreement atomic_3998_agreement
private def within_031_026 : CountSegment 1 10 13120 3998 4000 1847 1848 := CountSegment.join atomic_3999 atomic_4000
private theorem within_031_026_agreement : RowAgreement 3998 4000 := RowAgreement.join atomic_3999_agreement atomic_4000_agreement
private def within_031_027 : CountSegment 1 10 13120 3996 4000 1847 1848 := CountSegment.join within_031_025 within_031_026
private theorem within_031_027_agreement : RowAgreement 3996 4000 := RowAgreement.join within_031_025_agreement within_031_026_agreement
private def within_031_028 : CountSegment 1 10 13120 3992 4000 1846 1848 := CountSegment.join within_031_024 within_031_027
private theorem within_031_028_agreement : RowAgreement 3992 4000 := RowAgreement.join within_031_024_agreement within_031_027_agreement
private def within_031_029 : CountSegment 1 10 13120 3984 4000 1842 1848 := CountSegment.join within_031_021 within_031_028
private theorem within_031_029_agreement : RowAgreement 3984 4000 := RowAgreement.join within_031_021_agreement within_031_028_agreement
private def within_031_030 : CountSegment 1 10 13120 3968 4000 1836 1848 := CountSegment.join within_031_014 within_031_029
private theorem within_031_030_agreement : RowAgreement 3968 4000 := RowAgreement.join within_031_014_agreement within_031_029_agreement
private def within_031_031 : CountSegment 1 10 13120 4000 4002 1848 1848 := CountSegment.join atomic_4001 atomic_4002
private theorem within_031_031_agreement : RowAgreement 4000 4002 := RowAgreement.join atomic_4001_agreement atomic_4002_agreement
private def within_031_032 : CountSegment 1 10 13120 4002 4004 1848 1848 := CountSegment.join atomic_4003 atomic_4004
private theorem within_031_032_agreement : RowAgreement 4002 4004 := RowAgreement.join atomic_4003_agreement atomic_4004_agreement
private def within_031_033 : CountSegment 1 10 13120 4000 4004 1848 1848 := CountSegment.join within_031_031 within_031_032
private theorem within_031_033_agreement : RowAgreement 4000 4004 := RowAgreement.join within_031_031_agreement within_031_032_agreement
private def within_031_034 : CountSegment 1 10 13120 4004 4006 1848 1848 := CountSegment.join atomic_4005 atomic_4006
private theorem within_031_034_agreement : RowAgreement 4004 4006 := RowAgreement.join atomic_4005_agreement atomic_4006_agreement
private def within_031_035 : CountSegment 1 10 13120 4006 4008 1848 1850 := CountSegment.join atomic_4007 atomic_4008
private theorem within_031_035_agreement : RowAgreement 4006 4008 := RowAgreement.join atomic_4007_agreement atomic_4008_agreement
private def within_031_036 : CountSegment 1 10 13120 4004 4008 1848 1850 := CountSegment.join within_031_034 within_031_035
private theorem within_031_036_agreement : RowAgreement 4004 4008 := RowAgreement.join within_031_034_agreement within_031_035_agreement
private def within_031_037 : CountSegment 1 10 13120 4000 4008 1848 1850 := CountSegment.join within_031_033 within_031_036
private theorem within_031_037_agreement : RowAgreement 4000 4008 := RowAgreement.join within_031_033_agreement within_031_036_agreement
private def within_031_038 : CountSegment 1 10 13120 4008 4010 1850 1850 := CountSegment.join atomic_4009 atomic_4010
private theorem within_031_038_agreement : RowAgreement 4008 4010 := RowAgreement.join atomic_4009_agreement atomic_4010_agreement
private def within_031_039 : CountSegment 1 10 13120 4010 4012 1850 1850 := CountSegment.join atomic_4011 atomic_4012
private theorem within_031_039_agreement : RowAgreement 4010 4012 := RowAgreement.join atomic_4011_agreement atomic_4012_agreement
private def within_031_040 : CountSegment 1 10 13120 4008 4012 1850 1850 := CountSegment.join within_031_038 within_031_039
private theorem within_031_040_agreement : RowAgreement 4008 4012 := RowAgreement.join within_031_038_agreement within_031_039_agreement
private def within_031_041 : CountSegment 1 10 13120 4012 4014 1850 1851 := CountSegment.join atomic_4013 atomic_4014
private theorem within_031_041_agreement : RowAgreement 4012 4014 := RowAgreement.join atomic_4013_agreement atomic_4014_agreement
private def within_031_042 : CountSegment 1 10 13120 4014 4016 1851 1852 := CountSegment.join atomic_4015 atomic_4016
private theorem within_031_042_agreement : RowAgreement 4014 4016 := RowAgreement.join atomic_4015_agreement atomic_4016_agreement
private def within_031_043 : CountSegment 1 10 13120 4012 4016 1850 1852 := CountSegment.join within_031_041 within_031_042
private theorem within_031_043_agreement : RowAgreement 4012 4016 := RowAgreement.join within_031_041_agreement within_031_042_agreement
private def within_031_044 : CountSegment 1 10 13120 4008 4016 1850 1852 := CountSegment.join within_031_040 within_031_043
private theorem within_031_044_agreement : RowAgreement 4008 4016 := RowAgreement.join within_031_040_agreement within_031_043_agreement
private def within_031_045 : CountSegment 1 10 13120 4000 4016 1848 1852 := CountSegment.join within_031_037 within_031_044
private theorem within_031_045_agreement : RowAgreement 4000 4016 := RowAgreement.join within_031_037_agreement within_031_044_agreement
private def within_031_046 : CountSegment 1 10 13120 4016 4018 1852 1854 := CountSegment.join atomic_4017 atomic_4018
private theorem within_031_046_agreement : RowAgreement 4016 4018 := RowAgreement.join atomic_4017_agreement atomic_4018_agreement
private def within_031_047 : CountSegment 1 10 13120 4018 4020 1854 1854 := CountSegment.join atomic_4019 atomic_4020
private theorem within_031_047_agreement : RowAgreement 4018 4020 := RowAgreement.join atomic_4019_agreement atomic_4020_agreement
private def within_031_048 : CountSegment 1 10 13120 4016 4020 1852 1854 := CountSegment.join within_031_046 within_031_047
private theorem within_031_048_agreement : RowAgreement 4016 4020 := RowAgreement.join within_031_046_agreement within_031_047_agreement
private def within_031_049 : CountSegment 1 10 13120 4020 4022 1854 1855 := CountSegment.join atomic_4021 atomic_4022
private theorem within_031_049_agreement : RowAgreement 4020 4022 := RowAgreement.join atomic_4021_agreement atomic_4022_agreement
private def within_031_050 : CountSegment 1 10 13120 4022 4024 1855 1855 := CountSegment.join atomic_4023 atomic_4024
private theorem within_031_050_agreement : RowAgreement 4022 4024 := RowAgreement.join atomic_4023_agreement atomic_4024_agreement
private def within_031_051 : CountSegment 1 10 13120 4020 4024 1854 1855 := CountSegment.join within_031_049 within_031_050
private theorem within_031_051_agreement : RowAgreement 4020 4024 := RowAgreement.join within_031_049_agreement within_031_050_agreement
private def within_031_052 : CountSegment 1 10 13120 4016 4024 1852 1855 := CountSegment.join within_031_048 within_031_051
private theorem within_031_052_agreement : RowAgreement 4016 4024 := RowAgreement.join within_031_048_agreement within_031_051_agreement
private def within_031_053 : CountSegment 1 10 13120 4024 4026 1855 1857 := CountSegment.join atomic_4025 atomic_4026
private theorem within_031_053_agreement : RowAgreement 4024 4026 := RowAgreement.join atomic_4025_agreement atomic_4026_agreement
private def within_031_054 : CountSegment 1 10 13120 4026 4028 1857 1857 := CountSegment.join atomic_4027 atomic_4028
private theorem within_031_054_agreement : RowAgreement 4026 4028 := RowAgreement.join atomic_4027_agreement atomic_4028_agreement
private def within_031_055 : CountSegment 1 10 13120 4024 4028 1855 1857 := CountSegment.join within_031_053 within_031_054
private theorem within_031_055_agreement : RowAgreement 4024 4028 := RowAgreement.join within_031_053_agreement within_031_054_agreement
private def within_031_056 : CountSegment 1 10 13120 4028 4030 1857 1859 := CountSegment.join atomic_4029 atomic_4030
private theorem within_031_056_agreement : RowAgreement 4028 4030 := RowAgreement.join atomic_4029_agreement atomic_4030_agreement
private def within_031_057 : CountSegment 1 10 13120 4030 4032 1859 1861 := CountSegment.join atomic_4031 atomic_4032
private theorem within_031_057_agreement : RowAgreement 4030 4032 := RowAgreement.join atomic_4031_agreement atomic_4032_agreement
private def within_031_058 : CountSegment 1 10 13120 4028 4032 1857 1861 := CountSegment.join within_031_056 within_031_057
private theorem within_031_058_agreement : RowAgreement 4028 4032 := RowAgreement.join within_031_056_agreement within_031_057_agreement
private def within_031_059 : CountSegment 1 10 13120 4024 4032 1855 1861 := CountSegment.join within_031_055 within_031_058
private theorem within_031_059_agreement : RowAgreement 4024 4032 := RowAgreement.join within_031_055_agreement within_031_058_agreement
private def within_031_060 : CountSegment 1 10 13120 4016 4032 1852 1861 := CountSegment.join within_031_052 within_031_059
private theorem within_031_060_agreement : RowAgreement 4016 4032 := RowAgreement.join within_031_052_agreement within_031_059_agreement
private def within_031_061 : CountSegment 1 10 13120 4000 4032 1848 1861 := CountSegment.join within_031_045 within_031_060
private theorem within_031_061_agreement : RowAgreement 4000 4032 := RowAgreement.join within_031_045_agreement within_031_060_agreement
private def within_031_062 : CountSegment 1 10 13120 3968 4032 1836 1861 := CountSegment.join within_031_030 within_031_061
private theorem within_031_062_agreement : RowAgreement 3968 4032 := RowAgreement.join within_031_030_agreement within_031_061_agreement
private def within_031_063 : CountSegment 1 10 13120 4032 4034 1861 1861 := CountSegment.join atomic_4033 atomic_4034
private theorem within_031_063_agreement : RowAgreement 4032 4034 := RowAgreement.join atomic_4033_agreement atomic_4034_agreement
private def within_031_064 : CountSegment 1 10 13120 4034 4036 1861 1863 := CountSegment.join atomic_4035 atomic_4036
private theorem within_031_064_agreement : RowAgreement 4034 4036 := RowAgreement.join atomic_4035_agreement atomic_4036_agreement
private def within_031_065 : CountSegment 1 10 13120 4032 4036 1861 1863 := CountSegment.join within_031_063 within_031_064
private theorem within_031_065_agreement : RowAgreement 4032 4036 := RowAgreement.join within_031_063_agreement within_031_064_agreement
private def within_031_066 : CountSegment 1 10 13120 4036 4038 1863 1863 := CountSegment.join atomic_4037 atomic_4038
private theorem within_031_066_agreement : RowAgreement 4036 4038 := RowAgreement.join atomic_4037_agreement atomic_4038_agreement
private def within_031_067 : CountSegment 1 10 13120 4038 4040 1863 1866 := CountSegment.join atomic_4039 atomic_4040
private theorem within_031_067_agreement : RowAgreement 4038 4040 := RowAgreement.join atomic_4039_agreement atomic_4040_agreement
private def within_031_068 : CountSegment 1 10 13120 4036 4040 1863 1866 := CountSegment.join within_031_066 within_031_067
private theorem within_031_068_agreement : RowAgreement 4036 4040 := RowAgreement.join within_031_066_agreement within_031_067_agreement
private def within_031_069 : CountSegment 1 10 13120 4032 4040 1861 1866 := CountSegment.join within_031_065 within_031_068
private theorem within_031_069_agreement : RowAgreement 4032 4040 := RowAgreement.join within_031_065_agreement within_031_068_agreement
private def within_031_070 : CountSegment 1 10 13120 4040 4042 1866 1867 := CountSegment.join atomic_4041 atomic_4042
private theorem within_031_070_agreement : RowAgreement 4040 4042 := RowAgreement.join atomic_4041_agreement atomic_4042_agreement
private def within_031_071 : CountSegment 1 10 13120 4042 4044 1867 1868 := CountSegment.join atomic_4043 atomic_4044
private theorem within_031_071_agreement : RowAgreement 4042 4044 := RowAgreement.join atomic_4043_agreement atomic_4044_agreement
private def within_031_072 : CountSegment 1 10 13120 4040 4044 1866 1868 := CountSegment.join within_031_070 within_031_071
private theorem within_031_072_agreement : RowAgreement 4040 4044 := RowAgreement.join within_031_070_agreement within_031_071_agreement
private def within_031_073 : CountSegment 1 10 13120 4044 4046 1868 1868 := CountSegment.join atomic_4045 atomic_4046
private theorem within_031_073_agreement : RowAgreement 4044 4046 := RowAgreement.join atomic_4045_agreement atomic_4046_agreement
private def within_031_074 : CountSegment 1 10 13120 4046 4048 1868 1869 := CountSegment.join atomic_4047 atomic_4048
private theorem within_031_074_agreement : RowAgreement 4046 4048 := RowAgreement.join atomic_4047_agreement atomic_4048_agreement
private def within_031_075 : CountSegment 1 10 13120 4044 4048 1868 1869 := CountSegment.join within_031_073 within_031_074
private theorem within_031_075_agreement : RowAgreement 4044 4048 := RowAgreement.join within_031_073_agreement within_031_074_agreement
private def within_031_076 : CountSegment 1 10 13120 4040 4048 1866 1869 := CountSegment.join within_031_072 within_031_075
private theorem within_031_076_agreement : RowAgreement 4040 4048 := RowAgreement.join within_031_072_agreement within_031_075_agreement
private def within_031_077 : CountSegment 1 10 13120 4032 4048 1861 1869 := CountSegment.join within_031_069 within_031_076
private theorem within_031_077_agreement : RowAgreement 4032 4048 := RowAgreement.join within_031_069_agreement within_031_076_agreement
private def within_031_078 : CountSegment 1 10 13120 4048 4050 1869 1870 := CountSegment.join atomic_4049 atomic_4050
private theorem within_031_078_agreement : RowAgreement 4048 4050 := RowAgreement.join atomic_4049_agreement atomic_4050_agreement
private def within_031_079 : CountSegment 1 10 13120 4050 4052 1870 1870 := CountSegment.join atomic_4051 atomic_4052
private theorem within_031_079_agreement : RowAgreement 4050 4052 := RowAgreement.join atomic_4051_agreement atomic_4052_agreement
private def within_031_080 : CountSegment 1 10 13120 4048 4052 1869 1870 := CountSegment.join within_031_078 within_031_079
private theorem within_031_080_agreement : RowAgreement 4048 4052 := RowAgreement.join within_031_078_agreement within_031_079_agreement
private def within_031_081 : CountSegment 1 10 13120 4052 4054 1870 1871 := CountSegment.join atomic_4053 atomic_4054
private theorem within_031_081_agreement : RowAgreement 4052 4054 := RowAgreement.join atomic_4053_agreement atomic_4054_agreement
private def within_031_082 : CountSegment 1 10 13120 4054 4056 1871 1871 := CountSegment.join atomic_4055 atomic_4056
private theorem within_031_082_agreement : RowAgreement 4054 4056 := RowAgreement.join atomic_4055_agreement atomic_4056_agreement
private def within_031_083 : CountSegment 1 10 13120 4052 4056 1870 1871 := CountSegment.join within_031_081 within_031_082
private theorem within_031_083_agreement : RowAgreement 4052 4056 := RowAgreement.join within_031_081_agreement within_031_082_agreement
private def within_031_084 : CountSegment 1 10 13120 4048 4056 1869 1871 := CountSegment.join within_031_080 within_031_083
private theorem within_031_084_agreement : RowAgreement 4048 4056 := RowAgreement.join within_031_080_agreement within_031_083_agreement
private def within_031_085 : CountSegment 1 10 13120 4056 4058 1871 1874 := CountSegment.join atomic_4057 atomic_4058
private theorem within_031_085_agreement : RowAgreement 4056 4058 := RowAgreement.join atomic_4057_agreement atomic_4058_agreement
private def within_031_086 : CountSegment 1 10 13120 4058 4060 1874 1875 := CountSegment.join atomic_4059 atomic_4060
private theorem within_031_086_agreement : RowAgreement 4058 4060 := RowAgreement.join atomic_4059_agreement atomic_4060_agreement
private def within_031_087 : CountSegment 1 10 13120 4056 4060 1871 1875 := CountSegment.join within_031_085 within_031_086
private theorem within_031_087_agreement : RowAgreement 4056 4060 := RowAgreement.join within_031_085_agreement within_031_086_agreement
private def within_031_088 : CountSegment 1 10 13120 4060 4062 1875 1875 := CountSegment.join atomic_4061 atomic_4062
private theorem within_031_088_agreement : RowAgreement 4060 4062 := RowAgreement.join atomic_4061_agreement atomic_4062_agreement
private def within_031_089 : CountSegment 1 10 13120 4062 4064 1875 1875 := CountSegment.join atomic_4063 atomic_4064
private theorem within_031_089_agreement : RowAgreement 4062 4064 := RowAgreement.join atomic_4063_agreement atomic_4064_agreement
private def within_031_090 : CountSegment 1 10 13120 4060 4064 1875 1875 := CountSegment.join within_031_088 within_031_089
private theorem within_031_090_agreement : RowAgreement 4060 4064 := RowAgreement.join within_031_088_agreement within_031_089_agreement
private def within_031_091 : CountSegment 1 10 13120 4056 4064 1871 1875 := CountSegment.join within_031_087 within_031_090
private theorem within_031_091_agreement : RowAgreement 4056 4064 := RowAgreement.join within_031_087_agreement within_031_090_agreement
private def within_031_092 : CountSegment 1 10 13120 4048 4064 1869 1875 := CountSegment.join within_031_084 within_031_091
private theorem within_031_092_agreement : RowAgreement 4048 4064 := RowAgreement.join within_031_084_agreement within_031_091_agreement
private def within_031_093 : CountSegment 1 10 13120 4032 4064 1861 1875 := CountSegment.join within_031_077 within_031_092
private theorem within_031_093_agreement : RowAgreement 4032 4064 := RowAgreement.join within_031_077_agreement within_031_092_agreement
private def within_031_094 : CountSegment 1 10 13120 4064 4066 1875 1877 := CountSegment.join atomic_4065 atomic_4066
private theorem within_031_094_agreement : RowAgreement 4064 4066 := RowAgreement.join atomic_4065_agreement atomic_4066_agreement
private def within_031_095 : CountSegment 1 10 13120 4066 4068 1877 1877 := CountSegment.join atomic_4067 atomic_4068
private theorem within_031_095_agreement : RowAgreement 4066 4068 := RowAgreement.join atomic_4067_agreement atomic_4068_agreement
private def within_031_096 : CountSegment 1 10 13120 4064 4068 1875 1877 := CountSegment.join within_031_094 within_031_095
private theorem within_031_096_agreement : RowAgreement 4064 4068 := RowAgreement.join within_031_094_agreement within_031_095_agreement
private def within_031_097 : CountSegment 1 10 13120 4068 4070 1877 1877 := CountSegment.join atomic_4069 atomic_4070
private theorem within_031_097_agreement : RowAgreement 4068 4070 := RowAgreement.join atomic_4069_agreement atomic_4070_agreement
private def within_031_098 : CountSegment 1 10 13120 4070 4072 1877 1878 := CountSegment.join atomic_4071 atomic_4072
private theorem within_031_098_agreement : RowAgreement 4070 4072 := RowAgreement.join atomic_4071_agreement atomic_4072_agreement
private def within_031_099 : CountSegment 1 10 13120 4068 4072 1877 1878 := CountSegment.join within_031_097 within_031_098
private theorem within_031_099_agreement : RowAgreement 4068 4072 := RowAgreement.join within_031_097_agreement within_031_098_agreement
private def within_031_100 : CountSegment 1 10 13120 4064 4072 1875 1878 := CountSegment.join within_031_096 within_031_099
private theorem within_031_100_agreement : RowAgreement 4064 4072 := RowAgreement.join within_031_096_agreement within_031_099_agreement
private def within_031_101 : CountSegment 1 10 13120 4072 4074 1878 1879 := CountSegment.join atomic_4073 atomic_4074
private theorem within_031_101_agreement : RowAgreement 4072 4074 := RowAgreement.join atomic_4073_agreement atomic_4074_agreement
private def within_031_102 : CountSegment 1 10 13120 4074 4076 1879 1879 := CountSegment.join atomic_4075 atomic_4076
private theorem within_031_102_agreement : RowAgreement 4074 4076 := RowAgreement.join atomic_4075_agreement atomic_4076_agreement
private def within_031_103 : CountSegment 1 10 13120 4072 4076 1878 1879 := CountSegment.join within_031_101 within_031_102
private theorem within_031_103_agreement : RowAgreement 4072 4076 := RowAgreement.join within_031_101_agreement within_031_102_agreement
private def within_031_104 : CountSegment 1 10 13120 4076 4078 1879 1880 := CountSegment.join atomic_4077 atomic_4078
private theorem within_031_104_agreement : RowAgreement 4076 4078 := RowAgreement.join atomic_4077_agreement atomic_4078_agreement
private def within_031_105 : CountSegment 1 10 13120 4078 4080 1880 1880 := CountSegment.join atomic_4079 atomic_4080
private theorem within_031_105_agreement : RowAgreement 4078 4080 := RowAgreement.join atomic_4079_agreement atomic_4080_agreement
private def within_031_106 : CountSegment 1 10 13120 4076 4080 1879 1880 := CountSegment.join within_031_104 within_031_105
private theorem within_031_106_agreement : RowAgreement 4076 4080 := RowAgreement.join within_031_104_agreement within_031_105_agreement
private def within_031_107 : CountSegment 1 10 13120 4072 4080 1878 1880 := CountSegment.join within_031_103 within_031_106
private theorem within_031_107_agreement : RowAgreement 4072 4080 := RowAgreement.join within_031_103_agreement within_031_106_agreement
private def within_031_108 : CountSegment 1 10 13120 4064 4080 1875 1880 := CountSegment.join within_031_100 within_031_107
private theorem within_031_108_agreement : RowAgreement 4064 4080 := RowAgreement.join within_031_100_agreement within_031_107_agreement
private def within_031_109 : CountSegment 1 10 13120 4080 4082 1880 1880 := CountSegment.join atomic_4081 atomic_4082
private theorem within_031_109_agreement : RowAgreement 4080 4082 := RowAgreement.join atomic_4081_agreement atomic_4082_agreement
private def within_031_110 : CountSegment 1 10 13120 4082 4084 1880 1881 := CountSegment.join atomic_4083 atomic_4084
private theorem within_031_110_agreement : RowAgreement 4082 4084 := RowAgreement.join atomic_4083_agreement atomic_4084_agreement
private def within_031_111 : CountSegment 1 10 13120 4080 4084 1880 1881 := CountSegment.join within_031_109 within_031_110
private theorem within_031_111_agreement : RowAgreement 4080 4084 := RowAgreement.join within_031_109_agreement within_031_110_agreement
private def within_031_112 : CountSegment 1 10 13120 4084 4086 1881 1881 := CountSegment.join atomic_4085 atomic_4086
private theorem within_031_112_agreement : RowAgreement 4084 4086 := RowAgreement.join atomic_4085_agreement atomic_4086_agreement
private def within_031_113 : CountSegment 1 10 13120 4086 4088 1881 1884 := CountSegment.join atomic_4087 atomic_4088
private theorem within_031_113_agreement : RowAgreement 4086 4088 := RowAgreement.join atomic_4087_agreement atomic_4088_agreement
private def within_031_114 : CountSegment 1 10 13120 4084 4088 1881 1884 := CountSegment.join within_031_112 within_031_113
private theorem within_031_114_agreement : RowAgreement 4084 4088 := RowAgreement.join within_031_112_agreement within_031_113_agreement
private def within_031_115 : CountSegment 1 10 13120 4080 4088 1880 1884 := CountSegment.join within_031_111 within_031_114
private theorem within_031_115_agreement : RowAgreement 4080 4088 := RowAgreement.join within_031_111_agreement within_031_114_agreement
private def within_031_116 : CountSegment 1 10 13120 4088 4090 1884 1885 := CountSegment.join atomic_4089 atomic_4090
private theorem within_031_116_agreement : RowAgreement 4088 4090 := RowAgreement.join atomic_4089_agreement atomic_4090_agreement
private def within_031_117 : CountSegment 1 10 13120 4090 4092 1885 1886 := CountSegment.join atomic_4091 atomic_4092
private theorem within_031_117_agreement : RowAgreement 4090 4092 := RowAgreement.join atomic_4091_agreement atomic_4092_agreement
private def within_031_118 : CountSegment 1 10 13120 4088 4092 1884 1886 := CountSegment.join within_031_116 within_031_117
private theorem within_031_118_agreement : RowAgreement 4088 4092 := RowAgreement.join within_031_116_agreement within_031_117_agreement
private def within_031_119 : CountSegment 1 10 13120 4092 4094 1886 1886 := CountSegment.join atomic_4093 atomic_4094
private theorem within_031_119_agreement : RowAgreement 4092 4094 := RowAgreement.join atomic_4093_agreement atomic_4094_agreement
private def within_031_120 : CountSegment 1 10 13120 4094 4096 1886 1886 := CountSegment.join atomic_4095 atomic_4096
private theorem within_031_120_agreement : RowAgreement 4094 4096 := RowAgreement.join atomic_4095_agreement atomic_4096_agreement
private def within_031_121 : CountSegment 1 10 13120 4092 4096 1886 1886 := CountSegment.join within_031_119 within_031_120
private theorem within_031_121_agreement : RowAgreement 4092 4096 := RowAgreement.join within_031_119_agreement within_031_120_agreement
private def within_031_122 : CountSegment 1 10 13120 4088 4096 1884 1886 := CountSegment.join within_031_118 within_031_121
private theorem within_031_122_agreement : RowAgreement 4088 4096 := RowAgreement.join within_031_118_agreement within_031_121_agreement
private def within_031_123 : CountSegment 1 10 13120 4080 4096 1880 1886 := CountSegment.join within_031_115 within_031_122
private theorem within_031_123_agreement : RowAgreement 4080 4096 := RowAgreement.join within_031_115_agreement within_031_122_agreement
private def within_031_124 : CountSegment 1 10 13120 4064 4096 1875 1886 := CountSegment.join within_031_108 within_031_123
private theorem within_031_124_agreement : RowAgreement 4064 4096 := RowAgreement.join within_031_108_agreement within_031_123_agreement
private def within_031_125 : CountSegment 1 10 13120 4032 4096 1861 1886 := CountSegment.join within_031_093 within_031_124
private theorem within_031_125_agreement : RowAgreement 4032 4096 := RowAgreement.join within_031_093_agreement within_031_124_agreement
private def within_031_126 : CountSegment 1 10 13120 3968 4096 1836 1886 := CountSegment.join within_031_062 within_031_125
private theorem within_031_126_agreement : RowAgreement 3968 4096 := RowAgreement.join within_031_062_agreement within_031_125_agreement
def segment_031 : CountSegment 1 10 13120 3968 4096 1836 1886 := within_031_126
theorem agreement_031 : RowAgreement 3968 4096 := within_031_126_agreement
end ProjectEulerSolutions.P58Numerical
