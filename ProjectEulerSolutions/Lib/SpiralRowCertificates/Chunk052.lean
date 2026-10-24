import ProjectEulerSolutions.Lib.SpiralPrimeSegments
import ProjectEulerSolutions.Lib.SpiralPrimeCertificates.All
import ProjectEulerSolutions.Lib.MillerRabin
import ProjectEulerSolutions.Lib.SpiralMillerAgreement
/-! Finite spiral row and counter certificates. Python execution is checked separately. -/
namespace ProjectEulerSolutions.P58Numerical
open SpiralPrimes
set_option Elab.async false
def row_6657 : RowCertificate 6657 where
  first := true
  first_correct := by
    have prime : Nat.Prime (corner 6657 1) := by change Nat.Prime 177275911; exact ProjectEulerSolutions.P58PrimeNodes.p_177275911.prime
    exact decide_eq_true prime
  second := false
  second_correct := by
    have composite : ¬ Nat.Prime (corner 6657 2) := by change ¬ Nat.Prime 177262597; exact PrattCertificates.composite_of_product (factor := 37) (cofactor := 4790881) rfl (by decide) (by decide)
    exact decide_eq_false composite
  third := false
  third_correct := by
    have composite : ¬ Nat.Prime (corner 6657 3) := by change ¬ Nat.Prime 177249283; exact PrattCertificates.composite_of_product (factor := 4231) (cofactor := 41893) rfl (by decide) (by decide)
    exact decide_eq_false composite
theorem row_count_6657 : row 6657 = 1 := (row_6657.checked).trans rfl
theorem model_count_6657 : modelRow 6657 = 1 := by decide
theorem modelrow_agrees_6657 : modelRow 6657 = row 6657 := model_count_6657.trans row_count_6657.symm
def row_6658 : RowCertificate 6658 where
  first := false
  first_correct := by
    have composite : ¬ Nat.Prime (corner 6658 1) := by change ¬ Nat.Prime 177329173; exact PrattCertificates.composite_of_product (factor := 7) (cofactor := 25332739) rfl (by decide) (by decide)
    exact decide_eq_false composite
  second := false
  second_correct := by
    have composite : ¬ Nat.Prime (corner 6658 2) := by change ¬ Nat.Prime 177315857; exact PrattCertificates.composite_of_product (factor := 41) (cofactor := 4324777) rfl (by decide) (by decide)
    exact decide_eq_false composite
  third := false
  third_correct := by
    have composite : ¬ Nat.Prime (corner 6658 3) := by change ¬ Nat.Prime 177302541; exact PrattCertificates.composite_of_product (factor := 3) (cofactor := 59100847) rfl (by decide) (by decide)
    exact decide_eq_false composite
theorem row_count_6658 : row 6658 = 0 := (row_6658.checked).trans rfl
theorem model_count_6658 : modelRow 6658 = 0 := by decide
theorem modelrow_agrees_6658 : modelRow 6658 = row 6658 := model_count_6658.trans row_count_6658.symm
def row_6659 : RowCertificate 6659 where
  first := false
  first_correct := by
    have composite : ¬ Nat.Prime (corner 6659 1) := by change ¬ Nat.Prime 177382443; exact PrattCertificates.composite_of_product (factor := 3) (cofactor := 59127481) rfl (by decide) (by decide)
    exact decide_eq_false composite
  second := false
  second_correct := by
    have composite : ¬ Nat.Prime (corner 6659 2) := by change ¬ Nat.Prime 177369125; exact PrattCertificates.composite_of_product (factor := 5) (cofactor := 35473825) rfl (by decide) (by decide)
    exact decide_eq_false composite
  third := true
  third_correct := by
    have prime : Nat.Prime (corner 6659 3) := by change Nat.Prime 177355807; exact ProjectEulerSolutions.P58PrimeNodes.p_177355807.prime
    exact decide_eq_true prime
theorem row_count_6659 : row 6659 = 1 := (row_6659.checked).trans rfl
theorem model_count_6659 : modelRow 6659 = 1 := by decide
theorem modelrow_agrees_6659 : modelRow 6659 = row 6659 := model_count_6659.trans row_count_6659.symm
def row_6660 : RowCertificate 6660 where
  first := false
  first_correct := by
    have composite : ¬ Nat.Prime (corner 6660 1) := by change ¬ Nat.Prime 177435721; exact PrattCertificates.composite_of_product (factor := 151) (cofactor := 1175071) rfl (by decide) (by decide)
    exact decide_eq_false composite
  second := false
  second_correct := by
    have composite : ¬ Nat.Prime (corner 6660 2) := by change ¬ Nat.Prime 177422401; exact PrattCertificates.composite_of_product (factor := 13) (cofactor := 13647877) rfl (by decide) (by decide)
    exact decide_eq_false composite
  third := true
  third_correct := by
    have prime : Nat.Prime (corner 6660 3) := by change Nat.Prime 177409081; exact ProjectEulerSolutions.P58PrimeNodes.p_177409081.prime
    exact decide_eq_true prime
theorem row_count_6660 : row 6660 = 1 := (row_6660.checked).trans rfl
theorem model_count_6660 : modelRow 6660 = 1 := by decide
theorem modelrow_agrees_6660 : modelRow 6660 = row 6660 := model_count_6660.trans row_count_6660.symm
def row_6661 : RowCertificate 6661 where
  first := true
  first_correct := by
    have prime : Nat.Prime (corner 6661 1) := by change Nat.Prime 177489007; exact ProjectEulerSolutions.P58PrimeNodes.p_177489007.prime
    exact decide_eq_true prime
  second := false
  second_correct := by
    have composite : ¬ Nat.Prime (corner 6661 2) := by change ¬ Nat.Prime 177475685; exact PrattCertificates.composite_of_product (factor := 5) (cofactor := 35495137) rfl (by decide) (by decide)
    exact decide_eq_false composite
  third := false
  third_correct := by
    have composite : ¬ Nat.Prime (corner 6661 3) := by change ¬ Nat.Prime 177462363; exact PrattCertificates.composite_of_product (factor := 3) (cofactor := 59154121) rfl (by decide) (by decide)
    exact decide_eq_false composite
theorem row_count_6661 : row 6661 = 1 := (row_6661.checked).trans rfl
theorem model_count_6661 : modelRow 6661 = 1 := by decide
theorem modelrow_agrees_6661 : modelRow 6661 = row 6661 := model_count_6661.trans row_count_6661.symm
def row_6662 : RowCertificate 6662 where
  first := false
  first_correct := by
    have composite : ¬ Nat.Prime (corner 6662 1) := by change ¬ Nat.Prime 177542301; exact PrattCertificates.composite_of_product (factor := 3) (cofactor := 59180767) rfl (by decide) (by decide)
    exact decide_eq_false composite
  second := false
  second_correct := by
    have composite : ¬ Nat.Prime (corner 6662 2) := by change ¬ Nat.Prime 177528977; exact PrattCertificates.composite_of_product (factor := 17) (cofactor := 10442881) rfl (by decide) (by decide)
    exact decide_eq_false composite
  third := false
  third_correct := by
    have composite : ¬ Nat.Prime (corner 6662 3) := by change ¬ Nat.Prime 177515653; exact PrattCertificates.composite_of_product (factor := 7) (cofactor := 25359379) rfl (by decide) (by decide)
    exact decide_eq_false composite
theorem row_count_6662 : row 6662 = 0 := (row_6662.checked).trans rfl
theorem model_count_6662 : modelRow 6662 = 0 := by decide
theorem modelrow_agrees_6662 : modelRow 6662 = row 6662 := model_count_6662.trans row_count_6662.symm
def row_6663 : RowCertificate 6663 where
  first := false
  first_correct := by
    have composite : ¬ Nat.Prime (corner 6663 1) := by change ¬ Nat.Prime 177595603; exact PrattCertificates.composite_of_product (factor := 19) (cofactor := 9347137) rfl (by decide) (by decide)
    exact decide_eq_false composite
  second := false
  second_correct := by
    have composite : ¬ Nat.Prime (corner 6663 2) := by change ¬ Nat.Prime 177582277; exact PrattCertificates.composite_of_product (factor := 37) (cofactor := 4799521) rfl (by decide) (by decide)
    exact decide_eq_false composite
  third := false
  third_correct := by
    have composite : ¬ Nat.Prime (corner 6663 3) := by change ¬ Nat.Prime 177568951; exact PrattCertificates.composite_of_product (factor := 7) (cofactor := 25366993) rfl (by decide) (by decide)
    exact decide_eq_false composite
theorem row_count_6663 : row 6663 = 0 := (row_6663.checked).trans rfl
theorem model_count_6663 : modelRow 6663 = 0 := by decide
theorem modelrow_agrees_6663 : modelRow 6663 = row 6663 := model_count_6663.trans row_count_6663.symm
def row_6664 : RowCertificate 6664 where
  first := false
  first_correct := by
    have composite : ¬ Nat.Prime (corner 6664 1) := by change ¬ Nat.Prime 177648913; exact PrattCertificates.composite_of_product (factor := 13) (cofactor := 13665301) rfl (by decide) (by decide)
    exact decide_eq_false composite
  second := false
  second_correct := by
    have composite : ¬ Nat.Prime (corner 6664 2) := by change ¬ Nat.Prime 177635585; exact PrattCertificates.composite_of_product (factor := 5) (cofactor := 35527117) rfl (by decide) (by decide)
    exact decide_eq_false composite
  third := false
  third_correct := by
    have composite : ¬ Nat.Prime (corner 6664 3) := by change ¬ Nat.Prime 177622257; exact PrattCertificates.composite_of_product (factor := 3) (cofactor := 59207419) rfl (by decide) (by decide)
    exact decide_eq_false composite
theorem row_count_6664 : row 6664 = 0 := (row_6664.checked).trans rfl
theorem model_count_6664 : modelRow 6664 = 0 := by decide
theorem modelrow_agrees_6664 : modelRow 6664 = row 6664 := model_count_6664.trans row_count_6664.symm
def row_6665 : RowCertificate 6665 where
  first := false
  first_correct := by
    have composite : ¬ Nat.Prime (corner 6665 1) := by change ¬ Nat.Prime 177702231; exact PrattCertificates.composite_of_product (factor := 3) (cofactor := 59234077) rfl (by decide) (by decide)
    exact decide_eq_false composite
  second := false
  second_correct := by
    have composite : ¬ Nat.Prime (corner 6665 2) := by change ¬ Nat.Prime 177688901; exact PrattCertificates.composite_of_product (factor := 13) (cofactor := 13668377) rfl (by decide) (by decide)
    exact decide_eq_false composite
  third := true
  third_correct := by
    have prime : Nat.Prime (corner 6665 3) := by change Nat.Prime 177675571; exact ProjectEulerSolutions.P58PrimeNodes.p_177675571.prime
    exact decide_eq_true prime
theorem row_count_6665 : row 6665 = 1 := (row_6665.checked).trans rfl
theorem model_count_6665 : modelRow 6665 = 1 := by decide
theorem modelrow_agrees_6665 : modelRow 6665 = row 6665 := model_count_6665.trans row_count_6665.symm
def row_6666 : RowCertificate 6666 where
  first := false
  first_correct := by
    have composite : ¬ Nat.Prime (corner 6666 1) := by change ¬ Nat.Prime 177755557; exact PrattCertificates.composite_of_product (factor := 7) (cofactor := 25393651) rfl (by decide) (by decide)
    exact decide_eq_false composite
  second := false
  second_correct := by
    have composite : ¬ Nat.Prime (corner 6666 2) := by change ¬ Nat.Prime 177742225; exact PrattCertificates.composite_of_product (factor := 5) (cofactor := 35548445) rfl (by decide) (by decide)
    exact decide_eq_false composite
  third := true
  third_correct := by
    have prime : Nat.Prime (corner 6666 3) := by change Nat.Prime 177728893; exact ProjectEulerSolutions.P58PrimeNodes.p_177728893.prime
    exact decide_eq_true prime
theorem row_count_6666 : row 6666 = 1 := (row_6666.checked).trans rfl
theorem model_count_6666 : modelRow 6666 = 1 := by decide
theorem modelrow_agrees_6666 : modelRow 6666 = row 6666 := model_count_6666.trans row_count_6666.symm
def row_6667 : RowCertificate 6667 where
  first := false
  first_correct := by
    have composite : ¬ Nat.Prime (corner 6667 1) := by change ¬ Nat.Prime 177808891; exact PrattCertificates.composite_of_product (factor := 13) (cofactor := 13677607) rfl (by decide) (by decide)
    exact decide_eq_false composite
  second := false
  second_correct := by
    have composite : ¬ Nat.Prime (corner 6667 2) := by change ¬ Nat.Prime 177795557; exact PrattCertificates.composite_of_product (factor := 41) (cofactor := 4336477) rfl (by decide) (by decide)
    exact decide_eq_false composite
  third := false
  third_correct := by
    have composite : ¬ Nat.Prime (corner 6667 3) := by change ¬ Nat.Prime 177782223; exact PrattCertificates.composite_of_product (factor := 3) (cofactor := 59260741) rfl (by decide) (by decide)
    exact decide_eq_false composite
theorem row_count_6667 : row 6667 = 0 := (row_6667.checked).trans rfl
theorem model_count_6667 : modelRow 6667 = 0 := by decide
theorem modelrow_agrees_6667 : modelRow 6667 = row 6667 := model_count_6667.trans row_count_6667.symm
def row_6668 : RowCertificate 6668 where
  first := false
  first_correct := by
    have composite : ¬ Nat.Prime (corner 6668 1) := by change ¬ Nat.Prime 177862233; exact PrattCertificates.composite_of_product (factor := 3) (cofactor := 59287411) rfl (by decide) (by decide)
    exact decide_eq_false composite
  second := true
  second_correct := by
    have prime : Nat.Prime (corner 6668 2) := by change Nat.Prime 177848897; exact ProjectEulerSolutions.P58PrimeNodes.p_177848897.prime
    exact decide_eq_true prime
  third := false
  third_correct := by
    have composite : ¬ Nat.Prime (corner 6668 3) := by change ¬ Nat.Prime 177835561; exact PrattCertificates.composite_of_product (factor := 31) (cofactor := 5736631) rfl (by decide) (by decide)
    exact decide_eq_false composite
theorem row_count_6668 : row 6668 = 1 := (row_6668.checked).trans rfl
theorem model_count_6668 : modelRow 6668 = 1 := by decide
theorem modelrow_agrees_6668 : modelRow 6668 = row 6668 := model_count_6668.trans row_count_6668.symm
def row_6669 : RowCertificate 6669 where
  first := true
  first_correct := by
    have prime : Nat.Prime (corner 6669 1) := by change Nat.Prime 177915583; exact ProjectEulerSolutions.P58PrimeNodes.p_177915583.prime
    exact decide_eq_true prime
  second := false
  second_correct := by
    have composite : ¬ Nat.Prime (corner 6669 2) := by change ¬ Nat.Prime 177902245; exact PrattCertificates.composite_of_product (factor := 5) (cofactor := 35580449) rfl (by decide) (by decide)
    exact decide_eq_false composite
  third := false
  third_correct := by
    have composite : ¬ Nat.Prime (corner 6669 3) := by change ¬ Nat.Prime 177888907; exact PrattCertificates.composite_of_product (factor := 7) (cofactor := 25412701) rfl (by decide) (by decide)
    exact decide_eq_false composite
theorem row_count_6669 : row 6669 = 1 := (row_6669.checked).trans rfl
theorem model_count_6669 : modelRow 6669 = 1 := by decide
theorem modelrow_agrees_6669 : modelRow 6669 = row 6669 := model_count_6669.trans row_count_6669.symm
def row_6670 : RowCertificate 6670 where
  first := false
  first_correct := by
    have composite : ¬ Nat.Prime (corner 6670 1) := by change ¬ Nat.Prime 177968941; exact PrattCertificates.composite_of_product (factor := 223) (cofactor := 798067) rfl (by decide) (by decide)
    exact decide_eq_false composite
  second := false
  second_correct := by
    have composite : ¬ Nat.Prime (corner 6670 2) := by change ¬ Nat.Prime 177955601; exact PrattCertificates.composite_of_product (factor := 293) (cofactor := 607357) rfl (by decide) (by decide)
    exact decide_eq_false composite
  third := false
  third_correct := by
    have composite : ¬ Nat.Prime (corner 6670 3) := by change ¬ Nat.Prime 177942261; exact PrattCertificates.composite_of_product (factor := 3) (cofactor := 59314087) rfl (by decide) (by decide)
    exact decide_eq_false composite
theorem row_count_6670 : row 6670 = 0 := (row_6670.checked).trans rfl
theorem model_count_6670 : modelRow 6670 = 0 := by decide
theorem modelrow_agrees_6670 : modelRow 6670 = row 6670 := model_count_6670.trans row_count_6670.symm
def row_6671 : RowCertificate 6671 where
  first := false
  first_correct := by
    have composite : ¬ Nat.Prime (corner 6671 1) := by change ¬ Nat.Prime 178022307; exact PrattCertificates.composite_of_product (factor := 3) (cofactor := 59340769) rfl (by decide) (by decide)
    exact decide_eq_false composite
  second := false
  second_correct := by
    have composite : ¬ Nat.Prime (corner 6671 2) := by change ¬ Nat.Prime 178008965; exact PrattCertificates.composite_of_product (factor := 5) (cofactor := 35601793) rfl (by decide) (by decide)
    exact decide_eq_false composite
  third := false
  third_correct := by
    have composite : ¬ Nat.Prime (corner 6671 3) := by change ¬ Nat.Prime 177995623; exact PrattCertificates.composite_of_product (factor := 13) (cofactor := 13691971) rfl (by decide) (by decide)
    exact decide_eq_false composite
theorem row_count_6671 : row 6671 = 0 := (row_6671.checked).trans rfl
theorem model_count_6671 : modelRow 6671 = 0 := by decide
theorem modelrow_agrees_6671 : modelRow 6671 = row 6671 := model_count_6671.trans row_count_6671.symm
def row_6672 : RowCertificate 6672 where
  first := false
  first_correct := by
    have composite : ¬ Nat.Prime (corner 6672 1) := by change ¬ Nat.Prime 178075681; exact PrattCertificates.composite_of_product (factor := 7) (cofactor := 25439383) rfl (by decide) (by decide)
    exact decide_eq_false composite
  second := true
  second_correct := by
    have prime : Nat.Prime (corner 6672 2) := by change Nat.Prime 178062337; exact ProjectEulerSolutions.P58PrimeNodes.p_178062337.prime
    exact decide_eq_true prime
  third := false
  third_correct := by
    have composite : ¬ Nat.Prime (corner 6672 3) := by change ¬ Nat.Prime 178048993; exact PrattCertificates.composite_of_product (factor := 103) (cofactor := 1728631) rfl (by decide) (by decide)
    exact decide_eq_false composite
theorem row_count_6672 : row 6672 = 1 := (row_6672.checked).trans rfl
theorem model_count_6672 : modelRow 6672 = 1 := by decide
theorem modelrow_agrees_6672 : modelRow 6672 = row 6672 := model_count_6672.trans row_count_6672.symm
def row_6673 : RowCertificate 6673 where
  first := false
  first_correct := by
    have composite : ¬ Nat.Prime (corner 6673 1) := by change ¬ Nat.Prime 178129063; exact PrattCertificates.composite_of_product (factor := 7) (cofactor := 25447009) rfl (by decide) (by decide)
    exact decide_eq_false composite
  second := false
  second_correct := by
    have composite : ¬ Nat.Prime (corner 6673 2) := by change ¬ Nat.Prime 178115717; exact PrattCertificates.composite_of_product (factor := 13) (cofactor := 13701209) rfl (by decide) (by decide)
    exact decide_eq_false composite
  third := false
  third_correct := by
    have composite : ¬ Nat.Prime (corner 6673 3) := by change ¬ Nat.Prime 178102371; exact PrattCertificates.composite_of_product (factor := 3) (cofactor := 59367457) rfl (by decide) (by decide)
    exact decide_eq_false composite
theorem row_count_6673 : row 6673 = 0 := (row_6673.checked).trans rfl
theorem model_count_6673 : modelRow 6673 = 0 := by decide
theorem modelrow_agrees_6673 : modelRow 6673 = row 6673 := model_count_6673.trans row_count_6673.symm
def row_6674 : RowCertificate 6674 where
  first := false
  first_correct := by
    have composite : ¬ Nat.Prime (corner 6674 1) := by change ¬ Nat.Prime 178182453; exact PrattCertificates.composite_of_product (factor := 3) (cofactor := 59394151) rfl (by decide) (by decide)
    exact decide_eq_false composite
  second := false
  second_correct := by
    have composite : ¬ Nat.Prime (corner 6674 2) := by change ¬ Nat.Prime 178169105; exact PrattCertificates.composite_of_product (factor := 5) (cofactor := 35633821) rfl (by decide) (by decide)
    exact decide_eq_false composite
  third := false
  third_correct := by
    have composite : ¬ Nat.Prime (corner 6674 3) := by change ¬ Nat.Prime 178155757; exact PrattCertificates.composite_of_product (factor := 13) (cofactor := 13704289) rfl (by decide) (by decide)
    exact decide_eq_false composite
theorem row_count_6674 : row 6674 = 0 := (row_6674.checked).trans rfl
theorem model_count_6674 : modelRow 6674 = 0 := by decide
theorem modelrow_agrees_6674 : modelRow 6674 = row 6674 := model_count_6674.trans row_count_6674.symm
def row_6675 : RowCertificate 6675 where
  first := false
  first_correct := by
    have composite : ¬ Nat.Prime (corner 6675 1) := by change ¬ Nat.Prime 178235851; exact PrattCertificates.composite_of_product (factor := 73) (cofactor := 2441587) rfl (by decide) (by decide)
    exact decide_eq_false composite
  second := true
  second_correct := by
    have prime : Nat.Prime (corner 6675 2) := by change Nat.Prime 178222501; exact ProjectEulerSolutions.P58PrimeNodes.p_178222501.prime
    exact decide_eq_true prime
  third := false
  third_correct := by
    have composite : ¬ Nat.Prime (corner 6675 3) := by change ¬ Nat.Prime 178209151; exact PrattCertificates.composite_of_product (factor := 19) (cofactor := 9379429) rfl (by decide) (by decide)
    exact decide_eq_false composite
theorem row_count_6675 : row 6675 = 1 := (row_6675.checked).trans rfl
theorem model_count_6675 : modelRow 6675 = 1 := by decide
theorem modelrow_agrees_6675 : modelRow 6675 = row 6675 := model_count_6675.trans row_count_6675.symm
def row_6676 : RowCertificate 6676 where
  first := false
  first_correct := by
    have composite : ¬ Nat.Prime (corner 6676 1) := by change ¬ Nat.Prime 178289257; exact PrattCertificates.composite_of_product (factor := 8059) (cofactor := 22123) rfl (by decide) (by decide)
    exact decide_eq_false composite
  second := false
  second_correct := by
    have composite : ¬ Nat.Prime (corner 6676 2) := by change ¬ Nat.Prime 178275905; exact PrattCertificates.composite_of_product (factor := 5) (cofactor := 35655181) rfl (by decide) (by decide)
    exact decide_eq_false composite
  third := false
  third_correct := by
    have composite : ¬ Nat.Prime (corner 6676 3) := by change ¬ Nat.Prime 178262553; exact PrattCertificates.composite_of_product (factor := 3) (cofactor := 59420851) rfl (by decide) (by decide)
    exact decide_eq_false composite
theorem row_count_6676 : row 6676 = 0 := (row_6676.checked).trans rfl
theorem model_count_6676 : modelRow 6676 = 0 := by decide
theorem modelrow_agrees_6676 : modelRow 6676 = row 6676 := model_count_6676.trans row_count_6676.symm
def row_6677 : RowCertificate 6677 where
  first := false
  first_correct := by
    have composite : ¬ Nat.Prime (corner 6677 1) := by change ¬ Nat.Prime 178342671; exact PrattCertificates.composite_of_product (factor := 3) (cofactor := 59447557) rfl (by decide) (by decide)
    exact decide_eq_false composite
  second := false
  second_correct := by
    have composite : ¬ Nat.Prime (corner 6677 2) := by change ¬ Nat.Prime 178329317; exact PrattCertificates.composite_of_product (factor := 409) (cofactor := 436013) rfl (by decide) (by decide)
    exact decide_eq_false composite
  third := false
  third_correct := by
    have composite : ¬ Nat.Prime (corner 6677 3) := by change ¬ Nat.Prime 178315963; exact PrattCertificates.composite_of_product (factor := 7) (cofactor := 25473709) rfl (by decide) (by decide)
    exact decide_eq_false composite
theorem row_count_6677 : row 6677 = 0 := (row_6677.checked).trans rfl
theorem model_count_6677 : modelRow 6677 = 0 := by decide
theorem modelrow_agrees_6677 : modelRow 6677 = row 6677 := model_count_6677.trans row_count_6677.symm
def row_6678 : RowCertificate 6678 where
  first := true
  first_correct := by
    have prime : Nat.Prime (corner 6678 1) := by change Nat.Prime 178396093; exact ProjectEulerSolutions.P58PrimeNodes.p_178396093.prime
    exact decide_eq_true prime
  second := false
  second_correct := by
    have composite : ¬ Nat.Prime (corner 6678 2) := by change ¬ Nat.Prime 178382737; exact PrattCertificates.composite_of_product (factor := 13) (cofactor := 13721749) rfl (by decide) (by decide)
    exact decide_eq_false composite
  third := false
  third_correct := by
    have composite : ¬ Nat.Prime (corner 6678 3) := by change ¬ Nat.Prime 178369381; exact PrattCertificates.composite_of_product (factor := 31) (cofactor := 5753851) rfl (by decide) (by decide)
    exact decide_eq_false composite
theorem row_count_6678 : row 6678 = 1 := (row_6678.checked).trans rfl
theorem model_count_6678 : modelRow 6678 = 1 := by decide
theorem modelrow_agrees_6678 : modelRow 6678 = row 6678 := model_count_6678.trans row_count_6678.symm
def row_6679 : RowCertificate 6679 where
  first := false
  first_correct := by
    have composite : ¬ Nat.Prime (corner 6679 1) := by change ¬ Nat.Prime 178449523; exact PrattCertificates.composite_of_product (factor := 7) (cofactor := 25492789) rfl (by decide) (by decide)
    exact decide_eq_false composite
  second := false
  second_correct := by
    have composite : ¬ Nat.Prime (corner 6679 2) := by change ¬ Nat.Prime 178436165; exact PrattCertificates.composite_of_product (factor := 5) (cofactor := 35687233) rfl (by decide) (by decide)
    exact decide_eq_false composite
  third := false
  third_correct := by
    have composite : ¬ Nat.Prime (corner 6679 3) := by change ¬ Nat.Prime 178422807; exact PrattCertificates.composite_of_product (factor := 3) (cofactor := 59474269) rfl (by decide) (by decide)
    exact decide_eq_false composite
theorem row_count_6679 : row 6679 = 0 := (row_6679.checked).trans rfl
theorem model_count_6679 : modelRow 6679 = 0 := by decide
theorem modelrow_agrees_6679 : modelRow 6679 = row 6679 := model_count_6679.trans row_count_6679.symm
def row_6680 : RowCertificate 6680 where
  first := false
  first_correct := by
    have composite : ¬ Nat.Prime (corner 6680 1) := by change ¬ Nat.Prime 178502961; exact PrattCertificates.composite_of_product (factor := 3) (cofactor := 59500987) rfl (by decide) (by decide)
    exact decide_eq_false composite
  second := true
  second_correct := by
    have prime : Nat.Prime (corner 6680 2) := by change Nat.Prime 178489601; exact ProjectEulerSolutions.P58PrimeNodes.p_178489601.prime
    exact decide_eq_true prime
  third := true
  third_correct := by
    have prime : Nat.Prime (corner 6680 3) := by change Nat.Prime 178476241; exact ProjectEulerSolutions.P58PrimeNodes.p_178476241.prime
    exact decide_eq_true prime
theorem row_count_6680 : row 6680 = 2 := (row_6680.checked).trans rfl
theorem model_count_6680 : modelRow 6680 = 2 := by decide
theorem modelrow_agrees_6680 : modelRow 6680 = row 6680 := model_count_6680.trans row_count_6680.symm
def row_6681 : RowCertificate 6681 where
  first := false
  first_correct := by
    have composite : ¬ Nat.Prime (corner 6681 1) := by change ¬ Nat.Prime 178556407; exact PrattCertificates.composite_of_product (factor := 67) (cofactor := 2665021) rfl (by decide) (by decide)
    exact decide_eq_false composite
  second := false
  second_correct := by
    have composite : ¬ Nat.Prime (corner 6681 2) := by change ¬ Nat.Prime 178543045; exact PrattCertificates.composite_of_product (factor := 5) (cofactor := 35708609) rfl (by decide) (by decide)
    exact decide_eq_false composite
  third := false
  third_correct := by
    have composite : ¬ Nat.Prime (corner 6681 3) := by change ¬ Nat.Prime 178529683; exact PrattCertificates.composite_of_product (factor := 109) (cofactor := 1637887) rfl (by decide) (by decide)
    exact decide_eq_false composite
theorem row_count_6681 : row 6681 = 0 := (row_6681.checked).trans rfl
theorem model_count_6681 : modelRow 6681 = 0 := by decide
theorem modelrow_agrees_6681 : modelRow 6681 = row 6681 := model_count_6681.trans row_count_6681.symm
def row_6682 : RowCertificate 6682 where
  first := false
  first_correct := by
    have composite : ¬ Nat.Prime (corner 6682 1) := by change ¬ Nat.Prime 178609861; exact PrattCertificates.composite_of_product (factor := 19) (cofactor := 9400519) rfl (by decide) (by decide)
    exact decide_eq_false composite
  second := false
  second_correct := by
    have composite : ¬ Nat.Prime (corner 6682 2) := by change ¬ Nat.Prime 178596497; exact PrattCertificates.composite_of_product (factor := 97) (cofactor := 1841201) rfl (by decide) (by decide)
    exact decide_eq_false composite
  third := false
  third_correct := by
    have composite : ¬ Nat.Prime (corner 6682 3) := by change ¬ Nat.Prime 178583133; exact PrattCertificates.composite_of_product (factor := 3) (cofactor := 59527711) rfl (by decide) (by decide)
    exact decide_eq_false composite
theorem row_count_6682 : row 6682 = 0 := (row_6682.checked).trans rfl
theorem model_count_6682 : modelRow 6682 = 0 := by decide
theorem modelrow_agrees_6682 : modelRow 6682 = row 6682 := model_count_6682.trans row_count_6682.symm
def row_6683 : RowCertificate 6683 where
  first := false
  first_correct := by
    have composite : ¬ Nat.Prime (corner 6683 1) := by change ¬ Nat.Prime 178663323; exact PrattCertificates.composite_of_product (factor := 3) (cofactor := 59554441) rfl (by decide) (by decide)
    exact decide_eq_false composite
  second := false
  second_correct := by
    have composite : ¬ Nat.Prime (corner 6683 2) := by change ¬ Nat.Prime 178649957; exact PrattCertificates.composite_of_product (factor := 17) (cofactor := 10508821) rfl (by decide) (by decide)
    exact decide_eq_false composite
  third := false
  third_correct := by
    have composite : ¬ Nat.Prime (corner 6683 3) := by change ¬ Nat.Prime 178636591; exact PrattCertificates.composite_of_product (factor := 7) (cofactor := 25519513) rfl (by decide) (by decide)
    exact decide_eq_false composite
theorem row_count_6683 : row 6683 = 0 := (row_6683.checked).trans rfl
theorem model_count_6683 : modelRow 6683 = 0 := by decide
theorem modelrow_agrees_6683 : modelRow 6683 = row 6683 := model_count_6683.trans row_count_6683.symm
def row_6684 : RowCertificate 6684 where
  first := false
  first_correct := by
    have composite : ¬ Nat.Prime (corner 6684 1) := by change ¬ Nat.Prime 178716793; exact PrattCertificates.composite_of_product (factor := 19) (cofactor := 9406147) rfl (by decide) (by decide)
    exact decide_eq_false composite
  second := false
  second_correct := by
    have composite : ¬ Nat.Prime (corner 6684 2) := by change ¬ Nat.Prime 178703425; exact PrattCertificates.composite_of_product (factor := 5) (cofactor := 35740685) rfl (by decide) (by decide)
    exact decide_eq_false composite
  third := false
  third_correct := by
    have composite : ¬ Nat.Prime (corner 6684 3) := by change ¬ Nat.Prime 178690057; exact PrattCertificates.composite_of_product (factor := 7) (cofactor := 25527151) rfl (by decide) (by decide)
    exact decide_eq_false composite
theorem row_count_6684 : row 6684 = 0 := (row_6684.checked).trans rfl
theorem model_count_6684 : modelRow 6684 = 0 := by decide
theorem modelrow_agrees_6684 : modelRow 6684 = row 6684 := model_count_6684.trans row_count_6684.symm
def row_6685 : RowCertificate 6685 where
  first := false
  first_correct := by
    have composite : ¬ Nat.Prime (corner 6685 1) := by change ¬ Nat.Prime 178770271; exact PrattCertificates.composite_of_product (factor := 67) (cofactor := 2668213) rfl (by decide) (by decide)
    exact decide_eq_false composite
  second := false
  second_correct := by
    have composite : ¬ Nat.Prime (corner 6685 2) := by change ¬ Nat.Prime 178756901; exact PrattCertificates.composite_of_product (factor := 61) (cofactor := 2930441) rfl (by decide) (by decide)
    exact decide_eq_false composite
  third := false
  third_correct := by
    have composite : ¬ Nat.Prime (corner 6685 3) := by change ¬ Nat.Prime 178743531; exact PrattCertificates.composite_of_product (factor := 3) (cofactor := 59581177) rfl (by decide) (by decide)
    exact decide_eq_false composite
theorem row_count_6685 : row 6685 = 0 := (row_6685.checked).trans rfl
theorem model_count_6685 : modelRow 6685 = 0 := by decide
theorem modelrow_agrees_6685 : modelRow 6685 = row 6685 := model_count_6685.trans row_count_6685.symm
def row_6686 : RowCertificate 6686 where
  first := false
  first_correct := by
    have composite : ¬ Nat.Prime (corner 6686 1) := by change ¬ Nat.Prime 178823757; exact PrattCertificates.composite_of_product (factor := 3) (cofactor := 59607919) rfl (by decide) (by decide)
    exact decide_eq_false composite
  second := false
  second_correct := by
    have composite : ¬ Nat.Prime (corner 6686 2) := by change ¬ Nat.Prime 178810385; exact PrattCertificates.composite_of_product (factor := 5) (cofactor := 35762077) rfl (by decide) (by decide)
    exact decide_eq_false composite
  third := true
  third_correct := by
    have prime : Nat.Prime (corner 6686 3) := by change Nat.Prime 178797013; exact ProjectEulerSolutions.P58PrimeNodes.p_178797013.prime
    exact decide_eq_true prime
theorem row_count_6686 : row 6686 = 1 := (row_6686.checked).trans rfl
theorem model_count_6686 : modelRow 6686 = 1 := by decide
theorem modelrow_agrees_6686 : modelRow 6686 = row 6686 := model_count_6686.trans row_count_6686.symm
def row_6687 : RowCertificate 6687 where
  first := false
  first_correct := by
    have composite : ¬ Nat.Prime (corner 6687 1) := by change ¬ Nat.Prime 178877251; exact PrattCertificates.composite_of_product (factor := 7) (cofactor := 25553893) rfl (by decide) (by decide)
    exact decide_eq_false composite
  second := false
  second_correct := by
    have composite : ¬ Nat.Prime (corner 6687 2) := by change ¬ Nat.Prime 178863877; exact PrattCertificates.composite_of_product (factor := 109) (cofactor := 1640953) rfl (by decide) (by decide)
    exact decide_eq_false composite
  third := false
  third_correct := by
    have composite : ¬ Nat.Prime (corner 6687 3) := by change ¬ Nat.Prime 178850503; exact PrattCertificates.composite_of_product (factor := 13) (cofactor := 13757731) rfl (by decide) (by decide)
    exact decide_eq_false composite
theorem row_count_6687 : row 6687 = 0 := (row_6687.checked).trans rfl
theorem model_count_6687 : modelRow 6687 = 0 := by decide
theorem modelrow_agrees_6687 : modelRow 6687 = row 6687 := model_count_6687.trans row_count_6687.symm
def row_6688 : RowCertificate 6688 where
  first := false
  first_correct := by
    have composite : ¬ Nat.Prime (corner 6688 1) := by change ¬ Nat.Prime 178930753; exact PrattCertificates.composite_of_product (factor := 229) (cofactor := 781357) rfl (by decide) (by decide)
    exact decide_eq_false composite
  second := true
  second_correct := by
    have prime : Nat.Prime (corner 6688 2) := by change Nat.Prime 178917377; exact ProjectEulerSolutions.P58PrimeNodes.p_178917377.prime
    exact decide_eq_true prime
  third := false
  third_correct := by
    have composite : ¬ Nat.Prime (corner 6688 3) := by change ¬ Nat.Prime 178904001; exact PrattCertificates.composite_of_product (factor := 3) (cofactor := 59634667) rfl (by decide) (by decide)
    exact decide_eq_false composite
theorem row_count_6688 : row 6688 = 1 := (row_6688.checked).trans rfl
theorem model_count_6688 : modelRow 6688 = 1 := by decide
theorem modelrow_agrees_6688 : modelRow 6688 = row 6688 := model_count_6688.trans row_count_6688.symm
def row_6689 : RowCertificate 6689 where
  first := false
  first_correct := by
    have composite : ¬ Nat.Prime (corner 6689 1) := by change ¬ Nat.Prime 178984263; exact PrattCertificates.composite_of_product (factor := 3) (cofactor := 59661421) rfl (by decide) (by decide)
    exact decide_eq_false composite
  second := false
  second_correct := by
    have composite : ¬ Nat.Prime (corner 6689 2) := by change ¬ Nat.Prime 178970885; exact PrattCertificates.composite_of_product (factor := 5) (cofactor := 35794177) rfl (by decide) (by decide)
    exact decide_eq_false composite
  third := true
  third_correct := by
    have prime : Nat.Prime (corner 6689 3) := by change Nat.Prime 178957507; exact ProjectEulerSolutions.P58PrimeNodes.p_178957507.prime
    exact decide_eq_true prime
theorem row_count_6689 : row 6689 = 1 := (row_6689.checked).trans rfl
theorem model_count_6689 : modelRow 6689 = 1 := by decide
theorem modelrow_agrees_6689 : modelRow 6689 = row 6689 := model_count_6689.trans row_count_6689.symm
def row_6690 : RowCertificate 6690 where
  first := false
  first_correct := by
    have composite : ¬ Nat.Prime (corner 6690 1) := by change ¬ Nat.Prime 179037781; exact PrattCertificates.composite_of_product (factor := 13) (cofactor := 13772137) rfl (by decide) (by decide)
    exact decide_eq_false composite
  second := false
  second_correct := by
    have composite : ¬ Nat.Prime (corner 6690 2) := by change ¬ Nat.Prime 179024401; exact PrattCertificates.composite_of_product (factor := 257) (cofactor := 696593) rfl (by decide) (by decide)
    exact decide_eq_false composite
  third := false
  third_correct := by
    have composite : ¬ Nat.Prime (corner 6690 3) := by change ¬ Nat.Prime 179011021; exact PrattCertificates.composite_of_product (factor := 7) (cofactor := 25573003) rfl (by decide) (by decide)
    exact decide_eq_false composite
theorem row_count_6690 : row 6690 = 0 := (row_6690.checked).trans rfl
theorem model_count_6690 : modelRow 6690 = 0 := by decide
theorem modelrow_agrees_6690 : modelRow 6690 = row 6690 := model_count_6690.trans row_count_6690.symm
def row_6691 : RowCertificate 6691 where
  first := false
  first_correct := by
    have composite : ¬ Nat.Prime (corner 6691 1) := by change ¬ Nat.Prime 179091307; exact PrattCertificates.composite_of_product (factor := 691) (cofactor := 259177) rfl (by decide) (by decide)
    exact decide_eq_false composite
  second := false
  second_correct := by
    have composite : ¬ Nat.Prime (corner 6691 2) := by change ¬ Nat.Prime 179077925; exact PrattCertificates.composite_of_product (factor := 5) (cofactor := 35815585) rfl (by decide) (by decide)
    exact decide_eq_false composite
  third := false
  third_correct := by
    have composite : ¬ Nat.Prime (corner 6691 3) := by change ¬ Nat.Prime 179064543; exact PrattCertificates.composite_of_product (factor := 3) (cofactor := 59688181) rfl (by decide) (by decide)
    exact decide_eq_false composite
theorem row_count_6691 : row 6691 = 0 := (row_6691.checked).trans rfl
theorem model_count_6691 : modelRow 6691 = 0 := by decide
theorem modelrow_agrees_6691 : modelRow 6691 = row 6691 := model_count_6691.trans row_count_6691.symm
def row_6692 : RowCertificate 6692 where
  first := false
  first_correct := by
    have composite : ¬ Nat.Prime (corner 6692 1) := by change ¬ Nat.Prime 179144841; exact PrattCertificates.composite_of_product (factor := 3) (cofactor := 59714947) rfl (by decide) (by decide)
    exact decide_eq_false composite
  second := false
  second_correct := by
    have composite : ¬ Nat.Prime (corner 6692 2) := by change ¬ Nat.Prime 179131457; exact PrattCertificates.composite_of_product (factor := 89) (cofactor := 2012713) rfl (by decide) (by decide)
    exact decide_eq_false composite
  third := false
  third_correct := by
    have composite : ¬ Nat.Prime (corner 6692 3) := by change ¬ Nat.Prime 179118073; exact PrattCertificates.composite_of_product (factor := 19) (cofactor := 9427267) rfl (by decide) (by decide)
    exact decide_eq_false composite
theorem row_count_6692 : row 6692 = 0 := (row_6692.checked).trans rfl
theorem model_count_6692 : modelRow 6692 = 0 := by decide
theorem modelrow_agrees_6692 : modelRow 6692 = row 6692 := model_count_6692.trans row_count_6692.symm
def row_6693 : RowCertificate 6693 where
  first := false
  first_correct := by
    have composite : ¬ Nat.Prime (corner 6693 1) := by change ¬ Nat.Prime 179198383; exact PrattCertificates.composite_of_product (factor := 7) (cofactor := 25599769) rfl (by decide) (by decide)
    exact decide_eq_false composite
  second := false
  second_correct := by
    have composite : ¬ Nat.Prime (corner 6693 2) := by change ¬ Nat.Prime 179184997; exact PrattCertificates.composite_of_product (factor := 29) (cofactor := 6178793) rfl (by decide) (by decide)
    exact decide_eq_false composite
  third := false
  third_correct := by
    have composite : ¬ Nat.Prime (corner 6693 3) := by change ¬ Nat.Prime 179171611; exact PrattCertificates.composite_of_product (factor := 1483) (cofactor := 120817) rfl (by decide) (by decide)
    exact decide_eq_false composite
theorem row_count_6693 : row 6693 = 0 := (row_6693.checked).trans rfl
theorem model_count_6693 : modelRow 6693 = 0 := by decide
theorem modelrow_agrees_6693 : modelRow 6693 = row 6693 := model_count_6693.trans row_count_6693.symm
def row_6694 : RowCertificate 6694 where
  first := false
  first_correct := by
    have composite : ¬ Nat.Prime (corner 6694 1) := by change ¬ Nat.Prime 179251933; exact PrattCertificates.composite_of_product (factor := 7) (cofactor := 25607419) rfl (by decide) (by decide)
    exact decide_eq_false composite
  second := false
  second_correct := by
    have composite : ¬ Nat.Prime (corner 6694 2) := by change ¬ Nat.Prime 179238545; exact PrattCertificates.composite_of_product (factor := 5) (cofactor := 35847709) rfl (by decide) (by decide)
    exact decide_eq_false composite
  third := false
  third_correct := by
    have composite : ¬ Nat.Prime (corner 6694 3) := by change ¬ Nat.Prime 179225157; exact PrattCertificates.composite_of_product (factor := 3) (cofactor := 59741719) rfl (by decide) (by decide)
    exact decide_eq_false composite
theorem row_count_6694 : row 6694 = 0 := (row_6694.checked).trans rfl
theorem model_count_6694 : modelRow 6694 = 0 := by decide
theorem modelrow_agrees_6694 : modelRow 6694 = row 6694 := model_count_6694.trans row_count_6694.symm
def row_6695 : RowCertificate 6695 where
  first := false
  first_correct := by
    have composite : ¬ Nat.Prime (corner 6695 1) := by change ¬ Nat.Prime 179305491; exact PrattCertificates.composite_of_product (factor := 3) (cofactor := 59768497) rfl (by decide) (by decide)
    exact decide_eq_false composite
  second := true
  second_correct := by
    have prime : Nat.Prime (corner 6695 2) := by change Nat.Prime 179292101; exact ProjectEulerSolutions.P58PrimeNodes.p_179292101.prime
    exact decide_eq_true prime
  third := false
  third_correct := by
    have composite : ¬ Nat.Prime (corner 6695 3) := by change ¬ Nat.Prime 179278711; exact PrattCertificates.composite_of_product (factor := 829) (cofactor := 216259) rfl (by decide) (by decide)
    exact decide_eq_false composite
theorem row_count_6695 : row 6695 = 1 := (row_6695.checked).trans rfl
theorem model_count_6695 : modelRow 6695 = 1 := by decide
theorem modelrow_agrees_6695 : modelRow 6695 = row 6695 := model_count_6695.trans row_count_6695.symm
def row_6696 : RowCertificate 6696 where
  first := true
  first_correct := by
    have prime : Nat.Prime (corner 6696 1) := by change Nat.Prime 179359057; exact ProjectEulerSolutions.P58PrimeNodes.p_179359057.prime
    exact decide_eq_true prime
  second := false
  second_correct := by
    have composite : ¬ Nat.Prime (corner 6696 2) := by change ¬ Nat.Prime 179345665; exact PrattCertificates.composite_of_product (factor := 5) (cofactor := 35869133) rfl (by decide) (by decide)
    exact decide_eq_false composite
  third := false
  third_correct := by
    have composite : ¬ Nat.Prime (corner 6696 3) := by change ¬ Nat.Prime 179332273; exact PrattCertificates.composite_of_product (factor := 8017) (cofactor := 22369) rfl (by decide) (by decide)
    exact decide_eq_false composite
theorem row_count_6696 : row 6696 = 1 := (row_6696.checked).trans rfl
theorem model_count_6696 : modelRow 6696 = 1 := by decide
theorem modelrow_agrees_6696 : modelRow 6696 = row 6696 := model_count_6696.trans row_count_6696.symm
def row_6697 : RowCertificate 6697 where
  first := false
  first_correct := by
    have composite : ¬ Nat.Prime (corner 6697 1) := by change ¬ Nat.Prime 179412631; exact PrattCertificates.composite_of_product (factor := 6121) (cofactor := 29311) rfl (by decide) (by decide)
    exact decide_eq_false composite
  second := true
  second_correct := by
    have prime : Nat.Prime (corner 6697 2) := by change Nat.Prime 179399237; exact ProjectEulerSolutions.P58PrimeNodes.p_179399237.prime
    exact decide_eq_true prime
  third := false
  third_correct := by
    have composite : ¬ Nat.Prime (corner 6697 3) := by change ¬ Nat.Prime 179385843; exact PrattCertificates.composite_of_product (factor := 3) (cofactor := 59795281) rfl (by decide) (by decide)
    exact decide_eq_false composite
theorem row_count_6697 : row 6697 = 1 := (row_6697.checked).trans rfl
theorem model_count_6697 : modelRow 6697 = 1 := by decide
theorem modelrow_agrees_6697 : modelRow 6697 = row 6697 := model_count_6697.trans row_count_6697.symm
def row_6698 : RowCertificate 6698 where
  first := false
  first_correct := by
    have composite : ¬ Nat.Prime (corner 6698 1) := by change ¬ Nat.Prime 179466213; exact PrattCertificates.composite_of_product (factor := 3) (cofactor := 59822071) rfl (by decide) (by decide)
    exact decide_eq_false composite
  second := false
  second_correct := by
    have composite : ¬ Nat.Prime (corner 6698 2) := by change ¬ Nat.Prime 179452817; exact PrattCertificates.composite_of_product (factor := 2557) (cofactor := 70181) rfl (by decide) (by decide)
    exact decide_eq_false composite
  third := false
  third_correct := by
    have composite : ¬ Nat.Prime (corner 6698 3) := by change ¬ Nat.Prime 179439421; exact PrattCertificates.composite_of_product (factor := 7) (cofactor := 25634203) rfl (by decide) (by decide)
    exact decide_eq_false composite
theorem row_count_6698 : row 6698 = 0 := (row_6698.checked).trans rfl
theorem model_count_6698 : modelRow 6698 = 0 := by decide
theorem modelrow_agrees_6698 : modelRow 6698 = row 6698 := model_count_6698.trans row_count_6698.symm
def row_6699 : RowCertificate 6699 where
  first := true
  first_correct := by
    have prime : Nat.Prime (corner 6699 1) := by change Nat.Prime 179519803; exact ProjectEulerSolutions.P58PrimeNodes.p_179519803.prime
    exact decide_eq_true prime
  second := false
  second_correct := by
    have composite : ¬ Nat.Prime (corner 6699 2) := by change ¬ Nat.Prime 179506405; exact PrattCertificates.composite_of_product (factor := 5) (cofactor := 35901281) rfl (by decide) (by decide)
    exact decide_eq_false composite
  third := false
  third_correct := by
    have composite : ¬ Nat.Prime (corner 6699 3) := by change ¬ Nat.Prime 179493007; exact PrattCertificates.composite_of_product (factor := 31) (cofactor := 5790097) rfl (by decide) (by decide)
    exact decide_eq_false composite
theorem row_count_6699 : row 6699 = 1 := (row_6699.checked).trans rfl
theorem model_count_6699 : modelRow 6699 = 1 := by decide
theorem modelrow_agrees_6699 : modelRow 6699 = row 6699 := model_count_6699.trans row_count_6699.symm
def row_6700 : RowCertificate 6700 where
  first := false
  first_correct := by
    have composite : ¬ Nat.Prime (corner 6700 1) := by change ¬ Nat.Prime 179573401; exact PrattCertificates.composite_of_product (factor := 7) (cofactor := 25653343) rfl (by decide) (by decide)
    exact decide_eq_false composite
  second := false
  second_correct := by
    have composite : ¬ Nat.Prime (corner 6700 2) := by change ¬ Nat.Prime 179560001; exact PrattCertificates.composite_of_product (factor := 17) (cofactor := 10562353) rfl (by decide) (by decide)
    exact decide_eq_false composite
  third := false
  third_correct := by
    have composite : ¬ Nat.Prime (corner 6700 3) := by change ¬ Nat.Prime 179546601; exact PrattCertificates.composite_of_product (factor := 3) (cofactor := 59848867) rfl (by decide) (by decide)
    exact decide_eq_false composite
theorem row_count_6700 : row 6700 = 0 := (row_6700.checked).trans rfl
theorem model_count_6700 : modelRow 6700 = 0 := by decide
theorem modelrow_agrees_6700 : modelRow 6700 = row 6700 := model_count_6700.trans row_count_6700.symm
def row_6701 : RowCertificate 6701 where
  first := false
  first_correct := by
    have composite : ¬ Nat.Prime (corner 6701 1) := by change ¬ Nat.Prime 179627007; exact PrattCertificates.composite_of_product (factor := 3) (cofactor := 59875669) rfl (by decide) (by decide)
    exact decide_eq_false composite
  second := false
  second_correct := by
    have composite : ¬ Nat.Prime (corner 6701 2) := by change ¬ Nat.Prime 179613605; exact PrattCertificates.composite_of_product (factor := 5) (cofactor := 35922721) rfl (by decide) (by decide)
    exact decide_eq_false composite
  third := false
  third_correct := by
    have composite : ¬ Nat.Prime (corner 6701 3) := by change ¬ Nat.Prime 179600203; exact PrattCertificates.composite_of_product (factor := 193) (cofactor := 930571) rfl (by decide) (by decide)
    exact decide_eq_false composite
theorem row_count_6701 : row 6701 = 0 := (row_6701.checked).trans rfl
theorem model_count_6701 : modelRow 6701 = 0 := by decide
theorem modelrow_agrees_6701 : modelRow 6701 = row 6701 := model_count_6701.trans row_count_6701.symm
def row_6702 : RowCertificate 6702 where
  first := false
  first_correct := by
    have composite : ¬ Nat.Prime (corner 6702 1) := by change ¬ Nat.Prime 179680621; exact PrattCertificates.composite_of_product (factor := 37) (cofactor := 4856233) rfl (by decide) (by decide)
    exact decide_eq_false composite
  second := false
  second_correct := by
    have composite : ¬ Nat.Prime (corner 6702 2) := by change ¬ Nat.Prime 179667217; exact PrattCertificates.composite_of_product (factor := 229) (cofactor := 784573) rfl (by decide) (by decide)
    exact decide_eq_false composite
  third := false
  third_correct := by
    have composite : ¬ Nat.Prime (corner 6702 3) := by change ¬ Nat.Prime 179653813; exact PrattCertificates.composite_of_product (factor := 487) (cofactor := 368899) rfl (by decide) (by decide)
    exact decide_eq_false composite
theorem row_count_6702 : row 6702 = 0 := (row_6702.checked).trans rfl
theorem model_count_6702 : modelRow 6702 = 0 := by decide
theorem modelrow_agrees_6702 : modelRow 6702 = row 6702 := model_count_6702.trans row_count_6702.symm
def row_6703 : RowCertificate 6703 where
  first := false
  first_correct := by
    have composite : ¬ Nat.Prime (corner 6703 1) := by change ¬ Nat.Prime 179734243; exact PrattCertificates.composite_of_product (factor := 13) (cofactor := 13825711) rfl (by decide) (by decide)
    exact decide_eq_false composite
  second := false
  second_correct := by
    have composite : ¬ Nat.Prime (corner 6703 2) := by change ¬ Nat.Prime 179720837; exact PrattCertificates.composite_of_product (factor := 8069) (cofactor := 22273) rfl (by decide) (by decide)
    exact decide_eq_false composite
  third := false
  third_correct := by
    have composite : ¬ Nat.Prime (corner 6703 3) := by change ¬ Nat.Prime 179707431; exact PrattCertificates.composite_of_product (factor := 3) (cofactor := 59902477) rfl (by decide) (by decide)
    exact decide_eq_false composite
theorem row_count_6703 : row 6703 = 0 := (row_6703.checked).trans rfl
theorem model_count_6703 : modelRow 6703 = 0 := by decide
theorem modelrow_agrees_6703 : modelRow 6703 = row 6703 := model_count_6703.trans row_count_6703.symm
def row_6704 : RowCertificate 6704 where
  first := false
  first_correct := by
    have composite : ¬ Nat.Prime (corner 6704 1) := by change ¬ Nat.Prime 179787873; exact PrattCertificates.composite_of_product (factor := 3) (cofactor := 59929291) rfl (by decide) (by decide)
    exact decide_eq_false composite
  second := false
  second_correct := by
    have composite : ¬ Nat.Prime (corner 6704 2) := by change ¬ Nat.Prime 179774465; exact PrattCertificates.composite_of_product (factor := 5) (cofactor := 35954893) rfl (by decide) (by decide)
    exact decide_eq_false composite
  third := false
  third_correct := by
    have composite : ¬ Nat.Prime (corner 6704 3) := by change ¬ Nat.Prime 179761057; exact PrattCertificates.composite_of_product (factor := 7) (cofactor := 25680151) rfl (by decide) (by decide)
    exact decide_eq_false composite
theorem row_count_6704 : row 6704 = 0 := (row_6704.checked).trans rfl
theorem model_count_6704 : modelRow 6704 = 0 := by decide
theorem modelrow_agrees_6704 : modelRow 6704 = row 6704 := model_count_6704.trans row_count_6704.symm
def row_6705 : RowCertificate 6705 where
  first := true
  first_correct := by
    have prime : Nat.Prime (corner 6705 1) := by change Nat.Prime 179841511; exact ProjectEulerSolutions.P58PrimeNodes.p_179841511.prime
    exact decide_eq_true prime
  second := false
  second_correct := by
    have composite : ¬ Nat.Prime (corner 6705 2) := by change ¬ Nat.Prime 179828101; exact PrattCertificates.composite_of_product (factor := 29) (cofactor := 6200969) rfl (by decide) (by decide)
    exact decide_eq_false composite
  third := false
  third_correct := by
    have composite : ¬ Nat.Prime (corner 6705 3) := by change ¬ Nat.Prime 179814691; exact PrattCertificates.composite_of_product (factor := 7) (cofactor := 25687813) rfl (by decide) (by decide)
    exact decide_eq_false composite
theorem row_count_6705 : row 6705 = 1 := (row_6705.checked).trans rfl
theorem model_count_6705 : modelRow 6705 = 1 := by decide
theorem modelrow_agrees_6705 : modelRow 6705 = row 6705 := model_count_6705.trans row_count_6705.symm
def row_6706 : RowCertificate 6706 where
  first := false
  first_correct := by
    have composite : ¬ Nat.Prime (corner 6706 1) := by change ¬ Nat.Prime 179895157; exact PrattCertificates.composite_of_product (factor := 13) (cofactor := 13838089) rfl (by decide) (by decide)
    exact decide_eq_false composite
  second := false
  second_correct := by
    have composite : ¬ Nat.Prime (corner 6706 2) := by change ¬ Nat.Prime 179881745; exact PrattCertificates.composite_of_product (factor := 5) (cofactor := 35976349) rfl (by decide) (by decide)
    exact decide_eq_false composite
  third := false
  third_correct := by
    have composite : ¬ Nat.Prime (corner 6706 3) := by change ¬ Nat.Prime 179868333; exact PrattCertificates.composite_of_product (factor := 3) (cofactor := 59956111) rfl (by decide) (by decide)
    exact decide_eq_false composite
theorem row_count_6706 : row 6706 = 0 := (row_6706.checked).trans rfl
theorem model_count_6706 : modelRow 6706 = 0 := by decide
theorem modelrow_agrees_6706 : modelRow 6706 = row 6706 := model_count_6706.trans row_count_6706.symm
def row_6707 : RowCertificate 6707 where
  first := false
  first_correct := by
    have composite : ¬ Nat.Prime (corner 6707 1) := by change ¬ Nat.Prime 179948811; exact PrattCertificates.composite_of_product (factor := 3) (cofactor := 59982937) rfl (by decide) (by decide)
    exact decide_eq_false composite
  second := false
  second_correct := by
    have composite : ¬ Nat.Prime (corner 6707 2) := by change ¬ Nat.Prime 179935397; exact PrattCertificates.composite_of_product (factor := 173) (cofactor := 1040089) rfl (by decide) (by decide)
    exact decide_eq_false composite
  third := false
  third_correct := by
    have composite : ¬ Nat.Prime (corner 6707 3) := by change ¬ Nat.Prime 179921983; exact PrattCertificates.composite_of_product (factor := 4597) (cofactor := 39139) rfl (by decide) (by decide)
    exact decide_eq_false composite
theorem row_count_6707 : row 6707 = 0 := (row_6707.checked).trans rfl
theorem model_count_6707 : modelRow 6707 = 0 := by decide
theorem modelrow_agrees_6707 : modelRow 6707 = row 6707 := model_count_6707.trans row_count_6707.symm
def row_6708 : RowCertificate 6708 where
  first := false
  first_correct := by
    have composite : ¬ Nat.Prime (corner 6708 1) := by change ¬ Nat.Prime 180002473; exact PrattCertificates.composite_of_product (factor := 7) (cofactor := 25714639) rfl (by decide) (by decide)
    exact decide_eq_false composite
  second := false
  second_correct := by
    have composite : ¬ Nat.Prime (corner 6708 2) := by change ¬ Nat.Prime 179989057; exact PrattCertificates.composite_of_product (factor := 41) (cofactor := 4389977) rfl (by decide) (by decide)
    exact decide_eq_false composite
  third := true
  third_correct := by
    have prime : Nat.Prime (corner 6708 3) := by change Nat.Prime 179975641; exact ProjectEulerSolutions.P58PrimeNodes.p_179975641.prime
    exact decide_eq_true prime
theorem row_count_6708 : row 6708 = 1 := (row_6708.checked).trans rfl
theorem model_count_6708 : modelRow 6708 = 1 := by decide
theorem modelrow_agrees_6708 : modelRow 6708 = row 6708 := model_count_6708.trans row_count_6708.symm
def row_6709 : RowCertificate 6709 where
  first := false
  first_correct := by
    have composite : ¬ Nat.Prime (corner 6709 1) := by change ¬ Nat.Prime 180056143; exact PrattCertificates.composite_of_product (factor := 8161) (cofactor := 22063) rfl (by decide) (by decide)
    exact decide_eq_false composite
  second := false
  second_correct := by
    have composite : ¬ Nat.Prime (corner 6709 2) := by change ¬ Nat.Prime 180042725; exact PrattCertificates.composite_of_product (factor := 5) (cofactor := 36008545) rfl (by decide) (by decide)
    exact decide_eq_false composite
  third := false
  third_correct := by
    have composite : ¬ Nat.Prime (corner 6709 3) := by change ¬ Nat.Prime 180029307; exact PrattCertificates.composite_of_product (factor := 3) (cofactor := 60009769) rfl (by decide) (by decide)
    exact decide_eq_false composite
theorem row_count_6709 : row 6709 = 0 := (row_6709.checked).trans rfl
theorem model_count_6709 : modelRow 6709 = 0 := by decide
theorem modelrow_agrees_6709 : modelRow 6709 = row 6709 := model_count_6709.trans row_count_6709.symm
def row_6710 : RowCertificate 6710 where
  first := false
  first_correct := by
    have composite : ¬ Nat.Prime (corner 6710 1) := by change ¬ Nat.Prime 180109821; exact PrattCertificates.composite_of_product (factor := 3) (cofactor := 60036607) rfl (by decide) (by decide)
    exact decide_eq_false composite
  second := true
  second_correct := by
    have prime : Nat.Prime (corner 6710 2) := by change Nat.Prime 180096401; exact ProjectEulerSolutions.P58PrimeNodes.p_180096401.prime
    exact decide_eq_true prime
  third := false
  third_correct := by
    have composite : ¬ Nat.Prime (corner 6710 3) := by change ¬ Nat.Prime 180082981; exact PrattCertificates.composite_of_product (factor := 13) (cofactor := 13852537) rfl (by decide) (by decide)
    exact decide_eq_false composite
theorem row_count_6710 : row 6710 = 1 := (row_6710.checked).trans rfl
theorem model_count_6710 : modelRow 6710 = 1 := by decide
theorem modelrow_agrees_6710 : modelRow 6710 = row 6710 := model_count_6710.trans row_count_6710.symm
def row_6711 : RowCertificate 6711 where
  first := false
  first_correct := by
    have composite : ¬ Nat.Prime (corner 6711 1) := by change ¬ Nat.Prime 180163507; exact PrattCertificates.composite_of_product (factor := 43) (cofactor := 4189849) rfl (by decide) (by decide)
    exact decide_eq_false composite
  second := false
  second_correct := by
    have composite : ¬ Nat.Prime (corner 6711 2) := by change ¬ Nat.Prime 180150085; exact PrattCertificates.composite_of_product (factor := 5) (cofactor := 36030017) rfl (by decide) (by decide)
    exact decide_eq_false composite
  third := false
  third_correct := by
    have composite : ¬ Nat.Prime (corner 6711 3) := by change ¬ Nat.Prime 180136663; exact PrattCertificates.composite_of_product (factor := 7) (cofactor := 25733809) rfl (by decide) (by decide)
    exact decide_eq_false composite
theorem row_count_6711 : row 6711 = 0 := (row_6711.checked).trans rfl
theorem model_count_6711 : modelRow 6711 = 0 := by decide
theorem modelrow_agrees_6711 : modelRow 6711 = row 6711 := model_count_6711.trans row_count_6711.symm
def row_6712 : RowCertificate 6712 where
  first := false
  first_correct := by
    have composite : ¬ Nat.Prime (corner 6712 1) := by change ¬ Nat.Prime 180217201; exact PrattCertificates.composite_of_product (factor := 163) (cofactor := 1105627) rfl (by decide) (by decide)
    exact decide_eq_false composite
  second := false
  second_correct := by
    have composite : ¬ Nat.Prime (corner 6712 2) := by change ¬ Nat.Prime 180203777; exact PrattCertificates.composite_of_product (factor := 13) (cofactor := 13861829) rfl (by decide) (by decide)
    exact decide_eq_false composite
  third := false
  third_correct := by
    have composite : ¬ Nat.Prime (corner 6712 3) := by change ¬ Nat.Prime 180190353; exact PrattCertificates.composite_of_product (factor := 3) (cofactor := 60063451) rfl (by decide) (by decide)
    exact decide_eq_false composite
theorem row_count_6712 : row 6712 = 0 := (row_6712.checked).trans rfl
theorem model_count_6712 : modelRow 6712 = 0 := by decide
theorem modelrow_agrees_6712 : modelRow 6712 = row 6712 := model_count_6712.trans row_count_6712.symm
def row_6713 : RowCertificate 6713 where
  first := false
  first_correct := by
    have composite : ¬ Nat.Prime (corner 6713 1) := by change ¬ Nat.Prime 180270903; exact PrattCertificates.composite_of_product (factor := 3) (cofactor := 60090301) rfl (by decide) (by decide)
    exact decide_eq_false composite
  second := false
  second_correct := by
    have composite : ¬ Nat.Prime (corner 6713 2) := by change ¬ Nat.Prime 180257477; exact PrattCertificates.composite_of_product (factor := 17) (cofactor := 10603381) rfl (by decide) (by decide)
    exact decide_eq_false composite
  third := false
  third_correct := by
    have composite : ¬ Nat.Prime (corner 6713 3) := by change ¬ Nat.Prime 180244051; exact PrattCertificates.composite_of_product (factor := 13) (cofactor := 13864927) rfl (by decide) (by decide)
    exact decide_eq_false composite
theorem row_count_6713 : row 6713 = 0 := (row_6713.checked).trans rfl
theorem model_count_6713 : modelRow 6713 = 0 := by decide
theorem modelrow_agrees_6713 : modelRow 6713 = row 6713 := model_count_6713.trans row_count_6713.symm
def row_6714 : RowCertificate 6714 where
  first := false
  first_correct := by
    have composite : ¬ Nat.Prime (corner 6714 1) := by change ¬ Nat.Prime 180324613; exact PrattCertificates.composite_of_product (factor := 7) (cofactor := 25760659) rfl (by decide) (by decide)
    exact decide_eq_false composite
  second := false
  second_correct := by
    have composite : ¬ Nat.Prime (corner 6714 2) := by change ¬ Nat.Prime 180311185; exact PrattCertificates.composite_of_product (factor := 5) (cofactor := 36062237) rfl (by decide) (by decide)
    exact decide_eq_false composite
  third := false
  third_correct := by
    have composite : ¬ Nat.Prime (corner 6714 3) := by change ¬ Nat.Prime 180297757; exact PrattCertificates.composite_of_product (factor := 853) (cofactor := 211369) rfl (by decide) (by decide)
    exact decide_eq_false composite
theorem row_count_6714 : row 6714 = 0 := (row_6714.checked).trans rfl
theorem model_count_6714 : modelRow 6714 = 0 := by decide
theorem modelrow_agrees_6714 : modelRow 6714 = row 6714 := model_count_6714.trans row_count_6714.symm
def row_6715 : RowCertificate 6715 where
  first := false
  first_correct := by
    have composite : ¬ Nat.Prime (corner 6715 1) := by change ¬ Nat.Prime 180378331; exact PrattCertificates.composite_of_product (factor := 7) (cofactor := 25768333) rfl (by decide) (by decide)
    exact decide_eq_false composite
  second := true
  second_correct := by
    have prime : Nat.Prime (corner 6715 2) := by change Nat.Prime 180364901; exact ProjectEulerSolutions.P58PrimeNodes.p_180364901.prime
    exact decide_eq_true prime
  third := false
  third_correct := by
    have composite : ¬ Nat.Prime (corner 6715 3) := by change ¬ Nat.Prime 180351471; exact PrattCertificates.composite_of_product (factor := 3) (cofactor := 60117157) rfl (by decide) (by decide)
    exact decide_eq_false composite
theorem row_count_6715 : row 6715 = 1 := (row_6715.checked).trans rfl
theorem model_count_6715 : modelRow 6715 = 1 := by decide
theorem modelrow_agrees_6715 : modelRow 6715 = row 6715 := model_count_6715.trans row_count_6715.symm
def row_6716 : RowCertificate 6716 where
  first := false
  first_correct := by
    have composite : ¬ Nat.Prime (corner 6716 1) := by change ¬ Nat.Prime 180432057; exact PrattCertificates.composite_of_product (factor := 3) (cofactor := 60144019) rfl (by decide) (by decide)
    exact decide_eq_false composite
  second := false
  second_correct := by
    have composite : ¬ Nat.Prime (corner 6716 2) := by change ¬ Nat.Prime 180418625; exact PrattCertificates.composite_of_product (factor := 5) (cofactor := 36083725) rfl (by decide) (by decide)
    exact decide_eq_false composite
  third := true
  third_correct := by
    have prime : Nat.Prime (corner 6716 3) := by change Nat.Prime 180405193; exact ProjectEulerSolutions.P58PrimeNodes.p_180405193.prime
    exact decide_eq_true prime
theorem row_count_6716 : row 6716 = 1 := (row_6716.checked).trans rfl
theorem model_count_6716 : modelRow 6716 = 1 := by decide
theorem modelrow_agrees_6716 : modelRow 6716 = row 6716 := model_count_6716.trans row_count_6716.symm
def row_6717 : RowCertificate 6717 where
  first := true
  first_correct := by
    have prime : Nat.Prime (corner 6717 1) := by change Nat.Prime 180485791; exact ProjectEulerSolutions.P58PrimeNodes.p_180485791.prime
    exact decide_eq_true prime
  second := false
  second_correct := by
    have composite : ¬ Nat.Prime (corner 6717 2) := by change ¬ Nat.Prime 180472357; exact PrattCertificates.composite_of_product (factor := 13) (cofactor := 13882489) rfl (by decide) (by decide)
    exact decide_eq_false composite
  third := false
  third_correct := by
    have composite : ¬ Nat.Prime (corner 6717 3) := by change ¬ Nat.Prime 180458923; exact PrattCertificates.composite_of_product (factor := 61) (cofactor := 2958343) rfl (by decide) (by decide)
    exact decide_eq_false composite
theorem row_count_6717 : row 6717 = 1 := (row_6717.checked).trans rfl
theorem model_count_6717 : modelRow 6717 = 1 := by decide
theorem modelrow_agrees_6717 : modelRow 6717 = row 6717 := model_count_6717.trans row_count_6717.symm
def row_6718 : RowCertificate 6718 where
  first := false
  first_correct := by
    have composite : ¬ Nat.Prime (corner 6718 1) := by change ¬ Nat.Prime 180539533; exact PrattCertificates.composite_of_product (factor := 103) (cofactor := 1752811) rfl (by decide) (by decide)
    exact decide_eq_false composite
  second := true
  second_correct := by
    have prime : Nat.Prime (corner 6718 2) := by change Nat.Prime 180526097; exact ProjectEulerSolutions.P58PrimeNodes.p_180526097.prime
    exact decide_eq_true prime
  third := false
  third_correct := by
    have composite : ¬ Nat.Prime (corner 6718 3) := by change ¬ Nat.Prime 180512661; exact PrattCertificates.composite_of_product (factor := 3) (cofactor := 60170887) rfl (by decide) (by decide)
    exact decide_eq_false composite
theorem row_count_6718 : row 6718 = 1 := (row_6718.checked).trans rfl
theorem model_count_6718 : modelRow 6718 = 1 := by decide
theorem modelrow_agrees_6718 : modelRow 6718 = row 6718 := model_count_6718.trans row_count_6718.symm
def row_6719 : RowCertificate 6719 where
  first := false
  first_correct := by
    have composite : ¬ Nat.Prime (corner 6719 1) := by change ¬ Nat.Prime 180593283; exact PrattCertificates.composite_of_product (factor := 3) (cofactor := 60197761) rfl (by decide) (by decide)
    exact decide_eq_false composite
  second := false
  second_correct := by
    have composite : ¬ Nat.Prime (corner 6719 2) := by change ¬ Nat.Prime 180579845; exact PrattCertificates.composite_of_product (factor := 5) (cofactor := 36115969) rfl (by decide) (by decide)
    exact decide_eq_false composite
  third := false
  third_correct := by
    have composite : ¬ Nat.Prime (corner 6719 3) := by change ¬ Nat.Prime 180566407; exact PrattCertificates.composite_of_product (factor := 7) (cofactor := 25795201) rfl (by decide) (by decide)
    exact decide_eq_false composite
theorem row_count_6719 : row 6719 = 0 := (row_6719.checked).trans rfl
theorem model_count_6719 : modelRow 6719 = 0 := by decide
theorem modelrow_agrees_6719 : modelRow 6719 = row 6719 := model_count_6719.trans row_count_6719.symm
def row_6720 : RowCertificate 6720 where
  first := false
  first_correct := by
    have composite : ¬ Nat.Prime (corner 6720 1) := by change ¬ Nat.Prime 180647041; exact PrattCertificates.composite_of_product (factor := 19) (cofactor := 9507739) rfl (by decide) (by decide)
    exact decide_eq_false composite
  second := false
  second_correct := by
    have composite : ¬ Nat.Prime (corner 6720 2) := by change ¬ Nat.Prime 180633601; exact PrattCertificates.composite_of_product (factor := 109) (cofactor := 1657189) rfl (by decide) (by decide)
    exact decide_eq_false composite
  third := false
  third_correct := by
    have composite : ¬ Nat.Prime (corner 6720 3) := by change ¬ Nat.Prime 180620161; exact PrattCertificates.composite_of_product (factor := 199) (cofactor := 907639) rfl (by decide) (by decide)
    exact decide_eq_false composite
theorem row_count_6720 : row 6720 = 0 := (row_6720.checked).trans rfl
theorem model_count_6720 : modelRow 6720 = 0 := by decide
theorem modelrow_agrees_6720 : modelRow 6720 = row 6720 := model_count_6720.trans row_count_6720.symm
def row_6721 : RowCertificate 6721 where
  first := false
  first_correct := by
    have composite : ¬ Nat.Prime (corner 6721 1) := by change ¬ Nat.Prime 180700807; exact PrattCertificates.composite_of_product (factor := 7) (cofactor := 25814401) rfl (by decide) (by decide)
    exact decide_eq_false composite
  second := false
  second_correct := by
    have composite : ¬ Nat.Prime (corner 6721 2) := by change ¬ Nat.Prime 180687365; exact PrattCertificates.composite_of_product (factor := 5) (cofactor := 36137473) rfl (by decide) (by decide)
    exact decide_eq_false composite
  third := false
  third_correct := by
    have composite : ¬ Nat.Prime (corner 6721 3) := by change ¬ Nat.Prime 180673923; exact PrattCertificates.composite_of_product (factor := 3) (cofactor := 60224641) rfl (by decide) (by decide)
    exact decide_eq_false composite
theorem row_count_6721 : row 6721 = 0 := (row_6721.checked).trans rfl
theorem model_count_6721 : modelRow 6721 = 0 := by decide
theorem modelrow_agrees_6721 : modelRow 6721 = row 6721 := model_count_6721.trans row_count_6721.symm
def row_6722 : RowCertificate 6722 where
  first := false
  first_correct := by
    have composite : ¬ Nat.Prime (corner 6722 1) := by change ¬ Nat.Prime 180754581; exact PrattCertificates.composite_of_product (factor := 3) (cofactor := 60251527) rfl (by decide) (by decide)
    exact decide_eq_false composite
  second := false
  second_correct := by
    have composite : ¬ Nat.Prime (corner 6722 2) := by change ¬ Nat.Prime 180741137; exact PrattCertificates.composite_of_product (factor := 29) (cofactor := 6232453) rfl (by decide) (by decide)
    exact decide_eq_false composite
  third := false
  third_correct := by
    have composite : ¬ Nat.Prime (corner 6722 3) := by change ¬ Nat.Prime 180727693; exact PrattCertificates.composite_of_product (factor := 409) (cofactor := 441877) rfl (by decide) (by decide)
    exact decide_eq_false composite
theorem row_count_6722 : row 6722 = 0 := (row_6722.checked).trans rfl
theorem model_count_6722 : modelRow 6722 = 0 := by decide
theorem modelrow_agrees_6722 : modelRow 6722 = row 6722 := model_count_6722.trans row_count_6722.symm
def row_6723 : RowCertificate 6723 where
  first := false
  first_correct := by
    have composite : ¬ Nat.Prime (corner 6723 1) := by change ¬ Nat.Prime 180808363; exact PrattCertificates.composite_of_product (factor := 103) (cofactor := 1755421) rfl (by decide) (by decide)
    exact decide_eq_false composite
  second := false
  second_correct := by
    have composite : ¬ Nat.Prime (corner 6723 2) := by change ¬ Nat.Prime 180794917; exact PrattCertificates.composite_of_product (factor := 5233) (cofactor := 34549) rfl (by decide) (by decide)
    exact decide_eq_false composite
  third := false
  third_correct := by
    have composite : ¬ Nat.Prime (corner 6723 3) := by change ¬ Nat.Prime 180781471; exact PrattCertificates.composite_of_product (factor := 13) (cofactor := 13906267) rfl (by decide) (by decide)
    exact decide_eq_false composite
theorem row_count_6723 : row 6723 = 0 := (row_6723.checked).trans rfl
theorem model_count_6723 : modelRow 6723 = 0 := by decide
theorem modelrow_agrees_6723 : modelRow 6723 = row 6723 := model_count_6723.trans row_count_6723.symm
def row_6724 : RowCertificate 6724 where
  first := false
  first_correct := by
    have composite : ¬ Nat.Prime (corner 6724 1) := by change ¬ Nat.Prime 180862153; exact PrattCertificates.composite_of_product (factor := 31) (cofactor := 5834263) rfl (by decide) (by decide)
    exact decide_eq_false composite
  second := false
  second_correct := by
    have composite : ¬ Nat.Prime (corner 6724 2) := by change ¬ Nat.Prime 180848705; exact PrattCertificates.composite_of_product (factor := 5) (cofactor := 36169741) rfl (by decide) (by decide)
    exact decide_eq_false composite
  third := false
  third_correct := by
    have composite : ¬ Nat.Prime (corner 6724 3) := by change ¬ Nat.Prime 180835257; exact PrattCertificates.composite_of_product (factor := 3) (cofactor := 60278419) rfl (by decide) (by decide)
    exact decide_eq_false composite
theorem row_count_6724 : row 6724 = 0 := (row_6724.checked).trans rfl
theorem model_count_6724 : modelRow 6724 = 0 := by decide
theorem modelrow_agrees_6724 : modelRow 6724 = row 6724 := model_count_6724.trans row_count_6724.symm
def row_6725 : RowCertificate 6725 where
  first := false
  first_correct := by
    have composite : ¬ Nat.Prime (corner 6725 1) := by change ¬ Nat.Prime 180915951; exact PrattCertificates.composite_of_product (factor := 3) (cofactor := 60305317) rfl (by decide) (by decide)
    exact decide_eq_false composite
  second := false
  second_correct := by
    have composite : ¬ Nat.Prime (corner 6725 2) := by change ¬ Nat.Prime 180902501; exact PrattCertificates.composite_of_product (factor := 13) (cofactor := 13915577) rfl (by decide) (by decide)
    exact decide_eq_false composite
  third := false
  third_correct := by
    have composite : ¬ Nat.Prime (corner 6725 3) := by change ¬ Nat.Prime 180889051; exact PrattCertificates.composite_of_product (factor := 7) (cofactor := 25841293) rfl (by decide) (by decide)
    exact decide_eq_false composite
theorem row_count_6725 : row 6725 = 0 := (row_6725.checked).trans rfl
theorem model_count_6725 : modelRow 6725 = 0 := by decide
theorem modelrow_agrees_6725 : modelRow 6725 = row 6725 := model_count_6725.trans row_count_6725.symm
def row_6726 : RowCertificate 6726 where
  first := false
  first_correct := by
    have composite : ¬ Nat.Prime (corner 6726 1) := by change ¬ Nat.Prime 180969757; exact PrattCertificates.composite_of_product (factor := 43) (cofactor := 4208599) rfl (by decide) (by decide)
    exact decide_eq_false composite
  second := false
  second_correct := by
    have composite : ¬ Nat.Prime (corner 6726 2) := by change ¬ Nat.Prime 180956305; exact PrattCertificates.composite_of_product (factor := 5) (cofactor := 36191261) rfl (by decide) (by decide)
    exact decide_eq_false composite
  third := false
  third_correct := by
    have composite : ¬ Nat.Prime (corner 6726 3) := by change ¬ Nat.Prime 180942853; exact PrattCertificates.composite_of_product (factor := 7) (cofactor := 25848979) rfl (by decide) (by decide)
    exact decide_eq_false composite
theorem row_count_6726 : row 6726 = 0 := (row_6726.checked).trans rfl
theorem model_count_6726 : modelRow 6726 = 0 := by decide
theorem modelrow_agrees_6726 : modelRow 6726 = row 6726 := model_count_6726.trans row_count_6726.symm
def row_6727 : RowCertificate 6727 where
  first := true
  first_correct := by
    have prime : Nat.Prime (corner 6727 1) := by change Nat.Prime 181023571; exact ProjectEulerSolutions.P58PrimeNodes.p_181023571.prime
    exact decide_eq_true prime
  second := false
  second_correct := by
    have composite : ¬ Nat.Prime (corner 6727 2) := by change ¬ Nat.Prime 181010117; exact PrattCertificates.composite_of_product (factor := 149) (cofactor := 1214833) rfl (by decide) (by decide)
    exact decide_eq_false composite
  third := false
  third_correct := by
    have composite : ¬ Nat.Prime (corner 6727 3) := by change ¬ Nat.Prime 180996663; exact PrattCertificates.composite_of_product (factor := 3) (cofactor := 60332221) rfl (by decide) (by decide)
    exact decide_eq_false composite
theorem row_count_6727 : row 6727 = 1 := (row_6727.checked).trans rfl
theorem model_count_6727 : modelRow 6727 = 1 := by decide
theorem modelrow_agrees_6727 : modelRow 6727 = row 6727 := model_count_6727.trans row_count_6727.symm
def row_6728 : RowCertificate 6728 where
  first := false
  first_correct := by
    have composite : ¬ Nat.Prime (corner 6728 1) := by change ¬ Nat.Prime 181077393; exact PrattCertificates.composite_of_product (factor := 3) (cofactor := 60359131) rfl (by decide) (by decide)
    exact decide_eq_false composite
  second := false
  second_correct := by
    have composite : ¬ Nat.Prime (corner 6728 2) := by change ¬ Nat.Prime 181063937; exact PrattCertificates.composite_of_product (factor := 353) (cofactor := 512929) rfl (by decide) (by decide)
    exact decide_eq_false composite
  third := false
  third_correct := by
    have composite : ¬ Nat.Prime (corner 6728 3) := by change ¬ Nat.Prime 181050481; exact PrattCertificates.composite_of_product (factor := 1381) (cofactor := 131101) rfl (by decide) (by decide)
    exact decide_eq_false composite
theorem row_count_6728 : row 6728 = 0 := (row_6728.checked).trans rfl
theorem model_count_6728 : modelRow 6728 = 0 := by decide
theorem modelrow_agrees_6728 : modelRow 6728 = row 6728 := model_count_6728.trans row_count_6728.symm
def row_6729 : RowCertificate 6729 where
  first := false
  first_correct := by
    have composite : ¬ Nat.Prime (corner 6729 1) := by change ¬ Nat.Prime 181131223; exact PrattCertificates.composite_of_product (factor := 7) (cofactor := 25875889) rfl (by decide) (by decide)
    exact decide_eq_false composite
  second := false
  second_correct := by
    have composite : ¬ Nat.Prime (corner 6729 2) := by change ¬ Nat.Prime 181117765; exact PrattCertificates.composite_of_product (factor := 5) (cofactor := 36223553) rfl (by decide) (by decide)
    exact decide_eq_false composite
  third := false
  third_correct := by
    have composite : ¬ Nat.Prime (corner 6729 3) := by change ¬ Nat.Prime 181104307; exact PrattCertificates.composite_of_product (factor := 37) (cofactor := 4894711) rfl (by decide) (by decide)
    exact decide_eq_false composite
theorem row_count_6729 : row 6729 = 0 := (row_6729.checked).trans rfl
theorem model_count_6729 : modelRow 6729 = 0 := by decide
theorem modelrow_agrees_6729 : modelRow 6729 = row 6729 := model_count_6729.trans row_count_6729.symm
def row_6730 : RowCertificate 6730 where
  first := true
  first_correct := by
    have prime : Nat.Prime (corner 6730 1) := by change Nat.Prime 181185061; exact ProjectEulerSolutions.P58PrimeNodes.p_181185061.prime
    exact decide_eq_true prime
  second := false
  second_correct := by
    have composite : ¬ Nat.Prime (corner 6730 2) := by change ¬ Nat.Prime 181171601; exact PrattCertificates.composite_of_product (factor := 13) (cofactor := 13936277) rfl (by decide) (by decide)
    exact decide_eq_false composite
  third := false
  third_correct := by
    have composite : ¬ Nat.Prime (corner 6730 3) := by change ¬ Nat.Prime 181158141; exact PrattCertificates.composite_of_product (factor := 3) (cofactor := 60386047) rfl (by decide) (by decide)
    exact decide_eq_false composite
theorem row_count_6730 : row 6730 = 1 := (row_6730.checked).trans rfl
theorem model_count_6730 : modelRow 6730 = 1 := by decide
theorem modelrow_agrees_6730 : modelRow 6730 = row 6730 := model_count_6730.trans row_count_6730.symm
def row_6731 : RowCertificate 6731 where
  first := false
  first_correct := by
    have composite : ¬ Nat.Prime (corner 6731 1) := by change ¬ Nat.Prime 181238907; exact PrattCertificates.composite_of_product (factor := 3) (cofactor := 60412969) rfl (by decide) (by decide)
    exact decide_eq_false composite
  second := false
  second_correct := by
    have composite : ¬ Nat.Prime (corner 6731 2) := by change ¬ Nat.Prime 181225445; exact PrattCertificates.composite_of_product (factor := 5) (cofactor := 36245089) rfl (by decide) (by decide)
    exact decide_eq_false composite
  third := true
  third_correct := by
    have prime : Nat.Prime (corner 6731 3) := by change Nat.Prime 181211983; exact ProjectEulerSolutions.P58PrimeNodes.p_181211983.prime
    exact decide_eq_true prime
theorem row_count_6731 : row 6731 = 1 := (row_6731.checked).trans rfl
theorem model_count_6731 : modelRow 6731 = 1 := by decide
theorem modelrow_agrees_6731 : modelRow 6731 = row 6731 := model_count_6731.trans row_count_6731.symm
def row_6732 : RowCertificate 6732 where
  first := false
  first_correct := by
    have composite : ¬ Nat.Prime (corner 6732 1) := by change ¬ Nat.Prime 181292761; exact PrattCertificates.composite_of_product (factor := 13) (cofactor := 13945597) rfl (by decide) (by decide)
    exact decide_eq_false composite
  second := false
  second_correct := by
    have composite : ¬ Nat.Prime (corner 6732 2) := by change ¬ Nat.Prime 181279297; exact PrattCertificates.composite_of_product (factor := 7121) (cofactor := 25457) rfl (by decide) (by decide)
    exact decide_eq_false composite
  third := false
  third_correct := by
    have composite : ¬ Nat.Prime (corner 6732 3) := by change ¬ Nat.Prime 181265833; exact PrattCertificates.composite_of_product (factor := 7) (cofactor := 25895119) rfl (by decide) (by decide)
    exact decide_eq_false composite
theorem row_count_6732 : row 6732 = 0 := (row_6732.checked).trans rfl
theorem model_count_6732 : modelRow 6732 = 0 := by decide
theorem modelrow_agrees_6732 : modelRow 6732 = row 6732 := model_count_6732.trans row_count_6732.symm
def row_6733 : RowCertificate 6733 where
  first := false
  first_correct := by
    have composite : ¬ Nat.Prime (corner 6733 1) := by change ¬ Nat.Prime 181346623; exact PrattCertificates.composite_of_product (factor := 13267) (cofactor := 13669) rfl (by decide) (by decide)
    exact decide_eq_false composite
  second := true
  second_correct := by
    have prime : Nat.Prime (corner 6733 2) := by change Nat.Prime 181333157; exact ProjectEulerSolutions.P58PrimeNodes.p_181333157.prime
    exact decide_eq_true prime
  third := false
  third_correct := by
    have composite : ¬ Nat.Prime (corner 6733 3) := by change ¬ Nat.Prime 181319691; exact PrattCertificates.composite_of_product (factor := 3) (cofactor := 60439897) rfl (by decide) (by decide)
    exact decide_eq_false composite
theorem row_count_6733 : row 6733 = 1 := (row_6733.checked).trans rfl
theorem model_count_6733 : modelRow 6733 = 1 := by decide
theorem modelrow_agrees_6733 : modelRow 6733 = row 6733 := model_count_6733.trans row_count_6733.symm
def row_6734 : RowCertificate 6734 where
  first := false
  first_correct := by
    have composite : ¬ Nat.Prime (corner 6734 1) := by change ¬ Nat.Prime 181400493; exact PrattCertificates.composite_of_product (factor := 3) (cofactor := 60466831) rfl (by decide) (by decide)
    exact decide_eq_false composite
  second := false
  second_correct := by
    have composite : ¬ Nat.Prime (corner 6734 2) := by change ¬ Nat.Prime 181387025; exact PrattCertificates.composite_of_product (factor := 5) (cofactor := 36277405) rfl (by decide) (by decide)
    exact decide_eq_false composite
  third := false
  third_correct := by
    have composite : ¬ Nat.Prime (corner 6734 3) := by change ¬ Nat.Prime 181373557; exact PrattCertificates.composite_of_product (factor := 61) (cofactor := 2973337) rfl (by decide) (by decide)
    exact decide_eq_false composite
theorem row_count_6734 : row 6734 = 0 := (row_6734.checked).trans rfl
theorem model_count_6734 : modelRow 6734 = 0 := by decide
theorem modelrow_agrees_6734 : modelRow 6734 = row 6734 := model_count_6734.trans row_count_6734.symm
def row_6735 : RowCertificate 6735 where
  first := false
  first_correct := by
    have composite : ¬ Nat.Prime (corner 6735 1) := by change ¬ Nat.Prime 181454371; exact PrattCertificates.composite_of_product (factor := 7) (cofactor := 25922053) rfl (by decide) (by decide)
    exact decide_eq_false composite
  second := false
  second_correct := by
    have composite : ¬ Nat.Prime (corner 6735 2) := by change ¬ Nat.Prime 181440901; exact PrattCertificates.composite_of_product (factor := 61) (cofactor := 2974441) rfl (by decide) (by decide)
    exact decide_eq_false composite
  third := true
  third_correct := by
    have prime : Nat.Prime (corner 6735 3) := by change Nat.Prime 181427431; exact ProjectEulerSolutions.P58PrimeNodes.p_181427431.prime
    exact decide_eq_true prime
theorem row_count_6735 : row 6735 = 1 := (row_6735.checked).trans rfl
theorem model_count_6735 : modelRow 6735 = 1 := by decide
theorem modelrow_agrees_6735 : modelRow 6735 = row 6735 := model_count_6735.trans row_count_6735.symm
def row_6736 : RowCertificate 6736 where
  first := false
  first_correct := by
    have composite : ¬ Nat.Prime (corner 6736 1) := by change ¬ Nat.Prime 181508257; exact PrattCertificates.composite_of_product (factor := 7) (cofactor := 25929751) rfl (by decide) (by decide)
    exact decide_eq_false composite
  second := false
  second_correct := by
    have composite : ¬ Nat.Prime (corner 6736 2) := by change ¬ Nat.Prime 181494785; exact PrattCertificates.composite_of_product (factor := 5) (cofactor := 36298957) rfl (by decide) (by decide)
    exact decide_eq_false composite
  third := false
  third_correct := by
    have composite : ¬ Nat.Prime (corner 6736 3) := by change ¬ Nat.Prime 181481313; exact PrattCertificates.composite_of_product (factor := 3) (cofactor := 60493771) rfl (by decide) (by decide)
    exact decide_eq_false composite
theorem row_count_6736 : row 6736 = 0 := (row_6736.checked).trans rfl
theorem model_count_6736 : modelRow 6736 = 0 := by decide
theorem modelrow_agrees_6736 : modelRow 6736 = row 6736 := model_count_6736.trans row_count_6736.symm
def row_6737 : RowCertificate 6737 where
  first := false
  first_correct := by
    have composite : ¬ Nat.Prime (corner 6737 1) := by change ¬ Nat.Prime 181562151; exact PrattCertificates.composite_of_product (factor := 3) (cofactor := 60520717) rfl (by decide) (by decide)
    exact decide_eq_false composite
  second := false
  second_correct := by
    have composite : ¬ Nat.Prime (corner 6737 2) := by change ¬ Nat.Prime 181548677; exact PrattCertificates.composite_of_product (factor := 37) (cofactor := 4906721) rfl (by decide) (by decide)
    exact decide_eq_false composite
  third := true
  third_correct := by
    have prime : Nat.Prime (corner 6737 3) := by change Nat.Prime 181535203; exact ProjectEulerSolutions.P58PrimeNodes.p_181535203.prime
    exact decide_eq_true prime
theorem row_count_6737 : row 6737 = 1 := (row_6737.checked).trans rfl
theorem model_count_6737 : modelRow 6737 = 1 := by decide
theorem modelrow_agrees_6737 : modelRow 6737 = row 6737 := model_count_6737.trans row_count_6737.symm
def row_6738 : RowCertificate 6738 where
  first := true
  first_correct := by
    have prime : Nat.Prime (corner 6738 1) := by change Nat.Prime 181616053; exact ProjectEulerSolutions.P58PrimeNodes.p_181616053.prime
    exact decide_eq_true prime
  second := false
  second_correct := by
    have composite : ¬ Nat.Prime (corner 6738 2) := by change ¬ Nat.Prime 181602577; exact PrattCertificates.composite_of_product (factor := 13) (cofactor := 13969429) rfl (by decide) (by decide)
    exact decide_eq_false composite
  third := false
  third_correct := by
    have composite : ¬ Nat.Prime (corner 6738 3) := by change ¬ Nat.Prime 181589101; exact PrattCertificates.composite_of_product (factor := 1063) (cofactor := 170827) rfl (by decide) (by decide)
    exact decide_eq_false composite
theorem row_count_6738 : row 6738 = 1 := (row_6738.checked).trans rfl
theorem model_count_6738 : modelRow 6738 = 1 := by decide
theorem modelrow_agrees_6738 : modelRow 6738 = row 6738 := model_count_6738.trans row_count_6738.symm
def row_6739 : RowCertificate 6739 where
  first := false
  first_correct := by
    have composite : ¬ Nat.Prime (corner 6739 1) := by change ¬ Nat.Prime 181669963; exact PrattCertificates.composite_of_product (factor := 19) (cofactor := 9561577) rfl (by decide) (by decide)
    exact decide_eq_false composite
  second := false
  second_correct := by
    have composite : ¬ Nat.Prime (corner 6739 2) := by change ¬ Nat.Prime 181656485; exact PrattCertificates.composite_of_product (factor := 5) (cofactor := 36331297) rfl (by decide) (by decide)
    exact decide_eq_false composite
  third := false
  third_correct := by
    have composite : ¬ Nat.Prime (corner 6739 3) := by change ¬ Nat.Prime 181643007; exact PrattCertificates.composite_of_product (factor := 3) (cofactor := 60547669) rfl (by decide) (by decide)
    exact decide_eq_false composite
theorem row_count_6739 : row 6739 = 0 := (row_6739.checked).trans rfl
theorem model_count_6739 : modelRow 6739 = 0 := by decide
theorem modelrow_agrees_6739 : modelRow 6739 = row 6739 := model_count_6739.trans row_count_6739.symm
def row_6740 : RowCertificate 6740 where
  first := false
  first_correct := by
    have composite : ¬ Nat.Prime (corner 6740 1) := by change ¬ Nat.Prime 181723881; exact PrattCertificates.composite_of_product (factor := 3) (cofactor := 60574627) rfl (by decide) (by decide)
    exact decide_eq_false composite
  second := false
  second_correct := by
    have composite : ¬ Nat.Prime (corner 6740 2) := by change ¬ Nat.Prime 181710401; exact PrattCertificates.composite_of_product (factor := 41) (cofactor := 4431961) rfl (by decide) (by decide)
    exact decide_eq_false composite
  third := false
  third_correct := by
    have composite : ¬ Nat.Prime (corner 6740 3) := by change ¬ Nat.Prime 181696921; exact PrattCertificates.composite_of_product (factor := 7) (cofactor := 25956703) rfl (by decide) (by decide)
    exact decide_eq_false composite
theorem row_count_6740 : row 6740 = 0 := (row_6740.checked).trans rfl
theorem model_count_6740 : modelRow 6740 = 0 := by decide
theorem modelrow_agrees_6740 : modelRow 6740 = row 6740 := model_count_6740.trans row_count_6740.symm
def row_6741 : RowCertificate 6741 where
  first := false
  first_correct := by
    have composite : ¬ Nat.Prime (corner 6741 1) := by change ¬ Nat.Prime 181777807; exact PrattCertificates.composite_of_product (factor := 19) (cofactor := 9567253) rfl (by decide) (by decide)
    exact decide_eq_false composite
  second := false
  second_correct := by
    have composite : ¬ Nat.Prime (corner 6741 2) := by change ¬ Nat.Prime 181764325; exact PrattCertificates.composite_of_product (factor := 5) (cofactor := 36352865) rfl (by decide) (by decide)
    exact decide_eq_false composite
  third := false
  third_correct := by
    have composite : ¬ Nat.Prime (corner 6741 3) := by change ¬ Nat.Prime 181750843; exact PrattCertificates.composite_of_product (factor := 127) (cofactor := 1431109) rfl (by decide) (by decide)
    exact decide_eq_false composite
theorem row_count_6741 : row 6741 = 0 := (row_6741.checked).trans rfl
theorem model_count_6741 : modelRow 6741 = 0 := by decide
theorem modelrow_agrees_6741 : modelRow 6741 = row 6741 := model_count_6741.trans row_count_6741.symm
def row_6742 : RowCertificate 6742 where
  first := false
  first_correct := by
    have composite : ¬ Nat.Prime (corner 6742 1) := by change ¬ Nat.Prime 181831741; exact PrattCertificates.composite_of_product (factor := 7) (cofactor := 25975963) rfl (by decide) (by decide)
    exact decide_eq_false composite
  second := false
  second_correct := by
    have composite : ¬ Nat.Prime (corner 6742 2) := by change ¬ Nat.Prime 181818257; exact PrattCertificates.composite_of_product (factor := 313) (cofactor := 580889) rfl (by decide) (by decide)
    exact decide_eq_false composite
  third := false
  third_correct := by
    have composite : ¬ Nat.Prime (corner 6742 3) := by change ¬ Nat.Prime 181804773; exact PrattCertificates.composite_of_product (factor := 3) (cofactor := 60601591) rfl (by decide) (by decide)
    exact decide_eq_false composite
theorem row_count_6742 : row 6742 = 0 := (row_6742.checked).trans rfl
theorem model_count_6742 : modelRow 6742 = 0 := by decide
theorem modelrow_agrees_6742 : modelRow 6742 = row 6742 := model_count_6742.trans row_count_6742.symm
def row_6743 : RowCertificate 6743 where
  first := false
  first_correct := by
    have composite : ¬ Nat.Prime (corner 6743 1) := by change ¬ Nat.Prime 181885683; exact PrattCertificates.composite_of_product (factor := 3) (cofactor := 60628561) rfl (by decide) (by decide)
    exact decide_eq_false composite
  second := false
  second_correct := by
    have composite : ¬ Nat.Prime (corner 6743 2) := by change ¬ Nat.Prime 181872197; exact PrattCertificates.composite_of_product (factor := 13) (cofactor := 13990169) rfl (by decide) (by decide)
    exact decide_eq_false composite
  third := false
  third_correct := by
    have composite : ¬ Nat.Prime (corner 6743 3) := by change ¬ Nat.Prime 181858711; exact PrattCertificates.composite_of_product (factor := 79) (cofactor := 2302009) rfl (by decide) (by decide)
    exact decide_eq_false composite
theorem row_count_6743 : row 6743 = 0 := (row_6743.checked).trans rfl
theorem model_count_6743 : modelRow 6743 = 0 := by decide
theorem modelrow_agrees_6743 : modelRow 6743 = row 6743 := model_count_6743.trans row_count_6743.symm
def row_6744 : RowCertificate 6744 where
  first := false
  first_correct := by
    have composite : ¬ Nat.Prime (corner 6744 1) := by change ¬ Nat.Prime 181939633; exact PrattCertificates.composite_of_product (factor := 349) (cofactor := 521317) rfl (by decide) (by decide)
    exact decide_eq_false composite
  second := false
  second_correct := by
    have composite : ¬ Nat.Prime (corner 6744 2) := by change ¬ Nat.Prime 181926145; exact PrattCertificates.composite_of_product (factor := 5) (cofactor := 36385229) rfl (by decide) (by decide)
    exact decide_eq_false composite
  third := true
  third_correct := by
    have prime : Nat.Prime (corner 6744 3) := by change Nat.Prime 181912657; exact ProjectEulerSolutions.P58PrimeNodes.p_181912657.prime
    exact decide_eq_true prime
theorem row_count_6744 : row 6744 = 1 := (row_6744.checked).trans rfl
theorem model_count_6744 : modelRow 6744 = 1 := by decide
theorem modelrow_agrees_6744 : modelRow 6744 = row 6744 := model_count_6744.trans row_count_6744.symm
def row_6745 : RowCertificate 6745 where
  first := false
  first_correct := by
    have composite : ¬ Nat.Prime (corner 6745 1) := by change ¬ Nat.Prime 181993591; exact PrattCertificates.composite_of_product (factor := 13) (cofactor := 13999507) rfl (by decide) (by decide)
    exact decide_eq_false composite
  second := true
  second_correct := by
    have prime : Nat.Prime (corner 6745 2) := by change Nat.Prime 181980101; exact ProjectEulerSolutions.P58PrimeNodes.p_181980101.prime
    exact decide_eq_true prime
  third := false
  third_correct := by
    have composite : ¬ Nat.Prime (corner 6745 3) := by change ¬ Nat.Prime 181966611; exact PrattCertificates.composite_of_product (factor := 3) (cofactor := 60655537) rfl (by decide) (by decide)
    exact decide_eq_false composite
theorem row_count_6745 : row 6745 = 1 := (row_6745.checked).trans rfl
theorem model_count_6745 : modelRow 6745 = 1 := by decide
theorem modelrow_agrees_6745 : modelRow 6745 = row 6745 := model_count_6745.trans row_count_6745.symm
def row_6746 : RowCertificate 6746 where
  first := false
  first_correct := by
    have composite : ¬ Nat.Prime (corner 6746 1) := by change ¬ Nat.Prime 182047557; exact PrattCertificates.composite_of_product (factor := 3) (cofactor := 60682519) rfl (by decide) (by decide)
    exact decide_eq_false composite
  second := false
  second_correct := by
    have composite : ¬ Nat.Prime (corner 6746 2) := by change ¬ Nat.Prime 182034065; exact PrattCertificates.composite_of_product (factor := 5) (cofactor := 36406813) rfl (by decide) (by decide)
    exact decide_eq_false composite
  third := false
  third_correct := by
    have composite : ¬ Nat.Prime (corner 6746 3) := by change ¬ Nat.Prime 182020573; exact PrattCertificates.composite_of_product (factor := 7) (cofactor := 26002939) rfl (by decide) (by decide)
    exact decide_eq_false composite
theorem row_count_6746 : row 6746 = 0 := (row_6746.checked).trans rfl
theorem model_count_6746 : modelRow 6746 = 0 := by decide
theorem modelrow_agrees_6746 : modelRow 6746 = row 6746 := model_count_6746.trans row_count_6746.symm
def row_6747 : RowCertificate 6747 where
  first := false
  first_correct := by
    have composite : ¬ Nat.Prime (corner 6747 1) := by change ¬ Nat.Prime 182101531; exact PrattCertificates.composite_of_product (factor := 37) (cofactor := 4921663) rfl (by decide) (by decide)
    exact decide_eq_false composite
  second := false
  second_correct := by
    have composite : ¬ Nat.Prime (corner 6747 2) := by change ¬ Nat.Prime 182088037; exact PrattCertificates.composite_of_product (factor := 17) (cofactor := 10711061) rfl (by decide) (by decide)
    exact decide_eq_false composite
  third := false
  third_correct := by
    have composite : ¬ Nat.Prime (corner 6747 3) := by change ¬ Nat.Prime 182074543; exact PrattCertificates.composite_of_product (factor := 7) (cofactor := 26010649) rfl (by decide) (by decide)
    exact decide_eq_false composite
theorem row_count_6747 : row 6747 = 0 := (row_6747.checked).trans rfl
theorem model_count_6747 : modelRow 6747 = 0 := by decide
theorem modelrow_agrees_6747 : modelRow 6747 = row 6747 := model_count_6747.trans row_count_6747.symm
def row_6748 : RowCertificate 6748 where
  first := false
  first_correct := by
    have composite : ¬ Nat.Prime (corner 6748 1) := by change ¬ Nat.Prime 182155513; exact PrattCertificates.composite_of_product (factor := 67) (cofactor := 2718739) rfl (by decide) (by decide)
    exact decide_eq_false composite
  second := false
  second_correct := by
    have composite : ¬ Nat.Prime (corner 6748 2) := by change ¬ Nat.Prime 182142017; exact PrattCertificates.composite_of_product (factor := 9029) (cofactor := 20173) rfl (by decide) (by decide)
    exact decide_eq_false composite
  third := false
  third_correct := by
    have composite : ¬ Nat.Prime (corner 6748 3) := by change ¬ Nat.Prime 182128521; exact PrattCertificates.composite_of_product (factor := 3) (cofactor := 60709507) rfl (by decide) (by decide)
    exact decide_eq_false composite
theorem row_count_6748 : row 6748 = 0 := (row_6748.checked).trans rfl
theorem model_count_6748 : modelRow 6748 = 0 := by decide
theorem modelrow_agrees_6748 : modelRow 6748 = row 6748 := model_count_6748.trans row_count_6748.symm
def row_6749 : RowCertificate 6749 where
  first := false
  first_correct := by
    have composite : ¬ Nat.Prime (corner 6749 1) := by change ¬ Nat.Prime 182209503; exact PrattCertificates.composite_of_product (factor := 3) (cofactor := 60736501) rfl (by decide) (by decide)
    exact decide_eq_false composite
  second := false
  second_correct := by
    have composite : ¬ Nat.Prime (corner 6749 2) := by change ¬ Nat.Prime 182196005; exact PrattCertificates.composite_of_product (factor := 5) (cofactor := 36439201) rfl (by decide) (by decide)
    exact decide_eq_false composite
  third := false
  third_correct := by
    have composite : ¬ Nat.Prime (corner 6749 3) := by change ¬ Nat.Prime 182182507; exact PrattCertificates.composite_of_product (factor := 13) (cofactor := 14014039) rfl (by decide) (by decide)
    exact decide_eq_false composite
theorem row_count_6749 : row 6749 = 0 := (row_6749.checked).trans rfl
theorem model_count_6749 : modelRow 6749 = 0 := by decide
theorem modelrow_agrees_6749 : modelRow 6749 = row 6749 := model_count_6749.trans row_count_6749.symm
def row_6750 : RowCertificate 6750 where
  first := false
  first_correct := by
    have composite : ¬ Nat.Prime (corner 6750 1) := by change ¬ Nat.Prime 182263501; exact PrattCertificates.composite_of_product (factor := 7) (cofactor := 26037643) rfl (by decide) (by decide)
    exact decide_eq_false composite
  second := false
  second_correct := by
    have composite : ¬ Nat.Prime (corner 6750 2) := by change ¬ Nat.Prime 182250001; exact PrattCertificates.composite_of_product (factor := 389) (cofactor := 468509) rfl (by decide) (by decide)
    exact decide_eq_false composite
  third := true
  third_correct := by
    have prime : Nat.Prime (corner 6750 3) := by change Nat.Prime 182236501; exact ProjectEulerSolutions.P58PrimeNodes.p_182236501.prime
    exact decide_eq_true prime
theorem row_count_6750 : row 6750 = 1 := (row_6750.checked).trans rfl
theorem model_count_6750 : modelRow 6750 = 1 := by decide
theorem modelrow_agrees_6750 : modelRow 6750 = row 6750 := model_count_6750.trans row_count_6750.symm
def row_6751 : RowCertificate 6751 where
  first := true
  first_correct := by
    have prime : Nat.Prime (corner 6751 1) := by change Nat.Prime 182317507; exact ProjectEulerSolutions.P58PrimeNodes.p_182317507.prime
    exact decide_eq_true prime
  second := false
  second_correct := by
    have composite : ¬ Nat.Prime (corner 6751 2) := by change ¬ Nat.Prime 182304005; exact PrattCertificates.composite_of_product (factor := 5) (cofactor := 36460801) rfl (by decide) (by decide)
    exact decide_eq_false composite
  third := false
  third_correct := by
    have composite : ¬ Nat.Prime (corner 6751 3) := by change ¬ Nat.Prime 182290503; exact PrattCertificates.composite_of_product (factor := 3) (cofactor := 60763501) rfl (by decide) (by decide)
    exact decide_eq_false composite
theorem row_count_6751 : row 6751 = 1 := (row_6751.checked).trans rfl
theorem model_count_6751 : modelRow 6751 = 1 := by decide
theorem modelrow_agrees_6751 : modelRow 6751 = row 6751 := model_count_6751.trans row_count_6751.symm
def row_6752 : RowCertificate 6752 where
  first := false
  first_correct := by
    have composite : ¬ Nat.Prime (corner 6752 1) := by change ¬ Nat.Prime 182371521; exact PrattCertificates.composite_of_product (factor := 3) (cofactor := 60790507) rfl (by decide) (by decide)
    exact decide_eq_false composite
  second := false
  second_correct := by
    have composite : ¬ Nat.Prime (corner 6752 2) := by change ¬ Nat.Prime 182358017; exact PrattCertificates.composite_of_product (factor := 4261) (cofactor := 42797) rfl (by decide) (by decide)
    exact decide_eq_false composite
  third := false
  third_correct := by
    have composite : ¬ Nat.Prime (corner 6752 3) := by change ¬ Nat.Prime 182344513; exact PrattCertificates.composite_of_product (factor := 13) (cofactor := 14026501) rfl (by decide) (by decide)
    exact decide_eq_false composite
theorem row_count_6752 : row 6752 = 0 := (row_6752.checked).trans rfl
theorem model_count_6752 : modelRow 6752 = 0 := by decide
theorem modelrow_agrees_6752 : modelRow 6752 = row 6752 := model_count_6752.trans row_count_6752.symm
def row_6753 : RowCertificate 6753 where
  first := false
  first_correct := by
    have composite : ¬ Nat.Prime (corner 6753 1) := by change ¬ Nat.Prime 182425543; exact PrattCertificates.composite_of_product (factor := 2767) (cofactor := 65929) rfl (by decide) (by decide)
    exact decide_eq_false composite
  second := true
  second_correct := by
    have prime : Nat.Prime (corner 6753 2) := by change Nat.Prime 182412037; exact ProjectEulerSolutions.P58PrimeNodes.p_182412037.prime
    exact decide_eq_true prime
  third := false
  third_correct := by
    have composite : ¬ Nat.Prime (corner 6753 3) := by change ¬ Nat.Prime 182398531; exact PrattCertificates.composite_of_product (factor := 7) (cofactor := 26056933) rfl (by decide) (by decide)
    exact decide_eq_false composite
theorem row_count_6753 : row 6753 = 1 := (row_6753.checked).trans rfl
theorem model_count_6753 : modelRow 6753 = 1 := by decide
theorem modelrow_agrees_6753 : modelRow 6753 = row 6753 := model_count_6753.trans row_count_6753.symm
def row_6754 : RowCertificate 6754 where
  first := false
  first_correct := by
    have composite : ¬ Nat.Prime (corner 6754 1) := by change ¬ Nat.Prime 182479573; exact PrattCertificates.composite_of_product (factor := 43) (cofactor := 4243711) rfl (by decide) (by decide)
    exact decide_eq_false composite
  second := false
  second_correct := by
    have composite : ¬ Nat.Prime (corner 6754 2) := by change ¬ Nat.Prime 182466065; exact PrattCertificates.composite_of_product (factor := 5) (cofactor := 36493213) rfl (by decide) (by decide)
    exact decide_eq_false composite
  third := false
  third_correct := by
    have composite : ¬ Nat.Prime (corner 6754 3) := by change ¬ Nat.Prime 182452557; exact PrattCertificates.composite_of_product (factor := 3) (cofactor := 60817519) rfl (by decide) (by decide)
    exact decide_eq_false composite
theorem row_count_6754 : row 6754 = 0 := (row_6754.checked).trans rfl
theorem model_count_6754 : modelRow 6754 = 0 := by decide
theorem modelrow_agrees_6754 : modelRow 6754 = row 6754 := model_count_6754.trans row_count_6754.symm
def row_6755 : RowCertificate 6755 where
  first := false
  first_correct := by
    have composite : ¬ Nat.Prime (corner 6755 1) := by change ¬ Nat.Prime 182533611; exact PrattCertificates.composite_of_product (factor := 3) (cofactor := 60844537) rfl (by decide) (by decide)
    exact decide_eq_false composite
  second := false
  second_correct := by
    have composite : ¬ Nat.Prime (corner 6755 2) := by change ¬ Nat.Prime 182520101; exact PrattCertificates.composite_of_product (factor := 829) (cofactor := 220169) rfl (by decide) (by decide)
    exact decide_eq_false composite
  third := false
  third_correct := by
    have composite : ¬ Nat.Prime (corner 6755 3) := by change ¬ Nat.Prime 182506591; exact PrattCertificates.composite_of_product (factor := 1123) (cofactor := 162517) rfl (by decide) (by decide)
    exact decide_eq_false composite
theorem row_count_6755 : row 6755 = 0 := (row_6755.checked).trans rfl
theorem model_count_6755 : modelRow 6755 = 0 := by decide
theorem modelrow_agrees_6755 : modelRow 6755 = row 6755 := model_count_6755.trans row_count_6755.symm
def row_6756 : RowCertificate 6756 where
  first := false
  first_correct := by
    have composite : ¬ Nat.Prime (corner 6756 1) := by change ¬ Nat.Prime 182587657; exact PrattCertificates.composite_of_product (factor := 7) (cofactor := 26083951) rfl (by decide) (by decide)
    exact decide_eq_false composite
  second := false
  second_correct := by
    have composite : ¬ Nat.Prime (corner 6756 2) := by change ¬ Nat.Prime 182574145; exact PrattCertificates.composite_of_product (factor := 5) (cofactor := 36514829) rfl (by decide) (by decide)
    exact decide_eq_false composite
  third := false
  third_correct := by
    have composite : ¬ Nat.Prime (corner 6756 3) := by change ¬ Nat.Prime 182560633; exact PrattCertificates.composite_of_product (factor := 241) (cofactor := 757513) rfl (by decide) (by decide)
    exact decide_eq_false composite
theorem row_count_6756 : row 6756 = 0 := (row_6756.checked).trans rfl
theorem model_count_6756 : modelRow 6756 = 0 := by decide
theorem modelrow_agrees_6756 : modelRow 6756 = row 6756 := model_count_6756.trans row_count_6756.symm
def row_6757 : RowCertificate 6757 where
  first := false
  first_correct := by
    have composite : ¬ Nat.Prime (corner 6757 1) := by change ¬ Nat.Prime 182641711; exact PrattCertificates.composite_of_product (factor := 7) (cofactor := 26091673) rfl (by decide) (by decide)
    exact decide_eq_false composite
  second := false
  second_correct := by
    have composite : ¬ Nat.Prime (corner 6757 2) := by change ¬ Nat.Prime 182628197; exact PrattCertificates.composite_of_product (factor := 677) (cofactor := 269761) rfl (by decide) (by decide)
    exact decide_eq_false composite
  third := false
  third_correct := by
    have composite : ¬ Nat.Prime (corner 6757 3) := by change ¬ Nat.Prime 182614683; exact PrattCertificates.composite_of_product (factor := 3) (cofactor := 60871561) rfl (by decide) (by decide)
    exact decide_eq_false composite
theorem row_count_6757 : row 6757 = 0 := (row_6757.checked).trans rfl
theorem model_count_6757 : modelRow 6757 = 0 := by decide
theorem modelrow_agrees_6757 : modelRow 6757 = row 6757 := model_count_6757.trans row_count_6757.symm
def row_6758 : RowCertificate 6758 where
  first := false
  first_correct := by
    have composite : ¬ Nat.Prime (corner 6758 1) := by change ¬ Nat.Prime 182695773; exact PrattCertificates.composite_of_product (factor := 3) (cofactor := 60898591) rfl (by decide) (by decide)
    exact decide_eq_false composite
  second := true
  second_correct := by
    have prime : Nat.Prime (corner 6758 2) := by change Nat.Prime 182682257; exact ProjectEulerSolutions.P58PrimeNodes.p_182682257.prime
    exact decide_eq_true prime
  third := false
  third_correct := by
    have composite : ¬ Nat.Prime (corner 6758 3) := by change ¬ Nat.Prime 182668741; exact PrattCertificates.composite_of_product (factor := 37) (cofactor := 4936993) rfl (by decide) (by decide)
    exact decide_eq_false composite
theorem row_count_6758 : row 6758 = 1 := (row_6758.checked).trans rfl
theorem model_count_6758 : modelRow 6758 = 1 := by decide
theorem modelrow_agrees_6758 : modelRow 6758 = row 6758 := model_count_6758.trans row_count_6758.symm
def row_6759 : RowCertificate 6759 where
  first := false
  first_correct := by
    have composite : ¬ Nat.Prime (corner 6759 1) := by change ¬ Nat.Prime 182749843; exact PrattCertificates.composite_of_product (factor := 97) (cofactor := 1884019) rfl (by decide) (by decide)
    exact decide_eq_false composite
  second := false
  second_correct := by
    have composite : ¬ Nat.Prime (corner 6759 2) := by change ¬ Nat.Prime 182736325; exact PrattCertificates.composite_of_product (factor := 5) (cofactor := 36547265) rfl (by decide) (by decide)
    exact decide_eq_false composite
  third := true
  third_correct := by
    have prime : Nat.Prime (corner 6759 3) := by change Nat.Prime 182722807; exact ProjectEulerSolutions.P58PrimeNodes.p_182722807.prime
    exact decide_eq_true prime
theorem row_count_6759 : row 6759 = 1 := (row_6759.checked).trans rfl
theorem model_count_6759 : modelRow 6759 = 1 := by decide
theorem modelrow_agrees_6759 : modelRow 6759 = row 6759 := model_count_6759.trans row_count_6759.symm
def row_6760 : RowCertificate 6760 where
  first := false
  first_correct := by
    have composite : ¬ Nat.Prime (corner 6760 1) := by change ¬ Nat.Prime 182803921; exact PrattCertificates.composite_of_product (factor := 19) (cofactor := 9621259) rfl (by decide) (by decide)
    exact decide_eq_false composite
  second := true
  second_correct := by
    have prime : Nat.Prime (corner 6760 2) := by change Nat.Prime 182790401; exact ProjectEulerSolutions.P58PrimeNodes.p_182790401.prime
    exact decide_eq_true prime
  third := false
  third_correct := by
    have composite : ¬ Nat.Prime (corner 6760 3) := by change ¬ Nat.Prime 182776881; exact PrattCertificates.composite_of_product (factor := 3) (cofactor := 60925627) rfl (by decide) (by decide)
    exact decide_eq_false composite
theorem row_count_6760 : row 6760 = 1 := (row_6760.checked).trans rfl
theorem model_count_6760 : modelRow 6760 = 1 := by decide
theorem modelrow_agrees_6760 : modelRow 6760 = row 6760 := model_count_6760.trans row_count_6760.symm
def row_6761 : RowCertificate 6761 where
  first := false
  first_correct := by
    have composite : ¬ Nat.Prime (corner 6761 1) := by change ¬ Nat.Prime 182858007; exact PrattCertificates.composite_of_product (factor := 3) (cofactor := 60952669) rfl (by decide) (by decide)
    exact decide_eq_false composite
  second := false
  second_correct := by
    have composite : ¬ Nat.Prime (corner 6761 2) := by change ¬ Nat.Prime 182844485; exact PrattCertificates.composite_of_product (factor := 5) (cofactor := 36568897) rfl (by decide) (by decide)
    exact decide_eq_false composite
  third := false
  third_correct := by
    have composite : ¬ Nat.Prime (corner 6761 3) := by change ¬ Nat.Prime 182830963; exact PrattCertificates.composite_of_product (factor := 7) (cofactor := 26118709) rfl (by decide) (by decide)
    exact decide_eq_false composite
theorem row_count_6761 : row 6761 = 0 := (row_6761.checked).trans rfl
theorem model_count_6761 : modelRow 6761 = 0 := by decide
theorem modelrow_agrees_6761 : modelRow 6761 = row 6761 := model_count_6761.trans row_count_6761.symm
def row_6762 : RowCertificate 6762 where
  first := false
  first_correct := by
    have composite : ¬ Nat.Prime (corner 6762 1) := by change ¬ Nat.Prime 182912101; exact PrattCertificates.composite_of_product (factor := 1117) (cofactor := 163753) rfl (by decide) (by decide)
    exact decide_eq_false composite
  second := false
  second_correct := by
    have composite : ¬ Nat.Prime (corner 6762 2) := by change ¬ Nat.Prime 182898577; exact PrattCertificates.composite_of_product (factor := 101) (cofactor := 1810877) rfl (by decide) (by decide)
    exact decide_eq_false composite
  third := false
  third_correct := by
    have composite : ¬ Nat.Prime (corner 6762 3) := by change ¬ Nat.Prime 182885053; exact PrattCertificates.composite_of_product (factor := 13) (cofactor := 14068081) rfl (by decide) (by decide)
    exact decide_eq_false composite
theorem row_count_6762 : row 6762 = 0 := (row_6762.checked).trans rfl
theorem model_count_6762 : modelRow 6762 = 0 := by decide
theorem modelrow_agrees_6762 : modelRow 6762 = row 6762 := model_count_6762.trans row_count_6762.symm
def row_6763 : RowCertificate 6763 where
  first := false
  first_correct := by
    have composite : ¬ Nat.Prime (corner 6763 1) := by change ¬ Nat.Prime 182966203; exact PrattCertificates.composite_of_product (factor := 7) (cofactor := 26138029) rfl (by decide) (by decide)
    exact decide_eq_false composite
  second := false
  second_correct := by
    have composite : ¬ Nat.Prime (corner 6763 2) := by change ¬ Nat.Prime 182952677; exact PrattCertificates.composite_of_product (factor := 29) (cofactor := 6308713) rfl (by decide) (by decide)
    exact decide_eq_false composite
  third := false
  third_correct := by
    have composite : ¬ Nat.Prime (corner 6763 3) := by change ¬ Nat.Prime 182939151; exact PrattCertificates.composite_of_product (factor := 3) (cofactor := 60979717) rfl (by decide) (by decide)
    exact decide_eq_false composite
theorem row_count_6763 : row 6763 = 0 := (row_6763.checked).trans rfl
theorem model_count_6763 : modelRow 6763 = 0 := by decide
theorem modelrow_agrees_6763 : modelRow 6763 = row 6763 := model_count_6763.trans row_count_6763.symm
def row_6764 : RowCertificate 6764 where
  first := false
  first_correct := by
    have composite : ¬ Nat.Prime (corner 6764 1) := by change ¬ Nat.Prime 183020313; exact PrattCertificates.composite_of_product (factor := 3) (cofactor := 61006771) rfl (by decide) (by decide)
    exact decide_eq_false composite
  second := false
  second_correct := by
    have composite : ¬ Nat.Prime (corner 6764 2) := by change ¬ Nat.Prime 183006785; exact PrattCertificates.composite_of_product (factor := 5) (cofactor := 36601357) rfl (by decide) (by decide)
    exact decide_eq_false composite
  third := true
  third_correct := by
    have prime : Nat.Prime (corner 6764 3) := by change Nat.Prime 182993257; exact ProjectEulerSolutions.P58PrimeNodes.p_182993257.prime
    exact decide_eq_true prime
theorem row_count_6764 : row 6764 = 1 := (row_6764.checked).trans rfl
theorem model_count_6764 : modelRow 6764 = 1 := by decide
theorem modelrow_agrees_6764 : modelRow 6764 = row 6764 := model_count_6764.trans row_count_6764.symm
def row_6765 : RowCertificate 6765 where
  first := false
  first_correct := by
    have composite : ¬ Nat.Prime (corner 6765 1) := by change ¬ Nat.Prime 183074431; exact PrattCertificates.composite_of_product (factor := 7027) (cofactor := 26053) rfl (by decide) (by decide)
    exact decide_eq_false composite
  second := false
  second_correct := by
    have composite : ¬ Nat.Prime (corner 6765 2) := by change ¬ Nat.Prime 183060901; exact PrattCertificates.composite_of_product (factor := 157) (cofactor := 1165993) rfl (by decide) (by decide)
    exact decide_eq_false composite
  third := false
  third_correct := by
    have composite : ¬ Nat.Prime (corner 6765 3) := by change ¬ Nat.Prime 183047371; exact PrattCertificates.composite_of_product (factor := 13) (cofactor := 14080567) rfl (by decide) (by decide)
    exact decide_eq_false composite
theorem row_count_6765 : row 6765 = 0 := (row_6765.checked).trans rfl
theorem model_count_6765 : modelRow 6765 = 0 := by decide
theorem modelrow_agrees_6765 : modelRow 6765 = row 6765 := model_count_6765.trans row_count_6765.symm
def row_6766 : RowCertificate 6766 where
  first := false
  first_correct := by
    have composite : ¬ Nat.Prime (corner 6766 1) := by change ¬ Nat.Prime 183128557; exact PrattCertificates.composite_of_product (factor := 79) (cofactor := 2318083) rfl (by decide) (by decide)
    exact decide_eq_false composite
  second := false
  second_correct := by
    have composite : ¬ Nat.Prime (corner 6766 2) := by change ¬ Nat.Prime 183115025; exact PrattCertificates.composite_of_product (factor := 5) (cofactor := 36623005) rfl (by decide) (by decide)
    exact decide_eq_false composite
  third := false
  third_correct := by
    have composite : ¬ Nat.Prime (corner 6766 3) := by change ¬ Nat.Prime 183101493; exact PrattCertificates.composite_of_product (factor := 3) (cofactor := 61033831) rfl (by decide) (by decide)
    exact decide_eq_false composite
theorem row_count_6766 : row 6766 = 0 := (row_6766.checked).trans rfl
theorem model_count_6766 : modelRow 6766 = 0 := by decide
theorem modelrow_agrees_6766 : modelRow 6766 = row 6766 := model_count_6766.trans row_count_6766.symm
def row_6767 : RowCertificate 6767 where
  first := false
  first_correct := by
    have composite : ¬ Nat.Prime (corner 6767 1) := by change ¬ Nat.Prime 183182691; exact PrattCertificates.composite_of_product (factor := 3) (cofactor := 61060897) rfl (by decide) (by decide)
    exact decide_eq_false composite
  second := true
  second_correct := by
    have prime : Nat.Prime (corner 6767 2) := by change Nat.Prime 183169157; exact ProjectEulerSolutions.P58PrimeNodes.p_183169157.prime
    exact decide_eq_true prime
  third := false
  third_correct := by
    have composite : ¬ Nat.Prime (corner 6767 3) := by change ¬ Nat.Prime 183155623; exact PrattCertificates.composite_of_product (factor := 7) (cofactor := 26165089) rfl (by decide) (by decide)
    exact decide_eq_false composite
theorem row_count_6767 : row 6767 = 1 := (row_6767.checked).trans rfl
theorem model_count_6767 : modelRow 6767 = 1 := by decide
theorem modelrow_agrees_6767 : modelRow 6767 = row 6767 := model_count_6767.trans row_count_6767.symm
def row_6768 : RowCertificate 6768 where
  first := false
  first_correct := by
    have composite : ¬ Nat.Prime (corner 6768 1) := by change ¬ Nat.Prime 183236833; exact PrattCertificates.composite_of_product (factor := 13) (cofactor := 14095141) rfl (by decide) (by decide)
    exact decide_eq_false composite
  second := false
  second_correct := by
    have composite : ¬ Nat.Prime (corner 6768 2) := by change ¬ Nat.Prime 183223297; exact PrattCertificates.composite_of_product (factor := 17) (cofactor := 10777841) rfl (by decide) (by decide)
    exact decide_eq_false composite
  third := false
  third_correct := by
    have composite : ¬ Nat.Prime (corner 6768 3) := by change ¬ Nat.Prime 183209761; exact PrattCertificates.composite_of_product (factor := 7) (cofactor := 26172823) rfl (by decide) (by decide)
    exact decide_eq_false composite
theorem row_count_6768 : row 6768 = 0 := (row_6768.checked).trans rfl
theorem model_count_6768 : modelRow 6768 = 0 := by decide
theorem modelrow_agrees_6768 : modelRow 6768 = row 6768 := model_count_6768.trans row_count_6768.symm
def row_6769 : RowCertificate 6769 where
  first := false
  first_correct := by
    have composite : ¬ Nat.Prime (corner 6769 1) := by change ¬ Nat.Prime 183290983; exact PrattCertificates.composite_of_product (factor := 43) (cofactor := 4262581) rfl (by decide) (by decide)
    exact decide_eq_false composite
  second := false
  second_correct := by
    have composite : ¬ Nat.Prime (corner 6769 2) := by change ¬ Nat.Prime 183277445; exact PrattCertificates.composite_of_product (factor := 5) (cofactor := 36655489) rfl (by decide) (by decide)
    exact decide_eq_false composite
  third := false
  third_correct := by
    have composite : ¬ Nat.Prime (corner 6769 3) := by change ¬ Nat.Prime 183263907; exact PrattCertificates.composite_of_product (factor := 3) (cofactor := 61087969) rfl (by decide) (by decide)
    exact decide_eq_false composite
theorem row_count_6769 : row 6769 = 0 := (row_6769.checked).trans rfl
theorem model_count_6769 : modelRow 6769 = 0 := by decide
theorem modelrow_agrees_6769 : modelRow 6769 = row 6769 := model_count_6769.trans row_count_6769.symm
def row_6770 : RowCertificate 6770 where
  first := false
  first_correct := by
    have composite : ¬ Nat.Prime (corner 6770 1) := by change ¬ Nat.Prime 183345141; exact PrattCertificates.composite_of_product (factor := 3) (cofactor := 61115047) rfl (by decide) (by decide)
    exact decide_eq_false composite
  second := false
  second_correct := by
    have composite : ¬ Nat.Prime (corner 6770 2) := by change ¬ Nat.Prime 183331601; exact PrattCertificates.composite_of_product (factor := 13177) (cofactor := 13913) rfl (by decide) (by decide)
    exact decide_eq_false composite
  third := false
  third_correct := by
    have composite : ¬ Nat.Prime (corner 6770 3) := by change ¬ Nat.Prime 183318061; exact PrattCertificates.composite_of_product (factor := 19) (cofactor := 9648319) rfl (by decide) (by decide)
    exact decide_eq_false composite
theorem row_count_6770 : row 6770 = 0 := (row_6770.checked).trans rfl
theorem model_count_6770 : modelRow 6770 = 0 := by decide
theorem modelrow_agrees_6770 : modelRow 6770 = row 6770 := model_count_6770.trans row_count_6770.symm
def row_6771 : RowCertificate 6771 where
  first := false
  first_correct := by
    have composite : ¬ Nat.Prime (corner 6771 1) := by change ¬ Nat.Prime 183399307; exact PrattCertificates.composite_of_product (factor := 7) (cofactor := 26199901) rfl (by decide) (by decide)
    exact decide_eq_false composite
  second := false
  second_correct := by
    have composite : ¬ Nat.Prime (corner 6771 2) := by change ¬ Nat.Prime 183385765; exact PrattCertificates.composite_of_product (factor := 5) (cofactor := 36677153) rfl (by decide) (by decide)
    exact decide_eq_false composite
  third := false
  third_correct := by
    have composite : ¬ Nat.Prime (corner 6771 3) := by change ¬ Nat.Prime 183372223; exact PrattCertificates.composite_of_product (factor := 31) (cofactor := 5915233) rfl (by decide) (by decide)
    exact decide_eq_false composite
theorem row_count_6771 : row 6771 = 0 := (row_6771.checked).trans rfl
theorem model_count_6771 : modelRow 6771 = 0 := by decide
theorem modelrow_agrees_6771 : modelRow 6771 = row 6771 := model_count_6771.trans row_count_6771.symm
def row_6772 : RowCertificate 6772 where
  first := false
  first_correct := by
    have composite : ¬ Nat.Prime (corner 6772 1) := by change ¬ Nat.Prime 183453481; exact PrattCertificates.composite_of_product (factor := 97) (cofactor := 1891273) rfl (by decide) (by decide)
    exact decide_eq_false composite
  second := false
  second_correct := by
    have composite : ¬ Nat.Prime (corner 6772 2) := by change ¬ Nat.Prime 183439937; exact PrattCertificates.composite_of_product (factor := 101) (cofactor := 1816237) rfl (by decide) (by decide)
    exact decide_eq_false composite
  third := false
  third_correct := by
    have composite : ¬ Nat.Prime (corner 6772 3) := by change ¬ Nat.Prime 183426393; exact PrattCertificates.composite_of_product (factor := 3) (cofactor := 61142131) rfl (by decide) (by decide)
    exact decide_eq_false composite
theorem row_count_6772 : row 6772 = 0 := (row_6772.checked).trans rfl
theorem model_count_6772 : modelRow 6772 = 0 := by decide
theorem modelrow_agrees_6772 : modelRow 6772 = row 6772 := model_count_6772.trans row_count_6772.symm
def row_6773 : RowCertificate 6773 where
  first := false
  first_correct := by
    have composite : ¬ Nat.Prime (corner 6773 1) := by change ¬ Nat.Prime 183507663; exact PrattCertificates.composite_of_product (factor := 3) (cofactor := 61169221) rfl (by decide) (by decide)
    exact decide_eq_false composite
  second := true
  second_correct := by
    have prime : Nat.Prime (corner 6773 2) := by change Nat.Prime 183494117; exact ProjectEulerSolutions.P58PrimeNodes.p_183494117.prime
    exact decide_eq_true prime
  third := true
  third_correct := by
    have prime : Nat.Prime (corner 6773 3) := by change Nat.Prime 183480571; exact ProjectEulerSolutions.P58PrimeNodes.p_183480571.prime
    exact decide_eq_true prime
theorem row_count_6773 : row 6773 = 2 := (row_6773.checked).trans rfl
theorem model_count_6773 : modelRow 6773 = 2 := by decide
theorem modelrow_agrees_6773 : modelRow 6773 = row 6773 := model_count_6773.trans row_count_6773.symm
def row_6774 : RowCertificate 6774 where
  first := true
  first_correct := by
    have prime : Nat.Prime (corner 6774 1) := by change Nat.Prime 183561853; exact ProjectEulerSolutions.P58PrimeNodes.p_183561853.prime
    exact decide_eq_true prime
  second := false
  second_correct := by
    have composite : ¬ Nat.Prime (corner 6774 2) := by change ¬ Nat.Prime 183548305; exact PrattCertificates.composite_of_product (factor := 5) (cofactor := 36709661) rfl (by decide) (by decide)
    exact decide_eq_false composite
  third := false
  third_correct := by
    have composite : ¬ Nat.Prime (corner 6774 3) := by change ¬ Nat.Prime 183534757; exact PrattCertificates.composite_of_product (factor := 7) (cofactor := 26219251) rfl (by decide) (by decide)
    exact decide_eq_false composite
theorem row_count_6774 : row 6774 = 1 := (row_6774.checked).trans rfl
theorem model_count_6774 : modelRow 6774 = 1 := by decide
theorem modelrow_agrees_6774 : modelRow 6774 = row 6774 := model_count_6774.trans row_count_6774.symm
def row_6775 : RowCertificate 6775 where
  first := false
  first_correct := by
    have composite : ¬ Nat.Prime (corner 6775 1) := by change ¬ Nat.Prime 183616051; exact PrattCertificates.composite_of_product (factor := 409) (cofactor := 448939) rfl (by decide) (by decide)
    exact decide_eq_false composite
  second := true
  second_correct := by
    have prime : Nat.Prime (corner 6775 2) := by change Nat.Prime 183602501; exact ProjectEulerSolutions.P58PrimeNodes.p_183602501.prime
    exact decide_eq_true prime
  third := false
  third_correct := by
    have composite : ¬ Nat.Prime (corner 6775 3) := by change ¬ Nat.Prime 183588951; exact PrattCertificates.composite_of_product (factor := 3) (cofactor := 61196317) rfl (by decide) (by decide)
    exact decide_eq_false composite
theorem row_count_6775 : row 6775 = 1 := (row_6775.checked).trans rfl
theorem model_count_6775 : modelRow 6775 = 1 := by decide
theorem modelrow_agrees_6775 : modelRow 6775 = row 6775 := model_count_6775.trans row_count_6775.symm
def row_6776 : RowCertificate 6776 where
  first := false
  first_correct := by
    have composite : ¬ Nat.Prime (corner 6776 1) := by change ¬ Nat.Prime 183670257; exact PrattCertificates.composite_of_product (factor := 3) (cofactor := 61223419) rfl (by decide) (by decide)
    exact decide_eq_false composite
  second := false
  second_correct := by
    have composite : ¬ Nat.Prime (corner 6776 2) := by change ¬ Nat.Prime 183656705; exact PrattCertificates.composite_of_product (factor := 5) (cofactor := 36731341) rfl (by decide) (by decide)
    exact decide_eq_false composite
  third := false
  third_correct := by
    have composite : ¬ Nat.Prime (corner 6776 3) := by change ¬ Nat.Prime 183643153; exact PrattCertificates.composite_of_product (factor := 43) (cofactor := 4270771) rfl (by decide) (by decide)
    exact decide_eq_false composite
theorem row_count_6776 : row 6776 = 0 := (row_6776.checked).trans rfl
theorem model_count_6776 : modelRow 6776 = 0 := by decide
theorem modelrow_agrees_6776 : modelRow 6776 = row 6776 := model_count_6776.trans row_count_6776.symm
def row_6777 : RowCertificate 6777 where
  first := false
  first_correct := by
    have composite : ¬ Nat.Prime (corner 6777 1) := by change ¬ Nat.Prime 183724471; exact PrattCertificates.composite_of_product (factor := 7) (cofactor := 26246353) rfl (by decide) (by decide)
    exact decide_eq_false composite
  second := false
  second_correct := by
    have composite : ¬ Nat.Prime (corner 6777 2) := by change ¬ Nat.Prime 183710917; exact PrattCertificates.composite_of_product (factor := 13) (cofactor := 14131609) rfl (by decide) (by decide)
    exact decide_eq_false composite
  third := true
  third_correct := by
    have prime : Nat.Prime (corner 6777 3) := by change Nat.Prime 183697363; exact ProjectEulerSolutions.P58PrimeNodes.p_183697363.prime
    exact decide_eq_true prime
theorem row_count_6777 : row 6777 = 1 := (row_6777.checked).trans rfl
theorem model_count_6777 : modelRow 6777 = 1 := by decide
theorem modelrow_agrees_6777 : modelRow 6777 = row 6777 := model_count_6777.trans row_count_6777.symm
def row_6778 : RowCertificate 6778 where
  first := false
  first_correct := by
    have composite : ¬ Nat.Prime (corner 6778 1) := by change ¬ Nat.Prime 183778693; exact PrattCertificates.composite_of_product (factor := 7) (cofactor := 26254099) rfl (by decide) (by decide)
    exact decide_eq_false composite
  second := false
  second_correct := by
    have composite : ¬ Nat.Prime (corner 6778 2) := by change ¬ Nat.Prime 183765137; exact PrattCertificates.composite_of_product (factor := 181) (cofactor := 1015277) rfl (by decide) (by decide)
    exact decide_eq_false composite
  third := false
  third_correct := by
    have composite : ¬ Nat.Prime (corner 6778 3) := by change ¬ Nat.Prime 183751581; exact PrattCertificates.composite_of_product (factor := 3) (cofactor := 61250527) rfl (by decide) (by decide)
    exact decide_eq_false composite
theorem row_count_6778 : row 6778 = 0 := (row_6778.checked).trans rfl
theorem model_count_6778 : modelRow 6778 = 0 := by decide
theorem modelrow_agrees_6778 : modelRow 6778 = row 6778 := model_count_6778.trans row_count_6778.symm
def row_6779 : RowCertificate 6779 where
  first := false
  first_correct := by
    have composite : ¬ Nat.Prime (corner 6779 1) := by change ¬ Nat.Prime 183832923; exact PrattCertificates.composite_of_product (factor := 3) (cofactor := 61277641) rfl (by decide) (by decide)
    exact decide_eq_false composite
  second := false
  second_correct := by
    have composite : ¬ Nat.Prime (corner 6779 2) := by change ¬ Nat.Prime 183819365; exact PrattCertificates.composite_of_product (factor := 5) (cofactor := 36763873) rfl (by decide) (by decide)
    exact decide_eq_false composite
  third := false
  third_correct := by
    have composite : ¬ Nat.Prime (corner 6779 3) := by change ¬ Nat.Prime 183805807; exact PrattCertificates.composite_of_product (factor := 151) (cofactor := 1217257) rfl (by decide) (by decide)
    exact decide_eq_false composite
theorem row_count_6779 : row 6779 = 0 := (row_6779.checked).trans rfl
theorem model_count_6779 : modelRow 6779 = 0 := by decide
theorem modelrow_agrees_6779 : modelRow 6779 = row 6779 := model_count_6779.trans row_count_6779.symm
def row_6780 : RowCertificate 6780 where
  first := false
  first_correct := by
    have composite : ¬ Nat.Prime (corner 6780 1) := by change ¬ Nat.Prime 183887161; exact PrattCertificates.composite_of_product (factor := 1213) (cofactor := 151597) rfl (by decide) (by decide)
    exact decide_eq_false composite
  second := false
  second_correct := by
    have composite : ¬ Nat.Prime (corner 6780 2) := by change ¬ Nat.Prime 183873601; exact PrattCertificates.composite_of_product (factor := 29) (cofactor := 6340469) rfl (by decide) (by decide)
    exact decide_eq_false composite
  third := false
  third_correct := by
    have composite : ¬ Nat.Prime (corner 6780 3) := by change ¬ Nat.Prime 183860041; exact PrattCertificates.composite_of_product (factor := 2161) (cofactor := 85081) rfl (by decide) (by decide)
    exact decide_eq_false composite
theorem row_count_6780 : row 6780 = 0 := (row_6780.checked).trans rfl
theorem model_count_6780 : modelRow 6780 = 0 := by decide
theorem modelrow_agrees_6780 : modelRow 6780 = row 6780 := model_count_6780.trans row_count_6780.symm
def row_6781 : RowCertificate 6781 where
  first := false
  first_correct := by
    have composite : ¬ Nat.Prime (corner 6781 1) := by change ¬ Nat.Prime 183941407; exact PrattCertificates.composite_of_product (factor := 13) (cofactor := 14149339) rfl (by decide) (by decide)
    exact decide_eq_false composite
  second := false
  second_correct := by
    have composite : ¬ Nat.Prime (corner 6781 2) := by change ¬ Nat.Prime 183927845; exact PrattCertificates.composite_of_product (factor := 5) (cofactor := 36785569) rfl (by decide) (by decide)
    exact decide_eq_false composite
  third := false
  third_correct := by
    have composite : ¬ Nat.Prime (corner 6781 3) := by change ¬ Nat.Prime 183914283; exact PrattCertificates.composite_of_product (factor := 3) (cofactor := 61304761) rfl (by decide) (by decide)
    exact decide_eq_false composite
theorem row_count_6781 : row 6781 = 0 := (row_6781.checked).trans rfl
theorem model_count_6781 : modelRow 6781 = 0 := by decide
theorem modelrow_agrees_6781 : modelRow 6781 = row 6781 := model_count_6781.trans row_count_6781.symm
def row_6782 : RowCertificate 6782 where
  first := false
  first_correct := by
    have composite : ¬ Nat.Prime (corner 6782 1) := by change ¬ Nat.Prime 183995661; exact PrattCertificates.composite_of_product (factor := 3) (cofactor := 61331887) rfl (by decide) (by decide)
    exact decide_eq_false composite
  second := false
  second_correct := by
    have composite : ¬ Nat.Prime (corner 6782 2) := by change ¬ Nat.Prime 183982097; exact PrattCertificates.composite_of_product (factor := 13) (cofactor := 14152469) rfl (by decide) (by decide)
    exact decide_eq_false composite
  third := false
  third_correct := by
    have composite : ¬ Nat.Prime (corner 6782 3) := by change ¬ Nat.Prime 183968533; exact PrattCertificates.composite_of_product (factor := 7) (cofactor := 26281219) rfl (by decide) (by decide)
    exact decide_eq_false composite
theorem row_count_6782 : row 6782 = 0 := (row_6782.checked).trans rfl
theorem model_count_6782 : modelRow 6782 = 0 := by decide
theorem modelrow_agrees_6782 : modelRow 6782 = row 6782 := model_count_6782.trans row_count_6782.symm
def row_6783 : RowCertificate 6783 where
  first := false
  first_correct := by
    have composite : ¬ Nat.Prime (corner 6783 1) := by change ¬ Nat.Prime 184049923; exact PrattCertificates.composite_of_product (factor := 691) (cofactor := 266353) rfl (by decide) (by decide)
    exact decide_eq_false composite
  second := false
  second_correct := by
    have composite : ¬ Nat.Prime (corner 6783 2) := by change ¬ Nat.Prime 184036357; exact PrattCertificates.composite_of_product (factor := 677) (cofactor := 271841) rfl (by decide) (by decide)
    exact decide_eq_false composite
  third := true
  third_correct := by
    have prime : Nat.Prime (corner 6783 3) := by change Nat.Prime 184022791; exact ProjectEulerSolutions.P58PrimeNodes.p_184022791.prime
    exact decide_eq_true prime
theorem row_count_6783 : row 6783 = 1 := (row_6783.checked).trans rfl
theorem model_count_6783 : modelRow 6783 = 1 := by decide
theorem modelrow_agrees_6783 : modelRow 6783 = row 6783 := model_count_6783.trans row_count_6783.symm
def row_6784 : RowCertificate 6784 where
  first := false
  first_correct := by
    have composite : ¬ Nat.Prime (corner 6784 1) := by change ¬ Nat.Prime 184104193; exact PrattCertificates.composite_of_product (factor := 7) (cofactor := 26300599) rfl (by decide) (by decide)
    exact decide_eq_false composite
  second := false
  second_correct := by
    have composite : ¬ Nat.Prime (corner 6784 2) := by change ¬ Nat.Prime 184090625; exact PrattCertificates.composite_of_product (factor := 5) (cofactor := 36818125) rfl (by decide) (by decide)
    exact decide_eq_false composite
  third := false
  third_correct := by
    have composite : ¬ Nat.Prime (corner 6784 3) := by change ¬ Nat.Prime 184077057; exact PrattCertificates.composite_of_product (factor := 3) (cofactor := 61359019) rfl (by decide) (by decide)
    exact decide_eq_false composite
theorem row_count_6784 : row 6784 = 0 := (row_6784.checked).trans rfl
theorem model_count_6784 : modelRow 6784 = 0 := by decide
theorem modelrow_agrees_6784 : modelRow 6784 = row 6784 := model_count_6784.trans row_count_6784.symm
private def atomic_6657 : CountSegment 1 10 13120 6656 6657 2875 2876 := CountSegment.single (first := 6656) (initial := 2875) row_count_6657 (by decide)
private theorem atomic_6657_agreement : RowAgreement 6656 6657 := RowAgreement.single (first := 6656) modelrow_agrees_6657
private def atomic_6658 : CountSegment 1 10 13120 6657 6658 2876 2876 := CountSegment.single (first := 6657) (initial := 2876) row_count_6658 (by decide)
private theorem atomic_6658_agreement : RowAgreement 6657 6658 := RowAgreement.single (first := 6657) modelrow_agrees_6658
private def atomic_6659 : CountSegment 1 10 13120 6658 6659 2876 2877 := CountSegment.single (first := 6658) (initial := 2876) row_count_6659 (by decide)
private theorem atomic_6659_agreement : RowAgreement 6658 6659 := RowAgreement.single (first := 6658) modelrow_agrees_6659
private def atomic_6660 : CountSegment 1 10 13120 6659 6660 2877 2878 := CountSegment.single (first := 6659) (initial := 2877) row_count_6660 (by decide)
private theorem atomic_6660_agreement : RowAgreement 6659 6660 := RowAgreement.single (first := 6659) modelrow_agrees_6660
private def atomic_6661 : CountSegment 1 10 13120 6660 6661 2878 2879 := CountSegment.single (first := 6660) (initial := 2878) row_count_6661 (by decide)
private theorem atomic_6661_agreement : RowAgreement 6660 6661 := RowAgreement.single (first := 6660) modelrow_agrees_6661
private def atomic_6662 : CountSegment 1 10 13120 6661 6662 2879 2879 := CountSegment.single (first := 6661) (initial := 2879) row_count_6662 (by decide)
private theorem atomic_6662_agreement : RowAgreement 6661 6662 := RowAgreement.single (first := 6661) modelrow_agrees_6662
private def atomic_6663 : CountSegment 1 10 13120 6662 6663 2879 2879 := CountSegment.single (first := 6662) (initial := 2879) row_count_6663 (by decide)
private theorem atomic_6663_agreement : RowAgreement 6662 6663 := RowAgreement.single (first := 6662) modelrow_agrees_6663
private def atomic_6664 : CountSegment 1 10 13120 6663 6664 2879 2879 := CountSegment.single (first := 6663) (initial := 2879) row_count_6664 (by decide)
private theorem atomic_6664_agreement : RowAgreement 6663 6664 := RowAgreement.single (first := 6663) modelrow_agrees_6664
private def atomic_6665 : CountSegment 1 10 13120 6664 6665 2879 2880 := CountSegment.single (first := 6664) (initial := 2879) row_count_6665 (by decide)
private theorem atomic_6665_agreement : RowAgreement 6664 6665 := RowAgreement.single (first := 6664) modelrow_agrees_6665
private def atomic_6666 : CountSegment 1 10 13120 6665 6666 2880 2881 := CountSegment.single (first := 6665) (initial := 2880) row_count_6666 (by decide)
private theorem atomic_6666_agreement : RowAgreement 6665 6666 := RowAgreement.single (first := 6665) modelrow_agrees_6666
private def atomic_6667 : CountSegment 1 10 13120 6666 6667 2881 2881 := CountSegment.single (first := 6666) (initial := 2881) row_count_6667 (by decide)
private theorem atomic_6667_agreement : RowAgreement 6666 6667 := RowAgreement.single (first := 6666) modelrow_agrees_6667
private def atomic_6668 : CountSegment 1 10 13120 6667 6668 2881 2882 := CountSegment.single (first := 6667) (initial := 2881) row_count_6668 (by decide)
private theorem atomic_6668_agreement : RowAgreement 6667 6668 := RowAgreement.single (first := 6667) modelrow_agrees_6668
private def atomic_6669 : CountSegment 1 10 13120 6668 6669 2882 2883 := CountSegment.single (first := 6668) (initial := 2882) row_count_6669 (by decide)
private theorem atomic_6669_agreement : RowAgreement 6668 6669 := RowAgreement.single (first := 6668) modelrow_agrees_6669
private def atomic_6670 : CountSegment 1 10 13120 6669 6670 2883 2883 := CountSegment.single (first := 6669) (initial := 2883) row_count_6670 (by decide)
private theorem atomic_6670_agreement : RowAgreement 6669 6670 := RowAgreement.single (first := 6669) modelrow_agrees_6670
private def atomic_6671 : CountSegment 1 10 13120 6670 6671 2883 2883 := CountSegment.single (first := 6670) (initial := 2883) row_count_6671 (by decide)
private theorem atomic_6671_agreement : RowAgreement 6670 6671 := RowAgreement.single (first := 6670) modelrow_agrees_6671
private def atomic_6672 : CountSegment 1 10 13120 6671 6672 2883 2884 := CountSegment.single (first := 6671) (initial := 2883) row_count_6672 (by decide)
private theorem atomic_6672_agreement : RowAgreement 6671 6672 := RowAgreement.single (first := 6671) modelrow_agrees_6672
private def atomic_6673 : CountSegment 1 10 13120 6672 6673 2884 2884 := CountSegment.single (first := 6672) (initial := 2884) row_count_6673 (by decide)
private theorem atomic_6673_agreement : RowAgreement 6672 6673 := RowAgreement.single (first := 6672) modelrow_agrees_6673
private def atomic_6674 : CountSegment 1 10 13120 6673 6674 2884 2884 := CountSegment.single (first := 6673) (initial := 2884) row_count_6674 (by decide)
private theorem atomic_6674_agreement : RowAgreement 6673 6674 := RowAgreement.single (first := 6673) modelrow_agrees_6674
private def atomic_6675 : CountSegment 1 10 13120 6674 6675 2884 2885 := CountSegment.single (first := 6674) (initial := 2884) row_count_6675 (by decide)
private theorem atomic_6675_agreement : RowAgreement 6674 6675 := RowAgreement.single (first := 6674) modelrow_agrees_6675
private def atomic_6676 : CountSegment 1 10 13120 6675 6676 2885 2885 := CountSegment.single (first := 6675) (initial := 2885) row_count_6676 (by decide)
private theorem atomic_6676_agreement : RowAgreement 6675 6676 := RowAgreement.single (first := 6675) modelrow_agrees_6676
private def atomic_6677 : CountSegment 1 10 13120 6676 6677 2885 2885 := CountSegment.single (first := 6676) (initial := 2885) row_count_6677 (by decide)
private theorem atomic_6677_agreement : RowAgreement 6676 6677 := RowAgreement.single (first := 6676) modelrow_agrees_6677
private def atomic_6678 : CountSegment 1 10 13120 6677 6678 2885 2886 := CountSegment.single (first := 6677) (initial := 2885) row_count_6678 (by decide)
private theorem atomic_6678_agreement : RowAgreement 6677 6678 := RowAgreement.single (first := 6677) modelrow_agrees_6678
private def atomic_6679 : CountSegment 1 10 13120 6678 6679 2886 2886 := CountSegment.single (first := 6678) (initial := 2886) row_count_6679 (by decide)
private theorem atomic_6679_agreement : RowAgreement 6678 6679 := RowAgreement.single (first := 6678) modelrow_agrees_6679
private def atomic_6680 : CountSegment 1 10 13120 6679 6680 2886 2888 := CountSegment.single (first := 6679) (initial := 2886) row_count_6680 (by decide)
private theorem atomic_6680_agreement : RowAgreement 6679 6680 := RowAgreement.single (first := 6679) modelrow_agrees_6680
private def atomic_6681 : CountSegment 1 10 13120 6680 6681 2888 2888 := CountSegment.single (first := 6680) (initial := 2888) row_count_6681 (by decide)
private theorem atomic_6681_agreement : RowAgreement 6680 6681 := RowAgreement.single (first := 6680) modelrow_agrees_6681
private def atomic_6682 : CountSegment 1 10 13120 6681 6682 2888 2888 := CountSegment.single (first := 6681) (initial := 2888) row_count_6682 (by decide)
private theorem atomic_6682_agreement : RowAgreement 6681 6682 := RowAgreement.single (first := 6681) modelrow_agrees_6682
private def atomic_6683 : CountSegment 1 10 13120 6682 6683 2888 2888 := CountSegment.single (first := 6682) (initial := 2888) row_count_6683 (by decide)
private theorem atomic_6683_agreement : RowAgreement 6682 6683 := RowAgreement.single (first := 6682) modelrow_agrees_6683
private def atomic_6684 : CountSegment 1 10 13120 6683 6684 2888 2888 := CountSegment.single (first := 6683) (initial := 2888) row_count_6684 (by decide)
private theorem atomic_6684_agreement : RowAgreement 6683 6684 := RowAgreement.single (first := 6683) modelrow_agrees_6684
private def atomic_6685 : CountSegment 1 10 13120 6684 6685 2888 2888 := CountSegment.single (first := 6684) (initial := 2888) row_count_6685 (by decide)
private theorem atomic_6685_agreement : RowAgreement 6684 6685 := RowAgreement.single (first := 6684) modelrow_agrees_6685
private def atomic_6686 : CountSegment 1 10 13120 6685 6686 2888 2889 := CountSegment.single (first := 6685) (initial := 2888) row_count_6686 (by decide)
private theorem atomic_6686_agreement : RowAgreement 6685 6686 := RowAgreement.single (first := 6685) modelrow_agrees_6686
private def atomic_6687 : CountSegment 1 10 13120 6686 6687 2889 2889 := CountSegment.single (first := 6686) (initial := 2889) row_count_6687 (by decide)
private theorem atomic_6687_agreement : RowAgreement 6686 6687 := RowAgreement.single (first := 6686) modelrow_agrees_6687
private def atomic_6688 : CountSegment 1 10 13120 6687 6688 2889 2890 := CountSegment.single (first := 6687) (initial := 2889) row_count_6688 (by decide)
private theorem atomic_6688_agreement : RowAgreement 6687 6688 := RowAgreement.single (first := 6687) modelrow_agrees_6688
private def atomic_6689 : CountSegment 1 10 13120 6688 6689 2890 2891 := CountSegment.single (first := 6688) (initial := 2890) row_count_6689 (by decide)
private theorem atomic_6689_agreement : RowAgreement 6688 6689 := RowAgreement.single (first := 6688) modelrow_agrees_6689
private def atomic_6690 : CountSegment 1 10 13120 6689 6690 2891 2891 := CountSegment.single (first := 6689) (initial := 2891) row_count_6690 (by decide)
private theorem atomic_6690_agreement : RowAgreement 6689 6690 := RowAgreement.single (first := 6689) modelrow_agrees_6690
private def atomic_6691 : CountSegment 1 10 13120 6690 6691 2891 2891 := CountSegment.single (first := 6690) (initial := 2891) row_count_6691 (by decide)
private theorem atomic_6691_agreement : RowAgreement 6690 6691 := RowAgreement.single (first := 6690) modelrow_agrees_6691
private def atomic_6692 : CountSegment 1 10 13120 6691 6692 2891 2891 := CountSegment.single (first := 6691) (initial := 2891) row_count_6692 (by decide)
private theorem atomic_6692_agreement : RowAgreement 6691 6692 := RowAgreement.single (first := 6691) modelrow_agrees_6692
private def atomic_6693 : CountSegment 1 10 13120 6692 6693 2891 2891 := CountSegment.single (first := 6692) (initial := 2891) row_count_6693 (by decide)
private theorem atomic_6693_agreement : RowAgreement 6692 6693 := RowAgreement.single (first := 6692) modelrow_agrees_6693
private def atomic_6694 : CountSegment 1 10 13120 6693 6694 2891 2891 := CountSegment.single (first := 6693) (initial := 2891) row_count_6694 (by decide)
private theorem atomic_6694_agreement : RowAgreement 6693 6694 := RowAgreement.single (first := 6693) modelrow_agrees_6694
private def atomic_6695 : CountSegment 1 10 13120 6694 6695 2891 2892 := CountSegment.single (first := 6694) (initial := 2891) row_count_6695 (by decide)
private theorem atomic_6695_agreement : RowAgreement 6694 6695 := RowAgreement.single (first := 6694) modelrow_agrees_6695
private def atomic_6696 : CountSegment 1 10 13120 6695 6696 2892 2893 := CountSegment.single (first := 6695) (initial := 2892) row_count_6696 (by decide)
private theorem atomic_6696_agreement : RowAgreement 6695 6696 := RowAgreement.single (first := 6695) modelrow_agrees_6696
private def atomic_6697 : CountSegment 1 10 13120 6696 6697 2893 2894 := CountSegment.single (first := 6696) (initial := 2893) row_count_6697 (by decide)
private theorem atomic_6697_agreement : RowAgreement 6696 6697 := RowAgreement.single (first := 6696) modelrow_agrees_6697
private def atomic_6698 : CountSegment 1 10 13120 6697 6698 2894 2894 := CountSegment.single (first := 6697) (initial := 2894) row_count_6698 (by decide)
private theorem atomic_6698_agreement : RowAgreement 6697 6698 := RowAgreement.single (first := 6697) modelrow_agrees_6698
private def atomic_6699 : CountSegment 1 10 13120 6698 6699 2894 2895 := CountSegment.single (first := 6698) (initial := 2894) row_count_6699 (by decide)
private theorem atomic_6699_agreement : RowAgreement 6698 6699 := RowAgreement.single (first := 6698) modelrow_agrees_6699
private def atomic_6700 : CountSegment 1 10 13120 6699 6700 2895 2895 := CountSegment.single (first := 6699) (initial := 2895) row_count_6700 (by decide)
private theorem atomic_6700_agreement : RowAgreement 6699 6700 := RowAgreement.single (first := 6699) modelrow_agrees_6700
private def atomic_6701 : CountSegment 1 10 13120 6700 6701 2895 2895 := CountSegment.single (first := 6700) (initial := 2895) row_count_6701 (by decide)
private theorem atomic_6701_agreement : RowAgreement 6700 6701 := RowAgreement.single (first := 6700) modelrow_agrees_6701
private def atomic_6702 : CountSegment 1 10 13120 6701 6702 2895 2895 := CountSegment.single (first := 6701) (initial := 2895) row_count_6702 (by decide)
private theorem atomic_6702_agreement : RowAgreement 6701 6702 := RowAgreement.single (first := 6701) modelrow_agrees_6702
private def atomic_6703 : CountSegment 1 10 13120 6702 6703 2895 2895 := CountSegment.single (first := 6702) (initial := 2895) row_count_6703 (by decide)
private theorem atomic_6703_agreement : RowAgreement 6702 6703 := RowAgreement.single (first := 6702) modelrow_agrees_6703
private def atomic_6704 : CountSegment 1 10 13120 6703 6704 2895 2895 := CountSegment.single (first := 6703) (initial := 2895) row_count_6704 (by decide)
private theorem atomic_6704_agreement : RowAgreement 6703 6704 := RowAgreement.single (first := 6703) modelrow_agrees_6704
private def atomic_6705 : CountSegment 1 10 13120 6704 6705 2895 2896 := CountSegment.single (first := 6704) (initial := 2895) row_count_6705 (by decide)
private theorem atomic_6705_agreement : RowAgreement 6704 6705 := RowAgreement.single (first := 6704) modelrow_agrees_6705
private def atomic_6706 : CountSegment 1 10 13120 6705 6706 2896 2896 := CountSegment.single (first := 6705) (initial := 2896) row_count_6706 (by decide)
private theorem atomic_6706_agreement : RowAgreement 6705 6706 := RowAgreement.single (first := 6705) modelrow_agrees_6706
private def atomic_6707 : CountSegment 1 10 13120 6706 6707 2896 2896 := CountSegment.single (first := 6706) (initial := 2896) row_count_6707 (by decide)
private theorem atomic_6707_agreement : RowAgreement 6706 6707 := RowAgreement.single (first := 6706) modelrow_agrees_6707
private def atomic_6708 : CountSegment 1 10 13120 6707 6708 2896 2897 := CountSegment.single (first := 6707) (initial := 2896) row_count_6708 (by decide)
private theorem atomic_6708_agreement : RowAgreement 6707 6708 := RowAgreement.single (first := 6707) modelrow_agrees_6708
private def atomic_6709 : CountSegment 1 10 13120 6708 6709 2897 2897 := CountSegment.single (first := 6708) (initial := 2897) row_count_6709 (by decide)
private theorem atomic_6709_agreement : RowAgreement 6708 6709 := RowAgreement.single (first := 6708) modelrow_agrees_6709
private def atomic_6710 : CountSegment 1 10 13120 6709 6710 2897 2898 := CountSegment.single (first := 6709) (initial := 2897) row_count_6710 (by decide)
private theorem atomic_6710_agreement : RowAgreement 6709 6710 := RowAgreement.single (first := 6709) modelrow_agrees_6710
private def atomic_6711 : CountSegment 1 10 13120 6710 6711 2898 2898 := CountSegment.single (first := 6710) (initial := 2898) row_count_6711 (by decide)
private theorem atomic_6711_agreement : RowAgreement 6710 6711 := RowAgreement.single (first := 6710) modelrow_agrees_6711
private def atomic_6712 : CountSegment 1 10 13120 6711 6712 2898 2898 := CountSegment.single (first := 6711) (initial := 2898) row_count_6712 (by decide)
private theorem atomic_6712_agreement : RowAgreement 6711 6712 := RowAgreement.single (first := 6711) modelrow_agrees_6712
private def atomic_6713 : CountSegment 1 10 13120 6712 6713 2898 2898 := CountSegment.single (first := 6712) (initial := 2898) row_count_6713 (by decide)
private theorem atomic_6713_agreement : RowAgreement 6712 6713 := RowAgreement.single (first := 6712) modelrow_agrees_6713
private def atomic_6714 : CountSegment 1 10 13120 6713 6714 2898 2898 := CountSegment.single (first := 6713) (initial := 2898) row_count_6714 (by decide)
private theorem atomic_6714_agreement : RowAgreement 6713 6714 := RowAgreement.single (first := 6713) modelrow_agrees_6714
private def atomic_6715 : CountSegment 1 10 13120 6714 6715 2898 2899 := CountSegment.single (first := 6714) (initial := 2898) row_count_6715 (by decide)
private theorem atomic_6715_agreement : RowAgreement 6714 6715 := RowAgreement.single (first := 6714) modelrow_agrees_6715
private def atomic_6716 : CountSegment 1 10 13120 6715 6716 2899 2900 := CountSegment.single (first := 6715) (initial := 2899) row_count_6716 (by decide)
private theorem atomic_6716_agreement : RowAgreement 6715 6716 := RowAgreement.single (first := 6715) modelrow_agrees_6716
private def atomic_6717 : CountSegment 1 10 13120 6716 6717 2900 2901 := CountSegment.single (first := 6716) (initial := 2900) row_count_6717 (by decide)
private theorem atomic_6717_agreement : RowAgreement 6716 6717 := RowAgreement.single (first := 6716) modelrow_agrees_6717
private def atomic_6718 : CountSegment 1 10 13120 6717 6718 2901 2902 := CountSegment.single (first := 6717) (initial := 2901) row_count_6718 (by decide)
private theorem atomic_6718_agreement : RowAgreement 6717 6718 := RowAgreement.single (first := 6717) modelrow_agrees_6718
private def atomic_6719 : CountSegment 1 10 13120 6718 6719 2902 2902 := CountSegment.single (first := 6718) (initial := 2902) row_count_6719 (by decide)
private theorem atomic_6719_agreement : RowAgreement 6718 6719 := RowAgreement.single (first := 6718) modelrow_agrees_6719
private def atomic_6720 : CountSegment 1 10 13120 6719 6720 2902 2902 := CountSegment.single (first := 6719) (initial := 2902) row_count_6720 (by decide)
private theorem atomic_6720_agreement : RowAgreement 6719 6720 := RowAgreement.single (first := 6719) modelrow_agrees_6720
private def atomic_6721 : CountSegment 1 10 13120 6720 6721 2902 2902 := CountSegment.single (first := 6720) (initial := 2902) row_count_6721 (by decide)
private theorem atomic_6721_agreement : RowAgreement 6720 6721 := RowAgreement.single (first := 6720) modelrow_agrees_6721
private def atomic_6722 : CountSegment 1 10 13120 6721 6722 2902 2902 := CountSegment.single (first := 6721) (initial := 2902) row_count_6722 (by decide)
private theorem atomic_6722_agreement : RowAgreement 6721 6722 := RowAgreement.single (first := 6721) modelrow_agrees_6722
private def atomic_6723 : CountSegment 1 10 13120 6722 6723 2902 2902 := CountSegment.single (first := 6722) (initial := 2902) row_count_6723 (by decide)
private theorem atomic_6723_agreement : RowAgreement 6722 6723 := RowAgreement.single (first := 6722) modelrow_agrees_6723
private def atomic_6724 : CountSegment 1 10 13120 6723 6724 2902 2902 := CountSegment.single (first := 6723) (initial := 2902) row_count_6724 (by decide)
private theorem atomic_6724_agreement : RowAgreement 6723 6724 := RowAgreement.single (first := 6723) modelrow_agrees_6724
private def atomic_6725 : CountSegment 1 10 13120 6724 6725 2902 2902 := CountSegment.single (first := 6724) (initial := 2902) row_count_6725 (by decide)
private theorem atomic_6725_agreement : RowAgreement 6724 6725 := RowAgreement.single (first := 6724) modelrow_agrees_6725
private def atomic_6726 : CountSegment 1 10 13120 6725 6726 2902 2902 := CountSegment.single (first := 6725) (initial := 2902) row_count_6726 (by decide)
private theorem atomic_6726_agreement : RowAgreement 6725 6726 := RowAgreement.single (first := 6725) modelrow_agrees_6726
private def atomic_6727 : CountSegment 1 10 13120 6726 6727 2902 2903 := CountSegment.single (first := 6726) (initial := 2902) row_count_6727 (by decide)
private theorem atomic_6727_agreement : RowAgreement 6726 6727 := RowAgreement.single (first := 6726) modelrow_agrees_6727
private def atomic_6728 : CountSegment 1 10 13120 6727 6728 2903 2903 := CountSegment.single (first := 6727) (initial := 2903) row_count_6728 (by decide)
private theorem atomic_6728_agreement : RowAgreement 6727 6728 := RowAgreement.single (first := 6727) modelrow_agrees_6728
private def atomic_6729 : CountSegment 1 10 13120 6728 6729 2903 2903 := CountSegment.single (first := 6728) (initial := 2903) row_count_6729 (by decide)
private theorem atomic_6729_agreement : RowAgreement 6728 6729 := RowAgreement.single (first := 6728) modelrow_agrees_6729
private def atomic_6730 : CountSegment 1 10 13120 6729 6730 2903 2904 := CountSegment.single (first := 6729) (initial := 2903) row_count_6730 (by decide)
private theorem atomic_6730_agreement : RowAgreement 6729 6730 := RowAgreement.single (first := 6729) modelrow_agrees_6730
private def atomic_6731 : CountSegment 1 10 13120 6730 6731 2904 2905 := CountSegment.single (first := 6730) (initial := 2904) row_count_6731 (by decide)
private theorem atomic_6731_agreement : RowAgreement 6730 6731 := RowAgreement.single (first := 6730) modelrow_agrees_6731
private def atomic_6732 : CountSegment 1 10 13120 6731 6732 2905 2905 := CountSegment.single (first := 6731) (initial := 2905) row_count_6732 (by decide)
private theorem atomic_6732_agreement : RowAgreement 6731 6732 := RowAgreement.single (first := 6731) modelrow_agrees_6732
private def atomic_6733 : CountSegment 1 10 13120 6732 6733 2905 2906 := CountSegment.single (first := 6732) (initial := 2905) row_count_6733 (by decide)
private theorem atomic_6733_agreement : RowAgreement 6732 6733 := RowAgreement.single (first := 6732) modelrow_agrees_6733
private def atomic_6734 : CountSegment 1 10 13120 6733 6734 2906 2906 := CountSegment.single (first := 6733) (initial := 2906) row_count_6734 (by decide)
private theorem atomic_6734_agreement : RowAgreement 6733 6734 := RowAgreement.single (first := 6733) modelrow_agrees_6734
private def atomic_6735 : CountSegment 1 10 13120 6734 6735 2906 2907 := CountSegment.single (first := 6734) (initial := 2906) row_count_6735 (by decide)
private theorem atomic_6735_agreement : RowAgreement 6734 6735 := RowAgreement.single (first := 6734) modelrow_agrees_6735
private def atomic_6736 : CountSegment 1 10 13120 6735 6736 2907 2907 := CountSegment.single (first := 6735) (initial := 2907) row_count_6736 (by decide)
private theorem atomic_6736_agreement : RowAgreement 6735 6736 := RowAgreement.single (first := 6735) modelrow_agrees_6736
private def atomic_6737 : CountSegment 1 10 13120 6736 6737 2907 2908 := CountSegment.single (first := 6736) (initial := 2907) row_count_6737 (by decide)
private theorem atomic_6737_agreement : RowAgreement 6736 6737 := RowAgreement.single (first := 6736) modelrow_agrees_6737
private def atomic_6738 : CountSegment 1 10 13120 6737 6738 2908 2909 := CountSegment.single (first := 6737) (initial := 2908) row_count_6738 (by decide)
private theorem atomic_6738_agreement : RowAgreement 6737 6738 := RowAgreement.single (first := 6737) modelrow_agrees_6738
private def atomic_6739 : CountSegment 1 10 13120 6738 6739 2909 2909 := CountSegment.single (first := 6738) (initial := 2909) row_count_6739 (by decide)
private theorem atomic_6739_agreement : RowAgreement 6738 6739 := RowAgreement.single (first := 6738) modelrow_agrees_6739
private def atomic_6740 : CountSegment 1 10 13120 6739 6740 2909 2909 := CountSegment.single (first := 6739) (initial := 2909) row_count_6740 (by decide)
private theorem atomic_6740_agreement : RowAgreement 6739 6740 := RowAgreement.single (first := 6739) modelrow_agrees_6740
private def atomic_6741 : CountSegment 1 10 13120 6740 6741 2909 2909 := CountSegment.single (first := 6740) (initial := 2909) row_count_6741 (by decide)
private theorem atomic_6741_agreement : RowAgreement 6740 6741 := RowAgreement.single (first := 6740) modelrow_agrees_6741
private def atomic_6742 : CountSegment 1 10 13120 6741 6742 2909 2909 := CountSegment.single (first := 6741) (initial := 2909) row_count_6742 (by decide)
private theorem atomic_6742_agreement : RowAgreement 6741 6742 := RowAgreement.single (first := 6741) modelrow_agrees_6742
private def atomic_6743 : CountSegment 1 10 13120 6742 6743 2909 2909 := CountSegment.single (first := 6742) (initial := 2909) row_count_6743 (by decide)
private theorem atomic_6743_agreement : RowAgreement 6742 6743 := RowAgreement.single (first := 6742) modelrow_agrees_6743
private def atomic_6744 : CountSegment 1 10 13120 6743 6744 2909 2910 := CountSegment.single (first := 6743) (initial := 2909) row_count_6744 (by decide)
private theorem atomic_6744_agreement : RowAgreement 6743 6744 := RowAgreement.single (first := 6743) modelrow_agrees_6744
private def atomic_6745 : CountSegment 1 10 13120 6744 6745 2910 2911 := CountSegment.single (first := 6744) (initial := 2910) row_count_6745 (by decide)
private theorem atomic_6745_agreement : RowAgreement 6744 6745 := RowAgreement.single (first := 6744) modelrow_agrees_6745
private def atomic_6746 : CountSegment 1 10 13120 6745 6746 2911 2911 := CountSegment.single (first := 6745) (initial := 2911) row_count_6746 (by decide)
private theorem atomic_6746_agreement : RowAgreement 6745 6746 := RowAgreement.single (first := 6745) modelrow_agrees_6746
private def atomic_6747 : CountSegment 1 10 13120 6746 6747 2911 2911 := CountSegment.single (first := 6746) (initial := 2911) row_count_6747 (by decide)
private theorem atomic_6747_agreement : RowAgreement 6746 6747 := RowAgreement.single (first := 6746) modelrow_agrees_6747
private def atomic_6748 : CountSegment 1 10 13120 6747 6748 2911 2911 := CountSegment.single (first := 6747) (initial := 2911) row_count_6748 (by decide)
private theorem atomic_6748_agreement : RowAgreement 6747 6748 := RowAgreement.single (first := 6747) modelrow_agrees_6748
private def atomic_6749 : CountSegment 1 10 13120 6748 6749 2911 2911 := CountSegment.single (first := 6748) (initial := 2911) row_count_6749 (by decide)
private theorem atomic_6749_agreement : RowAgreement 6748 6749 := RowAgreement.single (first := 6748) modelrow_agrees_6749
private def atomic_6750 : CountSegment 1 10 13120 6749 6750 2911 2912 := CountSegment.single (first := 6749) (initial := 2911) row_count_6750 (by decide)
private theorem atomic_6750_agreement : RowAgreement 6749 6750 := RowAgreement.single (first := 6749) modelrow_agrees_6750
private def atomic_6751 : CountSegment 1 10 13120 6750 6751 2912 2913 := CountSegment.single (first := 6750) (initial := 2912) row_count_6751 (by decide)
private theorem atomic_6751_agreement : RowAgreement 6750 6751 := RowAgreement.single (first := 6750) modelrow_agrees_6751
private def atomic_6752 : CountSegment 1 10 13120 6751 6752 2913 2913 := CountSegment.single (first := 6751) (initial := 2913) row_count_6752 (by decide)
private theorem atomic_6752_agreement : RowAgreement 6751 6752 := RowAgreement.single (first := 6751) modelrow_agrees_6752
private def atomic_6753 : CountSegment 1 10 13120 6752 6753 2913 2914 := CountSegment.single (first := 6752) (initial := 2913) row_count_6753 (by decide)
private theorem atomic_6753_agreement : RowAgreement 6752 6753 := RowAgreement.single (first := 6752) modelrow_agrees_6753
private def atomic_6754 : CountSegment 1 10 13120 6753 6754 2914 2914 := CountSegment.single (first := 6753) (initial := 2914) row_count_6754 (by decide)
private theorem atomic_6754_agreement : RowAgreement 6753 6754 := RowAgreement.single (first := 6753) modelrow_agrees_6754
private def atomic_6755 : CountSegment 1 10 13120 6754 6755 2914 2914 := CountSegment.single (first := 6754) (initial := 2914) row_count_6755 (by decide)
private theorem atomic_6755_agreement : RowAgreement 6754 6755 := RowAgreement.single (first := 6754) modelrow_agrees_6755
private def atomic_6756 : CountSegment 1 10 13120 6755 6756 2914 2914 := CountSegment.single (first := 6755) (initial := 2914) row_count_6756 (by decide)
private theorem atomic_6756_agreement : RowAgreement 6755 6756 := RowAgreement.single (first := 6755) modelrow_agrees_6756
private def atomic_6757 : CountSegment 1 10 13120 6756 6757 2914 2914 := CountSegment.single (first := 6756) (initial := 2914) row_count_6757 (by decide)
private theorem atomic_6757_agreement : RowAgreement 6756 6757 := RowAgreement.single (first := 6756) modelrow_agrees_6757
private def atomic_6758 : CountSegment 1 10 13120 6757 6758 2914 2915 := CountSegment.single (first := 6757) (initial := 2914) row_count_6758 (by decide)
private theorem atomic_6758_agreement : RowAgreement 6757 6758 := RowAgreement.single (first := 6757) modelrow_agrees_6758
private def atomic_6759 : CountSegment 1 10 13120 6758 6759 2915 2916 := CountSegment.single (first := 6758) (initial := 2915) row_count_6759 (by decide)
private theorem atomic_6759_agreement : RowAgreement 6758 6759 := RowAgreement.single (first := 6758) modelrow_agrees_6759
private def atomic_6760 : CountSegment 1 10 13120 6759 6760 2916 2917 := CountSegment.single (first := 6759) (initial := 2916) row_count_6760 (by decide)
private theorem atomic_6760_agreement : RowAgreement 6759 6760 := RowAgreement.single (first := 6759) modelrow_agrees_6760
private def atomic_6761 : CountSegment 1 10 13120 6760 6761 2917 2917 := CountSegment.single (first := 6760) (initial := 2917) row_count_6761 (by decide)
private theorem atomic_6761_agreement : RowAgreement 6760 6761 := RowAgreement.single (first := 6760) modelrow_agrees_6761
private def atomic_6762 : CountSegment 1 10 13120 6761 6762 2917 2917 := CountSegment.single (first := 6761) (initial := 2917) row_count_6762 (by decide)
private theorem atomic_6762_agreement : RowAgreement 6761 6762 := RowAgreement.single (first := 6761) modelrow_agrees_6762
private def atomic_6763 : CountSegment 1 10 13120 6762 6763 2917 2917 := CountSegment.single (first := 6762) (initial := 2917) row_count_6763 (by decide)
private theorem atomic_6763_agreement : RowAgreement 6762 6763 := RowAgreement.single (first := 6762) modelrow_agrees_6763
private def atomic_6764 : CountSegment 1 10 13120 6763 6764 2917 2918 := CountSegment.single (first := 6763) (initial := 2917) row_count_6764 (by decide)
private theorem atomic_6764_agreement : RowAgreement 6763 6764 := RowAgreement.single (first := 6763) modelrow_agrees_6764
private def atomic_6765 : CountSegment 1 10 13120 6764 6765 2918 2918 := CountSegment.single (first := 6764) (initial := 2918) row_count_6765 (by decide)
private theorem atomic_6765_agreement : RowAgreement 6764 6765 := RowAgreement.single (first := 6764) modelrow_agrees_6765
private def atomic_6766 : CountSegment 1 10 13120 6765 6766 2918 2918 := CountSegment.single (first := 6765) (initial := 2918) row_count_6766 (by decide)
private theorem atomic_6766_agreement : RowAgreement 6765 6766 := RowAgreement.single (first := 6765) modelrow_agrees_6766
private def atomic_6767 : CountSegment 1 10 13120 6766 6767 2918 2919 := CountSegment.single (first := 6766) (initial := 2918) row_count_6767 (by decide)
private theorem atomic_6767_agreement : RowAgreement 6766 6767 := RowAgreement.single (first := 6766) modelrow_agrees_6767
private def atomic_6768 : CountSegment 1 10 13120 6767 6768 2919 2919 := CountSegment.single (first := 6767) (initial := 2919) row_count_6768 (by decide)
private theorem atomic_6768_agreement : RowAgreement 6767 6768 := RowAgreement.single (first := 6767) modelrow_agrees_6768
private def atomic_6769 : CountSegment 1 10 13120 6768 6769 2919 2919 := CountSegment.single (first := 6768) (initial := 2919) row_count_6769 (by decide)
private theorem atomic_6769_agreement : RowAgreement 6768 6769 := RowAgreement.single (first := 6768) modelrow_agrees_6769
private def atomic_6770 : CountSegment 1 10 13120 6769 6770 2919 2919 := CountSegment.single (first := 6769) (initial := 2919) row_count_6770 (by decide)
private theorem atomic_6770_agreement : RowAgreement 6769 6770 := RowAgreement.single (first := 6769) modelrow_agrees_6770
private def atomic_6771 : CountSegment 1 10 13120 6770 6771 2919 2919 := CountSegment.single (first := 6770) (initial := 2919) row_count_6771 (by decide)
private theorem atomic_6771_agreement : RowAgreement 6770 6771 := RowAgreement.single (first := 6770) modelrow_agrees_6771
private def atomic_6772 : CountSegment 1 10 13120 6771 6772 2919 2919 := CountSegment.single (first := 6771) (initial := 2919) row_count_6772 (by decide)
private theorem atomic_6772_agreement : RowAgreement 6771 6772 := RowAgreement.single (first := 6771) modelrow_agrees_6772
private def atomic_6773 : CountSegment 1 10 13120 6772 6773 2919 2921 := CountSegment.single (first := 6772) (initial := 2919) row_count_6773 (by decide)
private theorem atomic_6773_agreement : RowAgreement 6772 6773 := RowAgreement.single (first := 6772) modelrow_agrees_6773
private def atomic_6774 : CountSegment 1 10 13120 6773 6774 2921 2922 := CountSegment.single (first := 6773) (initial := 2921) row_count_6774 (by decide)
private theorem atomic_6774_agreement : RowAgreement 6773 6774 := RowAgreement.single (first := 6773) modelrow_agrees_6774
private def atomic_6775 : CountSegment 1 10 13120 6774 6775 2922 2923 := CountSegment.single (first := 6774) (initial := 2922) row_count_6775 (by decide)
private theorem atomic_6775_agreement : RowAgreement 6774 6775 := RowAgreement.single (first := 6774) modelrow_agrees_6775
private def atomic_6776 : CountSegment 1 10 13120 6775 6776 2923 2923 := CountSegment.single (first := 6775) (initial := 2923) row_count_6776 (by decide)
private theorem atomic_6776_agreement : RowAgreement 6775 6776 := RowAgreement.single (first := 6775) modelrow_agrees_6776
private def atomic_6777 : CountSegment 1 10 13120 6776 6777 2923 2924 := CountSegment.single (first := 6776) (initial := 2923) row_count_6777 (by decide)
private theorem atomic_6777_agreement : RowAgreement 6776 6777 := RowAgreement.single (first := 6776) modelrow_agrees_6777
private def atomic_6778 : CountSegment 1 10 13120 6777 6778 2924 2924 := CountSegment.single (first := 6777) (initial := 2924) row_count_6778 (by decide)
private theorem atomic_6778_agreement : RowAgreement 6777 6778 := RowAgreement.single (first := 6777) modelrow_agrees_6778
private def atomic_6779 : CountSegment 1 10 13120 6778 6779 2924 2924 := CountSegment.single (first := 6778) (initial := 2924) row_count_6779 (by decide)
private theorem atomic_6779_agreement : RowAgreement 6778 6779 := RowAgreement.single (first := 6778) modelrow_agrees_6779
private def atomic_6780 : CountSegment 1 10 13120 6779 6780 2924 2924 := CountSegment.single (first := 6779) (initial := 2924) row_count_6780 (by decide)
private theorem atomic_6780_agreement : RowAgreement 6779 6780 := RowAgreement.single (first := 6779) modelrow_agrees_6780
private def atomic_6781 : CountSegment 1 10 13120 6780 6781 2924 2924 := CountSegment.single (first := 6780) (initial := 2924) row_count_6781 (by decide)
private theorem atomic_6781_agreement : RowAgreement 6780 6781 := RowAgreement.single (first := 6780) modelrow_agrees_6781
private def atomic_6782 : CountSegment 1 10 13120 6781 6782 2924 2924 := CountSegment.single (first := 6781) (initial := 2924) row_count_6782 (by decide)
private theorem atomic_6782_agreement : RowAgreement 6781 6782 := RowAgreement.single (first := 6781) modelrow_agrees_6782
private def atomic_6783 : CountSegment 1 10 13120 6782 6783 2924 2925 := CountSegment.single (first := 6782) (initial := 2924) row_count_6783 (by decide)
private theorem atomic_6783_agreement : RowAgreement 6782 6783 := RowAgreement.single (first := 6782) modelrow_agrees_6783
private def atomic_6784 : CountSegment 1 10 13120 6783 6784 2925 2925 := CountSegment.single (first := 6783) (initial := 2925) row_count_6784 (by decide)
private theorem atomic_6784_agreement : RowAgreement 6783 6784 := RowAgreement.single (first := 6783) modelrow_agrees_6784
private def within_052_000 : CountSegment 1 10 13120 6656 6658 2875 2876 := CountSegment.join atomic_6657 atomic_6658
private theorem within_052_000_agreement : RowAgreement 6656 6658 := RowAgreement.join atomic_6657_agreement atomic_6658_agreement
private def within_052_001 : CountSegment 1 10 13120 6658 6660 2876 2878 := CountSegment.join atomic_6659 atomic_6660
private theorem within_052_001_agreement : RowAgreement 6658 6660 := RowAgreement.join atomic_6659_agreement atomic_6660_agreement
private def within_052_002 : CountSegment 1 10 13120 6656 6660 2875 2878 := CountSegment.join within_052_000 within_052_001
private theorem within_052_002_agreement : RowAgreement 6656 6660 := RowAgreement.join within_052_000_agreement within_052_001_agreement
private def within_052_003 : CountSegment 1 10 13120 6660 6662 2878 2879 := CountSegment.join atomic_6661 atomic_6662
private theorem within_052_003_agreement : RowAgreement 6660 6662 := RowAgreement.join atomic_6661_agreement atomic_6662_agreement
private def within_052_004 : CountSegment 1 10 13120 6662 6664 2879 2879 := CountSegment.join atomic_6663 atomic_6664
private theorem within_052_004_agreement : RowAgreement 6662 6664 := RowAgreement.join atomic_6663_agreement atomic_6664_agreement
private def within_052_005 : CountSegment 1 10 13120 6660 6664 2878 2879 := CountSegment.join within_052_003 within_052_004
private theorem within_052_005_agreement : RowAgreement 6660 6664 := RowAgreement.join within_052_003_agreement within_052_004_agreement
private def within_052_006 : CountSegment 1 10 13120 6656 6664 2875 2879 := CountSegment.join within_052_002 within_052_005
private theorem within_052_006_agreement : RowAgreement 6656 6664 := RowAgreement.join within_052_002_agreement within_052_005_agreement
private def within_052_007 : CountSegment 1 10 13120 6664 6666 2879 2881 := CountSegment.join atomic_6665 atomic_6666
private theorem within_052_007_agreement : RowAgreement 6664 6666 := RowAgreement.join atomic_6665_agreement atomic_6666_agreement
private def within_052_008 : CountSegment 1 10 13120 6666 6668 2881 2882 := CountSegment.join atomic_6667 atomic_6668
private theorem within_052_008_agreement : RowAgreement 6666 6668 := RowAgreement.join atomic_6667_agreement atomic_6668_agreement
private def within_052_009 : CountSegment 1 10 13120 6664 6668 2879 2882 := CountSegment.join within_052_007 within_052_008
private theorem within_052_009_agreement : RowAgreement 6664 6668 := RowAgreement.join within_052_007_agreement within_052_008_agreement
private def within_052_010 : CountSegment 1 10 13120 6668 6670 2882 2883 := CountSegment.join atomic_6669 atomic_6670
private theorem within_052_010_agreement : RowAgreement 6668 6670 := RowAgreement.join atomic_6669_agreement atomic_6670_agreement
private def within_052_011 : CountSegment 1 10 13120 6670 6672 2883 2884 := CountSegment.join atomic_6671 atomic_6672
private theorem within_052_011_agreement : RowAgreement 6670 6672 := RowAgreement.join atomic_6671_agreement atomic_6672_agreement
private def within_052_012 : CountSegment 1 10 13120 6668 6672 2882 2884 := CountSegment.join within_052_010 within_052_011
private theorem within_052_012_agreement : RowAgreement 6668 6672 := RowAgreement.join within_052_010_agreement within_052_011_agreement
private def within_052_013 : CountSegment 1 10 13120 6664 6672 2879 2884 := CountSegment.join within_052_009 within_052_012
private theorem within_052_013_agreement : RowAgreement 6664 6672 := RowAgreement.join within_052_009_agreement within_052_012_agreement
private def within_052_014 : CountSegment 1 10 13120 6656 6672 2875 2884 := CountSegment.join within_052_006 within_052_013
private theorem within_052_014_agreement : RowAgreement 6656 6672 := RowAgreement.join within_052_006_agreement within_052_013_agreement
private def within_052_015 : CountSegment 1 10 13120 6672 6674 2884 2884 := CountSegment.join atomic_6673 atomic_6674
private theorem within_052_015_agreement : RowAgreement 6672 6674 := RowAgreement.join atomic_6673_agreement atomic_6674_agreement
private def within_052_016 : CountSegment 1 10 13120 6674 6676 2884 2885 := CountSegment.join atomic_6675 atomic_6676
private theorem within_052_016_agreement : RowAgreement 6674 6676 := RowAgreement.join atomic_6675_agreement atomic_6676_agreement
private def within_052_017 : CountSegment 1 10 13120 6672 6676 2884 2885 := CountSegment.join within_052_015 within_052_016
private theorem within_052_017_agreement : RowAgreement 6672 6676 := RowAgreement.join within_052_015_agreement within_052_016_agreement
private def within_052_018 : CountSegment 1 10 13120 6676 6678 2885 2886 := CountSegment.join atomic_6677 atomic_6678
private theorem within_052_018_agreement : RowAgreement 6676 6678 := RowAgreement.join atomic_6677_agreement atomic_6678_agreement
private def within_052_019 : CountSegment 1 10 13120 6678 6680 2886 2888 := CountSegment.join atomic_6679 atomic_6680
private theorem within_052_019_agreement : RowAgreement 6678 6680 := RowAgreement.join atomic_6679_agreement atomic_6680_agreement
private def within_052_020 : CountSegment 1 10 13120 6676 6680 2885 2888 := CountSegment.join within_052_018 within_052_019
private theorem within_052_020_agreement : RowAgreement 6676 6680 := RowAgreement.join within_052_018_agreement within_052_019_agreement
private def within_052_021 : CountSegment 1 10 13120 6672 6680 2884 2888 := CountSegment.join within_052_017 within_052_020
private theorem within_052_021_agreement : RowAgreement 6672 6680 := RowAgreement.join within_052_017_agreement within_052_020_agreement
private def within_052_022 : CountSegment 1 10 13120 6680 6682 2888 2888 := CountSegment.join atomic_6681 atomic_6682
private theorem within_052_022_agreement : RowAgreement 6680 6682 := RowAgreement.join atomic_6681_agreement atomic_6682_agreement
private def within_052_023 : CountSegment 1 10 13120 6682 6684 2888 2888 := CountSegment.join atomic_6683 atomic_6684
private theorem within_052_023_agreement : RowAgreement 6682 6684 := RowAgreement.join atomic_6683_agreement atomic_6684_agreement
private def within_052_024 : CountSegment 1 10 13120 6680 6684 2888 2888 := CountSegment.join within_052_022 within_052_023
private theorem within_052_024_agreement : RowAgreement 6680 6684 := RowAgreement.join within_052_022_agreement within_052_023_agreement
private def within_052_025 : CountSegment 1 10 13120 6684 6686 2888 2889 := CountSegment.join atomic_6685 atomic_6686
private theorem within_052_025_agreement : RowAgreement 6684 6686 := RowAgreement.join atomic_6685_agreement atomic_6686_agreement
private def within_052_026 : CountSegment 1 10 13120 6686 6688 2889 2890 := CountSegment.join atomic_6687 atomic_6688
private theorem within_052_026_agreement : RowAgreement 6686 6688 := RowAgreement.join atomic_6687_agreement atomic_6688_agreement
private def within_052_027 : CountSegment 1 10 13120 6684 6688 2888 2890 := CountSegment.join within_052_025 within_052_026
private theorem within_052_027_agreement : RowAgreement 6684 6688 := RowAgreement.join within_052_025_agreement within_052_026_agreement
private def within_052_028 : CountSegment 1 10 13120 6680 6688 2888 2890 := CountSegment.join within_052_024 within_052_027
private theorem within_052_028_agreement : RowAgreement 6680 6688 := RowAgreement.join within_052_024_agreement within_052_027_agreement
private def within_052_029 : CountSegment 1 10 13120 6672 6688 2884 2890 := CountSegment.join within_052_021 within_052_028
private theorem within_052_029_agreement : RowAgreement 6672 6688 := RowAgreement.join within_052_021_agreement within_052_028_agreement
private def within_052_030 : CountSegment 1 10 13120 6656 6688 2875 2890 := CountSegment.join within_052_014 within_052_029
private theorem within_052_030_agreement : RowAgreement 6656 6688 := RowAgreement.join within_052_014_agreement within_052_029_agreement
private def within_052_031 : CountSegment 1 10 13120 6688 6690 2890 2891 := CountSegment.join atomic_6689 atomic_6690
private theorem within_052_031_agreement : RowAgreement 6688 6690 := RowAgreement.join atomic_6689_agreement atomic_6690_agreement
private def within_052_032 : CountSegment 1 10 13120 6690 6692 2891 2891 := CountSegment.join atomic_6691 atomic_6692
private theorem within_052_032_agreement : RowAgreement 6690 6692 := RowAgreement.join atomic_6691_agreement atomic_6692_agreement
private def within_052_033 : CountSegment 1 10 13120 6688 6692 2890 2891 := CountSegment.join within_052_031 within_052_032
private theorem within_052_033_agreement : RowAgreement 6688 6692 := RowAgreement.join within_052_031_agreement within_052_032_agreement
private def within_052_034 : CountSegment 1 10 13120 6692 6694 2891 2891 := CountSegment.join atomic_6693 atomic_6694
private theorem within_052_034_agreement : RowAgreement 6692 6694 := RowAgreement.join atomic_6693_agreement atomic_6694_agreement
private def within_052_035 : CountSegment 1 10 13120 6694 6696 2891 2893 := CountSegment.join atomic_6695 atomic_6696
private theorem within_052_035_agreement : RowAgreement 6694 6696 := RowAgreement.join atomic_6695_agreement atomic_6696_agreement
private def within_052_036 : CountSegment 1 10 13120 6692 6696 2891 2893 := CountSegment.join within_052_034 within_052_035
private theorem within_052_036_agreement : RowAgreement 6692 6696 := RowAgreement.join within_052_034_agreement within_052_035_agreement
private def within_052_037 : CountSegment 1 10 13120 6688 6696 2890 2893 := CountSegment.join within_052_033 within_052_036
private theorem within_052_037_agreement : RowAgreement 6688 6696 := RowAgreement.join within_052_033_agreement within_052_036_agreement
private def within_052_038 : CountSegment 1 10 13120 6696 6698 2893 2894 := CountSegment.join atomic_6697 atomic_6698
private theorem within_052_038_agreement : RowAgreement 6696 6698 := RowAgreement.join atomic_6697_agreement atomic_6698_agreement
private def within_052_039 : CountSegment 1 10 13120 6698 6700 2894 2895 := CountSegment.join atomic_6699 atomic_6700
private theorem within_052_039_agreement : RowAgreement 6698 6700 := RowAgreement.join atomic_6699_agreement atomic_6700_agreement
private def within_052_040 : CountSegment 1 10 13120 6696 6700 2893 2895 := CountSegment.join within_052_038 within_052_039
private theorem within_052_040_agreement : RowAgreement 6696 6700 := RowAgreement.join within_052_038_agreement within_052_039_agreement
private def within_052_041 : CountSegment 1 10 13120 6700 6702 2895 2895 := CountSegment.join atomic_6701 atomic_6702
private theorem within_052_041_agreement : RowAgreement 6700 6702 := RowAgreement.join atomic_6701_agreement atomic_6702_agreement
private def within_052_042 : CountSegment 1 10 13120 6702 6704 2895 2895 := CountSegment.join atomic_6703 atomic_6704
private theorem within_052_042_agreement : RowAgreement 6702 6704 := RowAgreement.join atomic_6703_agreement atomic_6704_agreement
private def within_052_043 : CountSegment 1 10 13120 6700 6704 2895 2895 := CountSegment.join within_052_041 within_052_042
private theorem within_052_043_agreement : RowAgreement 6700 6704 := RowAgreement.join within_052_041_agreement within_052_042_agreement
private def within_052_044 : CountSegment 1 10 13120 6696 6704 2893 2895 := CountSegment.join within_052_040 within_052_043
private theorem within_052_044_agreement : RowAgreement 6696 6704 := RowAgreement.join within_052_040_agreement within_052_043_agreement
private def within_052_045 : CountSegment 1 10 13120 6688 6704 2890 2895 := CountSegment.join within_052_037 within_052_044
private theorem within_052_045_agreement : RowAgreement 6688 6704 := RowAgreement.join within_052_037_agreement within_052_044_agreement
private def within_052_046 : CountSegment 1 10 13120 6704 6706 2895 2896 := CountSegment.join atomic_6705 atomic_6706
private theorem within_052_046_agreement : RowAgreement 6704 6706 := RowAgreement.join atomic_6705_agreement atomic_6706_agreement
private def within_052_047 : CountSegment 1 10 13120 6706 6708 2896 2897 := CountSegment.join atomic_6707 atomic_6708
private theorem within_052_047_agreement : RowAgreement 6706 6708 := RowAgreement.join atomic_6707_agreement atomic_6708_agreement
private def within_052_048 : CountSegment 1 10 13120 6704 6708 2895 2897 := CountSegment.join within_052_046 within_052_047
private theorem within_052_048_agreement : RowAgreement 6704 6708 := RowAgreement.join within_052_046_agreement within_052_047_agreement
private def within_052_049 : CountSegment 1 10 13120 6708 6710 2897 2898 := CountSegment.join atomic_6709 atomic_6710
private theorem within_052_049_agreement : RowAgreement 6708 6710 := RowAgreement.join atomic_6709_agreement atomic_6710_agreement
private def within_052_050 : CountSegment 1 10 13120 6710 6712 2898 2898 := CountSegment.join atomic_6711 atomic_6712
private theorem within_052_050_agreement : RowAgreement 6710 6712 := RowAgreement.join atomic_6711_agreement atomic_6712_agreement
private def within_052_051 : CountSegment 1 10 13120 6708 6712 2897 2898 := CountSegment.join within_052_049 within_052_050
private theorem within_052_051_agreement : RowAgreement 6708 6712 := RowAgreement.join within_052_049_agreement within_052_050_agreement
private def within_052_052 : CountSegment 1 10 13120 6704 6712 2895 2898 := CountSegment.join within_052_048 within_052_051
private theorem within_052_052_agreement : RowAgreement 6704 6712 := RowAgreement.join within_052_048_agreement within_052_051_agreement
private def within_052_053 : CountSegment 1 10 13120 6712 6714 2898 2898 := CountSegment.join atomic_6713 atomic_6714
private theorem within_052_053_agreement : RowAgreement 6712 6714 := RowAgreement.join atomic_6713_agreement atomic_6714_agreement
private def within_052_054 : CountSegment 1 10 13120 6714 6716 2898 2900 := CountSegment.join atomic_6715 atomic_6716
private theorem within_052_054_agreement : RowAgreement 6714 6716 := RowAgreement.join atomic_6715_agreement atomic_6716_agreement
private def within_052_055 : CountSegment 1 10 13120 6712 6716 2898 2900 := CountSegment.join within_052_053 within_052_054
private theorem within_052_055_agreement : RowAgreement 6712 6716 := RowAgreement.join within_052_053_agreement within_052_054_agreement
private def within_052_056 : CountSegment 1 10 13120 6716 6718 2900 2902 := CountSegment.join atomic_6717 atomic_6718
private theorem within_052_056_agreement : RowAgreement 6716 6718 := RowAgreement.join atomic_6717_agreement atomic_6718_agreement
private def within_052_057 : CountSegment 1 10 13120 6718 6720 2902 2902 := CountSegment.join atomic_6719 atomic_6720
private theorem within_052_057_agreement : RowAgreement 6718 6720 := RowAgreement.join atomic_6719_agreement atomic_6720_agreement
private def within_052_058 : CountSegment 1 10 13120 6716 6720 2900 2902 := CountSegment.join within_052_056 within_052_057
private theorem within_052_058_agreement : RowAgreement 6716 6720 := RowAgreement.join within_052_056_agreement within_052_057_agreement
private def within_052_059 : CountSegment 1 10 13120 6712 6720 2898 2902 := CountSegment.join within_052_055 within_052_058
private theorem within_052_059_agreement : RowAgreement 6712 6720 := RowAgreement.join within_052_055_agreement within_052_058_agreement
private def within_052_060 : CountSegment 1 10 13120 6704 6720 2895 2902 := CountSegment.join within_052_052 within_052_059
private theorem within_052_060_agreement : RowAgreement 6704 6720 := RowAgreement.join within_052_052_agreement within_052_059_agreement
private def within_052_061 : CountSegment 1 10 13120 6688 6720 2890 2902 := CountSegment.join within_052_045 within_052_060
private theorem within_052_061_agreement : RowAgreement 6688 6720 := RowAgreement.join within_052_045_agreement within_052_060_agreement
private def within_052_062 : CountSegment 1 10 13120 6656 6720 2875 2902 := CountSegment.join within_052_030 within_052_061
private theorem within_052_062_agreement : RowAgreement 6656 6720 := RowAgreement.join within_052_030_agreement within_052_061_agreement
private def within_052_063 : CountSegment 1 10 13120 6720 6722 2902 2902 := CountSegment.join atomic_6721 atomic_6722
private theorem within_052_063_agreement : RowAgreement 6720 6722 := RowAgreement.join atomic_6721_agreement atomic_6722_agreement
private def within_052_064 : CountSegment 1 10 13120 6722 6724 2902 2902 := CountSegment.join atomic_6723 atomic_6724
private theorem within_052_064_agreement : RowAgreement 6722 6724 := RowAgreement.join atomic_6723_agreement atomic_6724_agreement
private def within_052_065 : CountSegment 1 10 13120 6720 6724 2902 2902 := CountSegment.join within_052_063 within_052_064
private theorem within_052_065_agreement : RowAgreement 6720 6724 := RowAgreement.join within_052_063_agreement within_052_064_agreement
private def within_052_066 : CountSegment 1 10 13120 6724 6726 2902 2902 := CountSegment.join atomic_6725 atomic_6726
private theorem within_052_066_agreement : RowAgreement 6724 6726 := RowAgreement.join atomic_6725_agreement atomic_6726_agreement
private def within_052_067 : CountSegment 1 10 13120 6726 6728 2902 2903 := CountSegment.join atomic_6727 atomic_6728
private theorem within_052_067_agreement : RowAgreement 6726 6728 := RowAgreement.join atomic_6727_agreement atomic_6728_agreement
private def within_052_068 : CountSegment 1 10 13120 6724 6728 2902 2903 := CountSegment.join within_052_066 within_052_067
private theorem within_052_068_agreement : RowAgreement 6724 6728 := RowAgreement.join within_052_066_agreement within_052_067_agreement
private def within_052_069 : CountSegment 1 10 13120 6720 6728 2902 2903 := CountSegment.join within_052_065 within_052_068
private theorem within_052_069_agreement : RowAgreement 6720 6728 := RowAgreement.join within_052_065_agreement within_052_068_agreement
private def within_052_070 : CountSegment 1 10 13120 6728 6730 2903 2904 := CountSegment.join atomic_6729 atomic_6730
private theorem within_052_070_agreement : RowAgreement 6728 6730 := RowAgreement.join atomic_6729_agreement atomic_6730_agreement
private def within_052_071 : CountSegment 1 10 13120 6730 6732 2904 2905 := CountSegment.join atomic_6731 atomic_6732
private theorem within_052_071_agreement : RowAgreement 6730 6732 := RowAgreement.join atomic_6731_agreement atomic_6732_agreement
private def within_052_072 : CountSegment 1 10 13120 6728 6732 2903 2905 := CountSegment.join within_052_070 within_052_071
private theorem within_052_072_agreement : RowAgreement 6728 6732 := RowAgreement.join within_052_070_agreement within_052_071_agreement
private def within_052_073 : CountSegment 1 10 13120 6732 6734 2905 2906 := CountSegment.join atomic_6733 atomic_6734
private theorem within_052_073_agreement : RowAgreement 6732 6734 := RowAgreement.join atomic_6733_agreement atomic_6734_agreement
private def within_052_074 : CountSegment 1 10 13120 6734 6736 2906 2907 := CountSegment.join atomic_6735 atomic_6736
private theorem within_052_074_agreement : RowAgreement 6734 6736 := RowAgreement.join atomic_6735_agreement atomic_6736_agreement
private def within_052_075 : CountSegment 1 10 13120 6732 6736 2905 2907 := CountSegment.join within_052_073 within_052_074
private theorem within_052_075_agreement : RowAgreement 6732 6736 := RowAgreement.join within_052_073_agreement within_052_074_agreement
private def within_052_076 : CountSegment 1 10 13120 6728 6736 2903 2907 := CountSegment.join within_052_072 within_052_075
private theorem within_052_076_agreement : RowAgreement 6728 6736 := RowAgreement.join within_052_072_agreement within_052_075_agreement
private def within_052_077 : CountSegment 1 10 13120 6720 6736 2902 2907 := CountSegment.join within_052_069 within_052_076
private theorem within_052_077_agreement : RowAgreement 6720 6736 := RowAgreement.join within_052_069_agreement within_052_076_agreement
private def within_052_078 : CountSegment 1 10 13120 6736 6738 2907 2909 := CountSegment.join atomic_6737 atomic_6738
private theorem within_052_078_agreement : RowAgreement 6736 6738 := RowAgreement.join atomic_6737_agreement atomic_6738_agreement
private def within_052_079 : CountSegment 1 10 13120 6738 6740 2909 2909 := CountSegment.join atomic_6739 atomic_6740
private theorem within_052_079_agreement : RowAgreement 6738 6740 := RowAgreement.join atomic_6739_agreement atomic_6740_agreement
private def within_052_080 : CountSegment 1 10 13120 6736 6740 2907 2909 := CountSegment.join within_052_078 within_052_079
private theorem within_052_080_agreement : RowAgreement 6736 6740 := RowAgreement.join within_052_078_agreement within_052_079_agreement
private def within_052_081 : CountSegment 1 10 13120 6740 6742 2909 2909 := CountSegment.join atomic_6741 atomic_6742
private theorem within_052_081_agreement : RowAgreement 6740 6742 := RowAgreement.join atomic_6741_agreement atomic_6742_agreement
private def within_052_082 : CountSegment 1 10 13120 6742 6744 2909 2910 := CountSegment.join atomic_6743 atomic_6744
private theorem within_052_082_agreement : RowAgreement 6742 6744 := RowAgreement.join atomic_6743_agreement atomic_6744_agreement
private def within_052_083 : CountSegment 1 10 13120 6740 6744 2909 2910 := CountSegment.join within_052_081 within_052_082
private theorem within_052_083_agreement : RowAgreement 6740 6744 := RowAgreement.join within_052_081_agreement within_052_082_agreement
private def within_052_084 : CountSegment 1 10 13120 6736 6744 2907 2910 := CountSegment.join within_052_080 within_052_083
private theorem within_052_084_agreement : RowAgreement 6736 6744 := RowAgreement.join within_052_080_agreement within_052_083_agreement
private def within_052_085 : CountSegment 1 10 13120 6744 6746 2910 2911 := CountSegment.join atomic_6745 atomic_6746
private theorem within_052_085_agreement : RowAgreement 6744 6746 := RowAgreement.join atomic_6745_agreement atomic_6746_agreement
private def within_052_086 : CountSegment 1 10 13120 6746 6748 2911 2911 := CountSegment.join atomic_6747 atomic_6748
private theorem within_052_086_agreement : RowAgreement 6746 6748 := RowAgreement.join atomic_6747_agreement atomic_6748_agreement
private def within_052_087 : CountSegment 1 10 13120 6744 6748 2910 2911 := CountSegment.join within_052_085 within_052_086
private theorem within_052_087_agreement : RowAgreement 6744 6748 := RowAgreement.join within_052_085_agreement within_052_086_agreement
private def within_052_088 : CountSegment 1 10 13120 6748 6750 2911 2912 := CountSegment.join atomic_6749 atomic_6750
private theorem within_052_088_agreement : RowAgreement 6748 6750 := RowAgreement.join atomic_6749_agreement atomic_6750_agreement
private def within_052_089 : CountSegment 1 10 13120 6750 6752 2912 2913 := CountSegment.join atomic_6751 atomic_6752
private theorem within_052_089_agreement : RowAgreement 6750 6752 := RowAgreement.join atomic_6751_agreement atomic_6752_agreement
private def within_052_090 : CountSegment 1 10 13120 6748 6752 2911 2913 := CountSegment.join within_052_088 within_052_089
private theorem within_052_090_agreement : RowAgreement 6748 6752 := RowAgreement.join within_052_088_agreement within_052_089_agreement
private def within_052_091 : CountSegment 1 10 13120 6744 6752 2910 2913 := CountSegment.join within_052_087 within_052_090
private theorem within_052_091_agreement : RowAgreement 6744 6752 := RowAgreement.join within_052_087_agreement within_052_090_agreement
private def within_052_092 : CountSegment 1 10 13120 6736 6752 2907 2913 := CountSegment.join within_052_084 within_052_091
private theorem within_052_092_agreement : RowAgreement 6736 6752 := RowAgreement.join within_052_084_agreement within_052_091_agreement
private def within_052_093 : CountSegment 1 10 13120 6720 6752 2902 2913 := CountSegment.join within_052_077 within_052_092
private theorem within_052_093_agreement : RowAgreement 6720 6752 := RowAgreement.join within_052_077_agreement within_052_092_agreement
private def within_052_094 : CountSegment 1 10 13120 6752 6754 2913 2914 := CountSegment.join atomic_6753 atomic_6754
private theorem within_052_094_agreement : RowAgreement 6752 6754 := RowAgreement.join atomic_6753_agreement atomic_6754_agreement
private def within_052_095 : CountSegment 1 10 13120 6754 6756 2914 2914 := CountSegment.join atomic_6755 atomic_6756
private theorem within_052_095_agreement : RowAgreement 6754 6756 := RowAgreement.join atomic_6755_agreement atomic_6756_agreement
private def within_052_096 : CountSegment 1 10 13120 6752 6756 2913 2914 := CountSegment.join within_052_094 within_052_095
private theorem within_052_096_agreement : RowAgreement 6752 6756 := RowAgreement.join within_052_094_agreement within_052_095_agreement
private def within_052_097 : CountSegment 1 10 13120 6756 6758 2914 2915 := CountSegment.join atomic_6757 atomic_6758
private theorem within_052_097_agreement : RowAgreement 6756 6758 := RowAgreement.join atomic_6757_agreement atomic_6758_agreement
private def within_052_098 : CountSegment 1 10 13120 6758 6760 2915 2917 := CountSegment.join atomic_6759 atomic_6760
private theorem within_052_098_agreement : RowAgreement 6758 6760 := RowAgreement.join atomic_6759_agreement atomic_6760_agreement
private def within_052_099 : CountSegment 1 10 13120 6756 6760 2914 2917 := CountSegment.join within_052_097 within_052_098
private theorem within_052_099_agreement : RowAgreement 6756 6760 := RowAgreement.join within_052_097_agreement within_052_098_agreement
private def within_052_100 : CountSegment 1 10 13120 6752 6760 2913 2917 := CountSegment.join within_052_096 within_052_099
private theorem within_052_100_agreement : RowAgreement 6752 6760 := RowAgreement.join within_052_096_agreement within_052_099_agreement
private def within_052_101 : CountSegment 1 10 13120 6760 6762 2917 2917 := CountSegment.join atomic_6761 atomic_6762
private theorem within_052_101_agreement : RowAgreement 6760 6762 := RowAgreement.join atomic_6761_agreement atomic_6762_agreement
private def within_052_102 : CountSegment 1 10 13120 6762 6764 2917 2918 := CountSegment.join atomic_6763 atomic_6764
private theorem within_052_102_agreement : RowAgreement 6762 6764 := RowAgreement.join atomic_6763_agreement atomic_6764_agreement
private def within_052_103 : CountSegment 1 10 13120 6760 6764 2917 2918 := CountSegment.join within_052_101 within_052_102
private theorem within_052_103_agreement : RowAgreement 6760 6764 := RowAgreement.join within_052_101_agreement within_052_102_agreement
private def within_052_104 : CountSegment 1 10 13120 6764 6766 2918 2918 := CountSegment.join atomic_6765 atomic_6766
private theorem within_052_104_agreement : RowAgreement 6764 6766 := RowAgreement.join atomic_6765_agreement atomic_6766_agreement
private def within_052_105 : CountSegment 1 10 13120 6766 6768 2918 2919 := CountSegment.join atomic_6767 atomic_6768
private theorem within_052_105_agreement : RowAgreement 6766 6768 := RowAgreement.join atomic_6767_agreement atomic_6768_agreement
private def within_052_106 : CountSegment 1 10 13120 6764 6768 2918 2919 := CountSegment.join within_052_104 within_052_105
private theorem within_052_106_agreement : RowAgreement 6764 6768 := RowAgreement.join within_052_104_agreement within_052_105_agreement
private def within_052_107 : CountSegment 1 10 13120 6760 6768 2917 2919 := CountSegment.join within_052_103 within_052_106
private theorem within_052_107_agreement : RowAgreement 6760 6768 := RowAgreement.join within_052_103_agreement within_052_106_agreement
private def within_052_108 : CountSegment 1 10 13120 6752 6768 2913 2919 := CountSegment.join within_052_100 within_052_107
private theorem within_052_108_agreement : RowAgreement 6752 6768 := RowAgreement.join within_052_100_agreement within_052_107_agreement
private def within_052_109 : CountSegment 1 10 13120 6768 6770 2919 2919 := CountSegment.join atomic_6769 atomic_6770
private theorem within_052_109_agreement : RowAgreement 6768 6770 := RowAgreement.join atomic_6769_agreement atomic_6770_agreement
private def within_052_110 : CountSegment 1 10 13120 6770 6772 2919 2919 := CountSegment.join atomic_6771 atomic_6772
private theorem within_052_110_agreement : RowAgreement 6770 6772 := RowAgreement.join atomic_6771_agreement atomic_6772_agreement
private def within_052_111 : CountSegment 1 10 13120 6768 6772 2919 2919 := CountSegment.join within_052_109 within_052_110
private theorem within_052_111_agreement : RowAgreement 6768 6772 := RowAgreement.join within_052_109_agreement within_052_110_agreement
private def within_052_112 : CountSegment 1 10 13120 6772 6774 2919 2922 := CountSegment.join atomic_6773 atomic_6774
private theorem within_052_112_agreement : RowAgreement 6772 6774 := RowAgreement.join atomic_6773_agreement atomic_6774_agreement
private def within_052_113 : CountSegment 1 10 13120 6774 6776 2922 2923 := CountSegment.join atomic_6775 atomic_6776
private theorem within_052_113_agreement : RowAgreement 6774 6776 := RowAgreement.join atomic_6775_agreement atomic_6776_agreement
private def within_052_114 : CountSegment 1 10 13120 6772 6776 2919 2923 := CountSegment.join within_052_112 within_052_113
private theorem within_052_114_agreement : RowAgreement 6772 6776 := RowAgreement.join within_052_112_agreement within_052_113_agreement
private def within_052_115 : CountSegment 1 10 13120 6768 6776 2919 2923 := CountSegment.join within_052_111 within_052_114
private theorem within_052_115_agreement : RowAgreement 6768 6776 := RowAgreement.join within_052_111_agreement within_052_114_agreement
private def within_052_116 : CountSegment 1 10 13120 6776 6778 2923 2924 := CountSegment.join atomic_6777 atomic_6778
private theorem within_052_116_agreement : RowAgreement 6776 6778 := RowAgreement.join atomic_6777_agreement atomic_6778_agreement
private def within_052_117 : CountSegment 1 10 13120 6778 6780 2924 2924 := CountSegment.join atomic_6779 atomic_6780
private theorem within_052_117_agreement : RowAgreement 6778 6780 := RowAgreement.join atomic_6779_agreement atomic_6780_agreement
private def within_052_118 : CountSegment 1 10 13120 6776 6780 2923 2924 := CountSegment.join within_052_116 within_052_117
private theorem within_052_118_agreement : RowAgreement 6776 6780 := RowAgreement.join within_052_116_agreement within_052_117_agreement
private def within_052_119 : CountSegment 1 10 13120 6780 6782 2924 2924 := CountSegment.join atomic_6781 atomic_6782
private theorem within_052_119_agreement : RowAgreement 6780 6782 := RowAgreement.join atomic_6781_agreement atomic_6782_agreement
private def within_052_120 : CountSegment 1 10 13120 6782 6784 2924 2925 := CountSegment.join atomic_6783 atomic_6784
private theorem within_052_120_agreement : RowAgreement 6782 6784 := RowAgreement.join atomic_6783_agreement atomic_6784_agreement
private def within_052_121 : CountSegment 1 10 13120 6780 6784 2924 2925 := CountSegment.join within_052_119 within_052_120
private theorem within_052_121_agreement : RowAgreement 6780 6784 := RowAgreement.join within_052_119_agreement within_052_120_agreement
private def within_052_122 : CountSegment 1 10 13120 6776 6784 2923 2925 := CountSegment.join within_052_118 within_052_121
private theorem within_052_122_agreement : RowAgreement 6776 6784 := RowAgreement.join within_052_118_agreement within_052_121_agreement
private def within_052_123 : CountSegment 1 10 13120 6768 6784 2919 2925 := CountSegment.join within_052_115 within_052_122
private theorem within_052_123_agreement : RowAgreement 6768 6784 := RowAgreement.join within_052_115_agreement within_052_122_agreement
private def within_052_124 : CountSegment 1 10 13120 6752 6784 2913 2925 := CountSegment.join within_052_108 within_052_123
private theorem within_052_124_agreement : RowAgreement 6752 6784 := RowAgreement.join within_052_108_agreement within_052_123_agreement
private def within_052_125 : CountSegment 1 10 13120 6720 6784 2902 2925 := CountSegment.join within_052_093 within_052_124
private theorem within_052_125_agreement : RowAgreement 6720 6784 := RowAgreement.join within_052_093_agreement within_052_124_agreement
private def within_052_126 : CountSegment 1 10 13120 6656 6784 2875 2925 := CountSegment.join within_052_062 within_052_125
private theorem within_052_126_agreement : RowAgreement 6656 6784 := RowAgreement.join within_052_062_agreement within_052_125_agreement
def segment_052 : CountSegment 1 10 13120 6656 6784 2875 2925 := within_052_126
theorem agreement_052 : RowAgreement 6656 6784 := within_052_126_agreement
end ProjectEulerSolutions.P58Numerical
