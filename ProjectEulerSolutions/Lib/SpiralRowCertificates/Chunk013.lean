import ProjectEulerSolutions.Lib.SpiralPrimeSegments
import ProjectEulerSolutions.Lib.SpiralPrimeCertificates.All
import ProjectEulerSolutions.Lib.MillerRabin
import ProjectEulerSolutions.Lib.SpiralMillerAgreement
/-! Finite spiral row and counter certificates. Python execution is checked separately. -/
namespace ProjectEulerSolutions.P58Numerical
open SpiralPrimes
set_option Elab.async false
def row_1665 : RowCertificate 1665 where
  first := true
  first_correct := by
    have prime : Nat.Prime (corner 1665 1) := by change Nat.Prime 11092231; exact ProjectEulerSolutions.P58PrimeNodes.p_11092231.prime
    exact decide_eq_true prime
  second := false
  second_correct := by
    have composite : ¬ Nat.Prime (corner 1665 2) := by change ¬ Nat.Prime 11088901; exact PrattCertificates.composite_of_product (factor := 41) (cofactor := 270461) rfl (by decide) (by decide)
    exact decide_eq_false composite
  third := false
  third_correct := by
    have composite : ¬ Nat.Prime (corner 1665 3) := by change ¬ Nat.Prime 11085571; exact PrattCertificates.composite_of_product (factor := 7) (cofactor := 1583653) rfl (by decide) (by decide)
    exact decide_eq_false composite
theorem row_count_1665 : row 1665 = 1 := (row_1665.checked).trans rfl
theorem model_count_1665 : modelRow 1665 = 1 := by decide
theorem modelrow_agrees_1665 : modelRow 1665 = row 1665 := model_count_1665.trans row_count_1665.symm
def row_1666 : RowCertificate 1666 where
  first := false
  first_correct := by
    have composite : ¬ Nat.Prime (corner 1666 1) := by change ¬ Nat.Prime 11105557; exact PrattCertificates.composite_of_product (factor := 19) (cofactor := 584503) rfl (by decide) (by decide)
    exact decide_eq_false composite
  second := false
  second_correct := by
    have composite : ¬ Nat.Prime (corner 1666 2) := by change ¬ Nat.Prime 11102225; exact PrattCertificates.composite_of_product (factor := 5) (cofactor := 2220445) rfl (by decide) (by decide)
    exact decide_eq_false composite
  third := false
  third_correct := by
    have composite : ¬ Nat.Prime (corner 1666 3) := by change ¬ Nat.Prime 11098893; exact PrattCertificates.composite_of_product (factor := 3) (cofactor := 3699631) rfl (by decide) (by decide)
    exact decide_eq_false composite
theorem row_count_1666 : row 1666 = 0 := (row_1666.checked).trans rfl
theorem model_count_1666 : modelRow 1666 = 0 := by decide
theorem modelrow_agrees_1666 : modelRow 1666 = row 1666 := model_count_1666.trans row_count_1666.symm
def row_1667 : RowCertificate 1667 where
  first := false
  first_correct := by
    have composite : ¬ Nat.Prime (corner 1667 1) := by change ¬ Nat.Prime 11118891; exact PrattCertificates.composite_of_product (factor := 3) (cofactor := 3706297) rfl (by decide) (by decide)
    exact decide_eq_false composite
  second := true
  second_correct := by
    have prime : Nat.Prime (corner 1667 2) := by change Nat.Prime 11115557; exact ProjectEulerSolutions.P58PrimeNodes.p_11115557.prime
    exact decide_eq_true prime
  third := false
  third_correct := by
    have composite : ¬ Nat.Prime (corner 1667 3) := by change ¬ Nat.Prime 11112223; exact PrattCertificates.composite_of_product (factor := 97) (cofactor := 114559) rfl (by decide) (by decide)
    exact decide_eq_false composite
theorem row_count_1667 : row 1667 = 1 := (row_1667.checked).trans rfl
theorem model_count_1667 : modelRow 1667 = 1 := by decide
theorem modelrow_agrees_1667 : modelRow 1667 = row 1667 := model_count_1667.trans row_count_1667.symm
def row_1668 : RowCertificate 1668 where
  first := false
  first_correct := by
    have composite : ¬ Nat.Prime (corner 1668 1) := by change ¬ Nat.Prime 11132233; exact PrattCertificates.composite_of_product (factor := 7) (cofactor := 1590319) rfl (by decide) (by decide)
    exact decide_eq_false composite
  second := false
  second_correct := by
    have composite : ¬ Nat.Prime (corner 1668 2) := by change ¬ Nat.Prime 11128897; exact PrattCertificates.composite_of_product (factor := 13) (cofactor := 856069) rfl (by decide) (by decide)
    exact decide_eq_false composite
  third := false
  third_correct := by
    have composite : ¬ Nat.Prime (corner 1668 3) := by change ¬ Nat.Prime 11125561; exact PrattCertificates.composite_of_product (factor := 1123) (cofactor := 9907) rfl (by decide) (by decide)
    exact decide_eq_false composite
theorem row_count_1668 : row 1668 = 0 := (row_1668.checked).trans rfl
theorem model_count_1668 : modelRow 1668 = 0 := by decide
theorem modelrow_agrees_1668 : modelRow 1668 = row 1668 := model_count_1668.trans row_count_1668.symm
def row_1669 : RowCertificate 1669 where
  first := true
  first_correct := by
    have prime : Nat.Prime (corner 1669 1) := by change Nat.Prime 11145583; exact ProjectEulerSolutions.P58PrimeNodes.p_11145583.prime
    exact decide_eq_true prime
  second := false
  second_correct := by
    have composite : ¬ Nat.Prime (corner 1669 2) := by change ¬ Nat.Prime 11142245; exact PrattCertificates.composite_of_product (factor := 5) (cofactor := 2228449) rfl (by decide) (by decide)
    exact decide_eq_false composite
  third := false
  third_correct := by
    have composite : ¬ Nat.Prime (corner 1669 3) := by change ¬ Nat.Prime 11138907; exact PrattCertificates.composite_of_product (factor := 3) (cofactor := 3712969) rfl (by decide) (by decide)
    exact decide_eq_false composite
theorem row_count_1669 : row 1669 = 1 := (row_1669.checked).trans rfl
theorem model_count_1669 : modelRow 1669 = 1 := by decide
theorem modelrow_agrees_1669 : modelRow 1669 = row 1669 := model_count_1669.trans row_count_1669.symm
def row_1670 : RowCertificate 1670 where
  first := false
  first_correct := by
    have composite : ¬ Nat.Prime (corner 1670 1) := by change ¬ Nat.Prime 11158941; exact PrattCertificates.composite_of_product (factor := 3) (cofactor := 3719647) rfl (by decide) (by decide)
    exact decide_eq_false composite
  second := true
  second_correct := by
    have prime : Nat.Prime (corner 1670 2) := by change Nat.Prime 11155601; exact ProjectEulerSolutions.P58PrimeNodes.p_11155601.prime
    exact decide_eq_true prime
  third := true
  third_correct := by
    have prime : Nat.Prime (corner 1670 3) := by change Nat.Prime 11152261; exact ProjectEulerSolutions.P58PrimeNodes.p_11152261.prime
    exact decide_eq_true prime
theorem row_count_1670 : row 1670 = 2 := (row_1670.checked).trans rfl
theorem model_count_1670 : modelRow 1670 = 2 := by decide
theorem modelrow_agrees_1670 : modelRow 1670 = row 1670 := model_count_1670.trans row_count_1670.symm
def row_1671 : RowCertificate 1671 where
  first := false
  first_correct := by
    have composite : ¬ Nat.Prime (corner 1671 1) := by change ¬ Nat.Prime 11172307; exact PrattCertificates.composite_of_product (factor := 31) (cofactor := 360397) rfl (by decide) (by decide)
    exact decide_eq_false composite
  second := false
  second_correct := by
    have composite : ¬ Nat.Prime (corner 1671 2) := by change ¬ Nat.Prime 11168965; exact PrattCertificates.composite_of_product (factor := 5) (cofactor := 2233793) rfl (by decide) (by decide)
    exact decide_eq_false composite
  third := false
  third_correct := by
    have composite : ¬ Nat.Prime (corner 1671 3) := by change ¬ Nat.Prime 11165623; exact PrattCertificates.composite_of_product (factor := 7) (cofactor := 1595089) rfl (by decide) (by decide)
    exact decide_eq_false composite
theorem row_count_1671 : row 1671 = 0 := (row_1671.checked).trans rfl
theorem model_count_1671 : modelRow 1671 = 0 := by decide
theorem modelrow_agrees_1671 : modelRow 1671 = row 1671 := model_count_1671.trans row_count_1671.symm
def row_1672 : RowCertificate 1672 where
  first := false
  first_correct := by
    have composite : ¬ Nat.Prime (corner 1672 1) := by change ¬ Nat.Prime 11185681; exact PrattCertificates.composite_of_product (factor := 13) (cofactor := 860437) rfl (by decide) (by decide)
    exact decide_eq_false composite
  second := false
  second_correct := by
    have composite : ¬ Nat.Prime (corner 1672 2) := by change ¬ Nat.Prime 11182337; exact PrattCertificates.composite_of_product (factor := 61) (cofactor := 183317) rfl (by decide) (by decide)
    exact decide_eq_false composite
  third := false
  third_correct := by
    have composite : ¬ Nat.Prime (corner 1672 3) := by change ¬ Nat.Prime 11178993; exact PrattCertificates.composite_of_product (factor := 3) (cofactor := 3726331) rfl (by decide) (by decide)
    exact decide_eq_false composite
theorem row_count_1672 : row 1672 = 0 := (row_1672.checked).trans rfl
theorem model_count_1672 : modelRow 1672 = 0 := by decide
theorem modelrow_agrees_1672 : modelRow 1672 = row 1672 := model_count_1672.trans row_count_1672.symm
def row_1673 : RowCertificate 1673 where
  first := false
  first_correct := by
    have composite : ¬ Nat.Prime (corner 1673 1) := by change ¬ Nat.Prime 11199063; exact PrattCertificates.composite_of_product (factor := 3) (cofactor := 3733021) rfl (by decide) (by decide)
    exact decide_eq_false composite
  second := false
  second_correct := by
    have composite : ¬ Nat.Prime (corner 1673 2) := by change ¬ Nat.Prime 11195717; exact PrattCertificates.composite_of_product (factor := 13) (cofactor := 861209) rfl (by decide) (by decide)
    exact decide_eq_false composite
  third := true
  third_correct := by
    have prime : Nat.Prime (corner 1673 3) := by change Nat.Prime 11192371; exact ProjectEulerSolutions.P58PrimeNodes.p_11192371.prime
    exact decide_eq_true prime
theorem row_count_1673 : row 1673 = 1 := (row_1673.checked).trans rfl
theorem model_count_1673 : modelRow 1673 = 1 := by decide
theorem modelrow_agrees_1673 : modelRow 1673 = row 1673 := model_count_1673.trans row_count_1673.symm
def row_1674 : RowCertificate 1674 where
  first := false
  first_correct := by
    have composite : ¬ Nat.Prime (corner 1674 1) := by change ¬ Nat.Prime 11212453; exact PrattCertificates.composite_of_product (factor := 7) (cofactor := 1601779) rfl (by decide) (by decide)
    exact decide_eq_false composite
  second := false
  second_correct := by
    have composite : ¬ Nat.Prime (corner 1674 2) := by change ¬ Nat.Prime 11209105; exact PrattCertificates.composite_of_product (factor := 5) (cofactor := 2241821) rfl (by decide) (by decide)
    exact decide_eq_false composite
  third := false
  third_correct := by
    have composite : ¬ Nat.Prime (corner 1674 3) := by change ¬ Nat.Prime 11205757; exact PrattCertificates.composite_of_product (factor := 43) (cofactor := 260599) rfl (by decide) (by decide)
    exact decide_eq_false composite
theorem row_count_1674 : row 1674 = 0 := (row_1674.checked).trans rfl
theorem model_count_1674 : modelRow 1674 = 0 := by decide
theorem modelrow_agrees_1674 : modelRow 1674 = row 1674 := model_count_1674.trans row_count_1674.symm
def row_1675 : RowCertificate 1675 where
  first := false
  first_correct := by
    have composite : ¬ Nat.Prime (corner 1675 1) := by change ¬ Nat.Prime 11225851; exact PrattCertificates.composite_of_product (factor := 7) (cofactor := 1603693) rfl (by decide) (by decide)
    exact decide_eq_false composite
  second := true
  second_correct := by
    have prime : Nat.Prime (corner 1675 2) := by change Nat.Prime 11222501; exact ProjectEulerSolutions.P58PrimeNodes.p_11222501.prime
    exact decide_eq_true prime
  third := false
  third_correct := by
    have composite : ¬ Nat.Prime (corner 1675 3) := by change ¬ Nat.Prime 11219151; exact PrattCertificates.composite_of_product (factor := 3) (cofactor := 3739717) rfl (by decide) (by decide)
    exact decide_eq_false composite
theorem row_count_1675 : row 1675 = 1 := (row_1675.checked).trans rfl
theorem model_count_1675 : modelRow 1675 = 1 := by decide
theorem modelrow_agrees_1675 : modelRow 1675 = row 1675 := model_count_1675.trans row_count_1675.symm
def row_1676 : RowCertificate 1676 where
  first := false
  first_correct := by
    have composite : ¬ Nat.Prime (corner 1676 1) := by change ¬ Nat.Prime 11239257; exact PrattCertificates.composite_of_product (factor := 3) (cofactor := 3746419) rfl (by decide) (by decide)
    exact decide_eq_false composite
  second := false
  second_correct := by
    have composite : ¬ Nat.Prime (corner 1676 2) := by change ¬ Nat.Prime 11235905; exact PrattCertificates.composite_of_product (factor := 5) (cofactor := 2247181) rfl (by decide) (by decide)
    exact decide_eq_false composite
  third := false
  third_correct := by
    have composite : ¬ Nat.Prime (corner 1676 3) := by change ¬ Nat.Prime 11232553; exact PrattCertificates.composite_of_product (factor := 19) (cofactor := 591187) rfl (by decide) (by decide)
    exact decide_eq_false composite
theorem row_count_1676 : row 1676 = 0 := (row_1676.checked).trans rfl
theorem model_count_1676 : modelRow 1676 = 0 := by decide
theorem modelrow_agrees_1676 : modelRow 1676 = row 1676 := model_count_1676.trans row_count_1676.symm
def row_1677 : RowCertificate 1677 where
  first := false
  first_correct := by
    have composite : ¬ Nat.Prime (corner 1677 1) := by change ¬ Nat.Prime 11252671; exact PrattCertificates.composite_of_product (factor := 151) (cofactor := 74521) rfl (by decide) (by decide)
    exact decide_eq_false composite
  second := false
  second_correct := by
    have composite : ¬ Nat.Prime (corner 1677 2) := by change ¬ Nat.Prime 11249317; exact PrattCertificates.composite_of_product (factor := 349) (cofactor := 32233) rfl (by decide) (by decide)
    exact decide_eq_false composite
  third := false
  third_correct := by
    have composite : ¬ Nat.Prime (corner 1677 3) := by change ¬ Nat.Prime 11245963; exact PrattCertificates.composite_of_product (factor := 31) (cofactor := 362773) rfl (by decide) (by decide)
    exact decide_eq_false composite
theorem row_count_1677 : row 1677 = 0 := (row_1677.checked).trans rfl
theorem model_count_1677 : modelRow 1677 = 0 := by decide
theorem modelrow_agrees_1677 : modelRow 1677 = row 1677 := model_count_1677.trans row_count_1677.symm
def row_1678 : RowCertificate 1678 where
  first := false
  first_correct := by
    have composite : ¬ Nat.Prime (corner 1678 1) := by change ¬ Nat.Prime 11266093; exact PrattCertificates.composite_of_product (factor := 37) (cofactor := 304489) rfl (by decide) (by decide)
    exact decide_eq_false composite
  second := true
  second_correct := by
    have prime : Nat.Prime (corner 1678 2) := by change Nat.Prime 11262737; exact ProjectEulerSolutions.P58PrimeNodes.p_11262737.prime
    exact decide_eq_true prime
  third := false
  third_correct := by
    have composite : ¬ Nat.Prime (corner 1678 3) := by change ¬ Nat.Prime 11259381; exact PrattCertificates.composite_of_product (factor := 3) (cofactor := 3753127) rfl (by decide) (by decide)
    exact decide_eq_false composite
theorem row_count_1678 : row 1678 = 1 := (row_1678.checked).trans rfl
theorem model_count_1678 : modelRow 1678 = 1 := by decide
theorem modelrow_agrees_1678 : modelRow 1678 = row 1678 := model_count_1678.trans row_count_1678.symm
def row_1679 : RowCertificate 1679 where
  first := false
  first_correct := by
    have composite : ¬ Nat.Prime (corner 1679 1) := by change ¬ Nat.Prime 11279523; exact PrattCertificates.composite_of_product (factor := 3) (cofactor := 3759841) rfl (by decide) (by decide)
    exact decide_eq_false composite
  second := false
  second_correct := by
    have composite : ¬ Nat.Prime (corner 1679 2) := by change ¬ Nat.Prime 11276165; exact PrattCertificates.composite_of_product (factor := 5) (cofactor := 2255233) rfl (by decide) (by decide)
    exact decide_eq_false composite
  third := false
  third_correct := by
    have composite : ¬ Nat.Prime (corner 1679 3) := by change ¬ Nat.Prime 11272807; exact PrattCertificates.composite_of_product (factor := 7) (cofactor := 1610401) rfl (by decide) (by decide)
    exact decide_eq_false composite
theorem row_count_1679 : row 1679 = 0 := (row_1679.checked).trans rfl
theorem model_count_1679 : modelRow 1679 = 0 := by decide
theorem modelrow_agrees_1679 : modelRow 1679 = row 1679 := model_count_1679.trans row_count_1679.symm
def row_1680 : RowCertificate 1680 where
  first := false
  first_correct := by
    have composite : ¬ Nat.Prime (corner 1680 1) := by change ¬ Nat.Prime 11292961; exact PrattCertificates.composite_of_product (factor := 43) (cofactor := 262627) rfl (by decide) (by decide)
    exact decide_eq_false composite
  second := true
  second_correct := by
    have prime : Nat.Prime (corner 1680 2) := by change Nat.Prime 11289601; exact ProjectEulerSolutions.P58PrimeNodes.p_11289601.prime
    exact decide_eq_true prime
  third := false
  third_correct := by
    have composite : ¬ Nat.Prime (corner 1680 3) := by change ¬ Nat.Prime 11286241; exact PrattCertificates.composite_of_product (factor := 97) (cofactor := 116353) rfl (by decide) (by decide)
    exact decide_eq_false composite
theorem row_count_1680 : row 1680 = 1 := (row_1680.checked).trans rfl
theorem model_count_1680 : modelRow 1680 = 1 := by decide
theorem modelrow_agrees_1680 : modelRow 1680 = row 1680 := model_count_1680.trans row_count_1680.symm
def row_1681 : RowCertificate 1681 where
  first := false
  first_correct := by
    have composite : ¬ Nat.Prime (corner 1681 1) := by change ¬ Nat.Prime 11306407; exact PrattCertificates.composite_of_product (factor := 7) (cofactor := 1615201) rfl (by decide) (by decide)
    exact decide_eq_false composite
  second := false
  second_correct := by
    have composite : ¬ Nat.Prime (corner 1681 2) := by change ¬ Nat.Prime 11303045; exact PrattCertificates.composite_of_product (factor := 5) (cofactor := 2260609) rfl (by decide) (by decide)
    exact decide_eq_false composite
  third := false
  third_correct := by
    have composite : ¬ Nat.Prime (corner 1681 3) := by change ¬ Nat.Prime 11299683; exact PrattCertificates.composite_of_product (factor := 3) (cofactor := 3766561) rfl (by decide) (by decide)
    exact decide_eq_false composite
theorem row_count_1681 : row 1681 = 0 := (row_1681.checked).trans rfl
theorem model_count_1681 : modelRow 1681 = 0 := by decide
theorem modelrow_agrees_1681 : modelRow 1681 = row 1681 := model_count_1681.trans row_count_1681.symm
def row_1682 : RowCertificate 1682 where
  first := false
  first_correct := by
    have composite : ¬ Nat.Prime (corner 1682 1) := by change ¬ Nat.Prime 11319861; exact PrattCertificates.composite_of_product (factor := 3) (cofactor := 3773287) rfl (by decide) (by decide)
    exact decide_eq_false composite
  second := false
  second_correct := by
    have composite : ¬ Nat.Prime (corner 1682 2) := by change ¬ Nat.Prime 11316497; exact PrattCertificates.composite_of_product (factor := 2393) (cofactor := 4729) rfl (by decide) (by decide)
    exact decide_eq_false composite
  third := false
  third_correct := by
    have composite : ¬ Nat.Prime (corner 1682 3) := by change ¬ Nat.Prime 11313133; exact PrattCertificates.composite_of_product (factor := 13) (cofactor := 870241) rfl (by decide) (by decide)
    exact decide_eq_false composite
theorem row_count_1682 : row 1682 = 0 := (row_1682.checked).trans rfl
theorem model_count_1682 : modelRow 1682 = 0 := by decide
theorem modelrow_agrees_1682 : modelRow 1682 = row 1682 := model_count_1682.trans row_count_1682.symm
def row_1683 : RowCertificate 1683 where
  first := false
  first_correct := by
    have composite : ¬ Nat.Prime (corner 1683 1) := by change ¬ Nat.Prime 11333323; exact PrattCertificates.composite_of_product (factor := 73) (cofactor := 155251) rfl (by decide) (by decide)
    exact decide_eq_false composite
  second := false
  second_correct := by
    have composite : ¬ Nat.Prime (corner 1683 2) := by change ¬ Nat.Prime 11329957; exact PrattCertificates.composite_of_product (factor := 61) (cofactor := 185737) rfl (by decide) (by decide)
    exact decide_eq_false composite
  third := false
  third_correct := by
    have composite : ¬ Nat.Prime (corner 1683 3) := by change ¬ Nat.Prime 11326591; exact PrattCertificates.composite_of_product (factor := 193) (cofactor := 58687) rfl (by decide) (by decide)
    exact decide_eq_false composite
theorem row_count_1683 : row 1683 = 0 := (row_1683.checked).trans rfl
theorem model_count_1683 : modelRow 1683 = 0 := by decide
theorem modelrow_agrees_1683 : modelRow 1683 = row 1683 := model_count_1683.trans row_count_1683.symm
def row_1684 : RowCertificate 1684 where
  first := false
  first_correct := by
    have composite : ¬ Nat.Prime (corner 1684 1) := by change ¬ Nat.Prime 11346793; exact PrattCertificates.composite_of_product (factor := 61) (cofactor := 186013) rfl (by decide) (by decide)
    exact decide_eq_false composite
  second := false
  second_correct := by
    have composite : ¬ Nat.Prime (corner 1684 2) := by change ¬ Nat.Prime 11343425; exact PrattCertificates.composite_of_product (factor := 5) (cofactor := 2268685) rfl (by decide) (by decide)
    exact decide_eq_false composite
  third := false
  third_correct := by
    have composite : ¬ Nat.Prime (corner 1684 3) := by change ¬ Nat.Prime 11340057; exact PrattCertificates.composite_of_product (factor := 3) (cofactor := 3780019) rfl (by decide) (by decide)
    exact decide_eq_false composite
theorem row_count_1684 : row 1684 = 0 := (row_1684.checked).trans rfl
theorem model_count_1684 : modelRow 1684 = 0 := by decide
theorem modelrow_agrees_1684 : modelRow 1684 = row 1684 := model_count_1684.trans row_count_1684.symm
def row_1685 : RowCertificate 1685 where
  first := false
  first_correct := by
    have composite : ¬ Nat.Prime (corner 1685 1) := by change ¬ Nat.Prime 11360271; exact PrattCertificates.composite_of_product (factor := 3) (cofactor := 3786757) rfl (by decide) (by decide)
    exact decide_eq_false composite
  second := false
  second_correct := by
    have composite : ¬ Nat.Prime (corner 1685 2) := by change ¬ Nat.Prime 11356901; exact PrattCertificates.composite_of_product (factor := 17) (cofactor := 668053) rfl (by decide) (by decide)
    exact decide_eq_false composite
  third := false
  third_correct := by
    have composite : ¬ Nat.Prime (corner 1685 3) := by change ¬ Nat.Prime 11353531; exact PrattCertificates.composite_of_product (factor := 7) (cofactor := 1621933) rfl (by decide) (by decide)
    exact decide_eq_false composite
theorem row_count_1685 : row 1685 = 0 := (row_1685.checked).trans rfl
theorem model_count_1685 : modelRow 1685 = 0 := by decide
theorem modelrow_agrees_1685 : modelRow 1685 = row 1685 := model_count_1685.trans row_count_1685.symm
def row_1686 : RowCertificate 1686 where
  first := false
  first_correct := by
    have composite : ¬ Nat.Prime (corner 1686 1) := by change ¬ Nat.Prime 11373757; exact PrattCertificates.composite_of_product (factor := 1783) (cofactor := 6379) rfl (by decide) (by decide)
    exact decide_eq_false composite
  second := false
  second_correct := by
    have composite : ¬ Nat.Prime (corner 1686 2) := by change ¬ Nat.Prime 11370385; exact PrattCertificates.composite_of_product (factor := 5) (cofactor := 2274077) rfl (by decide) (by decide)
    exact decide_eq_false composite
  third := false
  third_correct := by
    have composite : ¬ Nat.Prime (corner 1686 3) := by change ¬ Nat.Prime 11367013; exact PrattCertificates.composite_of_product (factor := 7) (cofactor := 1623859) rfl (by decide) (by decide)
    exact decide_eq_false composite
theorem row_count_1686 : row 1686 = 0 := (row_1686.checked).trans rfl
theorem model_count_1686 : modelRow 1686 = 0 := by decide
theorem modelrow_agrees_1686 : modelRow 1686 = row 1686 := model_count_1686.trans row_count_1686.symm
def row_1687 : RowCertificate 1687 where
  first := false
  first_correct := by
    have composite : ¬ Nat.Prime (corner 1687 1) := by change ¬ Nat.Prime 11387251; exact PrattCertificates.composite_of_product (factor := 19) (cofactor := 599329) rfl (by decide) (by decide)
    exact decide_eq_false composite
  second := true
  second_correct := by
    have prime : Nat.Prime (corner 1687 2) := by change Nat.Prime 11383877; exact ProjectEulerSolutions.P58PrimeNodes.p_11383877.prime
    exact decide_eq_true prime
  third := false
  third_correct := by
    have composite : ¬ Nat.Prime (corner 1687 3) := by change ¬ Nat.Prime 11380503; exact PrattCertificates.composite_of_product (factor := 3) (cofactor := 3793501) rfl (by decide) (by decide)
    exact decide_eq_false composite
theorem row_count_1687 : row 1687 = 1 := (row_1687.checked).trans rfl
theorem model_count_1687 : modelRow 1687 = 1 := by decide
theorem modelrow_agrees_1687 : modelRow 1687 = row 1687 := model_count_1687.trans row_count_1687.symm
def row_1688 : RowCertificate 1688 where
  first := false
  first_correct := by
    have composite : ¬ Nat.Prime (corner 1688 1) := by change ¬ Nat.Prime 11400753; exact PrattCertificates.composite_of_product (factor := 3) (cofactor := 3800251) rfl (by decide) (by decide)
    exact decide_eq_false composite
  second := false
  second_correct := by
    have composite : ¬ Nat.Prime (corner 1688 2) := by change ¬ Nat.Prime 11397377; exact PrattCertificates.composite_of_product (factor := 29) (cofactor := 393013) rfl (by decide) (by decide)
    exact decide_eq_false composite
  third := false
  third_correct := by
    have composite : ¬ Nat.Prime (corner 1688 3) := by change ¬ Nat.Prime 11394001; exact PrattCertificates.composite_of_product (factor := 541) (cofactor := 21061) rfl (by decide) (by decide)
    exact decide_eq_false composite
theorem row_count_1688 : row 1688 = 0 := (row_1688.checked).trans rfl
theorem model_count_1688 : modelRow 1688 = 0 := by decide
theorem modelrow_agrees_1688 : modelRow 1688 = row 1688 := model_count_1688.trans row_count_1688.symm
def row_1689 : RowCertificate 1689 where
  first := false
  first_correct := by
    have composite : ¬ Nat.Prime (corner 1689 1) := by change ¬ Nat.Prime 11414263; exact PrattCertificates.composite_of_product (factor := 7) (cofactor := 1630609) rfl (by decide) (by decide)
    exact decide_eq_false composite
  second := false
  second_correct := by
    have composite : ¬ Nat.Prime (corner 1689 2) := by change ¬ Nat.Prime 11410885; exact PrattCertificates.composite_of_product (factor := 5) (cofactor := 2282177) rfl (by decide) (by decide)
    exact decide_eq_false composite
  third := false
  third_correct := by
    have composite : ¬ Nat.Prime (corner 1689 3) := by change ¬ Nat.Prime 11407507; exact PrattCertificates.composite_of_product (factor := 37) (cofactor := 308311) rfl (by decide) (by decide)
    exact decide_eq_false composite
theorem row_count_1689 : row 1689 = 0 := (row_1689.checked).trans rfl
theorem model_count_1689 : modelRow 1689 = 0 := by decide
theorem modelrow_agrees_1689 : modelRow 1689 = row 1689 := model_count_1689.trans row_count_1689.symm
def row_1690 : RowCertificate 1690 where
  first := true
  first_correct := by
    have prime : Nat.Prime (corner 1690 1) := by change Nat.Prime 11427781; exact ProjectEulerSolutions.P58PrimeNodes.p_11427781.prime
    exact decide_eq_true prime
  second := false
  second_correct := by
    have composite : ¬ Nat.Prime (corner 1690 2) := by change ¬ Nat.Prime 11424401; exact PrattCertificates.composite_of_product (factor := 173) (cofactor := 66037) rfl (by decide) (by decide)
    exact decide_eq_false composite
  third := false
  third_correct := by
    have composite : ¬ Nat.Prime (corner 1690 3) := by change ¬ Nat.Prime 11421021; exact PrattCertificates.composite_of_product (factor := 3) (cofactor := 3807007) rfl (by decide) (by decide)
    exact decide_eq_false composite
theorem row_count_1690 : row 1690 = 1 := (row_1690.checked).trans rfl
theorem model_count_1690 : modelRow 1690 = 1 := by decide
theorem modelrow_agrees_1690 : modelRow 1690 = row 1690 := model_count_1690.trans row_count_1690.symm
def row_1691 : RowCertificate 1691 where
  first := false
  first_correct := by
    have composite : ¬ Nat.Prime (corner 1691 1) := by change ¬ Nat.Prime 11441307; exact PrattCertificates.composite_of_product (factor := 3) (cofactor := 3813769) rfl (by decide) (by decide)
    exact decide_eq_false composite
  second := false
  second_correct := by
    have composite : ¬ Nat.Prime (corner 1691 2) := by change ¬ Nat.Prime 11437925; exact PrattCertificates.composite_of_product (factor := 5) (cofactor := 2287585) rfl (by decide) (by decide)
    exact decide_eq_false composite
  third := true
  third_correct := by
    have prime : Nat.Prime (corner 1691 3) := by change Nat.Prime 11434543; exact ProjectEulerSolutions.P58PrimeNodes.p_11434543.prime
    exact decide_eq_true prime
theorem row_count_1691 : row 1691 = 1 := (row_1691.checked).trans rfl
theorem model_count_1691 : modelRow 1691 = 1 := by decide
theorem modelrow_agrees_1691 : modelRow 1691 = row 1691 := model_count_1691.trans row_count_1691.symm
def row_1692 : RowCertificate 1692 where
  first := false
  first_correct := by
    have composite : ¬ Nat.Prime (corner 1692 1) := by change ¬ Nat.Prime 11454841; exact PrattCertificates.composite_of_product (factor := 31) (cofactor := 369511) rfl (by decide) (by decide)
    exact decide_eq_false composite
  second := false
  second_correct := by
    have composite : ¬ Nat.Prime (corner 1692 2) := by change ¬ Nat.Prime 11451457; exact PrattCertificates.composite_of_product (factor := 277) (cofactor := 41341) rfl (by decide) (by decide)
    exact decide_eq_false composite
  third := false
  third_correct := by
    have composite : ¬ Nat.Prime (corner 1692 3) := by change ¬ Nat.Prime 11448073; exact PrattCertificates.composite_of_product (factor := 7) (cofactor := 1635439) rfl (by decide) (by decide)
    exact decide_eq_false composite
theorem row_count_1692 : row 1692 = 0 := (row_1692.checked).trans rfl
theorem model_count_1692 : modelRow 1692 = 0 := by decide
theorem modelrow_agrees_1692 : modelRow 1692 = row 1692 := model_count_1692.trans row_count_1692.symm
def row_1693 : RowCertificate 1693 where
  first := false
  first_correct := by
    have composite : ¬ Nat.Prime (corner 1693 1) := by change ¬ Nat.Prime 11468383; exact PrattCertificates.composite_of_product (factor := 367) (cofactor := 31249) rfl (by decide) (by decide)
    exact decide_eq_false composite
  second := false
  second_correct := by
    have composite : ¬ Nat.Prime (corner 1693 2) := by change ¬ Nat.Prime 11464997; exact PrattCertificates.composite_of_product (factor := 389) (cofactor := 29473) rfl (by decide) (by decide)
    exact decide_eq_false composite
  third := false
  third_correct := by
    have composite : ¬ Nat.Prime (corner 1693 3) := by change ¬ Nat.Prime 11461611; exact PrattCertificates.composite_of_product (factor := 3) (cofactor := 3820537) rfl (by decide) (by decide)
    exact decide_eq_false composite
theorem row_count_1693 : row 1693 = 0 := (row_1693.checked).trans rfl
theorem model_count_1693 : modelRow 1693 = 0 := by decide
theorem modelrow_agrees_1693 : modelRow 1693 = row 1693 := model_count_1693.trans row_count_1693.symm
def row_1694 : RowCertificate 1694 where
  first := false
  first_correct := by
    have composite : ¬ Nat.Prime (corner 1694 1) := by change ¬ Nat.Prime 11481933; exact PrattCertificates.composite_of_product (factor := 3) (cofactor := 3827311) rfl (by decide) (by decide)
    exact decide_eq_false composite
  second := false
  second_correct := by
    have composite : ¬ Nat.Prime (corner 1694 2) := by change ¬ Nat.Prime 11478545; exact PrattCertificates.composite_of_product (factor := 5) (cofactor := 2295709) rfl (by decide) (by decide)
    exact decide_eq_false composite
  third := false
  third_correct := by
    have composite : ¬ Nat.Prime (corner 1694 3) := by change ¬ Nat.Prime 11475157; exact PrattCertificates.composite_of_product (factor := 67) (cofactor := 171271) rfl (by decide) (by decide)
    exact decide_eq_false composite
theorem row_count_1694 : row 1694 = 0 := (row_1694.checked).trans rfl
theorem model_count_1694 : modelRow 1694 = 0 := by decide
theorem modelrow_agrees_1694 : modelRow 1694 = row 1694 := model_count_1694.trans row_count_1694.symm
def row_1695 : RowCertificate 1695 where
  first := false
  first_correct := by
    have composite : ¬ Nat.Prime (corner 1695 1) := by change ¬ Nat.Prime 11495491; exact PrattCertificates.composite_of_product (factor := 7) (cofactor := 1642213) rfl (by decide) (by decide)
    exact decide_eq_false composite
  second := true
  second_correct := by
    have prime : Nat.Prime (corner 1695 2) := by change Nat.Prime 11492101; exact ProjectEulerSolutions.P58PrimeNodes.p_11492101.prime
    exact decide_eq_true prime
  third := false
  third_correct := by
    have composite : ¬ Nat.Prime (corner 1695 3) := by change ¬ Nat.Prime 11488711; exact PrattCertificates.composite_of_product (factor := 13) (cofactor := 883747) rfl (by decide) (by decide)
    exact decide_eq_false composite
theorem row_count_1695 : row 1695 = 1 := (row_1695.checked).trans rfl
theorem model_count_1695 : modelRow 1695 = 1 := by decide
theorem modelrow_agrees_1695 : modelRow 1695 = row 1695 := model_count_1695.trans row_count_1695.symm
def row_1696 : RowCertificate 1696 where
  first := false
  first_correct := by
    have composite : ¬ Nat.Prime (corner 1696 1) := by change ¬ Nat.Prime 11509057; exact PrattCertificates.composite_of_product (factor := 7) (cofactor := 1644151) rfl (by decide) (by decide)
    exact decide_eq_false composite
  second := false
  second_correct := by
    have composite : ¬ Nat.Prime (corner 1696 2) := by change ¬ Nat.Prime 11505665; exact PrattCertificates.composite_of_product (factor := 5) (cofactor := 2301133) rfl (by decide) (by decide)
    exact decide_eq_false composite
  third := false
  third_correct := by
    have composite : ¬ Nat.Prime (corner 1696 3) := by change ¬ Nat.Prime 11502273; exact PrattCertificates.composite_of_product (factor := 3) (cofactor := 3834091) rfl (by decide) (by decide)
    exact decide_eq_false composite
theorem row_count_1696 : row 1696 = 0 := (row_1696.checked).trans rfl
theorem model_count_1696 : modelRow 1696 = 0 := by decide
theorem modelrow_agrees_1696 : modelRow 1696 = row 1696 := model_count_1696.trans row_count_1696.symm
def row_1697 : RowCertificate 1697 where
  first := false
  first_correct := by
    have composite : ¬ Nat.Prime (corner 1697 1) := by change ¬ Nat.Prime 11522631; exact PrattCertificates.composite_of_product (factor := 3) (cofactor := 3840877) rfl (by decide) (by decide)
    exact decide_eq_false composite
  second := false
  second_correct := by
    have composite : ¬ Nat.Prime (corner 1697 2) := by change ¬ Nat.Prime 11519237; exact PrattCertificates.composite_of_product (factor := 41) (cofactor := 280957) rfl (by decide) (by decide)
    exact decide_eq_false composite
  third := false
  third_correct := by
    have composite : ¬ Nat.Prime (corner 1697 3) := by change ¬ Nat.Prime 11515843; exact PrattCertificates.composite_of_product (factor := 19) (cofactor := 606097) rfl (by decide) (by decide)
    exact decide_eq_false composite
theorem row_count_1697 : row 1697 = 0 := (row_1697.checked).trans rfl
theorem model_count_1697 : modelRow 1697 = 0 := by decide
theorem modelrow_agrees_1697 : modelRow 1697 = row 1697 := model_count_1697.trans row_count_1697.symm
def row_1698 : RowCertificate 1698 where
  first := false
  first_correct := by
    have composite : ¬ Nat.Prime (corner 1698 1) := by change ¬ Nat.Prime 11536213; exact PrattCertificates.composite_of_product (factor := 13) (cofactor := 887401) rfl (by decide) (by decide)
    exact decide_eq_false composite
  second := false
  second_correct := by
    have composite : ¬ Nat.Prime (corner 1698 2) := by change ¬ Nat.Prime 11532817; exact PrattCertificates.composite_of_product (factor := 17) (cofactor := 678401) rfl (by decide) (by decide)
    exact decide_eq_false composite
  third := true
  third_correct := by
    have prime : Nat.Prime (corner 1698 3) := by change Nat.Prime 11529421; exact ProjectEulerSolutions.P58PrimeNodes.p_11529421.prime
    exact decide_eq_true prime
theorem row_count_1698 : row 1698 = 1 := (row_1698.checked).trans rfl
theorem model_count_1698 : modelRow 1698 = 1 := by decide
theorem modelrow_agrees_1698 : modelRow 1698 = row 1698 := model_count_1698.trans row_count_1698.symm
def row_1699 : RowCertificate 1699 where
  first := true
  first_correct := by
    have prime : Nat.Prime (corner 1699 1) := by change Nat.Prime 11549803; exact ProjectEulerSolutions.P58PrimeNodes.p_11549803.prime
    exact decide_eq_true prime
  second := false
  second_correct := by
    have composite : ¬ Nat.Prime (corner 1699 2) := by change ¬ Nat.Prime 11546405; exact PrattCertificates.composite_of_product (factor := 5) (cofactor := 2309281) rfl (by decide) (by decide)
    exact decide_eq_false composite
  third := false
  third_correct := by
    have composite : ¬ Nat.Prime (corner 1699 3) := by change ¬ Nat.Prime 11543007; exact PrattCertificates.composite_of_product (factor := 3) (cofactor := 3847669) rfl (by decide) (by decide)
    exact decide_eq_false composite
theorem row_count_1699 : row 1699 = 1 := (row_1699.checked).trans rfl
theorem model_count_1699 : modelRow 1699 = 1 := by decide
theorem modelrow_agrees_1699 : modelRow 1699 = row 1699 := model_count_1699.trans row_count_1699.symm
def row_1700 : RowCertificate 1700 where
  first := false
  first_correct := by
    have composite : ¬ Nat.Prime (corner 1700 1) := by change ¬ Nat.Prime 11563401; exact PrattCertificates.composite_of_product (factor := 3) (cofactor := 3854467) rfl (by decide) (by decide)
    exact decide_eq_false composite
  second := false
  second_correct := by
    have composite : ¬ Nat.Prime (corner 1700 2) := by change ¬ Nat.Prime 11560001; exact PrattCertificates.composite_of_product (factor := 2797) (cofactor := 4133) rfl (by decide) (by decide)
    exact decide_eq_false composite
  third := false
  third_correct := by
    have composite : ¬ Nat.Prime (corner 1700 3) := by change ¬ Nat.Prime 11556601; exact PrattCertificates.composite_of_product (factor := 7) (cofactor := 1650943) rfl (by decide) (by decide)
    exact decide_eq_false composite
theorem row_count_1700 : row 1700 = 0 := (row_1700.checked).trans rfl
theorem model_count_1700 : modelRow 1700 = 0 := by decide
theorem modelrow_agrees_1700 : modelRow 1700 = row 1700 := model_count_1700.trans row_count_1700.symm
def row_1701 : RowCertificate 1701 where
  first := false
  first_correct := by
    have composite : ¬ Nat.Prime (corner 1701 1) := by change ¬ Nat.Prime 11577007; exact PrattCertificates.composite_of_product (factor := 13) (cofactor := 890539) rfl (by decide) (by decide)
    exact decide_eq_false composite
  second := false
  second_correct := by
    have composite : ¬ Nat.Prime (corner 1701 2) := by change ¬ Nat.Prime 11573605; exact PrattCertificates.composite_of_product (factor := 5) (cofactor := 2314721) rfl (by decide) (by decide)
    exact decide_eq_false composite
  third := false
  third_correct := by
    have composite : ¬ Nat.Prime (corner 1701 3) := by change ¬ Nat.Prime 11570203; exact PrattCertificates.composite_of_product (factor := 1009) (cofactor := 11467) rfl (by decide) (by decide)
    exact decide_eq_false composite
theorem row_count_1701 : row 1701 = 0 := (row_1701.checked).trans rfl
theorem model_count_1701 : modelRow 1701 = 0 := by decide
theorem modelrow_agrees_1701 : modelRow 1701 = row 1701 := model_count_1701.trans row_count_1701.symm
def row_1702 : RowCertificate 1702 where
  first := false
  first_correct := by
    have composite : ¬ Nat.Prime (corner 1702 1) := by change ¬ Nat.Prime 11590621; exact PrattCertificates.composite_of_product (factor := 7) (cofactor := 1655803) rfl (by decide) (by decide)
    exact decide_eq_false composite
  second := false
  second_correct := by
    have composite : ¬ Nat.Prime (corner 1702 2) := by change ¬ Nat.Prime 11587217; exact PrattCertificates.composite_of_product (factor := 17) (cofactor := 681601) rfl (by decide) (by decide)
    exact decide_eq_false composite
  third := false
  third_correct := by
    have composite : ¬ Nat.Prime (corner 1702 3) := by change ¬ Nat.Prime 11583813; exact PrattCertificates.composite_of_product (factor := 3) (cofactor := 3861271) rfl (by decide) (by decide)
    exact decide_eq_false composite
theorem row_count_1702 : row 1702 = 0 := (row_1702.checked).trans rfl
theorem model_count_1702 : modelRow 1702 = 0 := by decide
theorem modelrow_agrees_1702 : modelRow 1702 = row 1702 := model_count_1702.trans row_count_1702.symm
def row_1703 : RowCertificate 1703 where
  first := false
  first_correct := by
    have composite : ¬ Nat.Prime (corner 1703 1) := by change ¬ Nat.Prime 11604243; exact PrattCertificates.composite_of_product (factor := 3) (cofactor := 3868081) rfl (by decide) (by decide)
    exact decide_eq_false composite
  second := false
  second_correct := by
    have composite : ¬ Nat.Prime (corner 1703 2) := by change ¬ Nat.Prime 11600837; exact PrattCertificates.composite_of_product (factor := 233) (cofactor := 49789) rfl (by decide) (by decide)
    exact decide_eq_false composite
  third := true
  third_correct := by
    have prime : Nat.Prime (corner 1703 3) := by change Nat.Prime 11597431; exact ProjectEulerSolutions.P58PrimeNodes.p_11597431.prime
    exact decide_eq_true prime
theorem row_count_1703 : row 1703 = 1 := (row_1703.checked).trans rfl
theorem model_count_1703 : modelRow 1703 = 1 := by decide
theorem modelrow_agrees_1703 : modelRow 1703 = row 1703 := model_count_1703.trans row_count_1703.symm
def row_1704 : RowCertificate 1704 where
  first := false
  first_correct := by
    have composite : ¬ Nat.Prime (corner 1704 1) := by change ¬ Nat.Prime 11617873; exact PrattCertificates.composite_of_product (factor := 19) (cofactor := 611467) rfl (by decide) (by decide)
    exact decide_eq_false composite
  second := false
  second_correct := by
    have composite : ¬ Nat.Prime (corner 1704 2) := by change ¬ Nat.Prime 11614465; exact PrattCertificates.composite_of_product (factor := 5) (cofactor := 2322893) rfl (by decide) (by decide)
    exact decide_eq_false composite
  third := true
  third_correct := by
    have prime : Nat.Prime (corner 1704 3) := by change Nat.Prime 11611057; exact ProjectEulerSolutions.P58PrimeNodes.p_11611057.prime
    exact decide_eq_true prime
theorem row_count_1704 : row 1704 = 1 := (row_1704.checked).trans rfl
theorem model_count_1704 : modelRow 1704 = 1 := by decide
theorem modelrow_agrees_1704 : modelRow 1704 = row 1704 := model_count_1704.trans row_count_1704.symm
def row_1705 : RowCertificate 1705 where
  first := false
  first_correct := by
    have composite : ¬ Nat.Prime (corner 1705 1) := by change ¬ Nat.Prime 11631511; exact PrattCertificates.composite_of_product (factor := 1237) (cofactor := 9403) rfl (by decide) (by decide)
    exact decide_eq_false composite
  second := false
  second_correct := by
    have composite : ¬ Nat.Prime (corner 1705 2) := by change ¬ Nat.Prime 11628101; exact PrattCertificates.composite_of_product (factor := 29) (cofactor := 400969) rfl (by decide) (by decide)
    exact decide_eq_false composite
  third := false
  third_correct := by
    have composite : ¬ Nat.Prime (corner 1705 3) := by change ¬ Nat.Prime 11624691; exact PrattCertificates.composite_of_product (factor := 3) (cofactor := 3874897) rfl (by decide) (by decide)
    exact decide_eq_false composite
theorem row_count_1705 : row 1705 = 0 := (row_1705.checked).trans rfl
theorem model_count_1705 : modelRow 1705 = 0 := by decide
theorem modelrow_agrees_1705 : modelRow 1705 = row 1705 := model_count_1705.trans row_count_1705.symm
def row_1706 : RowCertificate 1706 where
  first := false
  first_correct := by
    have composite : ¬ Nat.Prime (corner 1706 1) := by change ¬ Nat.Prime 11645157; exact PrattCertificates.composite_of_product (factor := 3) (cofactor := 3881719) rfl (by decide) (by decide)
    exact decide_eq_false composite
  second := false
  second_correct := by
    have composite : ¬ Nat.Prime (corner 1706 2) := by change ¬ Nat.Prime 11641745; exact PrattCertificates.composite_of_product (factor := 5) (cofactor := 2328349) rfl (by decide) (by decide)
    exact decide_eq_false composite
  third := false
  third_correct := by
    have composite : ¬ Nat.Prime (corner 1706 3) := by change ¬ Nat.Prime 11638333; exact PrattCertificates.composite_of_product (factor := 7) (cofactor := 1662619) rfl (by decide) (by decide)
    exact decide_eq_false composite
theorem row_count_1706 : row 1706 = 0 := (row_1706.checked).trans rfl
theorem model_count_1706 : modelRow 1706 = 0 := by decide
theorem modelrow_agrees_1706 : modelRow 1706 = row 1706 := model_count_1706.trans row_count_1706.symm
def row_1707 : RowCertificate 1707 where
  first := false
  first_correct := by
    have composite : ¬ Nat.Prime (corner 1707 1) := by change ¬ Nat.Prime 11658811; exact PrattCertificates.composite_of_product (factor := 37) (cofactor := 315103) rfl (by decide) (by decide)
    exact decide_eq_false composite
  second := false
  second_correct := by
    have composite : ¬ Nat.Prime (corner 1707 2) := by change ¬ Nat.Prime 11655397; exact PrattCertificates.composite_of_product (factor := 13) (cofactor := 896569) rfl (by decide) (by decide)
    exact decide_eq_false composite
  third := false
  third_correct := by
    have composite : ¬ Nat.Prime (corner 1707 3) := by change ¬ Nat.Prime 11651983; exact PrattCertificates.composite_of_product (factor := 7) (cofactor := 1664569) rfl (by decide) (by decide)
    exact decide_eq_false composite
theorem row_count_1707 : row 1707 = 0 := (row_1707.checked).trans rfl
theorem model_count_1707 : modelRow 1707 = 0 := by decide
theorem modelrow_agrees_1707 : modelRow 1707 = row 1707 := model_count_1707.trans row_count_1707.symm
def row_1708 : RowCertificate 1708 where
  first := false
  first_correct := by
    have composite : ¬ Nat.Prime (corner 1708 1) := by change ¬ Nat.Prime 11672473; exact PrattCertificates.composite_of_product (factor := 1597) (cofactor := 7309) rfl (by decide) (by decide)
    exact decide_eq_false composite
  second := false
  second_correct := by
    have composite : ¬ Nat.Prime (corner 1708 2) := by change ¬ Nat.Prime 11669057; exact PrattCertificates.composite_of_product (factor := 89) (cofactor := 131113) rfl (by decide) (by decide)
    exact decide_eq_false composite
  third := false
  third_correct := by
    have composite : ¬ Nat.Prime (corner 1708 3) := by change ¬ Nat.Prime 11665641; exact PrattCertificates.composite_of_product (factor := 3) (cofactor := 3888547) rfl (by decide) (by decide)
    exact decide_eq_false composite
theorem row_count_1708 : row 1708 = 0 := (row_1708.checked).trans rfl
theorem model_count_1708 : modelRow 1708 = 0 := by decide
theorem modelrow_agrees_1708 : modelRow 1708 = row 1708 := model_count_1708.trans row_count_1708.symm
def row_1709 : RowCertificate 1709 where
  first := false
  first_correct := by
    have composite : ¬ Nat.Prime (corner 1709 1) := by change ¬ Nat.Prime 11686143; exact PrattCertificates.composite_of_product (factor := 3) (cofactor := 3895381) rfl (by decide) (by decide)
    exact decide_eq_false composite
  second := false
  second_correct := by
    have composite : ¬ Nat.Prime (corner 1709 2) := by change ¬ Nat.Prime 11682725; exact PrattCertificates.composite_of_product (factor := 5) (cofactor := 2336545) rfl (by decide) (by decide)
    exact decide_eq_false composite
  third := true
  third_correct := by
    have prime : Nat.Prime (corner 1709 3) := by change Nat.Prime 11679307; exact ProjectEulerSolutions.P58PrimeNodes.p_11679307.prime
    exact decide_eq_true prime
theorem row_count_1709 : row 1709 = 1 := (row_1709.checked).trans rfl
theorem model_count_1709 : modelRow 1709 = 1 := by decide
theorem modelrow_agrees_1709 : modelRow 1709 = row 1709 := model_count_1709.trans row_count_1709.symm
def row_1710 : RowCertificate 1710 where
  first := false
  first_correct := by
    have composite : ¬ Nat.Prime (corner 1710 1) := by change ¬ Nat.Prime 11699821; exact PrattCertificates.composite_of_product (factor := 7) (cofactor := 1671403) rfl (by decide) (by decide)
    exact decide_eq_false composite
  second := false
  second_correct := by
    have composite : ¬ Nat.Prime (corner 1710 2) := by change ¬ Nat.Prime 11696401; exact PrattCertificates.composite_of_product (factor := 181) (cofactor := 64621) rfl (by decide) (by decide)
    exact decide_eq_false composite
  third := true
  third_correct := by
    have prime : Nat.Prime (corner 1710 3) := by change Nat.Prime 11692981; exact ProjectEulerSolutions.P58PrimeNodes.p_11692981.prime
    exact decide_eq_true prime
theorem row_count_1710 : row 1710 = 1 := (row_1710.checked).trans rfl
theorem model_count_1710 : modelRow 1710 = 1 := by decide
theorem modelrow_agrees_1710 : modelRow 1710 = row 1710 := model_count_1710.trans row_count_1710.symm
def row_1711 : RowCertificate 1711 where
  first := false
  first_correct := by
    have composite : ¬ Nat.Prime (corner 1711 1) := by change ¬ Nat.Prime 11713507; exact PrattCertificates.composite_of_product (factor := 13) (cofactor := 901039) rfl (by decide) (by decide)
    exact decide_eq_false composite
  second := false
  second_correct := by
    have composite : ¬ Nat.Prime (corner 1711 2) := by change ¬ Nat.Prime 11710085; exact PrattCertificates.composite_of_product (factor := 5) (cofactor := 2342017) rfl (by decide) (by decide)
    exact decide_eq_false composite
  third := false
  third_correct := by
    have composite : ¬ Nat.Prime (corner 1711 3) := by change ¬ Nat.Prime 11706663; exact PrattCertificates.composite_of_product (factor := 3) (cofactor := 3902221) rfl (by decide) (by decide)
    exact decide_eq_false composite
theorem row_count_1711 : row 1711 = 0 := (row_1711.checked).trans rfl
theorem model_count_1711 : modelRow 1711 = 0 := by decide
theorem modelrow_agrees_1711 : modelRow 1711 = row 1711 := model_count_1711.trans row_count_1711.symm
def row_1712 : RowCertificate 1712 where
  first := false
  first_correct := by
    have composite : ¬ Nat.Prime (corner 1712 1) := by change ¬ Nat.Prime 11727201; exact PrattCertificates.composite_of_product (factor := 3) (cofactor := 3909067) rfl (by decide) (by decide)
    exact decide_eq_false composite
  second := false
  second_correct := by
    have composite : ¬ Nat.Prime (corner 1712 2) := by change ¬ Nat.Prime 11723777; exact PrattCertificates.composite_of_product (factor := 13) (cofactor := 901829) rfl (by decide) (by decide)
    exact decide_eq_false composite
  third := true
  third_correct := by
    have prime : Nat.Prime (corner 1712 3) := by change Nat.Prime 11720353; exact ProjectEulerSolutions.P58PrimeNodes.p_11720353.prime
    exact decide_eq_true prime
theorem row_count_1712 : row 1712 = 1 := (row_1712.checked).trans rfl
theorem model_count_1712 : modelRow 1712 = 1 := by decide
theorem modelrow_agrees_1712 : modelRow 1712 = row 1712 := model_count_1712.trans row_count_1712.symm
def row_1713 : RowCertificate 1713 where
  first := false
  first_correct := by
    have composite : ¬ Nat.Prime (corner 1713 1) := by change ¬ Nat.Prime 11740903; exact PrattCertificates.composite_of_product (factor := 1621) (cofactor := 7243) rfl (by decide) (by decide)
    exact decide_eq_false composite
  second := false
  second_correct := by
    have composite : ¬ Nat.Prime (corner 1713 2) := by change ¬ Nat.Prime 11737477; exact PrattCertificates.composite_of_product (factor := 157) (cofactor := 74761) rfl (by decide) (by decide)
    exact decide_eq_false composite
  third := false
  third_correct := by
    have composite : ¬ Nat.Prime (corner 1713 3) := by change ¬ Nat.Prime 11734051; exact PrattCertificates.composite_of_product (factor := 7) (cofactor := 1676293) rfl (by decide) (by decide)
    exact decide_eq_false composite
theorem row_count_1713 : row 1713 = 0 := (row_1713.checked).trans rfl
theorem model_count_1713 : modelRow 1713 = 0 := by decide
theorem modelrow_agrees_1713 : modelRow 1713 = row 1713 := model_count_1713.trans row_count_1713.symm
def row_1714 : RowCertificate 1714 where
  first := false
  first_correct := by
    have composite : ¬ Nat.Prime (corner 1714 1) := by change ¬ Nat.Prime 11754613; exact PrattCertificates.composite_of_product (factor := 13) (cofactor := 904201) rfl (by decide) (by decide)
    exact decide_eq_false composite
  second := false
  second_correct := by
    have composite : ¬ Nat.Prime (corner 1714 2) := by change ¬ Nat.Prime 11751185; exact PrattCertificates.composite_of_product (factor := 5) (cofactor := 2350237) rfl (by decide) (by decide)
    exact decide_eq_false composite
  third := false
  third_correct := by
    have composite : ¬ Nat.Prime (corner 1714 3) := by change ¬ Nat.Prime 11747757; exact PrattCertificates.composite_of_product (factor := 3) (cofactor := 3915919) rfl (by decide) (by decide)
    exact decide_eq_false composite
theorem row_count_1714 : row 1714 = 0 := (row_1714.checked).trans rfl
theorem model_count_1714 : modelRow 1714 = 0 := by decide
theorem modelrow_agrees_1714 : modelRow 1714 = row 1714 := model_count_1714.trans row_count_1714.symm
def row_1715 : RowCertificate 1715 where
  first := false
  first_correct := by
    have composite : ¬ Nat.Prime (corner 1715 1) := by change ¬ Nat.Prime 11768331; exact PrattCertificates.composite_of_product (factor := 3) (cofactor := 3922777) rfl (by decide) (by decide)
    exact decide_eq_false composite
  second := false
  second_correct := by
    have composite : ¬ Nat.Prime (corner 1715 2) := by change ¬ Nat.Prime 11764901; exact PrattCertificates.composite_of_product (factor := 17) (cofactor := 692053) rfl (by decide) (by decide)
    exact decide_eq_false composite
  third := false
  third_correct := by
    have composite : ¬ Nat.Prime (corner 1715 3) := by change ¬ Nat.Prime 11761471; exact PrattCertificates.composite_of_product (factor := 61) (cofactor := 192811) rfl (by decide) (by decide)
    exact decide_eq_false composite
theorem row_count_1715 : row 1715 = 0 := (row_1715.checked).trans rfl
theorem model_count_1715 : modelRow 1715 = 0 := by decide
theorem modelrow_agrees_1715 : modelRow 1715 = row 1715 := model_count_1715.trans row_count_1715.symm
def row_1716 : RowCertificate 1716 where
  first := false
  first_correct := by
    have composite : ¬ Nat.Prime (corner 1716 1) := by change ¬ Nat.Prime 11782057; exact PrattCertificates.composite_of_product (factor := 7) (cofactor := 1683151) rfl (by decide) (by decide)
    exact decide_eq_false composite
  second := false
  second_correct := by
    have composite : ¬ Nat.Prime (corner 1716 2) := by change ¬ Nat.Prime 11778625; exact PrattCertificates.composite_of_product (factor := 5) (cofactor := 2355725) rfl (by decide) (by decide)
    exact decide_eq_false composite
  third := false
  third_correct := by
    have composite : ¬ Nat.Prime (corner 1716 3) := by change ¬ Nat.Prime 11775193; exact PrattCertificates.composite_of_product (factor := 19) (cofactor := 619747) rfl (by decide) (by decide)
    exact decide_eq_false composite
theorem row_count_1716 : row 1716 = 0 := (row_1716.checked).trans rfl
theorem model_count_1716 : modelRow 1716 = 0 := by decide
theorem modelrow_agrees_1716 : modelRow 1716 = row 1716 := model_count_1716.trans row_count_1716.symm
def row_1717 : RowCertificate 1717 where
  first := false
  first_correct := by
    have composite : ¬ Nat.Prime (corner 1717 1) := by change ¬ Nat.Prime 11795791; exact PrattCertificates.composite_of_product (factor := 7) (cofactor := 1685113) rfl (by decide) (by decide)
    exact decide_eq_false composite
  second := false
  second_correct := by
    have composite : ¬ Nat.Prime (corner 1717 2) := by change ¬ Nat.Prime 11792357; exact PrattCertificates.composite_of_product (factor := 29) (cofactor := 406633) rfl (by decide) (by decide)
    exact decide_eq_false composite
  third := false
  third_correct := by
    have composite : ¬ Nat.Prime (corner 1717 3) := by change ¬ Nat.Prime 11788923; exact PrattCertificates.composite_of_product (factor := 3) (cofactor := 3929641) rfl (by decide) (by decide)
    exact decide_eq_false composite
theorem row_count_1717 : row 1717 = 0 := (row_1717.checked).trans rfl
theorem model_count_1717 : modelRow 1717 = 0 := by decide
theorem modelrow_agrees_1717 : modelRow 1717 = row 1717 := model_count_1717.trans row_count_1717.symm
def row_1718 : RowCertificate 1718 where
  first := false
  first_correct := by
    have composite : ¬ Nat.Prime (corner 1718 1) := by change ¬ Nat.Prime 11809533; exact PrattCertificates.composite_of_product (factor := 3) (cofactor := 3936511) rfl (by decide) (by decide)
    exact decide_eq_false composite
  second := true
  second_correct := by
    have prime : Nat.Prime (corner 1718 2) := by change Nat.Prime 11806097; exact ProjectEulerSolutions.P58PrimeNodes.p_11806097.prime
    exact decide_eq_true prime
  third := false
  third_correct := by
    have composite : ¬ Nat.Prime (corner 1718 3) := by change ¬ Nat.Prime 11802661; exact PrattCertificates.composite_of_product (factor := 13) (cofactor := 907897) rfl (by decide) (by decide)
    exact decide_eq_false composite
theorem row_count_1718 : row 1718 = 1 := (row_1718.checked).trans rfl
theorem model_count_1718 : modelRow 1718 = 1 := by decide
theorem modelrow_agrees_1718 : modelRow 1718 = row 1718 := model_count_1718.trans row_count_1718.symm
def row_1719 : RowCertificate 1719 where
  first := true
  first_correct := by
    have prime : Nat.Prime (corner 1719 1) := by change Nat.Prime 11823283; exact ProjectEulerSolutions.P58PrimeNodes.p_11823283.prime
    exact decide_eq_true prime
  second := false
  second_correct := by
    have composite : ¬ Nat.Prime (corner 1719 2) := by change ¬ Nat.Prime 11819845; exact PrattCertificates.composite_of_product (factor := 5) (cofactor := 2363969) rfl (by decide) (by decide)
    exact decide_eq_false composite
  third := true
  third_correct := by
    have prime : Nat.Prime (corner 1719 3) := by change Nat.Prime 11816407; exact ProjectEulerSolutions.P58PrimeNodes.p_11816407.prime
    exact decide_eq_true prime
theorem row_count_1719 : row 1719 = 2 := (row_1719.checked).trans rfl
theorem model_count_1719 : modelRow 1719 = 2 := by decide
theorem modelrow_agrees_1719 : modelRow 1719 = row 1719 := model_count_1719.trans row_count_1719.symm
def row_1720 : RowCertificate 1720 where
  first := false
  first_correct := by
    have composite : ¬ Nat.Prime (corner 1720 1) := by change ¬ Nat.Prime 11837041; exact PrattCertificates.composite_of_product (factor := 151) (cofactor := 78391) rfl (by decide) (by decide)
    exact decide_eq_false composite
  second := false
  second_correct := by
    have composite : ¬ Nat.Prime (corner 1720 2) := by change ¬ Nat.Prime 11833601; exact PrattCertificates.composite_of_product (factor := 13) (cofactor := 910277) rfl (by decide) (by decide)
    exact decide_eq_false composite
  third := false
  third_correct := by
    have composite : ¬ Nat.Prime (corner 1720 3) := by change ¬ Nat.Prime 11830161; exact PrattCertificates.composite_of_product (factor := 3) (cofactor := 3943387) rfl (by decide) (by decide)
    exact decide_eq_false composite
theorem row_count_1720 : row 1720 = 0 := (row_1720.checked).trans rfl
theorem model_count_1720 : modelRow 1720 = 0 := by decide
theorem modelrow_agrees_1720 : modelRow 1720 = row 1720 := model_count_1720.trans row_count_1720.symm
def row_1721 : RowCertificate 1721 where
  first := false
  first_correct := by
    have composite : ¬ Nat.Prime (corner 1721 1) := by change ¬ Nat.Prime 11850807; exact PrattCertificates.composite_of_product (factor := 3) (cofactor := 3950269) rfl (by decide) (by decide)
    exact decide_eq_false composite
  second := false
  second_correct := by
    have composite : ¬ Nat.Prime (corner 1721 2) := by change ¬ Nat.Prime 11847365; exact PrattCertificates.composite_of_product (factor := 5) (cofactor := 2369473) rfl (by decide) (by decide)
    exact decide_eq_false composite
  third := false
  third_correct := by
    have composite : ¬ Nat.Prime (corner 1721 3) := by change ¬ Nat.Prime 11843923; exact PrattCertificates.composite_of_product (factor := 7) (cofactor := 1691989) rfl (by decide) (by decide)
    exact decide_eq_false composite
theorem row_count_1721 : row 1721 = 0 := (row_1721.checked).trans rfl
theorem model_count_1721 : modelRow 1721 = 0 := by decide
theorem modelrow_agrees_1721 : modelRow 1721 = row 1721 := model_count_1721.trans row_count_1721.symm
def row_1722 : RowCertificate 1722 where
  first := true
  first_correct := by
    have prime : Nat.Prime (corner 1722 1) := by change Nat.Prime 11864581; exact ProjectEulerSolutions.P58PrimeNodes.p_11864581.prime
    exact decide_eq_true prime
  second := false
  second_correct := by
    have composite : ¬ Nat.Prime (corner 1722 2) := by change ¬ Nat.Prime 11861137; exact PrattCertificates.composite_of_product (factor := 101) (cofactor := 117437) rfl (by decide) (by decide)
    exact decide_eq_false composite
  third := true
  third_correct := by
    have prime : Nat.Prime (corner 1722 3) := by change Nat.Prime 11857693; exact ProjectEulerSolutions.P58PrimeNodes.p_11857693.prime
    exact decide_eq_true prime
theorem row_count_1722 : row 1722 = 2 := (row_1722.checked).trans rfl
theorem model_count_1722 : modelRow 1722 = 2 := by decide
theorem modelrow_agrees_1722 : modelRow 1722 = row 1722 := model_count_1722.trans row_count_1722.symm
def row_1723 : RowCertificate 1723 where
  first := false
  first_correct := by
    have composite : ¬ Nat.Prime (corner 1723 1) := by change ¬ Nat.Prime 11878363; exact PrattCertificates.composite_of_product (factor := 7) (cofactor := 1696909) rfl (by decide) (by decide)
    exact decide_eq_false composite
  second := true
  second_correct := by
    have prime : Nat.Prime (corner 1723 2) := by change Nat.Prime 11874917; exact ProjectEulerSolutions.P58PrimeNodes.p_11874917.prime
    exact decide_eq_true prime
  third := false
  third_correct := by
    have composite : ¬ Nat.Prime (corner 1723 3) := by change ¬ Nat.Prime 11871471; exact PrattCertificates.composite_of_product (factor := 3) (cofactor := 3957157) rfl (by decide) (by decide)
    exact decide_eq_false composite
theorem row_count_1723 : row 1723 = 1 := (row_1723.checked).trans rfl
theorem model_count_1723 : modelRow 1723 = 1 := by decide
theorem modelrow_agrees_1723 : modelRow 1723 = row 1723 := model_count_1723.trans row_count_1723.symm
def row_1724 : RowCertificate 1724 where
  first := false
  first_correct := by
    have composite : ¬ Nat.Prime (corner 1724 1) := by change ¬ Nat.Prime 11892153; exact PrattCertificates.composite_of_product (factor := 3) (cofactor := 3964051) rfl (by decide) (by decide)
    exact decide_eq_false composite
  second := false
  second_correct := by
    have composite : ¬ Nat.Prime (corner 1724 2) := by change ¬ Nat.Prime 11888705; exact PrattCertificates.composite_of_product (factor := 5) (cofactor := 2377741) rfl (by decide) (by decide)
    exact decide_eq_false composite
  third := false
  third_correct := by
    have composite : ¬ Nat.Prime (corner 1724 3) := by change ¬ Nat.Prime 11885257; exact PrattCertificates.composite_of_product (factor := 2341) (cofactor := 5077) rfl (by decide) (by decide)
    exact decide_eq_false composite
theorem row_count_1724 : row 1724 = 0 := (row_1724.checked).trans rfl
theorem model_count_1724 : modelRow 1724 = 0 := by decide
theorem modelrow_agrees_1724 : modelRow 1724 = row 1724 := model_count_1724.trans row_count_1724.symm
def row_1725 : RowCertificate 1725 where
  first := false
  first_correct := by
    have composite : ¬ Nat.Prime (corner 1725 1) := by change ¬ Nat.Prime 11905951; exact PrattCertificates.composite_of_product (factor := 19) (cofactor := 626629) rfl (by decide) (by decide)
    exact decide_eq_false composite
  second := false
  second_correct := by
    have composite : ¬ Nat.Prime (corner 1725 2) := by change ¬ Nat.Prime 11902501; exact PrattCertificates.composite_of_product (factor := 13) (cofactor := 915577) rfl (by decide) (by decide)
    exact decide_eq_false composite
  third := true
  third_correct := by
    have prime : Nat.Prime (corner 1725 3) := by change Nat.Prime 11899051; exact ProjectEulerSolutions.P58PrimeNodes.p_11899051.prime
    exact decide_eq_true prime
theorem row_count_1725 : row 1725 = 1 := (row_1725.checked).trans rfl
theorem model_count_1725 : modelRow 1725 = 1 := by decide
theorem modelrow_agrees_1725 : modelRow 1725 = row 1725 := model_count_1725.trans row_count_1725.symm
def row_1726 : RowCertificate 1726 where
  first := false
  first_correct := by
    have composite : ¬ Nat.Prime (corner 1726 1) := by change ¬ Nat.Prime 11919757; exact PrattCertificates.composite_of_product (factor := 79) (cofactor := 150883) rfl (by decide) (by decide)
    exact decide_eq_false composite
  second := false
  second_correct := by
    have composite : ¬ Nat.Prime (corner 1726 2) := by change ¬ Nat.Prime 11916305; exact PrattCertificates.composite_of_product (factor := 5) (cofactor := 2383261) rfl (by decide) (by decide)
    exact decide_eq_false composite
  third := false
  third_correct := by
    have composite : ¬ Nat.Prime (corner 1726 3) := by change ¬ Nat.Prime 11912853; exact PrattCertificates.composite_of_product (factor := 3) (cofactor := 3970951) rfl (by decide) (by decide)
    exact decide_eq_false composite
theorem row_count_1726 : row 1726 = 0 := (row_1726.checked).trans rfl
theorem model_count_1726 : modelRow 1726 = 0 := by decide
theorem modelrow_agrees_1726 : modelRow 1726 = row 1726 := model_count_1726.trans row_count_1726.symm
def row_1727 : RowCertificate 1727 where
  first := false
  first_correct := by
    have composite : ¬ Nat.Prime (corner 1727 1) := by change ¬ Nat.Prime 11933571; exact PrattCertificates.composite_of_product (factor := 3) (cofactor := 3977857) rfl (by decide) (by decide)
    exact decide_eq_false composite
  second := false
  second_correct := by
    have composite : ¬ Nat.Prime (corner 1727 2) := by change ¬ Nat.Prime 11930117; exact PrattCertificates.composite_of_product (factor := 1033) (cofactor := 11549) rfl (by decide) (by decide)
    exact decide_eq_false composite
  third := false
  third_correct := by
    have composite : ¬ Nat.Prime (corner 1727 3) := by change ¬ Nat.Prime 11926663; exact PrattCertificates.composite_of_product (factor := 7) (cofactor := 1703809) rfl (by decide) (by decide)
    exact decide_eq_false composite
theorem row_count_1727 : row 1727 = 0 := (row_1727.checked).trans rfl
theorem model_count_1727 : modelRow 1727 = 0 := by decide
theorem modelrow_agrees_1727 : modelRow 1727 = row 1727 := model_count_1727.trans row_count_1727.symm
def row_1728 : RowCertificate 1728 where
  first := false
  first_correct := by
    have composite : ¬ Nat.Prime (corner 1728 1) := by change ¬ Nat.Prime 11947393; exact PrattCertificates.composite_of_product (factor := 97) (cofactor := 123169) rfl (by decide) (by decide)
    exact decide_eq_false composite
  second := false
  second_correct := by
    have composite : ¬ Nat.Prime (corner 1728 2) := by change ¬ Nat.Prime 11943937; exact PrattCertificates.composite_of_product (factor := 1069) (cofactor := 11173) rfl (by decide) (by decide)
    exact decide_eq_false composite
  third := false
  third_correct := by
    have composite : ¬ Nat.Prime (corner 1728 3) := by change ¬ Nat.Prime 11940481; exact PrattCertificates.composite_of_product (factor := 7) (cofactor := 1705783) rfl (by decide) (by decide)
    exact decide_eq_false composite
theorem row_count_1728 : row 1728 = 0 := (row_1728.checked).trans rfl
theorem model_count_1728 : modelRow 1728 = 0 := by decide
theorem modelrow_agrees_1728 : modelRow 1728 = row 1728 := model_count_1728.trans row_count_1728.symm
def row_1729 : RowCertificate 1729 where
  first := true
  first_correct := by
    have prime : Nat.Prime (corner 1729 1) := by change Nat.Prime 11961223; exact ProjectEulerSolutions.P58PrimeNodes.p_11961223.prime
    exact decide_eq_true prime
  second := false
  second_correct := by
    have composite : ¬ Nat.Prime (corner 1729 2) := by change ¬ Nat.Prime 11957765; exact PrattCertificates.composite_of_product (factor := 5) (cofactor := 2391553) rfl (by decide) (by decide)
    exact decide_eq_false composite
  third := false
  third_correct := by
    have composite : ¬ Nat.Prime (corner 1729 3) := by change ¬ Nat.Prime 11954307; exact PrattCertificates.composite_of_product (factor := 3) (cofactor := 3984769) rfl (by decide) (by decide)
    exact decide_eq_false composite
theorem row_count_1729 : row 1729 = 1 := (row_1729.checked).trans rfl
theorem model_count_1729 : modelRow 1729 = 1 := by decide
theorem modelrow_agrees_1729 : modelRow 1729 = row 1729 := model_count_1729.trans row_count_1729.symm
def row_1730 : RowCertificate 1730 where
  first := false
  first_correct := by
    have composite : ¬ Nat.Prime (corner 1730 1) := by change ¬ Nat.Prime 11975061; exact PrattCertificates.composite_of_product (factor := 3) (cofactor := 3991687) rfl (by decide) (by decide)
    exact decide_eq_false composite
  second := false
  second_correct := by
    have composite : ¬ Nat.Prime (corner 1730 2) := by change ¬ Nat.Prime 11971601; exact PrattCertificates.composite_of_product (factor := 557) (cofactor := 21493) rfl (by decide) (by decide)
    exact decide_eq_false composite
  third := true
  third_correct := by
    have prime : Nat.Prime (corner 1730 3) := by change Nat.Prime 11968141; exact ProjectEulerSolutions.P58PrimeNodes.p_11968141.prime
    exact decide_eq_true prime
theorem row_count_1730 : row 1730 = 1 := (row_1730.checked).trans rfl
theorem model_count_1730 : modelRow 1730 = 1 := by decide
theorem modelrow_agrees_1730 : modelRow 1730 = row 1730 := model_count_1730.trans row_count_1730.symm
def row_1731 : RowCertificate 1731 where
  first := false
  first_correct := by
    have composite : ¬ Nat.Prime (corner 1731 1) := by change ¬ Nat.Prime 11988907; exact PrattCertificates.composite_of_product (factor := 7) (cofactor := 1712701) rfl (by decide) (by decide)
    exact decide_eq_false composite
  second := false
  second_correct := by
    have composite : ¬ Nat.Prime (corner 1731 2) := by change ¬ Nat.Prime 11985445; exact PrattCertificates.composite_of_product (factor := 5) (cofactor := 2397089) rfl (by decide) (by decide)
    exact decide_eq_false composite
  third := false
  third_correct := by
    have composite : ¬ Nat.Prime (corner 1731 3) := by change ¬ Nat.Prime 11981983; exact PrattCertificates.composite_of_product (factor := 13) (cofactor := 921691) rfl (by decide) (by decide)
    exact decide_eq_false composite
theorem row_count_1731 : row 1731 = 0 := (row_1731.checked).trans rfl
theorem model_count_1731 : modelRow 1731 = 0 := by decide
theorem modelrow_agrees_1731 : modelRow 1731 = row 1731 := model_count_1731.trans row_count_1731.symm
def row_1732 : RowCertificate 1732 where
  first := true
  first_correct := by
    have prime : Nat.Prime (corner 1732 1) := by change Nat.Prime 12002761; exact ProjectEulerSolutions.P58PrimeNodes.p_12002761.prime
    exact decide_eq_true prime
  second := false
  second_correct := by
    have composite : ¬ Nat.Prime (corner 1732 2) := by change ¬ Nat.Prime 11999297; exact PrattCertificates.composite_of_product (factor := 17) (cofactor := 705841) rfl (by decide) (by decide)
    exact decide_eq_false composite
  third := false
  third_correct := by
    have composite : ¬ Nat.Prime (corner 1732 3) := by change ¬ Nat.Prime 11995833; exact PrattCertificates.composite_of_product (factor := 3) (cofactor := 3998611) rfl (by decide) (by decide)
    exact decide_eq_false composite
theorem row_count_1732 : row 1732 = 1 := (row_1732.checked).trans rfl
theorem model_count_1732 : modelRow 1732 = 1 := by decide
theorem modelrow_agrees_1732 : modelRow 1732 = row 1732 := model_count_1732.trans row_count_1732.symm
def row_1733 : RowCertificate 1733 where
  first := false
  first_correct := by
    have composite : ¬ Nat.Prime (corner 1733 1) := by change ¬ Nat.Prime 12016623; exact PrattCertificates.composite_of_product (factor := 3) (cofactor := 4005541) rfl (by decide) (by decide)
    exact decide_eq_false composite
  second := false
  second_correct := by
    have composite : ¬ Nat.Prime (corner 1733 2) := by change ¬ Nat.Prime 12013157; exact PrattCertificates.composite_of_product (factor := 13) (cofactor := 924089) rfl (by decide) (by decide)
    exact decide_eq_false composite
  third := false
  third_correct := by
    have composite : ¬ Nat.Prime (corner 1733 3) := by change ¬ Nat.Prime 12009691; exact PrattCertificates.composite_of_product (factor := 19) (cofactor := 632089) rfl (by decide) (by decide)
    exact decide_eq_false composite
theorem row_count_1733 : row 1733 = 0 := (row_1733.checked).trans rfl
theorem model_count_1733 : modelRow 1733 = 0 := by decide
theorem modelrow_agrees_1733 : modelRow 1733 = row 1733 := model_count_1733.trans row_count_1733.symm
def row_1734 : RowCertificate 1734 where
  first := false
  first_correct := by
    have composite : ¬ Nat.Prime (corner 1734 1) := by change ¬ Nat.Prime 12030493; exact PrattCertificates.composite_of_product (factor := 1609) (cofactor := 7477) rfl (by decide) (by decide)
    exact decide_eq_false composite
  second := false
  second_correct := by
    have composite : ¬ Nat.Prime (corner 1734 2) := by change ¬ Nat.Prime 12027025; exact PrattCertificates.composite_of_product (factor := 5) (cofactor := 2405405) rfl (by decide) (by decide)
    exact decide_eq_false composite
  third := false
  third_correct := by
    have composite : ¬ Nat.Prime (corner 1734 3) := by change ¬ Nat.Prime 12023557; exact PrattCertificates.composite_of_product (factor := 7) (cofactor := 1717651) rfl (by decide) (by decide)
    exact decide_eq_false composite
theorem row_count_1734 : row 1734 = 0 := (row_1734.checked).trans rfl
theorem model_count_1734 : modelRow 1734 = 0 := by decide
theorem modelrow_agrees_1734 : modelRow 1734 = row 1734 := model_count_1734.trans row_count_1734.symm
def row_1735 : RowCertificate 1735 where
  first := true
  first_correct := by
    have prime : Nat.Prime (corner 1735 1) := by change Nat.Prime 12044371; exact ProjectEulerSolutions.P58PrimeNodes.p_12044371.prime
    exact decide_eq_true prime
  second := false
  second_correct := by
    have composite : ¬ Nat.Prime (corner 1735 2) := by change ¬ Nat.Prime 12040901; exact PrattCertificates.composite_of_product (factor := 97) (cofactor := 124133) rfl (by decide) (by decide)
    exact decide_eq_false composite
  third := false
  third_correct := by
    have composite : ¬ Nat.Prime (corner 1735 3) := by change ¬ Nat.Prime 12037431; exact PrattCertificates.composite_of_product (factor := 3) (cofactor := 4012477) rfl (by decide) (by decide)
    exact decide_eq_false composite
theorem row_count_1735 : row 1735 = 1 := (row_1735.checked).trans rfl
theorem model_count_1735 : modelRow 1735 = 1 := by decide
theorem modelrow_agrees_1735 : modelRow 1735 = row 1735 := model_count_1735.trans row_count_1735.symm
def row_1736 : RowCertificate 1736 where
  first := false
  first_correct := by
    have composite : ¬ Nat.Prime (corner 1736 1) := by change ¬ Nat.Prime 12058257; exact PrattCertificates.composite_of_product (factor := 3) (cofactor := 4019419) rfl (by decide) (by decide)
    exact decide_eq_false composite
  second := false
  second_correct := by
    have composite : ¬ Nat.Prime (corner 1736 2) := by change ¬ Nat.Prime 12054785; exact PrattCertificates.composite_of_product (factor := 5) (cofactor := 2410957) rfl (by decide) (by decide)
    exact decide_eq_false composite
  third := false
  third_correct := by
    have composite : ¬ Nat.Prime (corner 1736 3) := by change ¬ Nat.Prime 12051313; exact PrattCertificates.composite_of_product (factor := 1117) (cofactor := 10789) rfl (by decide) (by decide)
    exact decide_eq_false composite
theorem row_count_1736 : row 1736 = 0 := (row_1736.checked).trans rfl
theorem model_count_1736 : modelRow 1736 = 0 := by decide
theorem modelrow_agrees_1736 : modelRow 1736 = row 1736 := model_count_1736.trans row_count_1736.symm
def row_1737 : RowCertificate 1737 where
  first := false
  first_correct := by
    have composite : ¬ Nat.Prime (corner 1737 1) := by change ¬ Nat.Prime 12072151; exact PrattCertificates.composite_of_product (factor := 7) (cofactor := 1724593) rfl (by decide) (by decide)
    exact decide_eq_false composite
  second := true
  second_correct := by
    have prime : Nat.Prime (corner 1737 2) := by change Nat.Prime 12068677; exact ProjectEulerSolutions.P58PrimeNodes.p_12068677.prime
    exact decide_eq_true prime
  third := true
  third_correct := by
    have prime : Nat.Prime (corner 1737 3) := by change Nat.Prime 12065203; exact ProjectEulerSolutions.P58PrimeNodes.p_12065203.prime
    exact decide_eq_true prime
theorem row_count_1737 : row 1737 = 2 := (row_1737.checked).trans rfl
theorem model_count_1737 : modelRow 1737 = 2 := by decide
theorem modelrow_agrees_1737 : modelRow 1737 = row 1737 := model_count_1737.trans row_count_1737.symm
def row_1738 : RowCertificate 1738 where
  first := false
  first_correct := by
    have composite : ¬ Nat.Prime (corner 1738 1) := by change ¬ Nat.Prime 12086053; exact PrattCertificates.composite_of_product (factor := 7) (cofactor := 1726579) rfl (by decide) (by decide)
    exact decide_eq_false composite
  second := false
  second_correct := by
    have composite : ¬ Nat.Prime (corner 1738 2) := by change ¬ Nat.Prime 12082577; exact PrattCertificates.composite_of_product (factor := 13) (cofactor := 929429) rfl (by decide) (by decide)
    exact decide_eq_false composite
  third := false
  third_correct := by
    have composite : ¬ Nat.Prime (corner 1738 3) := by change ¬ Nat.Prime 12079101; exact PrattCertificates.composite_of_product (factor := 3) (cofactor := 4026367) rfl (by decide) (by decide)
    exact decide_eq_false composite
theorem row_count_1738 : row 1738 = 0 := (row_1738.checked).trans rfl
theorem model_count_1738 : modelRow 1738 = 0 := by decide
theorem modelrow_agrees_1738 : modelRow 1738 = row 1738 := model_count_1738.trans row_count_1738.symm
def row_1739 : RowCertificate 1739 where
  first := false
  first_correct := by
    have composite : ¬ Nat.Prime (corner 1739 1) := by change ¬ Nat.Prime 12099963; exact PrattCertificates.composite_of_product (factor := 3) (cofactor := 4033321) rfl (by decide) (by decide)
    exact decide_eq_false composite
  second := false
  second_correct := by
    have composite : ¬ Nat.Prime (corner 1739 2) := by change ¬ Nat.Prime 12096485; exact PrattCertificates.composite_of_product (factor := 5) (cofactor := 2419297) rfl (by decide) (by decide)
    exact decide_eq_false composite
  third := false
  third_correct := by
    have composite : ¬ Nat.Prime (corner 1739 3) := by change ¬ Nat.Prime 12093007; exact PrattCertificates.composite_of_product (factor := 31) (cofactor := 390097) rfl (by decide) (by decide)
    exact decide_eq_false composite
theorem row_count_1739 : row 1739 = 0 := (row_1739.checked).trans rfl
theorem model_count_1739 : modelRow 1739 = 0 := by decide
theorem modelrow_agrees_1739 : modelRow 1739 = row 1739 := model_count_1739.trans row_count_1739.symm
def row_1740 : RowCertificate 1740 where
  first := false
  first_correct := by
    have composite : ¬ Nat.Prime (corner 1740 1) := by change ¬ Nat.Prime 12113881; exact PrattCertificates.composite_of_product (factor := 13) (cofactor := 931837) rfl (by decide) (by decide)
    exact decide_eq_false composite
  second := true
  second_correct := by
    have prime : Nat.Prime (corner 1740 2) := by change Nat.Prime 12110401; exact ProjectEulerSolutions.P58PrimeNodes.p_12110401.prime
    exact decide_eq_true prime
  third := true
  third_correct := by
    have prime : Nat.Prime (corner 1740 3) := by change Nat.Prime 12106921; exact ProjectEulerSolutions.P58PrimeNodes.p_12106921.prime
    exact decide_eq_true prime
theorem row_count_1740 : row 1740 = 2 := (row_1740.checked).trans rfl
theorem model_count_1740 : modelRow 1740 = 2 := by decide
theorem modelrow_agrees_1740 : modelRow 1740 = row 1740 := model_count_1740.trans row_count_1740.symm
def row_1741 : RowCertificate 1741 where
  first := true
  first_correct := by
    have prime : Nat.Prime (corner 1741 1) := by change Nat.Prime 12127807; exact ProjectEulerSolutions.P58PrimeNodes.p_12127807.prime
    exact decide_eq_true prime
  second := false
  second_correct := by
    have composite : ¬ Nat.Prime (corner 1741 2) := by change ¬ Nat.Prime 12124325; exact PrattCertificates.composite_of_product (factor := 5) (cofactor := 2424865) rfl (by decide) (by decide)
    exact decide_eq_false composite
  third := false
  third_correct := by
    have composite : ¬ Nat.Prime (corner 1741 3) := by change ¬ Nat.Prime 12120843; exact PrattCertificates.composite_of_product (factor := 3) (cofactor := 4040281) rfl (by decide) (by decide)
    exact decide_eq_false composite
theorem row_count_1741 : row 1741 = 1 := (row_1741.checked).trans rfl
theorem model_count_1741 : modelRow 1741 = 1 := by decide
theorem modelrow_agrees_1741 : modelRow 1741 = row 1741 := model_count_1741.trans row_count_1741.symm
def row_1742 : RowCertificate 1742 where
  first := false
  first_correct := by
    have composite : ¬ Nat.Prime (corner 1742 1) := by change ¬ Nat.Prime 12141741; exact PrattCertificates.composite_of_product (factor := 3) (cofactor := 4047247) rfl (by decide) (by decide)
    exact decide_eq_false composite
  second := false
  second_correct := by
    have composite : ¬ Nat.Prime (corner 1742 2) := by change ¬ Nat.Prime 12138257; exact PrattCertificates.composite_of_product (factor := 37) (cofactor := 328061) rfl (by decide) (by decide)
    exact decide_eq_false composite
  third := false
  third_correct := by
    have composite : ¬ Nat.Prime (corner 1742 3) := by change ¬ Nat.Prime 12134773; exact PrattCertificates.composite_of_product (factor := 7) (cofactor := 1733539) rfl (by decide) (by decide)
    exact decide_eq_false composite
theorem row_count_1742 : row 1742 = 0 := (row_1742.checked).trans rfl
theorem model_count_1742 : modelRow 1742 = 0 := by decide
theorem modelrow_agrees_1742 : modelRow 1742 = row 1742 := model_count_1742.trans row_count_1742.symm
def row_1743 : RowCertificate 1743 where
  first := false
  first_correct := by
    have composite : ¬ Nat.Prime (corner 1743 1) := by change ¬ Nat.Prime 12155683; exact PrattCertificates.composite_of_product (factor := 2383) (cofactor := 5101) rfl (by decide) (by decide)
    exact decide_eq_false composite
  second := false
  second_correct := by
    have composite : ¬ Nat.Prime (corner 1743 2) := by change ¬ Nat.Prime 12152197; exact PrattCertificates.composite_of_product (factor := 577) (cofactor := 21061) rfl (by decide) (by decide)
    exact decide_eq_false composite
  third := true
  third_correct := by
    have prime : Nat.Prime (corner 1743 3) := by change Nat.Prime 12148711; exact ProjectEulerSolutions.P58PrimeNodes.p_12148711.prime
    exact decide_eq_true prime
theorem row_count_1743 : row 1743 = 1 := (row_1743.checked).trans rfl
theorem model_count_1743 : modelRow 1743 = 1 := by decide
theorem modelrow_agrees_1743 : modelRow 1743 = row 1743 := model_count_1743.trans row_count_1743.symm
def row_1744 : RowCertificate 1744 where
  first := false
  first_correct := by
    have composite : ¬ Nat.Prime (corner 1744 1) := by change ¬ Nat.Prime 12169633; exact PrattCertificates.composite_of_product (factor := 7) (cofactor := 1738519) rfl (by decide) (by decide)
    exact decide_eq_false composite
  second := false
  second_correct := by
    have composite : ¬ Nat.Prime (corner 1744 2) := by change ¬ Nat.Prime 12166145; exact PrattCertificates.composite_of_product (factor := 5) (cofactor := 2433229) rfl (by decide) (by decide)
    exact decide_eq_false composite
  third := false
  third_correct := by
    have composite : ¬ Nat.Prime (corner 1744 3) := by change ¬ Nat.Prime 12162657; exact PrattCertificates.composite_of_product (factor := 3) (cofactor := 4054219) rfl (by decide) (by decide)
    exact decide_eq_false composite
theorem row_count_1744 : row 1744 = 0 := (row_1744.checked).trans rfl
theorem model_count_1744 : modelRow 1744 = 0 := by decide
theorem modelrow_agrees_1744 : modelRow 1744 = row 1744 := model_count_1744.trans row_count_1744.symm
def row_1745 : RowCertificate 1745 where
  first := false
  first_correct := by
    have composite : ¬ Nat.Prime (corner 1745 1) := by change ¬ Nat.Prime 12183591; exact PrattCertificates.composite_of_product (factor := 3) (cofactor := 4061197) rfl (by decide) (by decide)
    exact decide_eq_false composite
  second := true
  second_correct := by
    have prime : Nat.Prime (corner 1745 2) := by change Nat.Prime 12180101; exact ProjectEulerSolutions.P58PrimeNodes.p_12180101.prime
    exact decide_eq_true prime
  third := false
  third_correct := by
    have composite : ¬ Nat.Prime (corner 1745 3) := by change ¬ Nat.Prime 12176611; exact PrattCertificates.composite_of_product (factor := 43) (cofactor := 283177) rfl (by decide) (by decide)
    exact decide_eq_false composite
theorem row_count_1745 : row 1745 = 1 := (row_1745.checked).trans rfl
theorem model_count_1745 : modelRow 1745 = 1 := by decide
theorem modelrow_agrees_1745 : modelRow 1745 = row 1745 := model_count_1745.trans row_count_1745.symm
def row_1746 : RowCertificate 1746 where
  first := true
  first_correct := by
    have prime : Nat.Prime (corner 1746 1) := by change Nat.Prime 12197557; exact ProjectEulerSolutions.P58PrimeNodes.p_12197557.prime
    exact decide_eq_true prime
  second := false
  second_correct := by
    have composite : ¬ Nat.Prime (corner 1746 2) := by change ¬ Nat.Prime 12194065; exact PrattCertificates.composite_of_product (factor := 5) (cofactor := 2438813) rfl (by decide) (by decide)
    exact decide_eq_false composite
  third := true
  third_correct := by
    have prime : Nat.Prime (corner 1746 3) := by change Nat.Prime 12190573; exact ProjectEulerSolutions.P58PrimeNodes.p_12190573.prime
    exact decide_eq_true prime
theorem row_count_1746 : row 1746 = 2 := (row_1746.checked).trans rfl
theorem model_count_1746 : modelRow 1746 = 2 := by decide
theorem modelrow_agrees_1746 : modelRow 1746 = row 1746 := model_count_1746.trans row_count_1746.symm
def row_1747 : RowCertificate 1747 where
  first := false
  first_correct := by
    have composite : ¬ Nat.Prime (corner 1747 1) := by change ¬ Nat.Prime 12211531; exact PrattCertificates.composite_of_product (factor := 271) (cofactor := 45061) rfl (by decide) (by decide)
    exact decide_eq_false composite
  second := false
  second_correct := by
    have composite : ¬ Nat.Prime (corner 1747 2) := by change ¬ Nat.Prime 12208037; exact PrattCertificates.composite_of_product (factor := 41) (cofactor := 297757) rfl (by decide) (by decide)
    exact decide_eq_false composite
  third := false
  third_correct := by
    have composite : ¬ Nat.Prime (corner 1747 3) := by change ¬ Nat.Prime 12204543; exact PrattCertificates.composite_of_product (factor := 3) (cofactor := 4068181) rfl (by decide) (by decide)
    exact decide_eq_false composite
theorem row_count_1747 : row 1747 = 0 := (row_1747.checked).trans rfl
theorem model_count_1747 : modelRow 1747 = 0 := by decide
theorem modelrow_agrees_1747 : modelRow 1747 = row 1747 := model_count_1747.trans row_count_1747.symm
def row_1748 : RowCertificate 1748 where
  first := false
  first_correct := by
    have composite : ¬ Nat.Prime (corner 1748 1) := by change ¬ Nat.Prime 12225513; exact PrattCertificates.composite_of_product (factor := 3) (cofactor := 4075171) rfl (by decide) (by decide)
    exact decide_eq_false composite
  second := false
  second_correct := by
    have composite : ¬ Nat.Prime (corner 1748 2) := by change ¬ Nat.Prime 12222017; exact PrattCertificates.composite_of_product (factor := 1009) (cofactor := 12113) rfl (by decide) (by decide)
    exact decide_eq_false composite
  third := false
  third_correct := by
    have composite : ¬ Nat.Prime (corner 1748 3) := by change ¬ Nat.Prime 12218521; exact PrattCertificates.composite_of_product (factor := 7) (cofactor := 1745503) rfl (by decide) (by decide)
    exact decide_eq_false composite
theorem row_count_1748 : row 1748 = 0 := (row_1748.checked).trans rfl
theorem model_count_1748 : modelRow 1748 = 0 := by decide
theorem modelrow_agrees_1748 : modelRow 1748 = row 1748 := model_count_1748.trans row_count_1748.symm
def row_1749 : RowCertificate 1749 where
  first := false
  first_correct := by
    have composite : ¬ Nat.Prime (corner 1749 1) := by change ¬ Nat.Prime 12239503; exact PrattCertificates.composite_of_product (factor := 337) (cofactor := 36319) rfl (by decide) (by decide)
    exact decide_eq_false composite
  second := false
  second_correct := by
    have composite : ¬ Nat.Prime (corner 1749 2) := by change ¬ Nat.Prime 12236005; exact PrattCertificates.composite_of_product (factor := 5) (cofactor := 2447201) rfl (by decide) (by decide)
    exact decide_eq_false composite
  third := false
  third_correct := by
    have composite : ¬ Nat.Prime (corner 1749 3) := by change ¬ Nat.Prime 12232507; exact PrattCertificates.composite_of_product (factor := 7) (cofactor := 1747501) rfl (by decide) (by decide)
    exact decide_eq_false composite
theorem row_count_1749 : row 1749 = 0 := (row_1749.checked).trans rfl
theorem model_count_1749 : modelRow 1749 = 0 := by decide
theorem modelrow_agrees_1749 : modelRow 1749 = row 1749 := model_count_1749.trans row_count_1749.symm
def row_1750 : RowCertificate 1750 where
  first := false
  first_correct := by
    have composite : ¬ Nat.Prime (corner 1750 1) := by change ¬ Nat.Prime 12253501; exact PrattCertificates.composite_of_product (factor := 13) (cofactor := 942577) rfl (by decide) (by decide)
    exact decide_eq_false composite
  second := false
  second_correct := by
    have composite : ¬ Nat.Prime (corner 1750 2) := by change ¬ Nat.Prime 12250001; exact PrattCertificates.composite_of_product (factor := 569) (cofactor := 21529) rfl (by decide) (by decide)
    exact decide_eq_false composite
  third := false
  third_correct := by
    have composite : ¬ Nat.Prime (corner 1750 3) := by change ¬ Nat.Prime 12246501; exact PrattCertificates.composite_of_product (factor := 3) (cofactor := 4082167) rfl (by decide) (by decide)
    exact decide_eq_false composite
theorem row_count_1750 : row 1750 = 0 := (row_1750.checked).trans rfl
theorem model_count_1750 : modelRow 1750 = 0 := by decide
theorem modelrow_agrees_1750 : modelRow 1750 = row 1750 := model_count_1750.trans row_count_1750.symm
def row_1751 : RowCertificate 1751 where
  first := false
  first_correct := by
    have composite : ¬ Nat.Prime (corner 1751 1) := by change ¬ Nat.Prime 12267507; exact PrattCertificates.composite_of_product (factor := 3) (cofactor := 4089169) rfl (by decide) (by decide)
    exact decide_eq_false composite
  second := false
  second_correct := by
    have composite : ¬ Nat.Prime (corner 1751 2) := by change ¬ Nat.Prime 12264005; exact PrattCertificates.composite_of_product (factor := 5) (cofactor := 2452801) rfl (by decide) (by decide)
    exact decide_eq_false composite
  third := true
  third_correct := by
    have prime : Nat.Prime (corner 1751 3) := by change Nat.Prime 12260503; exact ProjectEulerSolutions.P58PrimeNodes.p_12260503.prime
    exact decide_eq_true prime
theorem row_count_1751 : row 1751 = 1 := (row_1751.checked).trans rfl
theorem model_count_1751 : modelRow 1751 = 1 := by decide
theorem modelrow_agrees_1751 : modelRow 1751 = row 1751 := model_count_1751.trans row_count_1751.symm
def row_1752 : RowCertificate 1752 where
  first := false
  first_correct := by
    have composite : ¬ Nat.Prime (corner 1752 1) := by change ¬ Nat.Prime 12281521; exact PrattCertificates.composite_of_product (factor := 7) (cofactor := 1754503) rfl (by decide) (by decide)
    exact decide_eq_false composite
  second := true
  second_correct := by
    have prime : Nat.Prime (corner 1752 2) := by change Nat.Prime 12278017; exact ProjectEulerSolutions.P58PrimeNodes.p_12278017.prime
    exact decide_eq_true prime
  third := false
  third_correct := by
    have composite : ¬ Nat.Prime (corner 1752 3) := by change ¬ Nat.Prime 12274513; exact PrattCertificates.composite_of_product (factor := 19) (cofactor := 646027) rfl (by decide) (by decide)
    exact decide_eq_false composite
theorem row_count_1752 : row 1752 = 1 := (row_1752.checked).trans rfl
theorem model_count_1752 : modelRow 1752 = 1 := by decide
theorem modelrow_agrees_1752 : modelRow 1752 = row 1752 := model_count_1752.trans row_count_1752.symm
def row_1753 : RowCertificate 1753 where
  first := false
  first_correct := by
    have composite : ¬ Nat.Prime (corner 1753 1) := by change ¬ Nat.Prime 12295543; exact PrattCertificates.composite_of_product (factor := 13) (cofactor := 945811) rfl (by decide) (by decide)
    exact decide_eq_false composite
  second := false
  second_correct := by
    have composite : ¬ Nat.Prime (corner 1753 2) := by change ¬ Nat.Prime 12292037; exact PrattCertificates.composite_of_product (factor := 17) (cofactor := 723061) rfl (by decide) (by decide)
    exact decide_eq_false composite
  third := false
  third_correct := by
    have composite : ¬ Nat.Prime (corner 1753 3) := by change ¬ Nat.Prime 12288531; exact PrattCertificates.composite_of_product (factor := 3) (cofactor := 4096177) rfl (by decide) (by decide)
    exact decide_eq_false composite
theorem row_count_1753 : row 1753 = 0 := (row_1753.checked).trans rfl
theorem model_count_1753 : modelRow 1753 = 0 := by decide
theorem modelrow_agrees_1753 : modelRow 1753 = row 1753 := model_count_1753.trans row_count_1753.symm
def row_1754 : RowCertificate 1754 where
  first := false
  first_correct := by
    have composite : ¬ Nat.Prime (corner 1754 1) := by change ¬ Nat.Prime 12309573; exact PrattCertificates.composite_of_product (factor := 3) (cofactor := 4103191) rfl (by decide) (by decide)
    exact decide_eq_false composite
  second := false
  second_correct := by
    have composite : ¬ Nat.Prime (corner 1754 2) := by change ¬ Nat.Prime 12306065; exact PrattCertificates.composite_of_product (factor := 5) (cofactor := 2461213) rfl (by decide) (by decide)
    exact decide_eq_false composite
  third := false
  third_correct := by
    have composite : ¬ Nat.Prime (corner 1754 3) := by change ¬ Nat.Prime 12302557; exact PrattCertificates.composite_of_product (factor := 19) (cofactor := 647503) rfl (by decide) (by decide)
    exact decide_eq_false composite
theorem row_count_1754 : row 1754 = 0 := (row_1754.checked).trans rfl
theorem model_count_1754 : modelRow 1754 = 0 := by decide
theorem modelrow_agrees_1754 : modelRow 1754 = row 1754 := model_count_1754.trans row_count_1754.symm
def row_1755 : RowCertificate 1755 where
  first := true
  first_correct := by
    have prime : Nat.Prime (corner 1755 1) := by change Nat.Prime 12323611; exact ProjectEulerSolutions.P58PrimeNodes.p_12323611.prime
    exact decide_eq_true prime
  second := false
  second_correct := by
    have composite : ¬ Nat.Prime (corner 1755 2) := by change ¬ Nat.Prime 12320101; exact PrattCertificates.composite_of_product (factor := 397) (cofactor := 31033) rfl (by decide) (by decide)
    exact decide_eq_false composite
  third := false
  third_correct := by
    have composite : ¬ Nat.Prime (corner 1755 3) := by change ¬ Nat.Prime 12316591; exact PrattCertificates.composite_of_product (factor := 7) (cofactor := 1759513) rfl (by decide) (by decide)
    exact decide_eq_false composite
theorem row_count_1755 : row 1755 = 1 := (row_1755.checked).trans rfl
theorem model_count_1755 : modelRow 1755 = 1 := by decide
theorem modelrow_agrees_1755 : modelRow 1755 = row 1755 := model_count_1755.trans row_count_1755.symm
def row_1756 : RowCertificate 1756 where
  first := false
  first_correct := by
    have composite : ¬ Nat.Prime (corner 1756 1) := by change ¬ Nat.Prime 12337657; exact PrattCertificates.composite_of_product (factor := 73) (cofactor := 169009) rfl (by decide) (by decide)
    exact decide_eq_false composite
  second := false
  second_correct := by
    have composite : ¬ Nat.Prime (corner 1756 2) := by change ¬ Nat.Prime 12334145; exact PrattCertificates.composite_of_product (factor := 5) (cofactor := 2466829) rfl (by decide) (by decide)
    exact decide_eq_false composite
  third := false
  third_correct := by
    have composite : ¬ Nat.Prime (corner 1756 3) := by change ¬ Nat.Prime 12330633; exact PrattCertificates.composite_of_product (factor := 3) (cofactor := 4110211) rfl (by decide) (by decide)
    exact decide_eq_false composite
theorem row_count_1756 : row 1756 = 0 := (row_1756.checked).trans rfl
theorem model_count_1756 : modelRow 1756 = 0 := by decide
theorem modelrow_agrees_1756 : modelRow 1756 = row 1756 := model_count_1756.trans row_count_1756.symm
def row_1757 : RowCertificate 1757 where
  first := false
  first_correct := by
    have composite : ¬ Nat.Prime (corner 1757 1) := by change ¬ Nat.Prime 12351711; exact PrattCertificates.composite_of_product (factor := 3) (cofactor := 4117237) rfl (by decide) (by decide)
    exact decide_eq_false composite
  second := false
  second_correct := by
    have composite : ¬ Nat.Prime (corner 1757 2) := by change ¬ Nat.Prime 12348197; exact PrattCertificates.composite_of_product (factor := 97) (cofactor := 127301) rfl (by decide) (by decide)
    exact decide_eq_false composite
  third := false
  third_correct := by
    have composite : ¬ Nat.Prime (corner 1757 3) := by change ¬ Nat.Prime 12344683; exact PrattCertificates.composite_of_product (factor := 13) (cofactor := 949591) rfl (by decide) (by decide)
    exact decide_eq_false composite
theorem row_count_1757 : row 1757 = 0 := (row_1757.checked).trans rfl
theorem model_count_1757 : modelRow 1757 = 0 := by decide
theorem modelrow_agrees_1757 : modelRow 1757 = row 1757 := model_count_1757.trans row_count_1757.symm
def row_1758 : RowCertificate 1758 where
  first := false
  first_correct := by
    have composite : ¬ Nat.Prime (corner 1758 1) := by change ¬ Nat.Prime 12365773; exact PrattCertificates.composite_of_product (factor := 7) (cofactor := 1766539) rfl (by decide) (by decide)
    exact decide_eq_false composite
  second := true
  second_correct := by
    have prime : Nat.Prime (corner 1758 2) := by change Nat.Prime 12362257; exact ProjectEulerSolutions.P58PrimeNodes.p_12362257.prime
    exact decide_eq_true prime
  third := true
  third_correct := by
    have prime : Nat.Prime (corner 1758 3) := by change Nat.Prime 12358741; exact ProjectEulerSolutions.P58PrimeNodes.p_12358741.prime
    exact decide_eq_true prime
theorem row_count_1758 : row 1758 = 2 := (row_1758.checked).trans rfl
theorem model_count_1758 : modelRow 1758 = 2 := by decide
theorem modelrow_agrees_1758 : modelRow 1758 = row 1758 := model_count_1758.trans row_count_1758.symm
def row_1759 : RowCertificate 1759 where
  first := false
  first_correct := by
    have composite : ¬ Nat.Prime (corner 1759 1) := by change ¬ Nat.Prime 12379843; exact PrattCertificates.composite_of_product (factor := 7) (cofactor := 1768549) rfl (by decide) (by decide)
    exact decide_eq_false composite
  second := false
  second_correct := by
    have composite : ¬ Nat.Prime (corner 1759 2) := by change ¬ Nat.Prime 12376325; exact PrattCertificates.composite_of_product (factor := 5) (cofactor := 2475265) rfl (by decide) (by decide)
    exact decide_eq_false composite
  third := false
  third_correct := by
    have composite : ¬ Nat.Prime (corner 1759 3) := by change ¬ Nat.Prime 12372807; exact PrattCertificates.composite_of_product (factor := 3) (cofactor := 4124269) rfl (by decide) (by decide)
    exact decide_eq_false composite
theorem row_count_1759 : row 1759 = 0 := (row_1759.checked).trans rfl
theorem model_count_1759 : modelRow 1759 = 0 := by decide
theorem modelrow_agrees_1759 : modelRow 1759 = row 1759 := model_count_1759.trans row_count_1759.symm
def row_1760 : RowCertificate 1760 where
  first := false
  first_correct := by
    have composite : ¬ Nat.Prime (corner 1760 1) := by change ¬ Nat.Prime 12393921; exact PrattCertificates.composite_of_product (factor := 3) (cofactor := 4131307) rfl (by decide) (by decide)
    exact decide_eq_false composite
  second := true
  second_correct := by
    have prime : Nat.Prime (corner 1760 2) := by change Nat.Prime 12390401; exact ProjectEulerSolutions.P58PrimeNodes.p_12390401.prime
    exact decide_eq_true prime
  third := false
  third_correct := by
    have composite : ¬ Nat.Prime (corner 1760 3) := by change ¬ Nat.Prime 12386881; exact PrattCertificates.composite_of_product (factor := 13) (cofactor := 952837) rfl (by decide) (by decide)
    exact decide_eq_false composite
theorem row_count_1760 : row 1760 = 1 := (row_1760.checked).trans rfl
theorem model_count_1760 : modelRow 1760 = 1 := by decide
theorem modelrow_agrees_1760 : modelRow 1760 = row 1760 := model_count_1760.trans row_count_1760.symm
def row_1761 : RowCertificate 1761 where
  first := false
  first_correct := by
    have composite : ¬ Nat.Prime (corner 1761 1) := by change ¬ Nat.Prime 12408007; exact PrattCertificates.composite_of_product (factor := 19) (cofactor := 653053) rfl (by decide) (by decide)
    exact decide_eq_false composite
  second := false
  second_correct := by
    have composite : ¬ Nat.Prime (corner 1761 2) := by change ¬ Nat.Prime 12404485; exact PrattCertificates.composite_of_product (factor := 5) (cofactor := 2480897) rfl (by decide) (by decide)
    exact decide_eq_false composite
  third := false
  third_correct := by
    have composite : ¬ Nat.Prime (corner 1761 3) := by change ¬ Nat.Prime 12400963; exact PrattCertificates.composite_of_product (factor := 67) (cofactor := 185089) rfl (by decide) (by decide)
    exact decide_eq_false composite
theorem row_count_1761 : row 1761 = 0 := (row_1761.checked).trans rfl
theorem model_count_1761 : modelRow 1761 = 0 := by decide
theorem modelrow_agrees_1761 : modelRow 1761 = row 1761 := model_count_1761.trans row_count_1761.symm
def row_1762 : RowCertificate 1762 where
  first := false
  first_correct := by
    have composite : ¬ Nat.Prime (corner 1762 1) := by change ¬ Nat.Prime 12422101; exact PrattCertificates.composite_of_product (factor := 61) (cofactor := 203641) rfl (by decide) (by decide)
    exact decide_eq_false composite
  second := false
  second_correct := by
    have composite : ¬ Nat.Prime (corner 1762 2) := by change ¬ Nat.Prime 12418577; exact PrattCertificates.composite_of_product (factor := 3049) (cofactor := 4073) rfl (by decide) (by decide)
    exact decide_eq_false composite
  third := false
  third_correct := by
    have composite : ¬ Nat.Prime (corner 1762 3) := by change ¬ Nat.Prime 12415053; exact PrattCertificates.composite_of_product (factor := 3) (cofactor := 4138351) rfl (by decide) (by decide)
    exact decide_eq_false composite
theorem row_count_1762 : row 1762 = 0 := (row_1762.checked).trans rfl
theorem model_count_1762 : modelRow 1762 = 0 := by decide
theorem modelrow_agrees_1762 : modelRow 1762 = row 1762 := model_count_1762.trans row_count_1762.symm
def row_1763 : RowCertificate 1763 where
  first := false
  first_correct := by
    have composite : ¬ Nat.Prime (corner 1763 1) := by change ¬ Nat.Prime 12436203; exact PrattCertificates.composite_of_product (factor := 3) (cofactor := 4145401) rfl (by decide) (by decide)
    exact decide_eq_false composite
  second := false
  second_correct := by
    have composite : ¬ Nat.Prime (corner 1763 2) := by change ¬ Nat.Prime 12432677; exact PrattCertificates.composite_of_product (factor := 29) (cofactor := 428713) rfl (by decide) (by decide)
    exact decide_eq_false composite
  third := false
  third_correct := by
    have composite : ¬ Nat.Prime (corner 1763 3) := by change ¬ Nat.Prime 12429151; exact PrattCertificates.composite_of_product (factor := 7) (cofactor := 1775593) rfl (by decide) (by decide)
    exact decide_eq_false composite
theorem row_count_1763 : row 1763 = 0 := (row_1763.checked).trans rfl
theorem model_count_1763 : modelRow 1763 = 0 := by decide
theorem modelrow_agrees_1763 : modelRow 1763 = row 1763 := model_count_1763.trans row_count_1763.symm
def row_1764 : RowCertificate 1764 where
  first := false
  first_correct := by
    have composite : ¬ Nat.Prime (corner 1764 1) := by change ¬ Nat.Prime 12450313; exact PrattCertificates.composite_of_product (factor := 31) (cofactor := 401623) rfl (by decide) (by decide)
    exact decide_eq_false composite
  second := false
  second_correct := by
    have composite : ¬ Nat.Prime (corner 1764 2) := by change ¬ Nat.Prime 12446785; exact PrattCertificates.composite_of_product (factor := 5) (cofactor := 2489357) rfl (by decide) (by decide)
    exact decide_eq_false composite
  third := false
  third_correct := by
    have composite : ¬ Nat.Prime (corner 1764 3) := by change ¬ Nat.Prime 12443257; exact PrattCertificates.composite_of_product (factor := 97) (cofactor := 128281) rfl (by decide) (by decide)
    exact decide_eq_false composite
theorem row_count_1764 : row 1764 = 0 := (row_1764.checked).trans rfl
theorem model_count_1764 : modelRow 1764 = 0 := by decide
theorem modelrow_agrees_1764 : modelRow 1764 = row 1764 := model_count_1764.trans row_count_1764.symm
def row_1765 : RowCertificate 1765 where
  first := false
  first_correct := by
    have composite : ¬ Nat.Prime (corner 1765 1) := by change ¬ Nat.Prime 12464431; exact PrattCertificates.composite_of_product (factor := 7) (cofactor := 1780633) rfl (by decide) (by decide)
    exact decide_eq_false composite
  second := true
  second_correct := by
    have prime : Nat.Prime (corner 1765 2) := by change Nat.Prime 12460901; exact ProjectEulerSolutions.P58PrimeNodes.p_12460901.prime
    exact decide_eq_true prime
  third := false
  third_correct := by
    have composite : ¬ Nat.Prime (corner 1765 3) := by change ¬ Nat.Prime 12457371; exact PrattCertificates.composite_of_product (factor := 3) (cofactor := 4152457) rfl (by decide) (by decide)
    exact decide_eq_false composite
theorem row_count_1765 : row 1765 = 1 := (row_1765.checked).trans rfl
theorem model_count_1765 : modelRow 1765 = 1 := by decide
theorem modelrow_agrees_1765 : modelRow 1765 = row 1765 := model_count_1765.trans row_count_1765.symm
def row_1766 : RowCertificate 1766 where
  first := false
  first_correct := by
    have composite : ¬ Nat.Prime (corner 1766 1) := by change ¬ Nat.Prime 12478557; exact PrattCertificates.composite_of_product (factor := 3) (cofactor := 4159519) rfl (by decide) (by decide)
    exact decide_eq_false composite
  second := false
  second_correct := by
    have composite : ¬ Nat.Prime (corner 1766 2) := by change ¬ Nat.Prime 12475025; exact PrattCertificates.composite_of_product (factor := 5) (cofactor := 2495005) rfl (by decide) (by decide)
    exact decide_eq_false composite
  third := false
  third_correct := by
    have composite : ¬ Nat.Prime (corner 1766 3) := by change ¬ Nat.Prime 12471493; exact PrattCertificates.composite_of_product (factor := 79) (cofactor := 157867) rfl (by decide) (by decide)
    exact decide_eq_false composite
theorem row_count_1766 : row 1766 = 0 := (row_1766.checked).trans rfl
theorem model_count_1766 : modelRow 1766 = 0 := by decide
theorem modelrow_agrees_1766 : modelRow 1766 = row 1766 := model_count_1766.trans row_count_1766.symm
def row_1767 : RowCertificate 1767 where
  first := true
  first_correct := by
    have prime : Nat.Prime (corner 1767 1) := by change Nat.Prime 12492691; exact ProjectEulerSolutions.P58PrimeNodes.p_12492691.prime
    exact decide_eq_true prime
  second := true
  second_correct := by
    have prime : Nat.Prime (corner 1767 2) := by change Nat.Prime 12489157; exact ProjectEulerSolutions.P58PrimeNodes.p_12489157.prime
    exact decide_eq_true prime
  third := false
  third_correct := by
    have composite : ¬ Nat.Prime (corner 1767 3) := by change ¬ Nat.Prime 12485623; exact PrattCertificates.composite_of_product (factor := 109) (cofactor := 114547) rfl (by decide) (by decide)
    exact decide_eq_false composite
theorem row_count_1767 : row 1767 = 2 := (row_1767.checked).trans rfl
theorem model_count_1767 : modelRow 1767 = 2 := by decide
theorem modelrow_agrees_1767 : modelRow 1767 = row 1767 := model_count_1767.trans row_count_1767.symm
def row_1768 : RowCertificate 1768 where
  first := false
  first_correct := by
    have composite : ¬ Nat.Prime (corner 1768 1) := by change ¬ Nat.Prime 12506833; exact PrattCertificates.composite_of_product (factor := 127) (cofactor := 98479) rfl (by decide) (by decide)
    exact decide_eq_false composite
  second := true
  second_correct := by
    have prime : Nat.Prime (corner 1768 2) := by change Nat.Prime 12503297; exact ProjectEulerSolutions.P58PrimeNodes.p_12503297.prime
    exact decide_eq_true prime
  third := false
  third_correct := by
    have composite : ¬ Nat.Prime (corner 1768 3) := by change ¬ Nat.Prime 12499761; exact PrattCertificates.composite_of_product (factor := 3) (cofactor := 4166587) rfl (by decide) (by decide)
    exact decide_eq_false composite
theorem row_count_1768 : row 1768 = 1 := (row_1768.checked).trans rfl
theorem model_count_1768 : modelRow 1768 = 1 := by decide
theorem modelrow_agrees_1768 : modelRow 1768 = row 1768 := model_count_1768.trans row_count_1768.symm
def row_1769 : RowCertificate 1769 where
  first := false
  first_correct := by
    have composite : ¬ Nat.Prime (corner 1769 1) := by change ¬ Nat.Prime 12520983; exact PrattCertificates.composite_of_product (factor := 3) (cofactor := 4173661) rfl (by decide) (by decide)
    exact decide_eq_false composite
  second := false
  second_correct := by
    have composite : ¬ Nat.Prime (corner 1769 2) := by change ¬ Nat.Prime 12517445; exact PrattCertificates.composite_of_product (factor := 5) (cofactor := 2503489) rfl (by decide) (by decide)
    exact decide_eq_false composite
  third := false
  third_correct := by
    have composite : ¬ Nat.Prime (corner 1769 3) := by change ¬ Nat.Prime 12513907; exact PrattCertificates.composite_of_product (factor := 7) (cofactor := 1787701) rfl (by decide) (by decide)
    exact decide_eq_false composite
theorem row_count_1769 : row 1769 = 0 := (row_1769.checked).trans rfl
theorem model_count_1769 : modelRow 1769 = 0 := by decide
theorem modelrow_agrees_1769 : modelRow 1769 = row 1769 := model_count_1769.trans row_count_1769.symm
def row_1770 : RowCertificate 1770 where
  first := true
  first_correct := by
    have prime : Nat.Prime (corner 1770 1) := by change Nat.Prime 12535141; exact ProjectEulerSolutions.P58PrimeNodes.p_12535141.prime
    exact decide_eq_true prime
  second := false
  second_correct := by
    have composite : ¬ Nat.Prime (corner 1770 2) := by change ¬ Nat.Prime 12531601; exact PrattCertificates.composite_of_product (factor := 17) (cofactor := 737153) rfl (by decide) (by decide)
    exact decide_eq_false composite
  third := false
  third_correct := by
    have composite : ¬ Nat.Prime (corner 1770 3) := by change ¬ Nat.Prime 12528061; exact PrattCertificates.composite_of_product (factor := 7) (cofactor := 1789723) rfl (by decide) (by decide)
    exact decide_eq_false composite
theorem row_count_1770 : row 1770 = 1 := (row_1770.checked).trans rfl
theorem model_count_1770 : modelRow 1770 = 1 := by decide
theorem modelrow_agrees_1770 : modelRow 1770 = row 1770 := model_count_1770.trans row_count_1770.symm
def row_1771 : RowCertificate 1771 where
  first := true
  first_correct := by
    have prime : Nat.Prime (corner 1771 1) := by change Nat.Prime 12549307; exact ProjectEulerSolutions.P58PrimeNodes.p_12549307.prime
    exact decide_eq_true prime
  second := false
  second_correct := by
    have composite : ¬ Nat.Prime (corner 1771 2) := by change ¬ Nat.Prime 12545765; exact PrattCertificates.composite_of_product (factor := 5) (cofactor := 2509153) rfl (by decide) (by decide)
    exact decide_eq_false composite
  third := false
  third_correct := by
    have composite : ¬ Nat.Prime (corner 1771 3) := by change ¬ Nat.Prime 12542223; exact PrattCertificates.composite_of_product (factor := 3) (cofactor := 4180741) rfl (by decide) (by decide)
    exact decide_eq_false composite
theorem row_count_1771 : row 1771 = 1 := (row_1771.checked).trans rfl
theorem model_count_1771 : modelRow 1771 = 1 := by decide
theorem modelrow_agrees_1771 : modelRow 1771 = row 1771 := model_count_1771.trans row_count_1771.symm
def row_1772 : RowCertificate 1772 where
  first := false
  first_correct := by
    have composite : ¬ Nat.Prime (corner 1772 1) := by change ¬ Nat.Prime 12563481; exact PrattCertificates.composite_of_product (factor := 3) (cofactor := 4187827) rfl (by decide) (by decide)
    exact decide_eq_false composite
  second := false
  second_correct := by
    have composite : ¬ Nat.Prime (corner 1772 2) := by change ¬ Nat.Prime 12559937; exact PrattCertificates.composite_of_product (factor := 13) (cofactor := 966149) rfl (by decide) (by decide)
    exact decide_eq_false composite
  third := true
  third_correct := by
    have prime : Nat.Prime (corner 1772 3) := by change Nat.Prime 12556393; exact ProjectEulerSolutions.P58PrimeNodes.p_12556393.prime
    exact decide_eq_true prime
theorem row_count_1772 : row 1772 = 1 := (row_1772.checked).trans rfl
theorem model_count_1772 : modelRow 1772 = 1 := by decide
theorem modelrow_agrees_1772 : modelRow 1772 = row 1772 := model_count_1772.trans row_count_1772.symm
def row_1773 : RowCertificate 1773 where
  first := false
  first_correct := by
    have composite : ¬ Nat.Prime (corner 1773 1) := by change ¬ Nat.Prime 12577663; exact PrattCertificates.composite_of_product (factor := 7) (cofactor := 1796809) rfl (by decide) (by decide)
    exact decide_eq_false composite
  second := false
  second_correct := by
    have composite : ¬ Nat.Prime (corner 1773 2) := by change ¬ Nat.Prime 12574117; exact PrattCertificates.composite_of_product (factor := 37) (cofactor := 339841) rfl (by decide) (by decide)
    exact decide_eq_false composite
  third := false
  third_correct := by
    have composite : ¬ Nat.Prime (corner 1773 3) := by change ¬ Nat.Prime 12570571; exact PrattCertificates.composite_of_product (factor := 13) (cofactor := 966967) rfl (by decide) (by decide)
    exact decide_eq_false composite
theorem row_count_1773 : row 1773 = 0 := (row_1773.checked).trans rfl
theorem model_count_1773 : modelRow 1773 = 0 := by decide
theorem modelrow_agrees_1773 : modelRow 1773 = row 1773 := model_count_1773.trans row_count_1773.symm
def row_1774 : RowCertificate 1774 where
  first := false
  first_correct := by
    have composite : ¬ Nat.Prime (corner 1774 1) := by change ¬ Nat.Prime 12591853; exact PrattCertificates.composite_of_product (factor := 103) (cofactor := 122251) rfl (by decide) (by decide)
    exact decide_eq_false composite
  second := false
  second_correct := by
    have composite : ¬ Nat.Prime (corner 1774 2) := by change ¬ Nat.Prime 12588305; exact PrattCertificates.composite_of_product (factor := 5) (cofactor := 2517661) rfl (by decide) (by decide)
    exact decide_eq_false composite
  third := false
  third_correct := by
    have composite : ¬ Nat.Prime (corner 1774 3) := by change ¬ Nat.Prime 12584757; exact PrattCertificates.composite_of_product (factor := 3) (cofactor := 4194919) rfl (by decide) (by decide)
    exact decide_eq_false composite
theorem row_count_1774 : row 1774 = 0 := (row_1774.checked).trans rfl
theorem model_count_1774 : modelRow 1774 = 0 := by decide
theorem modelrow_agrees_1774 : modelRow 1774 = row 1774 := model_count_1774.trans row_count_1774.symm
def row_1775 : RowCertificate 1775 where
  first := false
  first_correct := by
    have composite : ¬ Nat.Prime (corner 1775 1) := by change ¬ Nat.Prime 12606051; exact PrattCertificates.composite_of_product (factor := 3) (cofactor := 4202017) rfl (by decide) (by decide)
    exact decide_eq_false composite
  second := false
  second_correct := by
    have composite : ¬ Nat.Prime (corner 1775 2) := by change ¬ Nat.Prime 12602501; exact PrattCertificates.composite_of_product (factor := 29) (cofactor := 434569) rfl (by decide) (by decide)
    exact decide_eq_false composite
  third := true
  third_correct := by
    have prime : Nat.Prime (corner 1775 3) := by change Nat.Prime 12598951; exact ProjectEulerSolutions.P58PrimeNodes.p_12598951.prime
    exact decide_eq_true prime
theorem row_count_1775 : row 1775 = 1 := (row_1775.checked).trans rfl
theorem model_count_1775 : modelRow 1775 = 1 := by decide
theorem modelrow_agrees_1775 : modelRow 1775 = row 1775 := model_count_1775.trans row_count_1775.symm
def row_1776 : RowCertificate 1776 where
  first := false
  first_correct := by
    have composite : ¬ Nat.Prime (corner 1776 1) := by change ¬ Nat.Prime 12620257; exact PrattCertificates.composite_of_product (factor := 13) (cofactor := 970789) rfl (by decide) (by decide)
    exact decide_eq_false composite
  second := false
  second_correct := by
    have composite : ¬ Nat.Prime (corner 1776 2) := by change ¬ Nat.Prime 12616705; exact PrattCertificates.composite_of_product (factor := 5) (cofactor := 2523341) rfl (by decide) (by decide)
    exact decide_eq_false composite
  third := false
  third_correct := by
    have composite : ¬ Nat.Prime (corner 1776 3) := by change ¬ Nat.Prime 12613153; exact PrattCertificates.composite_of_product (factor := 7) (cofactor := 1801879) rfl (by decide) (by decide)
    exact decide_eq_false composite
theorem row_count_1776 : row 1776 = 0 := (row_1776.checked).trans rfl
theorem model_count_1776 : modelRow 1776 = 0 := by decide
theorem modelrow_agrees_1776 : modelRow 1776 = row 1776 := model_count_1776.trans row_count_1776.symm
def row_1777 : RowCertificate 1777 where
  first := true
  first_correct := by
    have prime : Nat.Prime (corner 1777 1) := by change Nat.Prime 12634471; exact ProjectEulerSolutions.P58PrimeNodes.p_12634471.prime
    exact decide_eq_true prime
  second := false
  second_correct := by
    have composite : ¬ Nat.Prime (corner 1777 2) := by change ¬ Nat.Prime 12630917; exact PrattCertificates.composite_of_product (factor := 13) (cofactor := 971609) rfl (by decide) (by decide)
    exact decide_eq_false composite
  third := false
  third_correct := by
    have composite : ¬ Nat.Prime (corner 1777 3) := by change ¬ Nat.Prime 12627363; exact PrattCertificates.composite_of_product (factor := 3) (cofactor := 4209121) rfl (by decide) (by decide)
    exact decide_eq_false composite
theorem row_count_1777 : row 1777 = 1 := (row_1777.checked).trans rfl
theorem model_count_1777 : modelRow 1777 = 1 := by decide
theorem modelrow_agrees_1777 : modelRow 1777 = row 1777 := model_count_1777.trans row_count_1777.symm
def row_1778 : RowCertificate 1778 where
  first := false
  first_correct := by
    have composite : ¬ Nat.Prime (corner 1778 1) := by change ¬ Nat.Prime 12648693; exact PrattCertificates.composite_of_product (factor := 3) (cofactor := 4216231) rfl (by decide) (by decide)
    exact decide_eq_false composite
  second := false
  second_correct := by
    have composite : ¬ Nat.Prime (corner 1778 2) := by change ¬ Nat.Prime 12645137; exact PrattCertificates.composite_of_product (factor := 2269) (cofactor := 5573) rfl (by decide) (by decide)
    exact decide_eq_false composite
  third := false
  third_correct := by
    have composite : ¬ Nat.Prime (corner 1778 3) := by change ¬ Nat.Prime 12641581; exact PrattCertificates.composite_of_product (factor := 787) (cofactor := 16063) rfl (by decide) (by decide)
    exact decide_eq_false composite
theorem row_count_1778 : row 1778 = 0 := (row_1778.checked).trans rfl
theorem model_count_1778 : modelRow 1778 = 0 := by decide
theorem modelrow_agrees_1778 : modelRow 1778 = row 1778 := model_count_1778.trans row_count_1778.symm
def row_1779 : RowCertificate 1779 where
  first := false
  first_correct := by
    have composite : ¬ Nat.Prime (corner 1779 1) := by change ¬ Nat.Prime 12662923; exact PrattCertificates.composite_of_product (factor := 7) (cofactor := 1808989) rfl (by decide) (by decide)
    exact decide_eq_false composite
  second := false
  second_correct := by
    have composite : ¬ Nat.Prime (corner 1779 2) := by change ¬ Nat.Prime 12659365; exact PrattCertificates.composite_of_product (factor := 5) (cofactor := 2531873) rfl (by decide) (by decide)
    exact decide_eq_false composite
  third := true
  third_correct := by
    have prime : Nat.Prime (corner 1779 3) := by change Nat.Prime 12655807; exact ProjectEulerSolutions.P58PrimeNodes.p_12655807.prime
    exact decide_eq_true prime
theorem row_count_1779 : row 1779 = 1 := (row_1779.checked).trans rfl
theorem model_count_1779 : modelRow 1779 = 1 := by decide
theorem modelrow_agrees_1779 : modelRow 1779 = row 1779 := model_count_1779.trans row_count_1779.symm
def row_1780 : RowCertificate 1780 where
  first := false
  first_correct := by
    have composite : ¬ Nat.Prime (corner 1780 1) := by change ¬ Nat.Prime 12677161; exact PrattCertificates.composite_of_product (factor := 7) (cofactor := 1811023) rfl (by decide) (by decide)
    exact decide_eq_false composite
  second := false
  second_correct := by
    have composite : ¬ Nat.Prime (corner 1780 2) := by change ¬ Nat.Prime 12673601; exact PrattCertificates.composite_of_product (factor := 197) (cofactor := 64333) rfl (by decide) (by decide)
    exact decide_eq_false composite
  third := false
  third_correct := by
    have composite : ¬ Nat.Prime (corner 1780 3) := by change ¬ Nat.Prime 12670041; exact PrattCertificates.composite_of_product (factor := 3) (cofactor := 4223347) rfl (by decide) (by decide)
    exact decide_eq_false composite
theorem row_count_1780 : row 1780 = 0 := (row_1780.checked).trans rfl
theorem model_count_1780 : modelRow 1780 = 0 := by decide
theorem modelrow_agrees_1780 : modelRow 1780 = row 1780 := model_count_1780.trans row_count_1780.symm
def row_1781 : RowCertificate 1781 where
  first := false
  first_correct := by
    have composite : ¬ Nat.Prime (corner 1781 1) := by change ¬ Nat.Prime 12691407; exact PrattCertificates.composite_of_product (factor := 3) (cofactor := 4230469) rfl (by decide) (by decide)
    exact decide_eq_false composite
  second := false
  second_correct := by
    have composite : ¬ Nat.Prime (corner 1781 2) := by change ¬ Nat.Prime 12687845; exact PrattCertificates.composite_of_product (factor := 5) (cofactor := 2537569) rfl (by decide) (by decide)
    exact decide_eq_false composite
  third := true
  third_correct := by
    have prime : Nat.Prime (corner 1781 3) := by change Nat.Prime 12684283; exact ProjectEulerSolutions.P58PrimeNodes.p_12684283.prime
    exact decide_eq_true prime
theorem row_count_1781 : row 1781 = 1 := (row_1781.checked).trans rfl
theorem model_count_1781 : modelRow 1781 = 1 := by decide
theorem modelrow_agrees_1781 : modelRow 1781 = row 1781 := model_count_1781.trans row_count_1781.symm
def row_1782 : RowCertificate 1782 where
  first := false
  first_correct := by
    have composite : ¬ Nat.Prime (corner 1782 1) := by change ¬ Nat.Prime 12705661; exact PrattCertificates.composite_of_product (factor := 19) (cofactor := 668719) rfl (by decide) (by decide)
    exact decide_eq_false composite
  second := false
  second_correct := by
    have composite : ¬ Nat.Prime (corner 1782 2) := by change ¬ Nat.Prime 12702097; exact PrattCertificates.composite_of_product (factor := 109) (cofactor := 116533) rfl (by decide) (by decide)
    exact decide_eq_false composite
  third := true
  third_correct := by
    have prime : Nat.Prime (corner 1782 3) := by change Nat.Prime 12698533; exact ProjectEulerSolutions.P58PrimeNodes.p_12698533.prime
    exact decide_eq_true prime
theorem row_count_1782 : row 1782 = 1 := (row_1782.checked).trans rfl
theorem model_count_1782 : modelRow 1782 = 1 := by decide
theorem modelrow_agrees_1782 : modelRow 1782 = row 1782 := model_count_1782.trans row_count_1782.symm
def row_1783 : RowCertificate 1783 where
  first := false
  first_correct := by
    have composite : ¬ Nat.Prime (corner 1783 1) := by change ¬ Nat.Prime 12719923; exact PrattCertificates.composite_of_product (factor := 1231) (cofactor := 10333) rfl (by decide) (by decide)
    exact decide_eq_false composite
  second := false
  second_correct := by
    have composite : ¬ Nat.Prime (corner 1783 2) := by change ¬ Nat.Prime 12716357; exact PrattCertificates.composite_of_product (factor := 17) (cofactor := 748021) rfl (by decide) (by decide)
    exact decide_eq_false composite
  third := false
  third_correct := by
    have composite : ¬ Nat.Prime (corner 1783 3) := by change ¬ Nat.Prime 12712791; exact PrattCertificates.composite_of_product (factor := 3) (cofactor := 4237597) rfl (by decide) (by decide)
    exact decide_eq_false composite
theorem row_count_1783 : row 1783 = 0 := (row_1783.checked).trans rfl
theorem model_count_1783 : modelRow 1783 = 0 := by decide
theorem modelrow_agrees_1783 : modelRow 1783 = row 1783 := model_count_1783.trans row_count_1783.symm
def row_1784 : RowCertificate 1784 where
  first := false
  first_correct := by
    have composite : ¬ Nat.Prime (corner 1784 1) := by change ¬ Nat.Prime 12734193; exact PrattCertificates.composite_of_product (factor := 3) (cofactor := 4244731) rfl (by decide) (by decide)
    exact decide_eq_false composite
  second := false
  second_correct := by
    have composite : ¬ Nat.Prime (corner 1784 2) := by change ¬ Nat.Prime 12730625; exact PrattCertificates.composite_of_product (factor := 5) (cofactor := 2546125) rfl (by decide) (by decide)
    exact decide_eq_false composite
  third := false
  third_correct := by
    have composite : ¬ Nat.Prime (corner 1784 3) := by change ¬ Nat.Prime 12727057; exact PrattCertificates.composite_of_product (factor := 7) (cofactor := 1818151) rfl (by decide) (by decide)
    exact decide_eq_false composite
theorem row_count_1784 : row 1784 = 0 := (row_1784.checked).trans rfl
theorem model_count_1784 : modelRow 1784 = 0 := by decide
theorem modelrow_agrees_1784 : modelRow 1784 = row 1784 := model_count_1784.trans row_count_1784.symm
def row_1785 : RowCertificate 1785 where
  first := false
  first_correct := by
    have composite : ¬ Nat.Prime (corner 1785 1) := by change ¬ Nat.Prime 12748471; exact PrattCertificates.composite_of_product (factor := 31) (cofactor := 411241) rfl (by decide) (by decide)
    exact decide_eq_false composite
  second := false
  second_correct := by
    have composite : ¬ Nat.Prime (corner 1785 2) := by change ¬ Nat.Prime 12744901; exact PrattCertificates.composite_of_product (factor := 13) (cofactor := 980377) rfl (by decide) (by decide)
    exact decide_eq_false composite
  third := false
  third_correct := by
    have composite : ¬ Nat.Prime (corner 1785 3) := by change ¬ Nat.Prime 12741331; exact PrattCertificates.composite_of_product (factor := 229) (cofactor := 55639) rfl (by decide) (by decide)
    exact decide_eq_false composite
theorem row_count_1785 : row 1785 = 0 := (row_1785.checked).trans rfl
theorem model_count_1785 : modelRow 1785 = 0 := by decide
theorem modelrow_agrees_1785 : modelRow 1785 = row 1785 := model_count_1785.trans row_count_1785.symm
def row_1786 : RowCertificate 1786 where
  first := false
  first_correct := by
    have composite : ¬ Nat.Prime (corner 1786 1) := by change ¬ Nat.Prime 12762757; exact PrattCertificates.composite_of_product (factor := 7) (cofactor := 1823251) rfl (by decide) (by decide)
    exact decide_eq_false composite
  second := false
  second_correct := by
    have composite : ¬ Nat.Prime (corner 1786 2) := by change ¬ Nat.Prime 12759185; exact PrattCertificates.composite_of_product (factor := 5) (cofactor := 2551837) rfl (by decide) (by decide)
    exact decide_eq_false composite
  third := false
  third_correct := by
    have composite : ¬ Nat.Prime (corner 1786 3) := by change ¬ Nat.Prime 12755613; exact PrattCertificates.composite_of_product (factor := 3) (cofactor := 4251871) rfl (by decide) (by decide)
    exact decide_eq_false composite
theorem row_count_1786 : row 1786 = 0 := (row_1786.checked).trans rfl
theorem model_count_1786 : modelRow 1786 = 0 := by decide
theorem modelrow_agrees_1786 : modelRow 1786 = row 1786 := model_count_1786.trans row_count_1786.symm
def row_1787 : RowCertificate 1787 where
  first := false
  first_correct := by
    have composite : ¬ Nat.Prime (corner 1787 1) := by change ¬ Nat.Prime 12777051; exact PrattCertificates.composite_of_product (factor := 3) (cofactor := 4259017) rfl (by decide) (by decide)
    exact decide_eq_false composite
  second := false
  second_correct := by
    have composite : ¬ Nat.Prime (corner 1787 2) := by change ¬ Nat.Prime 12773477; exact PrattCertificates.composite_of_product (factor := 17) (cofactor := 751381) rfl (by decide) (by decide)
    exact decide_eq_false composite
  third := true
  third_correct := by
    have prime : Nat.Prime (corner 1787 3) := by change Nat.Prime 12769903; exact ProjectEulerSolutions.P58PrimeNodes.p_12769903.prime
    exact decide_eq_true prime
theorem row_count_1787 : row 1787 = 1 := (row_1787.checked).trans rfl
theorem model_count_1787 : modelRow 1787 = 1 := by decide
theorem modelrow_agrees_1787 : modelRow 1787 = row 1787 := model_count_1787.trans row_count_1787.symm
def row_1788 : RowCertificate 1788 where
  first := true
  first_correct := by
    have prime : Nat.Prime (corner 1788 1) := by change Nat.Prime 12791353; exact ProjectEulerSolutions.P58PrimeNodes.p_12791353.prime
    exact decide_eq_true prime
  second := false
  second_correct := by
    have composite : ¬ Nat.Prime (corner 1788 2) := by change ¬ Nat.Prime 12787777; exact PrattCertificates.composite_of_product (factor := 41) (cofactor := 311897) rfl (by decide) (by decide)
    exact decide_eq_false composite
  third := false
  third_correct := by
    have composite : ¬ Nat.Prime (corner 1788 3) := by change ¬ Nat.Prime 12784201; exact PrattCertificates.composite_of_product (factor := 43) (cofactor := 297307) rfl (by decide) (by decide)
    exact decide_eq_false composite
theorem row_count_1788 : row 1788 = 1 := (row_1788.checked).trans rfl
theorem model_count_1788 : modelRow 1788 = 1 := by decide
theorem modelrow_agrees_1788 : modelRow 1788 = row 1788 := model_count_1788.trans row_count_1788.symm
def row_1789 : RowCertificate 1789 where
  first := false
  first_correct := by
    have composite : ¬ Nat.Prime (corner 1789 1) := by change ¬ Nat.Prime 12805663; exact PrattCertificates.composite_of_product (factor := 13) (cofactor := 985051) rfl (by decide) (by decide)
    exact decide_eq_false composite
  second := false
  second_correct := by
    have composite : ¬ Nat.Prime (corner 1789 2) := by change ¬ Nat.Prime 12802085; exact PrattCertificates.composite_of_product (factor := 5) (cofactor := 2560417) rfl (by decide) (by decide)
    exact decide_eq_false composite
  third := false
  third_correct := by
    have composite : ¬ Nat.Prime (corner 1789 3) := by change ¬ Nat.Prime 12798507; exact PrattCertificates.composite_of_product (factor := 3) (cofactor := 4266169) rfl (by decide) (by decide)
    exact decide_eq_false composite
theorem row_count_1789 : row 1789 = 0 := (row_1789.checked).trans rfl
theorem model_count_1789 : modelRow 1789 = 0 := by decide
theorem modelrow_agrees_1789 : modelRow 1789 = row 1789 := model_count_1789.trans row_count_1789.symm
def row_1790 : RowCertificate 1790 where
  first := false
  first_correct := by
    have composite : ¬ Nat.Prime (corner 1790 1) := by change ¬ Nat.Prime 12819981; exact PrattCertificates.composite_of_product (factor := 3) (cofactor := 4273327) rfl (by decide) (by decide)
    exact decide_eq_false composite
  second := false
  second_correct := by
    have composite : ¬ Nat.Prime (corner 1790 2) := by change ¬ Nat.Prime 12816401; exact PrattCertificates.composite_of_product (factor := 13) (cofactor := 985877) rfl (by decide) (by decide)
    exact decide_eq_false composite
  third := false
  third_correct := by
    have composite : ¬ Nat.Prime (corner 1790 3) := by change ¬ Nat.Prime 12812821; exact PrattCertificates.composite_of_product (factor := 7) (cofactor := 1830403) rfl (by decide) (by decide)
    exact decide_eq_false composite
theorem row_count_1790 : row 1790 = 0 := (row_1790.checked).trans rfl
theorem model_count_1790 : modelRow 1790 = 0 := by decide
theorem modelrow_agrees_1790 : modelRow 1790 = row 1790 := model_count_1790.trans row_count_1790.symm
def row_1791 : RowCertificate 1791 where
  first := false
  first_correct := by
    have composite : ¬ Nat.Prime (corner 1791 1) := by change ¬ Nat.Prime 12834307; exact PrattCertificates.composite_of_product (factor := 193) (cofactor := 66499) rfl (by decide) (by decide)
    exact decide_eq_false composite
  second := false
  second_correct := by
    have composite : ¬ Nat.Prime (corner 1791 2) := by change ¬ Nat.Prime 12830725; exact PrattCertificates.composite_of_product (factor := 5) (cofactor := 2566145) rfl (by decide) (by decide)
    exact decide_eq_false composite
  third := false
  third_correct := by
    have composite : ¬ Nat.Prime (corner 1791 3) := by change ¬ Nat.Prime 12827143; exact PrattCertificates.composite_of_product (factor := 7) (cofactor := 1832449) rfl (by decide) (by decide)
    exact decide_eq_false composite
theorem row_count_1791 : row 1791 = 0 := (row_1791.checked).trans rfl
theorem model_count_1791 : modelRow 1791 = 0 := by decide
theorem modelrow_agrees_1791 : modelRow 1791 = row 1791 := model_count_1791.trans row_count_1791.symm
def row_1792 : RowCertificate 1792 where
  first := false
  first_correct := by
    have composite : ¬ Nat.Prime (corner 1792 1) := by change ¬ Nat.Prime 12848641; exact PrattCertificates.composite_of_product (factor := 13) (cofactor := 988357) rfl (by decide) (by decide)
    exact decide_eq_false composite
  second := false
  second_correct := by
    have composite : ¬ Nat.Prime (corner 1792 2) := by change ¬ Nat.Prime 12845057; exact PrattCertificates.composite_of_product (factor := 29) (cofactor := 442933) rfl (by decide) (by decide)
    exact decide_eq_false composite
  third := false
  third_correct := by
    have composite : ¬ Nat.Prime (corner 1792 3) := by change ¬ Nat.Prime 12841473; exact PrattCertificates.composite_of_product (factor := 3) (cofactor := 4280491) rfl (by decide) (by decide)
    exact decide_eq_false composite
theorem row_count_1792 : row 1792 = 0 := (row_1792.checked).trans rfl
theorem model_count_1792 : modelRow 1792 = 0 := by decide
theorem modelrow_agrees_1792 : modelRow 1792 = row 1792 := model_count_1792.trans row_count_1792.symm
private def atomic_1665 : CountSegment 1 10 13120 1664 1665 873 874 := CountSegment.single (first := 1664) (initial := 873) row_count_1665 (by decide)
private theorem atomic_1665_agreement : RowAgreement 1664 1665 := RowAgreement.single (first := 1664) modelrow_agrees_1665
private def atomic_1666 : CountSegment 1 10 13120 1665 1666 874 874 := CountSegment.single (first := 1665) (initial := 874) row_count_1666 (by decide)
private theorem atomic_1666_agreement : RowAgreement 1665 1666 := RowAgreement.single (first := 1665) modelrow_agrees_1666
private def atomic_1667 : CountSegment 1 10 13120 1666 1667 874 875 := CountSegment.single (first := 1666) (initial := 874) row_count_1667 (by decide)
private theorem atomic_1667_agreement : RowAgreement 1666 1667 := RowAgreement.single (first := 1666) modelrow_agrees_1667
private def atomic_1668 : CountSegment 1 10 13120 1667 1668 875 875 := CountSegment.single (first := 1667) (initial := 875) row_count_1668 (by decide)
private theorem atomic_1668_agreement : RowAgreement 1667 1668 := RowAgreement.single (first := 1667) modelrow_agrees_1668
private def atomic_1669 : CountSegment 1 10 13120 1668 1669 875 876 := CountSegment.single (first := 1668) (initial := 875) row_count_1669 (by decide)
private theorem atomic_1669_agreement : RowAgreement 1668 1669 := RowAgreement.single (first := 1668) modelrow_agrees_1669
private def atomic_1670 : CountSegment 1 10 13120 1669 1670 876 878 := CountSegment.single (first := 1669) (initial := 876) row_count_1670 (by decide)
private theorem atomic_1670_agreement : RowAgreement 1669 1670 := RowAgreement.single (first := 1669) modelrow_agrees_1670
private def atomic_1671 : CountSegment 1 10 13120 1670 1671 878 878 := CountSegment.single (first := 1670) (initial := 878) row_count_1671 (by decide)
private theorem atomic_1671_agreement : RowAgreement 1670 1671 := RowAgreement.single (first := 1670) modelrow_agrees_1671
private def atomic_1672 : CountSegment 1 10 13120 1671 1672 878 878 := CountSegment.single (first := 1671) (initial := 878) row_count_1672 (by decide)
private theorem atomic_1672_agreement : RowAgreement 1671 1672 := RowAgreement.single (first := 1671) modelrow_agrees_1672
private def atomic_1673 : CountSegment 1 10 13120 1672 1673 878 879 := CountSegment.single (first := 1672) (initial := 878) row_count_1673 (by decide)
private theorem atomic_1673_agreement : RowAgreement 1672 1673 := RowAgreement.single (first := 1672) modelrow_agrees_1673
private def atomic_1674 : CountSegment 1 10 13120 1673 1674 879 879 := CountSegment.single (first := 1673) (initial := 879) row_count_1674 (by decide)
private theorem atomic_1674_agreement : RowAgreement 1673 1674 := RowAgreement.single (first := 1673) modelrow_agrees_1674
private def atomic_1675 : CountSegment 1 10 13120 1674 1675 879 880 := CountSegment.single (first := 1674) (initial := 879) row_count_1675 (by decide)
private theorem atomic_1675_agreement : RowAgreement 1674 1675 := RowAgreement.single (first := 1674) modelrow_agrees_1675
private def atomic_1676 : CountSegment 1 10 13120 1675 1676 880 880 := CountSegment.single (first := 1675) (initial := 880) row_count_1676 (by decide)
private theorem atomic_1676_agreement : RowAgreement 1675 1676 := RowAgreement.single (first := 1675) modelrow_agrees_1676
private def atomic_1677 : CountSegment 1 10 13120 1676 1677 880 880 := CountSegment.single (first := 1676) (initial := 880) row_count_1677 (by decide)
private theorem atomic_1677_agreement : RowAgreement 1676 1677 := RowAgreement.single (first := 1676) modelrow_agrees_1677
private def atomic_1678 : CountSegment 1 10 13120 1677 1678 880 881 := CountSegment.single (first := 1677) (initial := 880) row_count_1678 (by decide)
private theorem atomic_1678_agreement : RowAgreement 1677 1678 := RowAgreement.single (first := 1677) modelrow_agrees_1678
private def atomic_1679 : CountSegment 1 10 13120 1678 1679 881 881 := CountSegment.single (first := 1678) (initial := 881) row_count_1679 (by decide)
private theorem atomic_1679_agreement : RowAgreement 1678 1679 := RowAgreement.single (first := 1678) modelrow_agrees_1679
private def atomic_1680 : CountSegment 1 10 13120 1679 1680 881 882 := CountSegment.single (first := 1679) (initial := 881) row_count_1680 (by decide)
private theorem atomic_1680_agreement : RowAgreement 1679 1680 := RowAgreement.single (first := 1679) modelrow_agrees_1680
private def atomic_1681 : CountSegment 1 10 13120 1680 1681 882 882 := CountSegment.single (first := 1680) (initial := 882) row_count_1681 (by decide)
private theorem atomic_1681_agreement : RowAgreement 1680 1681 := RowAgreement.single (first := 1680) modelrow_agrees_1681
private def atomic_1682 : CountSegment 1 10 13120 1681 1682 882 882 := CountSegment.single (first := 1681) (initial := 882) row_count_1682 (by decide)
private theorem atomic_1682_agreement : RowAgreement 1681 1682 := RowAgreement.single (first := 1681) modelrow_agrees_1682
private def atomic_1683 : CountSegment 1 10 13120 1682 1683 882 882 := CountSegment.single (first := 1682) (initial := 882) row_count_1683 (by decide)
private theorem atomic_1683_agreement : RowAgreement 1682 1683 := RowAgreement.single (first := 1682) modelrow_agrees_1683
private def atomic_1684 : CountSegment 1 10 13120 1683 1684 882 882 := CountSegment.single (first := 1683) (initial := 882) row_count_1684 (by decide)
private theorem atomic_1684_agreement : RowAgreement 1683 1684 := RowAgreement.single (first := 1683) modelrow_agrees_1684
private def atomic_1685 : CountSegment 1 10 13120 1684 1685 882 882 := CountSegment.single (first := 1684) (initial := 882) row_count_1685 (by decide)
private theorem atomic_1685_agreement : RowAgreement 1684 1685 := RowAgreement.single (first := 1684) modelrow_agrees_1685
private def atomic_1686 : CountSegment 1 10 13120 1685 1686 882 882 := CountSegment.single (first := 1685) (initial := 882) row_count_1686 (by decide)
private theorem atomic_1686_agreement : RowAgreement 1685 1686 := RowAgreement.single (first := 1685) modelrow_agrees_1686
private def atomic_1687 : CountSegment 1 10 13120 1686 1687 882 883 := CountSegment.single (first := 1686) (initial := 882) row_count_1687 (by decide)
private theorem atomic_1687_agreement : RowAgreement 1686 1687 := RowAgreement.single (first := 1686) modelrow_agrees_1687
private def atomic_1688 : CountSegment 1 10 13120 1687 1688 883 883 := CountSegment.single (first := 1687) (initial := 883) row_count_1688 (by decide)
private theorem atomic_1688_agreement : RowAgreement 1687 1688 := RowAgreement.single (first := 1687) modelrow_agrees_1688
private def atomic_1689 : CountSegment 1 10 13120 1688 1689 883 883 := CountSegment.single (first := 1688) (initial := 883) row_count_1689 (by decide)
private theorem atomic_1689_agreement : RowAgreement 1688 1689 := RowAgreement.single (first := 1688) modelrow_agrees_1689
private def atomic_1690 : CountSegment 1 10 13120 1689 1690 883 884 := CountSegment.single (first := 1689) (initial := 883) row_count_1690 (by decide)
private theorem atomic_1690_agreement : RowAgreement 1689 1690 := RowAgreement.single (first := 1689) modelrow_agrees_1690
private def atomic_1691 : CountSegment 1 10 13120 1690 1691 884 885 := CountSegment.single (first := 1690) (initial := 884) row_count_1691 (by decide)
private theorem atomic_1691_agreement : RowAgreement 1690 1691 := RowAgreement.single (first := 1690) modelrow_agrees_1691
private def atomic_1692 : CountSegment 1 10 13120 1691 1692 885 885 := CountSegment.single (first := 1691) (initial := 885) row_count_1692 (by decide)
private theorem atomic_1692_agreement : RowAgreement 1691 1692 := RowAgreement.single (first := 1691) modelrow_agrees_1692
private def atomic_1693 : CountSegment 1 10 13120 1692 1693 885 885 := CountSegment.single (first := 1692) (initial := 885) row_count_1693 (by decide)
private theorem atomic_1693_agreement : RowAgreement 1692 1693 := RowAgreement.single (first := 1692) modelrow_agrees_1693
private def atomic_1694 : CountSegment 1 10 13120 1693 1694 885 885 := CountSegment.single (first := 1693) (initial := 885) row_count_1694 (by decide)
private theorem atomic_1694_agreement : RowAgreement 1693 1694 := RowAgreement.single (first := 1693) modelrow_agrees_1694
private def atomic_1695 : CountSegment 1 10 13120 1694 1695 885 886 := CountSegment.single (first := 1694) (initial := 885) row_count_1695 (by decide)
private theorem atomic_1695_agreement : RowAgreement 1694 1695 := RowAgreement.single (first := 1694) modelrow_agrees_1695
private def atomic_1696 : CountSegment 1 10 13120 1695 1696 886 886 := CountSegment.single (first := 1695) (initial := 886) row_count_1696 (by decide)
private theorem atomic_1696_agreement : RowAgreement 1695 1696 := RowAgreement.single (first := 1695) modelrow_agrees_1696
private def atomic_1697 : CountSegment 1 10 13120 1696 1697 886 886 := CountSegment.single (first := 1696) (initial := 886) row_count_1697 (by decide)
private theorem atomic_1697_agreement : RowAgreement 1696 1697 := RowAgreement.single (first := 1696) modelrow_agrees_1697
private def atomic_1698 : CountSegment 1 10 13120 1697 1698 886 887 := CountSegment.single (first := 1697) (initial := 886) row_count_1698 (by decide)
private theorem atomic_1698_agreement : RowAgreement 1697 1698 := RowAgreement.single (first := 1697) modelrow_agrees_1698
private def atomic_1699 : CountSegment 1 10 13120 1698 1699 887 888 := CountSegment.single (first := 1698) (initial := 887) row_count_1699 (by decide)
private theorem atomic_1699_agreement : RowAgreement 1698 1699 := RowAgreement.single (first := 1698) modelrow_agrees_1699
private def atomic_1700 : CountSegment 1 10 13120 1699 1700 888 888 := CountSegment.single (first := 1699) (initial := 888) row_count_1700 (by decide)
private theorem atomic_1700_agreement : RowAgreement 1699 1700 := RowAgreement.single (first := 1699) modelrow_agrees_1700
private def atomic_1701 : CountSegment 1 10 13120 1700 1701 888 888 := CountSegment.single (first := 1700) (initial := 888) row_count_1701 (by decide)
private theorem atomic_1701_agreement : RowAgreement 1700 1701 := RowAgreement.single (first := 1700) modelrow_agrees_1701
private def atomic_1702 : CountSegment 1 10 13120 1701 1702 888 888 := CountSegment.single (first := 1701) (initial := 888) row_count_1702 (by decide)
private theorem atomic_1702_agreement : RowAgreement 1701 1702 := RowAgreement.single (first := 1701) modelrow_agrees_1702
private def atomic_1703 : CountSegment 1 10 13120 1702 1703 888 889 := CountSegment.single (first := 1702) (initial := 888) row_count_1703 (by decide)
private theorem atomic_1703_agreement : RowAgreement 1702 1703 := RowAgreement.single (first := 1702) modelrow_agrees_1703
private def atomic_1704 : CountSegment 1 10 13120 1703 1704 889 890 := CountSegment.single (first := 1703) (initial := 889) row_count_1704 (by decide)
private theorem atomic_1704_agreement : RowAgreement 1703 1704 := RowAgreement.single (first := 1703) modelrow_agrees_1704
private def atomic_1705 : CountSegment 1 10 13120 1704 1705 890 890 := CountSegment.single (first := 1704) (initial := 890) row_count_1705 (by decide)
private theorem atomic_1705_agreement : RowAgreement 1704 1705 := RowAgreement.single (first := 1704) modelrow_agrees_1705
private def atomic_1706 : CountSegment 1 10 13120 1705 1706 890 890 := CountSegment.single (first := 1705) (initial := 890) row_count_1706 (by decide)
private theorem atomic_1706_agreement : RowAgreement 1705 1706 := RowAgreement.single (first := 1705) modelrow_agrees_1706
private def atomic_1707 : CountSegment 1 10 13120 1706 1707 890 890 := CountSegment.single (first := 1706) (initial := 890) row_count_1707 (by decide)
private theorem atomic_1707_agreement : RowAgreement 1706 1707 := RowAgreement.single (first := 1706) modelrow_agrees_1707
private def atomic_1708 : CountSegment 1 10 13120 1707 1708 890 890 := CountSegment.single (first := 1707) (initial := 890) row_count_1708 (by decide)
private theorem atomic_1708_agreement : RowAgreement 1707 1708 := RowAgreement.single (first := 1707) modelrow_agrees_1708
private def atomic_1709 : CountSegment 1 10 13120 1708 1709 890 891 := CountSegment.single (first := 1708) (initial := 890) row_count_1709 (by decide)
private theorem atomic_1709_agreement : RowAgreement 1708 1709 := RowAgreement.single (first := 1708) modelrow_agrees_1709
private def atomic_1710 : CountSegment 1 10 13120 1709 1710 891 892 := CountSegment.single (first := 1709) (initial := 891) row_count_1710 (by decide)
private theorem atomic_1710_agreement : RowAgreement 1709 1710 := RowAgreement.single (first := 1709) modelrow_agrees_1710
private def atomic_1711 : CountSegment 1 10 13120 1710 1711 892 892 := CountSegment.single (first := 1710) (initial := 892) row_count_1711 (by decide)
private theorem atomic_1711_agreement : RowAgreement 1710 1711 := RowAgreement.single (first := 1710) modelrow_agrees_1711
private def atomic_1712 : CountSegment 1 10 13120 1711 1712 892 893 := CountSegment.single (first := 1711) (initial := 892) row_count_1712 (by decide)
private theorem atomic_1712_agreement : RowAgreement 1711 1712 := RowAgreement.single (first := 1711) modelrow_agrees_1712
private def atomic_1713 : CountSegment 1 10 13120 1712 1713 893 893 := CountSegment.single (first := 1712) (initial := 893) row_count_1713 (by decide)
private theorem atomic_1713_agreement : RowAgreement 1712 1713 := RowAgreement.single (first := 1712) modelrow_agrees_1713
private def atomic_1714 : CountSegment 1 10 13120 1713 1714 893 893 := CountSegment.single (first := 1713) (initial := 893) row_count_1714 (by decide)
private theorem atomic_1714_agreement : RowAgreement 1713 1714 := RowAgreement.single (first := 1713) modelrow_agrees_1714
private def atomic_1715 : CountSegment 1 10 13120 1714 1715 893 893 := CountSegment.single (first := 1714) (initial := 893) row_count_1715 (by decide)
private theorem atomic_1715_agreement : RowAgreement 1714 1715 := RowAgreement.single (first := 1714) modelrow_agrees_1715
private def atomic_1716 : CountSegment 1 10 13120 1715 1716 893 893 := CountSegment.single (first := 1715) (initial := 893) row_count_1716 (by decide)
private theorem atomic_1716_agreement : RowAgreement 1715 1716 := RowAgreement.single (first := 1715) modelrow_agrees_1716
private def atomic_1717 : CountSegment 1 10 13120 1716 1717 893 893 := CountSegment.single (first := 1716) (initial := 893) row_count_1717 (by decide)
private theorem atomic_1717_agreement : RowAgreement 1716 1717 := RowAgreement.single (first := 1716) modelrow_agrees_1717
private def atomic_1718 : CountSegment 1 10 13120 1717 1718 893 894 := CountSegment.single (first := 1717) (initial := 893) row_count_1718 (by decide)
private theorem atomic_1718_agreement : RowAgreement 1717 1718 := RowAgreement.single (first := 1717) modelrow_agrees_1718
private def atomic_1719 : CountSegment 1 10 13120 1718 1719 894 896 := CountSegment.single (first := 1718) (initial := 894) row_count_1719 (by decide)
private theorem atomic_1719_agreement : RowAgreement 1718 1719 := RowAgreement.single (first := 1718) modelrow_agrees_1719
private def atomic_1720 : CountSegment 1 10 13120 1719 1720 896 896 := CountSegment.single (first := 1719) (initial := 896) row_count_1720 (by decide)
private theorem atomic_1720_agreement : RowAgreement 1719 1720 := RowAgreement.single (first := 1719) modelrow_agrees_1720
private def atomic_1721 : CountSegment 1 10 13120 1720 1721 896 896 := CountSegment.single (first := 1720) (initial := 896) row_count_1721 (by decide)
private theorem atomic_1721_agreement : RowAgreement 1720 1721 := RowAgreement.single (first := 1720) modelrow_agrees_1721
private def atomic_1722 : CountSegment 1 10 13120 1721 1722 896 898 := CountSegment.single (first := 1721) (initial := 896) row_count_1722 (by decide)
private theorem atomic_1722_agreement : RowAgreement 1721 1722 := RowAgreement.single (first := 1721) modelrow_agrees_1722
private def atomic_1723 : CountSegment 1 10 13120 1722 1723 898 899 := CountSegment.single (first := 1722) (initial := 898) row_count_1723 (by decide)
private theorem atomic_1723_agreement : RowAgreement 1722 1723 := RowAgreement.single (first := 1722) modelrow_agrees_1723
private def atomic_1724 : CountSegment 1 10 13120 1723 1724 899 899 := CountSegment.single (first := 1723) (initial := 899) row_count_1724 (by decide)
private theorem atomic_1724_agreement : RowAgreement 1723 1724 := RowAgreement.single (first := 1723) modelrow_agrees_1724
private def atomic_1725 : CountSegment 1 10 13120 1724 1725 899 900 := CountSegment.single (first := 1724) (initial := 899) row_count_1725 (by decide)
private theorem atomic_1725_agreement : RowAgreement 1724 1725 := RowAgreement.single (first := 1724) modelrow_agrees_1725
private def atomic_1726 : CountSegment 1 10 13120 1725 1726 900 900 := CountSegment.single (first := 1725) (initial := 900) row_count_1726 (by decide)
private theorem atomic_1726_agreement : RowAgreement 1725 1726 := RowAgreement.single (first := 1725) modelrow_agrees_1726
private def atomic_1727 : CountSegment 1 10 13120 1726 1727 900 900 := CountSegment.single (first := 1726) (initial := 900) row_count_1727 (by decide)
private theorem atomic_1727_agreement : RowAgreement 1726 1727 := RowAgreement.single (first := 1726) modelrow_agrees_1727
private def atomic_1728 : CountSegment 1 10 13120 1727 1728 900 900 := CountSegment.single (first := 1727) (initial := 900) row_count_1728 (by decide)
private theorem atomic_1728_agreement : RowAgreement 1727 1728 := RowAgreement.single (first := 1727) modelrow_agrees_1728
private def atomic_1729 : CountSegment 1 10 13120 1728 1729 900 901 := CountSegment.single (first := 1728) (initial := 900) row_count_1729 (by decide)
private theorem atomic_1729_agreement : RowAgreement 1728 1729 := RowAgreement.single (first := 1728) modelrow_agrees_1729
private def atomic_1730 : CountSegment 1 10 13120 1729 1730 901 902 := CountSegment.single (first := 1729) (initial := 901) row_count_1730 (by decide)
private theorem atomic_1730_agreement : RowAgreement 1729 1730 := RowAgreement.single (first := 1729) modelrow_agrees_1730
private def atomic_1731 : CountSegment 1 10 13120 1730 1731 902 902 := CountSegment.single (first := 1730) (initial := 902) row_count_1731 (by decide)
private theorem atomic_1731_agreement : RowAgreement 1730 1731 := RowAgreement.single (first := 1730) modelrow_agrees_1731
private def atomic_1732 : CountSegment 1 10 13120 1731 1732 902 903 := CountSegment.single (first := 1731) (initial := 902) row_count_1732 (by decide)
private theorem atomic_1732_agreement : RowAgreement 1731 1732 := RowAgreement.single (first := 1731) modelrow_agrees_1732
private def atomic_1733 : CountSegment 1 10 13120 1732 1733 903 903 := CountSegment.single (first := 1732) (initial := 903) row_count_1733 (by decide)
private theorem atomic_1733_agreement : RowAgreement 1732 1733 := RowAgreement.single (first := 1732) modelrow_agrees_1733
private def atomic_1734 : CountSegment 1 10 13120 1733 1734 903 903 := CountSegment.single (first := 1733) (initial := 903) row_count_1734 (by decide)
private theorem atomic_1734_agreement : RowAgreement 1733 1734 := RowAgreement.single (first := 1733) modelrow_agrees_1734
private def atomic_1735 : CountSegment 1 10 13120 1734 1735 903 904 := CountSegment.single (first := 1734) (initial := 903) row_count_1735 (by decide)
private theorem atomic_1735_agreement : RowAgreement 1734 1735 := RowAgreement.single (first := 1734) modelrow_agrees_1735
private def atomic_1736 : CountSegment 1 10 13120 1735 1736 904 904 := CountSegment.single (first := 1735) (initial := 904) row_count_1736 (by decide)
private theorem atomic_1736_agreement : RowAgreement 1735 1736 := RowAgreement.single (first := 1735) modelrow_agrees_1736
private def atomic_1737 : CountSegment 1 10 13120 1736 1737 904 906 := CountSegment.single (first := 1736) (initial := 904) row_count_1737 (by decide)
private theorem atomic_1737_agreement : RowAgreement 1736 1737 := RowAgreement.single (first := 1736) modelrow_agrees_1737
private def atomic_1738 : CountSegment 1 10 13120 1737 1738 906 906 := CountSegment.single (first := 1737) (initial := 906) row_count_1738 (by decide)
private theorem atomic_1738_agreement : RowAgreement 1737 1738 := RowAgreement.single (first := 1737) modelrow_agrees_1738
private def atomic_1739 : CountSegment 1 10 13120 1738 1739 906 906 := CountSegment.single (first := 1738) (initial := 906) row_count_1739 (by decide)
private theorem atomic_1739_agreement : RowAgreement 1738 1739 := RowAgreement.single (first := 1738) modelrow_agrees_1739
private def atomic_1740 : CountSegment 1 10 13120 1739 1740 906 908 := CountSegment.single (first := 1739) (initial := 906) row_count_1740 (by decide)
private theorem atomic_1740_agreement : RowAgreement 1739 1740 := RowAgreement.single (first := 1739) modelrow_agrees_1740
private def atomic_1741 : CountSegment 1 10 13120 1740 1741 908 909 := CountSegment.single (first := 1740) (initial := 908) row_count_1741 (by decide)
private theorem atomic_1741_agreement : RowAgreement 1740 1741 := RowAgreement.single (first := 1740) modelrow_agrees_1741
private def atomic_1742 : CountSegment 1 10 13120 1741 1742 909 909 := CountSegment.single (first := 1741) (initial := 909) row_count_1742 (by decide)
private theorem atomic_1742_agreement : RowAgreement 1741 1742 := RowAgreement.single (first := 1741) modelrow_agrees_1742
private def atomic_1743 : CountSegment 1 10 13120 1742 1743 909 910 := CountSegment.single (first := 1742) (initial := 909) row_count_1743 (by decide)
private theorem atomic_1743_agreement : RowAgreement 1742 1743 := RowAgreement.single (first := 1742) modelrow_agrees_1743
private def atomic_1744 : CountSegment 1 10 13120 1743 1744 910 910 := CountSegment.single (first := 1743) (initial := 910) row_count_1744 (by decide)
private theorem atomic_1744_agreement : RowAgreement 1743 1744 := RowAgreement.single (first := 1743) modelrow_agrees_1744
private def atomic_1745 : CountSegment 1 10 13120 1744 1745 910 911 := CountSegment.single (first := 1744) (initial := 910) row_count_1745 (by decide)
private theorem atomic_1745_agreement : RowAgreement 1744 1745 := RowAgreement.single (first := 1744) modelrow_agrees_1745
private def atomic_1746 : CountSegment 1 10 13120 1745 1746 911 913 := CountSegment.single (first := 1745) (initial := 911) row_count_1746 (by decide)
private theorem atomic_1746_agreement : RowAgreement 1745 1746 := RowAgreement.single (first := 1745) modelrow_agrees_1746
private def atomic_1747 : CountSegment 1 10 13120 1746 1747 913 913 := CountSegment.single (first := 1746) (initial := 913) row_count_1747 (by decide)
private theorem atomic_1747_agreement : RowAgreement 1746 1747 := RowAgreement.single (first := 1746) modelrow_agrees_1747
private def atomic_1748 : CountSegment 1 10 13120 1747 1748 913 913 := CountSegment.single (first := 1747) (initial := 913) row_count_1748 (by decide)
private theorem atomic_1748_agreement : RowAgreement 1747 1748 := RowAgreement.single (first := 1747) modelrow_agrees_1748
private def atomic_1749 : CountSegment 1 10 13120 1748 1749 913 913 := CountSegment.single (first := 1748) (initial := 913) row_count_1749 (by decide)
private theorem atomic_1749_agreement : RowAgreement 1748 1749 := RowAgreement.single (first := 1748) modelrow_agrees_1749
private def atomic_1750 : CountSegment 1 10 13120 1749 1750 913 913 := CountSegment.single (first := 1749) (initial := 913) row_count_1750 (by decide)
private theorem atomic_1750_agreement : RowAgreement 1749 1750 := RowAgreement.single (first := 1749) modelrow_agrees_1750
private def atomic_1751 : CountSegment 1 10 13120 1750 1751 913 914 := CountSegment.single (first := 1750) (initial := 913) row_count_1751 (by decide)
private theorem atomic_1751_agreement : RowAgreement 1750 1751 := RowAgreement.single (first := 1750) modelrow_agrees_1751
private def atomic_1752 : CountSegment 1 10 13120 1751 1752 914 915 := CountSegment.single (first := 1751) (initial := 914) row_count_1752 (by decide)
private theorem atomic_1752_agreement : RowAgreement 1751 1752 := RowAgreement.single (first := 1751) modelrow_agrees_1752
private def atomic_1753 : CountSegment 1 10 13120 1752 1753 915 915 := CountSegment.single (first := 1752) (initial := 915) row_count_1753 (by decide)
private theorem atomic_1753_agreement : RowAgreement 1752 1753 := RowAgreement.single (first := 1752) modelrow_agrees_1753
private def atomic_1754 : CountSegment 1 10 13120 1753 1754 915 915 := CountSegment.single (first := 1753) (initial := 915) row_count_1754 (by decide)
private theorem atomic_1754_agreement : RowAgreement 1753 1754 := RowAgreement.single (first := 1753) modelrow_agrees_1754
private def atomic_1755 : CountSegment 1 10 13120 1754 1755 915 916 := CountSegment.single (first := 1754) (initial := 915) row_count_1755 (by decide)
private theorem atomic_1755_agreement : RowAgreement 1754 1755 := RowAgreement.single (first := 1754) modelrow_agrees_1755
private def atomic_1756 : CountSegment 1 10 13120 1755 1756 916 916 := CountSegment.single (first := 1755) (initial := 916) row_count_1756 (by decide)
private theorem atomic_1756_agreement : RowAgreement 1755 1756 := RowAgreement.single (first := 1755) modelrow_agrees_1756
private def atomic_1757 : CountSegment 1 10 13120 1756 1757 916 916 := CountSegment.single (first := 1756) (initial := 916) row_count_1757 (by decide)
private theorem atomic_1757_agreement : RowAgreement 1756 1757 := RowAgreement.single (first := 1756) modelrow_agrees_1757
private def atomic_1758 : CountSegment 1 10 13120 1757 1758 916 918 := CountSegment.single (first := 1757) (initial := 916) row_count_1758 (by decide)
private theorem atomic_1758_agreement : RowAgreement 1757 1758 := RowAgreement.single (first := 1757) modelrow_agrees_1758
private def atomic_1759 : CountSegment 1 10 13120 1758 1759 918 918 := CountSegment.single (first := 1758) (initial := 918) row_count_1759 (by decide)
private theorem atomic_1759_agreement : RowAgreement 1758 1759 := RowAgreement.single (first := 1758) modelrow_agrees_1759
private def atomic_1760 : CountSegment 1 10 13120 1759 1760 918 919 := CountSegment.single (first := 1759) (initial := 918) row_count_1760 (by decide)
private theorem atomic_1760_agreement : RowAgreement 1759 1760 := RowAgreement.single (first := 1759) modelrow_agrees_1760
private def atomic_1761 : CountSegment 1 10 13120 1760 1761 919 919 := CountSegment.single (first := 1760) (initial := 919) row_count_1761 (by decide)
private theorem atomic_1761_agreement : RowAgreement 1760 1761 := RowAgreement.single (first := 1760) modelrow_agrees_1761
private def atomic_1762 : CountSegment 1 10 13120 1761 1762 919 919 := CountSegment.single (first := 1761) (initial := 919) row_count_1762 (by decide)
private theorem atomic_1762_agreement : RowAgreement 1761 1762 := RowAgreement.single (first := 1761) modelrow_agrees_1762
private def atomic_1763 : CountSegment 1 10 13120 1762 1763 919 919 := CountSegment.single (first := 1762) (initial := 919) row_count_1763 (by decide)
private theorem atomic_1763_agreement : RowAgreement 1762 1763 := RowAgreement.single (first := 1762) modelrow_agrees_1763
private def atomic_1764 : CountSegment 1 10 13120 1763 1764 919 919 := CountSegment.single (first := 1763) (initial := 919) row_count_1764 (by decide)
private theorem atomic_1764_agreement : RowAgreement 1763 1764 := RowAgreement.single (first := 1763) modelrow_agrees_1764
private def atomic_1765 : CountSegment 1 10 13120 1764 1765 919 920 := CountSegment.single (first := 1764) (initial := 919) row_count_1765 (by decide)
private theorem atomic_1765_agreement : RowAgreement 1764 1765 := RowAgreement.single (first := 1764) modelrow_agrees_1765
private def atomic_1766 : CountSegment 1 10 13120 1765 1766 920 920 := CountSegment.single (first := 1765) (initial := 920) row_count_1766 (by decide)
private theorem atomic_1766_agreement : RowAgreement 1765 1766 := RowAgreement.single (first := 1765) modelrow_agrees_1766
private def atomic_1767 : CountSegment 1 10 13120 1766 1767 920 922 := CountSegment.single (first := 1766) (initial := 920) row_count_1767 (by decide)
private theorem atomic_1767_agreement : RowAgreement 1766 1767 := RowAgreement.single (first := 1766) modelrow_agrees_1767
private def atomic_1768 : CountSegment 1 10 13120 1767 1768 922 923 := CountSegment.single (first := 1767) (initial := 922) row_count_1768 (by decide)
private theorem atomic_1768_agreement : RowAgreement 1767 1768 := RowAgreement.single (first := 1767) modelrow_agrees_1768
private def atomic_1769 : CountSegment 1 10 13120 1768 1769 923 923 := CountSegment.single (first := 1768) (initial := 923) row_count_1769 (by decide)
private theorem atomic_1769_agreement : RowAgreement 1768 1769 := RowAgreement.single (first := 1768) modelrow_agrees_1769
private def atomic_1770 : CountSegment 1 10 13120 1769 1770 923 924 := CountSegment.single (first := 1769) (initial := 923) row_count_1770 (by decide)
private theorem atomic_1770_agreement : RowAgreement 1769 1770 := RowAgreement.single (first := 1769) modelrow_agrees_1770
private def atomic_1771 : CountSegment 1 10 13120 1770 1771 924 925 := CountSegment.single (first := 1770) (initial := 924) row_count_1771 (by decide)
private theorem atomic_1771_agreement : RowAgreement 1770 1771 := RowAgreement.single (first := 1770) modelrow_agrees_1771
private def atomic_1772 : CountSegment 1 10 13120 1771 1772 925 926 := CountSegment.single (first := 1771) (initial := 925) row_count_1772 (by decide)
private theorem atomic_1772_agreement : RowAgreement 1771 1772 := RowAgreement.single (first := 1771) modelrow_agrees_1772
private def atomic_1773 : CountSegment 1 10 13120 1772 1773 926 926 := CountSegment.single (first := 1772) (initial := 926) row_count_1773 (by decide)
private theorem atomic_1773_agreement : RowAgreement 1772 1773 := RowAgreement.single (first := 1772) modelrow_agrees_1773
private def atomic_1774 : CountSegment 1 10 13120 1773 1774 926 926 := CountSegment.single (first := 1773) (initial := 926) row_count_1774 (by decide)
private theorem atomic_1774_agreement : RowAgreement 1773 1774 := RowAgreement.single (first := 1773) modelrow_agrees_1774
private def atomic_1775 : CountSegment 1 10 13120 1774 1775 926 927 := CountSegment.single (first := 1774) (initial := 926) row_count_1775 (by decide)
private theorem atomic_1775_agreement : RowAgreement 1774 1775 := RowAgreement.single (first := 1774) modelrow_agrees_1775
private def atomic_1776 : CountSegment 1 10 13120 1775 1776 927 927 := CountSegment.single (first := 1775) (initial := 927) row_count_1776 (by decide)
private theorem atomic_1776_agreement : RowAgreement 1775 1776 := RowAgreement.single (first := 1775) modelrow_agrees_1776
private def atomic_1777 : CountSegment 1 10 13120 1776 1777 927 928 := CountSegment.single (first := 1776) (initial := 927) row_count_1777 (by decide)
private theorem atomic_1777_agreement : RowAgreement 1776 1777 := RowAgreement.single (first := 1776) modelrow_agrees_1777
private def atomic_1778 : CountSegment 1 10 13120 1777 1778 928 928 := CountSegment.single (first := 1777) (initial := 928) row_count_1778 (by decide)
private theorem atomic_1778_agreement : RowAgreement 1777 1778 := RowAgreement.single (first := 1777) modelrow_agrees_1778
private def atomic_1779 : CountSegment 1 10 13120 1778 1779 928 929 := CountSegment.single (first := 1778) (initial := 928) row_count_1779 (by decide)
private theorem atomic_1779_agreement : RowAgreement 1778 1779 := RowAgreement.single (first := 1778) modelrow_agrees_1779
private def atomic_1780 : CountSegment 1 10 13120 1779 1780 929 929 := CountSegment.single (first := 1779) (initial := 929) row_count_1780 (by decide)
private theorem atomic_1780_agreement : RowAgreement 1779 1780 := RowAgreement.single (first := 1779) modelrow_agrees_1780
private def atomic_1781 : CountSegment 1 10 13120 1780 1781 929 930 := CountSegment.single (first := 1780) (initial := 929) row_count_1781 (by decide)
private theorem atomic_1781_agreement : RowAgreement 1780 1781 := RowAgreement.single (first := 1780) modelrow_agrees_1781
private def atomic_1782 : CountSegment 1 10 13120 1781 1782 930 931 := CountSegment.single (first := 1781) (initial := 930) row_count_1782 (by decide)
private theorem atomic_1782_agreement : RowAgreement 1781 1782 := RowAgreement.single (first := 1781) modelrow_agrees_1782
private def atomic_1783 : CountSegment 1 10 13120 1782 1783 931 931 := CountSegment.single (first := 1782) (initial := 931) row_count_1783 (by decide)
private theorem atomic_1783_agreement : RowAgreement 1782 1783 := RowAgreement.single (first := 1782) modelrow_agrees_1783
private def atomic_1784 : CountSegment 1 10 13120 1783 1784 931 931 := CountSegment.single (first := 1783) (initial := 931) row_count_1784 (by decide)
private theorem atomic_1784_agreement : RowAgreement 1783 1784 := RowAgreement.single (first := 1783) modelrow_agrees_1784
private def atomic_1785 : CountSegment 1 10 13120 1784 1785 931 931 := CountSegment.single (first := 1784) (initial := 931) row_count_1785 (by decide)
private theorem atomic_1785_agreement : RowAgreement 1784 1785 := RowAgreement.single (first := 1784) modelrow_agrees_1785
private def atomic_1786 : CountSegment 1 10 13120 1785 1786 931 931 := CountSegment.single (first := 1785) (initial := 931) row_count_1786 (by decide)
private theorem atomic_1786_agreement : RowAgreement 1785 1786 := RowAgreement.single (first := 1785) modelrow_agrees_1786
private def atomic_1787 : CountSegment 1 10 13120 1786 1787 931 932 := CountSegment.single (first := 1786) (initial := 931) row_count_1787 (by decide)
private theorem atomic_1787_agreement : RowAgreement 1786 1787 := RowAgreement.single (first := 1786) modelrow_agrees_1787
private def atomic_1788 : CountSegment 1 10 13120 1787 1788 932 933 := CountSegment.single (first := 1787) (initial := 932) row_count_1788 (by decide)
private theorem atomic_1788_agreement : RowAgreement 1787 1788 := RowAgreement.single (first := 1787) modelrow_agrees_1788
private def atomic_1789 : CountSegment 1 10 13120 1788 1789 933 933 := CountSegment.single (first := 1788) (initial := 933) row_count_1789 (by decide)
private theorem atomic_1789_agreement : RowAgreement 1788 1789 := RowAgreement.single (first := 1788) modelrow_agrees_1789
private def atomic_1790 : CountSegment 1 10 13120 1789 1790 933 933 := CountSegment.single (first := 1789) (initial := 933) row_count_1790 (by decide)
private theorem atomic_1790_agreement : RowAgreement 1789 1790 := RowAgreement.single (first := 1789) modelrow_agrees_1790
private def atomic_1791 : CountSegment 1 10 13120 1790 1791 933 933 := CountSegment.single (first := 1790) (initial := 933) row_count_1791 (by decide)
private theorem atomic_1791_agreement : RowAgreement 1790 1791 := RowAgreement.single (first := 1790) modelrow_agrees_1791
private def atomic_1792 : CountSegment 1 10 13120 1791 1792 933 933 := CountSegment.single (first := 1791) (initial := 933) row_count_1792 (by decide)
private theorem atomic_1792_agreement : RowAgreement 1791 1792 := RowAgreement.single (first := 1791) modelrow_agrees_1792
private def within_013_000 : CountSegment 1 10 13120 1664 1666 873 874 := CountSegment.join atomic_1665 atomic_1666
private theorem within_013_000_agreement : RowAgreement 1664 1666 := RowAgreement.join atomic_1665_agreement atomic_1666_agreement
private def within_013_001 : CountSegment 1 10 13120 1666 1668 874 875 := CountSegment.join atomic_1667 atomic_1668
private theorem within_013_001_agreement : RowAgreement 1666 1668 := RowAgreement.join atomic_1667_agreement atomic_1668_agreement
private def within_013_002 : CountSegment 1 10 13120 1664 1668 873 875 := CountSegment.join within_013_000 within_013_001
private theorem within_013_002_agreement : RowAgreement 1664 1668 := RowAgreement.join within_013_000_agreement within_013_001_agreement
private def within_013_003 : CountSegment 1 10 13120 1668 1670 875 878 := CountSegment.join atomic_1669 atomic_1670
private theorem within_013_003_agreement : RowAgreement 1668 1670 := RowAgreement.join atomic_1669_agreement atomic_1670_agreement
private def within_013_004 : CountSegment 1 10 13120 1670 1672 878 878 := CountSegment.join atomic_1671 atomic_1672
private theorem within_013_004_agreement : RowAgreement 1670 1672 := RowAgreement.join atomic_1671_agreement atomic_1672_agreement
private def within_013_005 : CountSegment 1 10 13120 1668 1672 875 878 := CountSegment.join within_013_003 within_013_004
private theorem within_013_005_agreement : RowAgreement 1668 1672 := RowAgreement.join within_013_003_agreement within_013_004_agreement
private def within_013_006 : CountSegment 1 10 13120 1664 1672 873 878 := CountSegment.join within_013_002 within_013_005
private theorem within_013_006_agreement : RowAgreement 1664 1672 := RowAgreement.join within_013_002_agreement within_013_005_agreement
private def within_013_007 : CountSegment 1 10 13120 1672 1674 878 879 := CountSegment.join atomic_1673 atomic_1674
private theorem within_013_007_agreement : RowAgreement 1672 1674 := RowAgreement.join atomic_1673_agreement atomic_1674_agreement
private def within_013_008 : CountSegment 1 10 13120 1674 1676 879 880 := CountSegment.join atomic_1675 atomic_1676
private theorem within_013_008_agreement : RowAgreement 1674 1676 := RowAgreement.join atomic_1675_agreement atomic_1676_agreement
private def within_013_009 : CountSegment 1 10 13120 1672 1676 878 880 := CountSegment.join within_013_007 within_013_008
private theorem within_013_009_agreement : RowAgreement 1672 1676 := RowAgreement.join within_013_007_agreement within_013_008_agreement
private def within_013_010 : CountSegment 1 10 13120 1676 1678 880 881 := CountSegment.join atomic_1677 atomic_1678
private theorem within_013_010_agreement : RowAgreement 1676 1678 := RowAgreement.join atomic_1677_agreement atomic_1678_agreement
private def within_013_011 : CountSegment 1 10 13120 1678 1680 881 882 := CountSegment.join atomic_1679 atomic_1680
private theorem within_013_011_agreement : RowAgreement 1678 1680 := RowAgreement.join atomic_1679_agreement atomic_1680_agreement
private def within_013_012 : CountSegment 1 10 13120 1676 1680 880 882 := CountSegment.join within_013_010 within_013_011
private theorem within_013_012_agreement : RowAgreement 1676 1680 := RowAgreement.join within_013_010_agreement within_013_011_agreement
private def within_013_013 : CountSegment 1 10 13120 1672 1680 878 882 := CountSegment.join within_013_009 within_013_012
private theorem within_013_013_agreement : RowAgreement 1672 1680 := RowAgreement.join within_013_009_agreement within_013_012_agreement
private def within_013_014 : CountSegment 1 10 13120 1664 1680 873 882 := CountSegment.join within_013_006 within_013_013
private theorem within_013_014_agreement : RowAgreement 1664 1680 := RowAgreement.join within_013_006_agreement within_013_013_agreement
private def within_013_015 : CountSegment 1 10 13120 1680 1682 882 882 := CountSegment.join atomic_1681 atomic_1682
private theorem within_013_015_agreement : RowAgreement 1680 1682 := RowAgreement.join atomic_1681_agreement atomic_1682_agreement
private def within_013_016 : CountSegment 1 10 13120 1682 1684 882 882 := CountSegment.join atomic_1683 atomic_1684
private theorem within_013_016_agreement : RowAgreement 1682 1684 := RowAgreement.join atomic_1683_agreement atomic_1684_agreement
private def within_013_017 : CountSegment 1 10 13120 1680 1684 882 882 := CountSegment.join within_013_015 within_013_016
private theorem within_013_017_agreement : RowAgreement 1680 1684 := RowAgreement.join within_013_015_agreement within_013_016_agreement
private def within_013_018 : CountSegment 1 10 13120 1684 1686 882 882 := CountSegment.join atomic_1685 atomic_1686
private theorem within_013_018_agreement : RowAgreement 1684 1686 := RowAgreement.join atomic_1685_agreement atomic_1686_agreement
private def within_013_019 : CountSegment 1 10 13120 1686 1688 882 883 := CountSegment.join atomic_1687 atomic_1688
private theorem within_013_019_agreement : RowAgreement 1686 1688 := RowAgreement.join atomic_1687_agreement atomic_1688_agreement
private def within_013_020 : CountSegment 1 10 13120 1684 1688 882 883 := CountSegment.join within_013_018 within_013_019
private theorem within_013_020_agreement : RowAgreement 1684 1688 := RowAgreement.join within_013_018_agreement within_013_019_agreement
private def within_013_021 : CountSegment 1 10 13120 1680 1688 882 883 := CountSegment.join within_013_017 within_013_020
private theorem within_013_021_agreement : RowAgreement 1680 1688 := RowAgreement.join within_013_017_agreement within_013_020_agreement
private def within_013_022 : CountSegment 1 10 13120 1688 1690 883 884 := CountSegment.join atomic_1689 atomic_1690
private theorem within_013_022_agreement : RowAgreement 1688 1690 := RowAgreement.join atomic_1689_agreement atomic_1690_agreement
private def within_013_023 : CountSegment 1 10 13120 1690 1692 884 885 := CountSegment.join atomic_1691 atomic_1692
private theorem within_013_023_agreement : RowAgreement 1690 1692 := RowAgreement.join atomic_1691_agreement atomic_1692_agreement
private def within_013_024 : CountSegment 1 10 13120 1688 1692 883 885 := CountSegment.join within_013_022 within_013_023
private theorem within_013_024_agreement : RowAgreement 1688 1692 := RowAgreement.join within_013_022_agreement within_013_023_agreement
private def within_013_025 : CountSegment 1 10 13120 1692 1694 885 885 := CountSegment.join atomic_1693 atomic_1694
private theorem within_013_025_agreement : RowAgreement 1692 1694 := RowAgreement.join atomic_1693_agreement atomic_1694_agreement
private def within_013_026 : CountSegment 1 10 13120 1694 1696 885 886 := CountSegment.join atomic_1695 atomic_1696
private theorem within_013_026_agreement : RowAgreement 1694 1696 := RowAgreement.join atomic_1695_agreement atomic_1696_agreement
private def within_013_027 : CountSegment 1 10 13120 1692 1696 885 886 := CountSegment.join within_013_025 within_013_026
private theorem within_013_027_agreement : RowAgreement 1692 1696 := RowAgreement.join within_013_025_agreement within_013_026_agreement
private def within_013_028 : CountSegment 1 10 13120 1688 1696 883 886 := CountSegment.join within_013_024 within_013_027
private theorem within_013_028_agreement : RowAgreement 1688 1696 := RowAgreement.join within_013_024_agreement within_013_027_agreement
private def within_013_029 : CountSegment 1 10 13120 1680 1696 882 886 := CountSegment.join within_013_021 within_013_028
private theorem within_013_029_agreement : RowAgreement 1680 1696 := RowAgreement.join within_013_021_agreement within_013_028_agreement
private def within_013_030 : CountSegment 1 10 13120 1664 1696 873 886 := CountSegment.join within_013_014 within_013_029
private theorem within_013_030_agreement : RowAgreement 1664 1696 := RowAgreement.join within_013_014_agreement within_013_029_agreement
private def within_013_031 : CountSegment 1 10 13120 1696 1698 886 887 := CountSegment.join atomic_1697 atomic_1698
private theorem within_013_031_agreement : RowAgreement 1696 1698 := RowAgreement.join atomic_1697_agreement atomic_1698_agreement
private def within_013_032 : CountSegment 1 10 13120 1698 1700 887 888 := CountSegment.join atomic_1699 atomic_1700
private theorem within_013_032_agreement : RowAgreement 1698 1700 := RowAgreement.join atomic_1699_agreement atomic_1700_agreement
private def within_013_033 : CountSegment 1 10 13120 1696 1700 886 888 := CountSegment.join within_013_031 within_013_032
private theorem within_013_033_agreement : RowAgreement 1696 1700 := RowAgreement.join within_013_031_agreement within_013_032_agreement
private def within_013_034 : CountSegment 1 10 13120 1700 1702 888 888 := CountSegment.join atomic_1701 atomic_1702
private theorem within_013_034_agreement : RowAgreement 1700 1702 := RowAgreement.join atomic_1701_agreement atomic_1702_agreement
private def within_013_035 : CountSegment 1 10 13120 1702 1704 888 890 := CountSegment.join atomic_1703 atomic_1704
private theorem within_013_035_agreement : RowAgreement 1702 1704 := RowAgreement.join atomic_1703_agreement atomic_1704_agreement
private def within_013_036 : CountSegment 1 10 13120 1700 1704 888 890 := CountSegment.join within_013_034 within_013_035
private theorem within_013_036_agreement : RowAgreement 1700 1704 := RowAgreement.join within_013_034_agreement within_013_035_agreement
private def within_013_037 : CountSegment 1 10 13120 1696 1704 886 890 := CountSegment.join within_013_033 within_013_036
private theorem within_013_037_agreement : RowAgreement 1696 1704 := RowAgreement.join within_013_033_agreement within_013_036_agreement
private def within_013_038 : CountSegment 1 10 13120 1704 1706 890 890 := CountSegment.join atomic_1705 atomic_1706
private theorem within_013_038_agreement : RowAgreement 1704 1706 := RowAgreement.join atomic_1705_agreement atomic_1706_agreement
private def within_013_039 : CountSegment 1 10 13120 1706 1708 890 890 := CountSegment.join atomic_1707 atomic_1708
private theorem within_013_039_agreement : RowAgreement 1706 1708 := RowAgreement.join atomic_1707_agreement atomic_1708_agreement
private def within_013_040 : CountSegment 1 10 13120 1704 1708 890 890 := CountSegment.join within_013_038 within_013_039
private theorem within_013_040_agreement : RowAgreement 1704 1708 := RowAgreement.join within_013_038_agreement within_013_039_agreement
private def within_013_041 : CountSegment 1 10 13120 1708 1710 890 892 := CountSegment.join atomic_1709 atomic_1710
private theorem within_013_041_agreement : RowAgreement 1708 1710 := RowAgreement.join atomic_1709_agreement atomic_1710_agreement
private def within_013_042 : CountSegment 1 10 13120 1710 1712 892 893 := CountSegment.join atomic_1711 atomic_1712
private theorem within_013_042_agreement : RowAgreement 1710 1712 := RowAgreement.join atomic_1711_agreement atomic_1712_agreement
private def within_013_043 : CountSegment 1 10 13120 1708 1712 890 893 := CountSegment.join within_013_041 within_013_042
private theorem within_013_043_agreement : RowAgreement 1708 1712 := RowAgreement.join within_013_041_agreement within_013_042_agreement
private def within_013_044 : CountSegment 1 10 13120 1704 1712 890 893 := CountSegment.join within_013_040 within_013_043
private theorem within_013_044_agreement : RowAgreement 1704 1712 := RowAgreement.join within_013_040_agreement within_013_043_agreement
private def within_013_045 : CountSegment 1 10 13120 1696 1712 886 893 := CountSegment.join within_013_037 within_013_044
private theorem within_013_045_agreement : RowAgreement 1696 1712 := RowAgreement.join within_013_037_agreement within_013_044_agreement
private def within_013_046 : CountSegment 1 10 13120 1712 1714 893 893 := CountSegment.join atomic_1713 atomic_1714
private theorem within_013_046_agreement : RowAgreement 1712 1714 := RowAgreement.join atomic_1713_agreement atomic_1714_agreement
private def within_013_047 : CountSegment 1 10 13120 1714 1716 893 893 := CountSegment.join atomic_1715 atomic_1716
private theorem within_013_047_agreement : RowAgreement 1714 1716 := RowAgreement.join atomic_1715_agreement atomic_1716_agreement
private def within_013_048 : CountSegment 1 10 13120 1712 1716 893 893 := CountSegment.join within_013_046 within_013_047
private theorem within_013_048_agreement : RowAgreement 1712 1716 := RowAgreement.join within_013_046_agreement within_013_047_agreement
private def within_013_049 : CountSegment 1 10 13120 1716 1718 893 894 := CountSegment.join atomic_1717 atomic_1718
private theorem within_013_049_agreement : RowAgreement 1716 1718 := RowAgreement.join atomic_1717_agreement atomic_1718_agreement
private def within_013_050 : CountSegment 1 10 13120 1718 1720 894 896 := CountSegment.join atomic_1719 atomic_1720
private theorem within_013_050_agreement : RowAgreement 1718 1720 := RowAgreement.join atomic_1719_agreement atomic_1720_agreement
private def within_013_051 : CountSegment 1 10 13120 1716 1720 893 896 := CountSegment.join within_013_049 within_013_050
private theorem within_013_051_agreement : RowAgreement 1716 1720 := RowAgreement.join within_013_049_agreement within_013_050_agreement
private def within_013_052 : CountSegment 1 10 13120 1712 1720 893 896 := CountSegment.join within_013_048 within_013_051
private theorem within_013_052_agreement : RowAgreement 1712 1720 := RowAgreement.join within_013_048_agreement within_013_051_agreement
private def within_013_053 : CountSegment 1 10 13120 1720 1722 896 898 := CountSegment.join atomic_1721 atomic_1722
private theorem within_013_053_agreement : RowAgreement 1720 1722 := RowAgreement.join atomic_1721_agreement atomic_1722_agreement
private def within_013_054 : CountSegment 1 10 13120 1722 1724 898 899 := CountSegment.join atomic_1723 atomic_1724
private theorem within_013_054_agreement : RowAgreement 1722 1724 := RowAgreement.join atomic_1723_agreement atomic_1724_agreement
private def within_013_055 : CountSegment 1 10 13120 1720 1724 896 899 := CountSegment.join within_013_053 within_013_054
private theorem within_013_055_agreement : RowAgreement 1720 1724 := RowAgreement.join within_013_053_agreement within_013_054_agreement
private def within_013_056 : CountSegment 1 10 13120 1724 1726 899 900 := CountSegment.join atomic_1725 atomic_1726
private theorem within_013_056_agreement : RowAgreement 1724 1726 := RowAgreement.join atomic_1725_agreement atomic_1726_agreement
private def within_013_057 : CountSegment 1 10 13120 1726 1728 900 900 := CountSegment.join atomic_1727 atomic_1728
private theorem within_013_057_agreement : RowAgreement 1726 1728 := RowAgreement.join atomic_1727_agreement atomic_1728_agreement
private def within_013_058 : CountSegment 1 10 13120 1724 1728 899 900 := CountSegment.join within_013_056 within_013_057
private theorem within_013_058_agreement : RowAgreement 1724 1728 := RowAgreement.join within_013_056_agreement within_013_057_agreement
private def within_013_059 : CountSegment 1 10 13120 1720 1728 896 900 := CountSegment.join within_013_055 within_013_058
private theorem within_013_059_agreement : RowAgreement 1720 1728 := RowAgreement.join within_013_055_agreement within_013_058_agreement
private def within_013_060 : CountSegment 1 10 13120 1712 1728 893 900 := CountSegment.join within_013_052 within_013_059
private theorem within_013_060_agreement : RowAgreement 1712 1728 := RowAgreement.join within_013_052_agreement within_013_059_agreement
private def within_013_061 : CountSegment 1 10 13120 1696 1728 886 900 := CountSegment.join within_013_045 within_013_060
private theorem within_013_061_agreement : RowAgreement 1696 1728 := RowAgreement.join within_013_045_agreement within_013_060_agreement
private def within_013_062 : CountSegment 1 10 13120 1664 1728 873 900 := CountSegment.join within_013_030 within_013_061
private theorem within_013_062_agreement : RowAgreement 1664 1728 := RowAgreement.join within_013_030_agreement within_013_061_agreement
private def within_013_063 : CountSegment 1 10 13120 1728 1730 900 902 := CountSegment.join atomic_1729 atomic_1730
private theorem within_013_063_agreement : RowAgreement 1728 1730 := RowAgreement.join atomic_1729_agreement atomic_1730_agreement
private def within_013_064 : CountSegment 1 10 13120 1730 1732 902 903 := CountSegment.join atomic_1731 atomic_1732
private theorem within_013_064_agreement : RowAgreement 1730 1732 := RowAgreement.join atomic_1731_agreement atomic_1732_agreement
private def within_013_065 : CountSegment 1 10 13120 1728 1732 900 903 := CountSegment.join within_013_063 within_013_064
private theorem within_013_065_agreement : RowAgreement 1728 1732 := RowAgreement.join within_013_063_agreement within_013_064_agreement
private def within_013_066 : CountSegment 1 10 13120 1732 1734 903 903 := CountSegment.join atomic_1733 atomic_1734
private theorem within_013_066_agreement : RowAgreement 1732 1734 := RowAgreement.join atomic_1733_agreement atomic_1734_agreement
private def within_013_067 : CountSegment 1 10 13120 1734 1736 903 904 := CountSegment.join atomic_1735 atomic_1736
private theorem within_013_067_agreement : RowAgreement 1734 1736 := RowAgreement.join atomic_1735_agreement atomic_1736_agreement
private def within_013_068 : CountSegment 1 10 13120 1732 1736 903 904 := CountSegment.join within_013_066 within_013_067
private theorem within_013_068_agreement : RowAgreement 1732 1736 := RowAgreement.join within_013_066_agreement within_013_067_agreement
private def within_013_069 : CountSegment 1 10 13120 1728 1736 900 904 := CountSegment.join within_013_065 within_013_068
private theorem within_013_069_agreement : RowAgreement 1728 1736 := RowAgreement.join within_013_065_agreement within_013_068_agreement
private def within_013_070 : CountSegment 1 10 13120 1736 1738 904 906 := CountSegment.join atomic_1737 atomic_1738
private theorem within_013_070_agreement : RowAgreement 1736 1738 := RowAgreement.join atomic_1737_agreement atomic_1738_agreement
private def within_013_071 : CountSegment 1 10 13120 1738 1740 906 908 := CountSegment.join atomic_1739 atomic_1740
private theorem within_013_071_agreement : RowAgreement 1738 1740 := RowAgreement.join atomic_1739_agreement atomic_1740_agreement
private def within_013_072 : CountSegment 1 10 13120 1736 1740 904 908 := CountSegment.join within_013_070 within_013_071
private theorem within_013_072_agreement : RowAgreement 1736 1740 := RowAgreement.join within_013_070_agreement within_013_071_agreement
private def within_013_073 : CountSegment 1 10 13120 1740 1742 908 909 := CountSegment.join atomic_1741 atomic_1742
private theorem within_013_073_agreement : RowAgreement 1740 1742 := RowAgreement.join atomic_1741_agreement atomic_1742_agreement
private def within_013_074 : CountSegment 1 10 13120 1742 1744 909 910 := CountSegment.join atomic_1743 atomic_1744
private theorem within_013_074_agreement : RowAgreement 1742 1744 := RowAgreement.join atomic_1743_agreement atomic_1744_agreement
private def within_013_075 : CountSegment 1 10 13120 1740 1744 908 910 := CountSegment.join within_013_073 within_013_074
private theorem within_013_075_agreement : RowAgreement 1740 1744 := RowAgreement.join within_013_073_agreement within_013_074_agreement
private def within_013_076 : CountSegment 1 10 13120 1736 1744 904 910 := CountSegment.join within_013_072 within_013_075
private theorem within_013_076_agreement : RowAgreement 1736 1744 := RowAgreement.join within_013_072_agreement within_013_075_agreement
private def within_013_077 : CountSegment 1 10 13120 1728 1744 900 910 := CountSegment.join within_013_069 within_013_076
private theorem within_013_077_agreement : RowAgreement 1728 1744 := RowAgreement.join within_013_069_agreement within_013_076_agreement
private def within_013_078 : CountSegment 1 10 13120 1744 1746 910 913 := CountSegment.join atomic_1745 atomic_1746
private theorem within_013_078_agreement : RowAgreement 1744 1746 := RowAgreement.join atomic_1745_agreement atomic_1746_agreement
private def within_013_079 : CountSegment 1 10 13120 1746 1748 913 913 := CountSegment.join atomic_1747 atomic_1748
private theorem within_013_079_agreement : RowAgreement 1746 1748 := RowAgreement.join atomic_1747_agreement atomic_1748_agreement
private def within_013_080 : CountSegment 1 10 13120 1744 1748 910 913 := CountSegment.join within_013_078 within_013_079
private theorem within_013_080_agreement : RowAgreement 1744 1748 := RowAgreement.join within_013_078_agreement within_013_079_agreement
private def within_013_081 : CountSegment 1 10 13120 1748 1750 913 913 := CountSegment.join atomic_1749 atomic_1750
private theorem within_013_081_agreement : RowAgreement 1748 1750 := RowAgreement.join atomic_1749_agreement atomic_1750_agreement
private def within_013_082 : CountSegment 1 10 13120 1750 1752 913 915 := CountSegment.join atomic_1751 atomic_1752
private theorem within_013_082_agreement : RowAgreement 1750 1752 := RowAgreement.join atomic_1751_agreement atomic_1752_agreement
private def within_013_083 : CountSegment 1 10 13120 1748 1752 913 915 := CountSegment.join within_013_081 within_013_082
private theorem within_013_083_agreement : RowAgreement 1748 1752 := RowAgreement.join within_013_081_agreement within_013_082_agreement
private def within_013_084 : CountSegment 1 10 13120 1744 1752 910 915 := CountSegment.join within_013_080 within_013_083
private theorem within_013_084_agreement : RowAgreement 1744 1752 := RowAgreement.join within_013_080_agreement within_013_083_agreement
private def within_013_085 : CountSegment 1 10 13120 1752 1754 915 915 := CountSegment.join atomic_1753 atomic_1754
private theorem within_013_085_agreement : RowAgreement 1752 1754 := RowAgreement.join atomic_1753_agreement atomic_1754_agreement
private def within_013_086 : CountSegment 1 10 13120 1754 1756 915 916 := CountSegment.join atomic_1755 atomic_1756
private theorem within_013_086_agreement : RowAgreement 1754 1756 := RowAgreement.join atomic_1755_agreement atomic_1756_agreement
private def within_013_087 : CountSegment 1 10 13120 1752 1756 915 916 := CountSegment.join within_013_085 within_013_086
private theorem within_013_087_agreement : RowAgreement 1752 1756 := RowAgreement.join within_013_085_agreement within_013_086_agreement
private def within_013_088 : CountSegment 1 10 13120 1756 1758 916 918 := CountSegment.join atomic_1757 atomic_1758
private theorem within_013_088_agreement : RowAgreement 1756 1758 := RowAgreement.join atomic_1757_agreement atomic_1758_agreement
private def within_013_089 : CountSegment 1 10 13120 1758 1760 918 919 := CountSegment.join atomic_1759 atomic_1760
private theorem within_013_089_agreement : RowAgreement 1758 1760 := RowAgreement.join atomic_1759_agreement atomic_1760_agreement
private def within_013_090 : CountSegment 1 10 13120 1756 1760 916 919 := CountSegment.join within_013_088 within_013_089
private theorem within_013_090_agreement : RowAgreement 1756 1760 := RowAgreement.join within_013_088_agreement within_013_089_agreement
private def within_013_091 : CountSegment 1 10 13120 1752 1760 915 919 := CountSegment.join within_013_087 within_013_090
private theorem within_013_091_agreement : RowAgreement 1752 1760 := RowAgreement.join within_013_087_agreement within_013_090_agreement
private def within_013_092 : CountSegment 1 10 13120 1744 1760 910 919 := CountSegment.join within_013_084 within_013_091
private theorem within_013_092_agreement : RowAgreement 1744 1760 := RowAgreement.join within_013_084_agreement within_013_091_agreement
private def within_013_093 : CountSegment 1 10 13120 1728 1760 900 919 := CountSegment.join within_013_077 within_013_092
private theorem within_013_093_agreement : RowAgreement 1728 1760 := RowAgreement.join within_013_077_agreement within_013_092_agreement
private def within_013_094 : CountSegment 1 10 13120 1760 1762 919 919 := CountSegment.join atomic_1761 atomic_1762
private theorem within_013_094_agreement : RowAgreement 1760 1762 := RowAgreement.join atomic_1761_agreement atomic_1762_agreement
private def within_013_095 : CountSegment 1 10 13120 1762 1764 919 919 := CountSegment.join atomic_1763 atomic_1764
private theorem within_013_095_agreement : RowAgreement 1762 1764 := RowAgreement.join atomic_1763_agreement atomic_1764_agreement
private def within_013_096 : CountSegment 1 10 13120 1760 1764 919 919 := CountSegment.join within_013_094 within_013_095
private theorem within_013_096_agreement : RowAgreement 1760 1764 := RowAgreement.join within_013_094_agreement within_013_095_agreement
private def within_013_097 : CountSegment 1 10 13120 1764 1766 919 920 := CountSegment.join atomic_1765 atomic_1766
private theorem within_013_097_agreement : RowAgreement 1764 1766 := RowAgreement.join atomic_1765_agreement atomic_1766_agreement
private def within_013_098 : CountSegment 1 10 13120 1766 1768 920 923 := CountSegment.join atomic_1767 atomic_1768
private theorem within_013_098_agreement : RowAgreement 1766 1768 := RowAgreement.join atomic_1767_agreement atomic_1768_agreement
private def within_013_099 : CountSegment 1 10 13120 1764 1768 919 923 := CountSegment.join within_013_097 within_013_098
private theorem within_013_099_agreement : RowAgreement 1764 1768 := RowAgreement.join within_013_097_agreement within_013_098_agreement
private def within_013_100 : CountSegment 1 10 13120 1760 1768 919 923 := CountSegment.join within_013_096 within_013_099
private theorem within_013_100_agreement : RowAgreement 1760 1768 := RowAgreement.join within_013_096_agreement within_013_099_agreement
private def within_013_101 : CountSegment 1 10 13120 1768 1770 923 924 := CountSegment.join atomic_1769 atomic_1770
private theorem within_013_101_agreement : RowAgreement 1768 1770 := RowAgreement.join atomic_1769_agreement atomic_1770_agreement
private def within_013_102 : CountSegment 1 10 13120 1770 1772 924 926 := CountSegment.join atomic_1771 atomic_1772
private theorem within_013_102_agreement : RowAgreement 1770 1772 := RowAgreement.join atomic_1771_agreement atomic_1772_agreement
private def within_013_103 : CountSegment 1 10 13120 1768 1772 923 926 := CountSegment.join within_013_101 within_013_102
private theorem within_013_103_agreement : RowAgreement 1768 1772 := RowAgreement.join within_013_101_agreement within_013_102_agreement
private def within_013_104 : CountSegment 1 10 13120 1772 1774 926 926 := CountSegment.join atomic_1773 atomic_1774
private theorem within_013_104_agreement : RowAgreement 1772 1774 := RowAgreement.join atomic_1773_agreement atomic_1774_agreement
private def within_013_105 : CountSegment 1 10 13120 1774 1776 926 927 := CountSegment.join atomic_1775 atomic_1776
private theorem within_013_105_agreement : RowAgreement 1774 1776 := RowAgreement.join atomic_1775_agreement atomic_1776_agreement
private def within_013_106 : CountSegment 1 10 13120 1772 1776 926 927 := CountSegment.join within_013_104 within_013_105
private theorem within_013_106_agreement : RowAgreement 1772 1776 := RowAgreement.join within_013_104_agreement within_013_105_agreement
private def within_013_107 : CountSegment 1 10 13120 1768 1776 923 927 := CountSegment.join within_013_103 within_013_106
private theorem within_013_107_agreement : RowAgreement 1768 1776 := RowAgreement.join within_013_103_agreement within_013_106_agreement
private def within_013_108 : CountSegment 1 10 13120 1760 1776 919 927 := CountSegment.join within_013_100 within_013_107
private theorem within_013_108_agreement : RowAgreement 1760 1776 := RowAgreement.join within_013_100_agreement within_013_107_agreement
private def within_013_109 : CountSegment 1 10 13120 1776 1778 927 928 := CountSegment.join atomic_1777 atomic_1778
private theorem within_013_109_agreement : RowAgreement 1776 1778 := RowAgreement.join atomic_1777_agreement atomic_1778_agreement
private def within_013_110 : CountSegment 1 10 13120 1778 1780 928 929 := CountSegment.join atomic_1779 atomic_1780
private theorem within_013_110_agreement : RowAgreement 1778 1780 := RowAgreement.join atomic_1779_agreement atomic_1780_agreement
private def within_013_111 : CountSegment 1 10 13120 1776 1780 927 929 := CountSegment.join within_013_109 within_013_110
private theorem within_013_111_agreement : RowAgreement 1776 1780 := RowAgreement.join within_013_109_agreement within_013_110_agreement
private def within_013_112 : CountSegment 1 10 13120 1780 1782 929 931 := CountSegment.join atomic_1781 atomic_1782
private theorem within_013_112_agreement : RowAgreement 1780 1782 := RowAgreement.join atomic_1781_agreement atomic_1782_agreement
private def within_013_113 : CountSegment 1 10 13120 1782 1784 931 931 := CountSegment.join atomic_1783 atomic_1784
private theorem within_013_113_agreement : RowAgreement 1782 1784 := RowAgreement.join atomic_1783_agreement atomic_1784_agreement
private def within_013_114 : CountSegment 1 10 13120 1780 1784 929 931 := CountSegment.join within_013_112 within_013_113
private theorem within_013_114_agreement : RowAgreement 1780 1784 := RowAgreement.join within_013_112_agreement within_013_113_agreement
private def within_013_115 : CountSegment 1 10 13120 1776 1784 927 931 := CountSegment.join within_013_111 within_013_114
private theorem within_013_115_agreement : RowAgreement 1776 1784 := RowAgreement.join within_013_111_agreement within_013_114_agreement
private def within_013_116 : CountSegment 1 10 13120 1784 1786 931 931 := CountSegment.join atomic_1785 atomic_1786
private theorem within_013_116_agreement : RowAgreement 1784 1786 := RowAgreement.join atomic_1785_agreement atomic_1786_agreement
private def within_013_117 : CountSegment 1 10 13120 1786 1788 931 933 := CountSegment.join atomic_1787 atomic_1788
private theorem within_013_117_agreement : RowAgreement 1786 1788 := RowAgreement.join atomic_1787_agreement atomic_1788_agreement
private def within_013_118 : CountSegment 1 10 13120 1784 1788 931 933 := CountSegment.join within_013_116 within_013_117
private theorem within_013_118_agreement : RowAgreement 1784 1788 := RowAgreement.join within_013_116_agreement within_013_117_agreement
private def within_013_119 : CountSegment 1 10 13120 1788 1790 933 933 := CountSegment.join atomic_1789 atomic_1790
private theorem within_013_119_agreement : RowAgreement 1788 1790 := RowAgreement.join atomic_1789_agreement atomic_1790_agreement
private def within_013_120 : CountSegment 1 10 13120 1790 1792 933 933 := CountSegment.join atomic_1791 atomic_1792
private theorem within_013_120_agreement : RowAgreement 1790 1792 := RowAgreement.join atomic_1791_agreement atomic_1792_agreement
private def within_013_121 : CountSegment 1 10 13120 1788 1792 933 933 := CountSegment.join within_013_119 within_013_120
private theorem within_013_121_agreement : RowAgreement 1788 1792 := RowAgreement.join within_013_119_agreement within_013_120_agreement
private def within_013_122 : CountSegment 1 10 13120 1784 1792 931 933 := CountSegment.join within_013_118 within_013_121
private theorem within_013_122_agreement : RowAgreement 1784 1792 := RowAgreement.join within_013_118_agreement within_013_121_agreement
private def within_013_123 : CountSegment 1 10 13120 1776 1792 927 933 := CountSegment.join within_013_115 within_013_122
private theorem within_013_123_agreement : RowAgreement 1776 1792 := RowAgreement.join within_013_115_agreement within_013_122_agreement
private def within_013_124 : CountSegment 1 10 13120 1760 1792 919 933 := CountSegment.join within_013_108 within_013_123
private theorem within_013_124_agreement : RowAgreement 1760 1792 := RowAgreement.join within_013_108_agreement within_013_123_agreement
private def within_013_125 : CountSegment 1 10 13120 1728 1792 900 933 := CountSegment.join within_013_093 within_013_124
private theorem within_013_125_agreement : RowAgreement 1728 1792 := RowAgreement.join within_013_093_agreement within_013_124_agreement
private def within_013_126 : CountSegment 1 10 13120 1664 1792 873 933 := CountSegment.join within_013_062 within_013_125
private theorem within_013_126_agreement : RowAgreement 1664 1792 := RowAgreement.join within_013_062_agreement within_013_125_agreement
def segment_013 : CountSegment 1 10 13120 1664 1792 873 933 := within_013_126
theorem agreement_013 : RowAgreement 1664 1792 := within_013_126_agreement
end ProjectEulerSolutions.P58Numerical
