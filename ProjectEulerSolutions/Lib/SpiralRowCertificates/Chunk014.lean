import ProjectEulerSolutions.Lib.SpiralPrimeSegments
import ProjectEulerSolutions.Lib.SpiralPrimeCertificates.All
import ProjectEulerSolutions.Lib.MillerRabin
import ProjectEulerSolutions.Lib.SpiralMillerAgreement
/-! Finite spiral row and counter certificates. Python execution is checked separately. -/
namespace ProjectEulerSolutions.P58Numerical
open SpiralPrimes
set_option Elab.async false
def row_1793 : RowCertificate 1793 where
  first := false
  first_correct := by
    have composite : ¬ Nat.Prime (corner 1793 1) := by change ¬ Nat.Prime 12862983; exact PrattCertificates.composite_of_product (factor := 3) (cofactor := 4287661) rfl (by decide) (by decide)
    exact decide_eq_false composite
  second := false
  second_correct := by
    have composite : ¬ Nat.Prime (corner 1793 2) := by change ¬ Nat.Prime 12859397; exact PrattCertificates.composite_of_product (factor := 193) (cofactor := 66629) rfl (by decide) (by decide)
    exact decide_eq_false composite
  third := false
  third_correct := by
    have composite : ¬ Nat.Prime (corner 1793 3) := by change ¬ Nat.Prime 12855811; exact PrattCertificates.composite_of_product (factor := 61) (cofactor := 210751) rfl (by decide) (by decide)
    exact decide_eq_false composite
theorem row_count_1793 : row 1793 = 0 := (row_1793.checked).trans rfl
theorem model_count_1793 : modelRow 1793 = 0 := by decide
theorem modelrow_agrees_1793 : modelRow 1793 = row 1793 := model_count_1793.trans row_count_1793.symm
def row_1794 : RowCertificate 1794 where
  first := false
  first_correct := by
    have composite : ¬ Nat.Prime (corner 1794 1) := by change ¬ Nat.Prime 12877333; exact PrattCertificates.composite_of_product (factor := 7) (cofactor := 1839619) rfl (by decide) (by decide)
    exact decide_eq_false composite
  second := false
  second_correct := by
    have composite : ¬ Nat.Prime (corner 1794 2) := by change ¬ Nat.Prime 12873745; exact PrattCertificates.composite_of_product (factor := 5) (cofactor := 2574749) rfl (by decide) (by decide)
    exact decide_eq_false composite
  third := true
  third_correct := by
    have prime : Nat.Prime (corner 1794 3) := by change Nat.Prime 12870157; exact ProjectEulerSolutions.P58PrimeNodes.p_12870157.prime
    exact decide_eq_true prime
theorem row_count_1794 : row 1794 = 1 := (row_1794.checked).trans rfl
theorem model_count_1794 : modelRow 1794 = 1 := by decide
theorem modelrow_agrees_1794 : modelRow 1794 = row 1794 := model_count_1794.trans row_count_1794.symm
def row_1795 : RowCertificate 1795 where
  first := false
  first_correct := by
    have composite : ¬ Nat.Prime (corner 1795 1) := by change ¬ Nat.Prime 12891691; exact PrattCertificates.composite_of_product (factor := 31) (cofactor := 415861) rfl (by decide) (by decide)
    exact decide_eq_false composite
  second := false
  second_correct := by
    have composite : ¬ Nat.Prime (corner 1795 2) := by change ¬ Nat.Prime 12888101; exact PrattCertificates.composite_of_product (factor := 1637) (cofactor := 7873) rfl (by decide) (by decide)
    exact decide_eq_false composite
  third := false
  third_correct := by
    have composite : ¬ Nat.Prime (corner 1795 3) := by change ¬ Nat.Prime 12884511; exact PrattCertificates.composite_of_product (factor := 3) (cofactor := 4294837) rfl (by decide) (by decide)
    exact decide_eq_false composite
theorem row_count_1795 : row 1795 = 0 := (row_1795.checked).trans rfl
theorem model_count_1795 : modelRow 1795 = 0 := by decide
theorem modelrow_agrees_1795 : modelRow 1795 = row 1795 := model_count_1795.trans row_count_1795.symm
def row_1796 : RowCertificate 1796 where
  first := false
  first_correct := by
    have composite : ¬ Nat.Prime (corner 1796 1) := by change ¬ Nat.Prime 12906057; exact PrattCertificates.composite_of_product (factor := 3) (cofactor := 4302019) rfl (by decide) (by decide)
    exact decide_eq_false composite
  second := false
  second_correct := by
    have composite : ¬ Nat.Prime (corner 1796 2) := by change ¬ Nat.Prime 12902465; exact PrattCertificates.composite_of_product (factor := 5) (cofactor := 2580493) rfl (by decide) (by decide)
    exact decide_eq_false composite
  third := false
  third_correct := by
    have composite : ¬ Nat.Prime (corner 1796 3) := by change ¬ Nat.Prime 12898873; exact PrattCertificates.composite_of_product (factor := 13) (cofactor := 992221) rfl (by decide) (by decide)
    exact decide_eq_false composite
theorem row_count_1796 : row 1796 = 0 := (row_1796.checked).trans rfl
theorem model_count_1796 : modelRow 1796 = 0 := by decide
theorem modelrow_agrees_1796 : modelRow 1796 = row 1796 := model_count_1796.trans row_count_1796.symm
def row_1797 : RowCertificate 1797 where
  first := true
  first_correct := by
    have prime : Nat.Prime (corner 1797 1) := by change Nat.Prime 12920431; exact ProjectEulerSolutions.P58PrimeNodes.p_12920431.prime
    exact decide_eq_true prime
  second := false
  second_correct := by
    have composite : ¬ Nat.Prime (corner 1797 2) := by change ¬ Nat.Prime 12916837; exact PrattCertificates.composite_of_product (factor := 89) (cofactor := 145133) rfl (by decide) (by decide)
    exact decide_eq_false composite
  third := false
  third_correct := by
    have composite : ¬ Nat.Prime (corner 1797 3) := by change ¬ Nat.Prime 12913243; exact PrattCertificates.composite_of_product (factor := 7) (cofactor := 1844749) rfl (by decide) (by decide)
    exact decide_eq_false composite
theorem row_count_1797 : row 1797 = 1 := (row_1797.checked).trans rfl
theorem model_count_1797 : modelRow 1797 = 1 := by decide
theorem modelrow_agrees_1797 : modelRow 1797 = row 1797 := model_count_1797.trans row_count_1797.symm
def row_1798 : RowCertificate 1798 where
  first := false
  first_correct := by
    have composite : ¬ Nat.Prime (corner 1798 1) := by change ¬ Nat.Prime 12934813; exact PrattCertificates.composite_of_product (factor := 2293) (cofactor := 5641) rfl (by decide) (by decide)
    exact decide_eq_false composite
  second := false
  second_correct := by
    have composite : ¬ Nat.Prime (corner 1798 2) := by change ¬ Nat.Prime 12931217; exact PrattCertificates.composite_of_product (factor := 13) (cofactor := 994709) rfl (by decide) (by decide)
    exact decide_eq_false composite
  third := false
  third_correct := by
    have composite : ¬ Nat.Prime (corner 1798 3) := by change ¬ Nat.Prime 12927621; exact PrattCertificates.composite_of_product (factor := 3) (cofactor := 4309207) rfl (by decide) (by decide)
    exact decide_eq_false composite
theorem row_count_1798 : row 1798 = 0 := (row_1798.checked).trans rfl
theorem model_count_1798 : modelRow 1798 = 0 := by decide
theorem modelrow_agrees_1798 : modelRow 1798 = row 1798 := model_count_1798.trans row_count_1798.symm
def row_1799 : RowCertificate 1799 where
  first := false
  first_correct := by
    have composite : ¬ Nat.Prime (corner 1799 1) := by change ¬ Nat.Prime 12949203; exact PrattCertificates.composite_of_product (factor := 3) (cofactor := 4316401) rfl (by decide) (by decide)
    exact decide_eq_false composite
  second := false
  second_correct := by
    have composite : ¬ Nat.Prime (corner 1799 2) := by change ¬ Nat.Prime 12945605; exact PrattCertificates.composite_of_product (factor := 5) (cofactor := 2589121) rfl (by decide) (by decide)
    exact decide_eq_false composite
  third := false
  third_correct := by
    have composite : ¬ Nat.Prime (corner 1799 3) := by change ¬ Nat.Prime 12942007; exact PrattCertificates.composite_of_product (factor := 13) (cofactor := 995539) rfl (by decide) (by decide)
    exact decide_eq_false composite
theorem row_count_1799 : row 1799 = 0 := (row_1799.checked).trans rfl
theorem model_count_1799 : modelRow 1799 = 0 := by decide
theorem modelrow_agrees_1799 : modelRow 1799 = row 1799 := model_count_1799.trans row_count_1799.symm
def row_1800 : RowCertificate 1800 where
  first := false
  first_correct := by
    have composite : ¬ Nat.Prime (corner 1800 1) := by change ¬ Nat.Prime 12963601; exact PrattCertificates.composite_of_product (factor := 7) (cofactor := 1851943) rfl (by decide) (by decide)
    exact decide_eq_false composite
  second := false
  second_correct := by
    have composite : ¬ Nat.Prime (corner 1800 2) := by change ¬ Nat.Prime 12960001; exact PrattCertificates.composite_of_product (factor := 17) (cofactor := 762353) rfl (by decide) (by decide)
    exact decide_eq_false composite
  third := false
  third_correct := by
    have composite : ¬ Nat.Prime (corner 1800 3) := by change ¬ Nat.Prime 12956401; exact PrattCertificates.composite_of_product (factor := 37) (cofactor := 350173) rfl (by decide) (by decide)
    exact decide_eq_false composite
theorem row_count_1800 : row 1800 = 0 := (row_1800.checked).trans rfl
theorem model_count_1800 : modelRow 1800 = 0 := by decide
theorem modelrow_agrees_1800 : modelRow 1800 = row 1800 := model_count_1800.trans row_count_1800.symm
def row_1801 : RowCertificate 1801 where
  first := false
  first_correct := by
    have composite : ¬ Nat.Prime (corner 1801 1) := by change ¬ Nat.Prime 12978007; exact PrattCertificates.composite_of_product (factor := 7) (cofactor := 1854001) rfl (by decide) (by decide)
    exact decide_eq_false composite
  second := false
  second_correct := by
    have composite : ¬ Nat.Prime (corner 1801 2) := by change ¬ Nat.Prime 12974405; exact PrattCertificates.composite_of_product (factor := 5) (cofactor := 2594881) rfl (by decide) (by decide)
    exact decide_eq_false composite
  third := false
  third_correct := by
    have composite : ¬ Nat.Prime (corner 1801 3) := by change ¬ Nat.Prime 12970803; exact PrattCertificates.composite_of_product (factor := 3) (cofactor := 4323601) rfl (by decide) (by decide)
    exact decide_eq_false composite
theorem row_count_1801 : row 1801 = 0 := (row_1801.checked).trans rfl
theorem model_count_1801 : modelRow 1801 = 0 := by decide
theorem modelrow_agrees_1801 : modelRow 1801 = row 1801 := model_count_1801.trans row_count_1801.symm
def row_1802 : RowCertificate 1802 where
  first := false
  first_correct := by
    have composite : ¬ Nat.Prime (corner 1802 1) := by change ¬ Nat.Prime 12992421; exact PrattCertificates.composite_of_product (factor := 3) (cofactor := 4330807) rfl (by decide) (by decide)
    exact decide_eq_false composite
  second := false
  second_correct := by
    have composite : ¬ Nat.Prime (corner 1802 2) := by change ¬ Nat.Prime 12988817; exact PrattCertificates.composite_of_product (factor := 73) (cofactor := 177929) rfl (by decide) (by decide)
    exact decide_eq_false composite
  third := true
  third_correct := by
    have prime : Nat.Prime (corner 1802 3) := by change Nat.Prime 12985213; exact ProjectEulerSolutions.P58PrimeNodes.p_12985213.prime
    exact decide_eq_true prime
theorem row_count_1802 : row 1802 = 1 := (row_1802.checked).trans rfl
theorem model_count_1802 : modelRow 1802 = 1 := by decide
theorem modelrow_agrees_1802 : modelRow 1802 = row 1802 := model_count_1802.trans row_count_1802.symm
def row_1803 : RowCertificate 1803 where
  first := true
  first_correct := by
    have prime : Nat.Prime (corner 1803 1) := by change Nat.Prime 13006843; exact ProjectEulerSolutions.P58PrimeNodes.p_13006843.prime
    exact decide_eq_true prime
  second := false
  second_correct := by
    have composite : ¬ Nat.Prime (corner 1803 2) := by change ¬ Nat.Prime 13003237; exact PrattCertificates.composite_of_product (factor := 13) (cofactor := 1000249) rfl (by decide) (by decide)
    exact decide_eq_false composite
  third := false
  third_correct := by
    have composite : ¬ Nat.Prime (corner 1803 3) := by change ¬ Nat.Prime 12999631; exact PrattCertificates.composite_of_product (factor := 43) (cofactor := 302317) rfl (by decide) (by decide)
    exact decide_eq_false composite
theorem row_count_1803 : row 1803 = 1 := (row_1803.checked).trans rfl
theorem model_count_1803 : modelRow 1803 = 1 := by decide
theorem modelrow_agrees_1803 : modelRow 1803 = row 1803 := model_count_1803.trans row_count_1803.symm
def row_1804 : RowCertificate 1804 where
  first := true
  first_correct := by
    have prime : Nat.Prime (corner 1804 1) := by change Nat.Prime 13021273; exact ProjectEulerSolutions.P58PrimeNodes.p_13021273.prime
    exact decide_eq_true prime
  second := false
  second_correct := by
    have composite : ¬ Nat.Prime (corner 1804 2) := by change ¬ Nat.Prime 13017665; exact PrattCertificates.composite_of_product (factor := 5) (cofactor := 2603533) rfl (by decide) (by decide)
    exact decide_eq_false composite
  third := false
  third_correct := by
    have composite : ¬ Nat.Prime (corner 1804 3) := by change ¬ Nat.Prime 13014057; exact PrattCertificates.composite_of_product (factor := 3) (cofactor := 4338019) rfl (by decide) (by decide)
    exact decide_eq_false composite
theorem row_count_1804 : row 1804 = 1 := (row_1804.checked).trans rfl
theorem model_count_1804 : modelRow 1804 = 1 := by decide
theorem modelrow_agrees_1804 : modelRow 1804 = row 1804 := model_count_1804.trans row_count_1804.symm
def row_1805 : RowCertificate 1805 where
  first := false
  first_correct := by
    have composite : ¬ Nat.Prime (corner 1805 1) := by change ¬ Nat.Prime 13035711; exact PrattCertificates.composite_of_product (factor := 3) (cofactor := 4345237) rfl (by decide) (by decide)
    exact decide_eq_false composite
  second := false
  second_correct := by
    have composite : ¬ Nat.Prime (corner 1805 2) := by change ¬ Nat.Prime 13032101; exact PrattCertificates.composite_of_product (factor := 61) (cofactor := 213641) rfl (by decide) (by decide)
    exact decide_eq_false composite
  third := false
  third_correct := by
    have composite : ¬ Nat.Prime (corner 1805 3) := by change ¬ Nat.Prime 13028491; exact PrattCertificates.composite_of_product (factor := 7) (cofactor := 1861213) rfl (by decide) (by decide)
    exact decide_eq_false composite
theorem row_count_1805 : row 1805 = 0 := (row_1805.checked).trans rfl
theorem model_count_1805 : modelRow 1805 = 0 := by decide
theorem modelrow_agrees_1805 : modelRow 1805 = row 1805 := model_count_1805.trans row_count_1805.symm
def row_1806 : RowCertificate 1806 where
  first := false
  first_correct := by
    have composite : ¬ Nat.Prime (corner 1806 1) := by change ¬ Nat.Prime 13050157; exact PrattCertificates.composite_of_product (factor := 61) (cofactor := 213937) rfl (by decide) (by decide)
    exact decide_eq_false composite
  second := false
  second_correct := by
    have composite : ¬ Nat.Prime (corner 1806 2) := by change ¬ Nat.Prime 13046545; exact PrattCertificates.composite_of_product (factor := 5) (cofactor := 2609309) rfl (by decide) (by decide)
    exact decide_eq_false composite
  third := true
  third_correct := by
    have prime : Nat.Prime (corner 1806 3) := by change Nat.Prime 13042933; exact ProjectEulerSolutions.P58PrimeNodes.p_13042933.prime
    exact decide_eq_true prime
theorem row_count_1806 : row 1806 = 1 := (row_1806.checked).trans rfl
theorem model_count_1806 : modelRow 1806 = 1 := by decide
theorem modelrow_agrees_1806 : modelRow 1806 = row 1806 := model_count_1806.trans row_count_1806.symm
def row_1807 : RowCertificate 1807 where
  first := false
  first_correct := by
    have composite : ¬ Nat.Prime (corner 1807 1) := by change ¬ Nat.Prime 13064611; exact PrattCertificates.composite_of_product (factor := 7) (cofactor := 1866373) rfl (by decide) (by decide)
    exact decide_eq_false composite
  second := false
  second_correct := by
    have composite : ¬ Nat.Prime (corner 1807 2) := by change ¬ Nat.Prime 13060997; exact PrattCertificates.composite_of_product (factor := 257) (cofactor := 50821) rfl (by decide) (by decide)
    exact decide_eq_false composite
  third := false
  third_correct := by
    have composite : ¬ Nat.Prime (corner 1807 3) := by change ¬ Nat.Prime 13057383; exact PrattCertificates.composite_of_product (factor := 3) (cofactor := 4352461) rfl (by decide) (by decide)
    exact decide_eq_false composite
theorem row_count_1807 : row 1807 = 0 := (row_1807.checked).trans rfl
theorem model_count_1807 : modelRow 1807 = 0 := by decide
theorem modelrow_agrees_1807 : modelRow 1807 = row 1807 := model_count_1807.trans row_count_1807.symm
def row_1808 : RowCertificate 1808 where
  first := false
  first_correct := by
    have composite : ¬ Nat.Prime (corner 1808 1) := by change ¬ Nat.Prime 13079073; exact PrattCertificates.composite_of_product (factor := 3) (cofactor := 4359691) rfl (by decide) (by decide)
    exact decide_eq_false composite
  second := false
  second_correct := by
    have composite : ¬ Nat.Prime (corner 1808 2) := by change ¬ Nat.Prime 13075457; exact PrattCertificates.composite_of_product (factor := 389) (cofactor := 33613) rfl (by decide) (by decide)
    exact decide_eq_false composite
  third := false
  third_correct := by
    have composite : ¬ Nat.Prime (corner 1808 3) := by change ¬ Nat.Prime 13071841; exact PrattCertificates.composite_of_product (factor := 37) (cofactor := 353293) rfl (by decide) (by decide)
    exact decide_eq_false composite
theorem row_count_1808 : row 1808 = 0 := (row_1808.checked).trans rfl
theorem model_count_1808 : modelRow 1808 = 0 := by decide
theorem modelrow_agrees_1808 : modelRow 1808 = row 1808 := model_count_1808.trans row_count_1808.symm
def row_1809 : RowCertificate 1809 where
  first := false
  first_correct := by
    have composite : ¬ Nat.Prime (corner 1809 1) := by change ¬ Nat.Prime 13093543; exact PrattCertificates.composite_of_product (factor := 43) (cofactor := 304501) rfl (by decide) (by decide)
    exact decide_eq_false composite
  second := false
  second_correct := by
    have composite : ¬ Nat.Prime (corner 1809 2) := by change ¬ Nat.Prime 13089925; exact PrattCertificates.composite_of_product (factor := 5) (cofactor := 2617985) rfl (by decide) (by decide)
    exact decide_eq_false composite
  third := false
  third_correct := by
    have composite : ¬ Nat.Prime (corner 1809 3) := by change ¬ Nat.Prime 13086307; exact PrattCertificates.composite_of_product (factor := 13) (cofactor := 1006639) rfl (by decide) (by decide)
    exact decide_eq_false composite
theorem row_count_1809 : row 1809 = 0 := (row_1809.checked).trans rfl
theorem model_count_1809 : modelRow 1809 = 0 := by decide
theorem modelrow_agrees_1809 : modelRow 1809 = row 1809 := model_count_1809.trans row_count_1809.symm
def row_1810 : RowCertificate 1810 where
  first := false
  first_correct := by
    have composite : ¬ Nat.Prime (corner 1810 1) := by change ¬ Nat.Prime 13108021; exact PrattCertificates.composite_of_product (factor := 673) (cofactor := 19477) rfl (by decide) (by decide)
    exact decide_eq_false composite
  second := false
  second_correct := by
    have composite : ¬ Nat.Prime (corner 1810 2) := by change ¬ Nat.Prime 13104401; exact PrattCertificates.composite_of_product (factor := 37) (cofactor := 354173) rfl (by decide) (by decide)
    exact decide_eq_false composite
  third := false
  third_correct := by
    have composite : ¬ Nat.Prime (corner 1810 3) := by change ¬ Nat.Prime 13100781; exact PrattCertificates.composite_of_product (factor := 3) (cofactor := 4366927) rfl (by decide) (by decide)
    exact decide_eq_false composite
theorem row_count_1810 : row 1810 = 0 := (row_1810.checked).trans rfl
theorem model_count_1810 : modelRow 1810 = 0 := by decide
theorem modelrow_agrees_1810 : modelRow 1810 = row 1810 := model_count_1810.trans row_count_1810.symm
def row_1811 : RowCertificate 1811 where
  first := false
  first_correct := by
    have composite : ¬ Nat.Prime (corner 1811 1) := by change ¬ Nat.Prime 13122507; exact PrattCertificates.composite_of_product (factor := 3) (cofactor := 4374169) rfl (by decide) (by decide)
    exact decide_eq_false composite
  second := false
  second_correct := by
    have composite : ¬ Nat.Prime (corner 1811 2) := by change ¬ Nat.Prime 13118885; exact PrattCertificates.composite_of_product (factor := 5) (cofactor := 2623777) rfl (by decide) (by decide)
    exact decide_eq_false composite
  third := false
  third_correct := by
    have composite : ¬ Nat.Prime (corner 1811 3) := by change ¬ Nat.Prime 13115263; exact PrattCertificates.composite_of_product (factor := 7) (cofactor := 1873609) rfl (by decide) (by decide)
    exact decide_eq_false composite
theorem row_count_1811 : row 1811 = 0 := (row_1811.checked).trans rfl
theorem model_count_1811 : modelRow 1811 = 0 := by decide
theorem modelrow_agrees_1811 : modelRow 1811 = row 1811 := model_count_1811.trans row_count_1811.symm
def row_1812 : RowCertificate 1812 where
  first := false
  first_correct := by
    have composite : ¬ Nat.Prime (corner 1812 1) := by change ¬ Nat.Prime 13137001; exact PrattCertificates.composite_of_product (factor := 97) (cofactor := 135433) rfl (by decide) (by decide)
    exact decide_eq_false composite
  second := true
  second_correct := by
    have prime : Nat.Prime (corner 1812 2) := by change Nat.Prime 13133377; exact ProjectEulerSolutions.P58PrimeNodes.p_13133377.prime
    exact decide_eq_true prime
  third := false
  third_correct := by
    have composite : ¬ Nat.Prime (corner 1812 3) := by change ¬ Nat.Prime 13129753; exact PrattCertificates.composite_of_product (factor := 7) (cofactor := 1875679) rfl (by decide) (by decide)
    exact decide_eq_false composite
theorem row_count_1812 : row 1812 = 1 := (row_1812.checked).trans rfl
theorem model_count_1812 : modelRow 1812 = 1 := by decide
theorem modelrow_agrees_1812 : modelRow 1812 = row 1812 := model_count_1812.trans row_count_1812.symm
def row_1813 : RowCertificate 1813 where
  first := true
  first_correct := by
    have prime : Nat.Prime (corner 1813 1) := by change Nat.Prime 13151503; exact ProjectEulerSolutions.P58PrimeNodes.p_13151503.prime
    exact decide_eq_true prime
  second := false
  second_correct := by
    have composite : ¬ Nat.Prime (corner 1813 2) := by change ¬ Nat.Prime 13147877; exact PrattCertificates.composite_of_product (factor := 101) (cofactor := 130177) rfl (by decide) (by decide)
    exact decide_eq_false composite
  third := false
  third_correct := by
    have composite : ¬ Nat.Prime (corner 1813 3) := by change ¬ Nat.Prime 13144251; exact PrattCertificates.composite_of_product (factor := 3) (cofactor := 4381417) rfl (by decide) (by decide)
    exact decide_eq_false composite
theorem row_count_1813 : row 1813 = 1 := (row_1813.checked).trans rfl
theorem model_count_1813 : modelRow 1813 = 1 := by decide
theorem modelrow_agrees_1813 : modelRow 1813 = row 1813 := model_count_1813.trans row_count_1813.symm
def row_1814 : RowCertificate 1814 where
  first := false
  first_correct := by
    have composite : ¬ Nat.Prime (corner 1814 1) := by change ¬ Nat.Prime 13166013; exact PrattCertificates.composite_of_product (factor := 3) (cofactor := 4388671) rfl (by decide) (by decide)
    exact decide_eq_false composite
  second := false
  second_correct := by
    have composite : ¬ Nat.Prime (corner 1814 2) := by change ¬ Nat.Prime 13162385; exact PrattCertificates.composite_of_product (factor := 5) (cofactor := 2632477) rfl (by decide) (by decide)
    exact decide_eq_false composite
  third := false
  third_correct := by
    have composite : ¬ Nat.Prime (corner 1814 3) := by change ¬ Nat.Prime 13158757; exact PrattCertificates.composite_of_product (factor := 409) (cofactor := 32173) rfl (by decide) (by decide)
    exact decide_eq_false composite
theorem row_count_1814 : row 1814 = 0 := (row_1814.checked).trans rfl
theorem model_count_1814 : modelRow 1814 = 0 := by decide
theorem modelrow_agrees_1814 : modelRow 1814 = row 1814 := model_count_1814.trans row_count_1814.symm
def row_1815 : RowCertificate 1815 where
  first := false
  first_correct := by
    have composite : ¬ Nat.Prime (corner 1815 1) := by change ¬ Nat.Prime 13180531; exact PrattCertificates.composite_of_product (factor := 7) (cofactor := 1882933) rfl (by decide) (by decide)
    exact decide_eq_false composite
  second := false
  second_correct := by
    have composite : ¬ Nat.Prime (corner 1815 2) := by change ¬ Nat.Prime 13176901; exact PrattCertificates.composite_of_product (factor := 109) (cofactor := 120889) rfl (by decide) (by decide)
    exact decide_eq_false composite
  third := true
  third_correct := by
    have prime : Nat.Prime (corner 1815 3) := by change Nat.Prime 13173271; exact ProjectEulerSolutions.P58PrimeNodes.p_13173271.prime
    exact decide_eq_true prime
theorem row_count_1815 : row 1815 = 1 := (row_1815.checked).trans rfl
theorem model_count_1815 : modelRow 1815 = 1 := by decide
theorem modelrow_agrees_1815 : modelRow 1815 = row 1815 := model_count_1815.trans row_count_1815.symm
def row_1816 : RowCertificate 1816 where
  first := false
  first_correct := by
    have composite : ¬ Nat.Prime (corner 1816 1) := by change ¬ Nat.Prime 13195057; exact PrattCertificates.composite_of_product (factor := 31) (cofactor := 425647) rfl (by decide) (by decide)
    exact decide_eq_false composite
  second := false
  second_correct := by
    have composite : ¬ Nat.Prime (corner 1816 2) := by change ¬ Nat.Prime 13191425; exact PrattCertificates.composite_of_product (factor := 5) (cofactor := 2638285) rfl (by decide) (by decide)
    exact decide_eq_false composite
  third := false
  third_correct := by
    have composite : ¬ Nat.Prime (corner 1816 3) := by change ¬ Nat.Prime 13187793; exact PrattCertificates.composite_of_product (factor := 3) (cofactor := 4395931) rfl (by decide) (by decide)
    exact decide_eq_false composite
theorem row_count_1816 : row 1816 = 0 := (row_1816.checked).trans rfl
theorem model_count_1816 : modelRow 1816 = 0 := by decide
theorem modelrow_agrees_1816 : modelRow 1816 = row 1816 := model_count_1816.trans row_count_1816.symm
def row_1817 : RowCertificate 1817 where
  first := false
  first_correct := by
    have composite : ¬ Nat.Prime (corner 1817 1) := by change ¬ Nat.Prime 13209591; exact PrattCertificates.composite_of_product (factor := 3) (cofactor := 4403197) rfl (by decide) (by decide)
    exact decide_eq_false composite
  second := false
  second_correct := by
    have composite : ¬ Nat.Prime (corner 1817 2) := by change ¬ Nat.Prime 13205957; exact PrattCertificates.composite_of_product (factor := 17) (cofactor := 776821) rfl (by decide) (by decide)
    exact decide_eq_false composite
  third := false
  third_correct := by
    have composite : ¬ Nat.Prime (corner 1817 3) := by change ¬ Nat.Prime 13202323; exact PrattCertificates.composite_of_product (factor := 1327) (cofactor := 9949) rfl (by decide) (by decide)
    exact decide_eq_false composite
theorem row_count_1817 : row 1817 = 0 := (row_1817.checked).trans rfl
theorem model_count_1817 : modelRow 1817 = 0 := by decide
theorem modelrow_agrees_1817 : modelRow 1817 = row 1817 := model_count_1817.trans row_count_1817.symm
def row_1818 : RowCertificate 1818 where
  first := false
  first_correct := by
    have composite : ¬ Nat.Prime (corner 1818 1) := by change ¬ Nat.Prime 13224133; exact PrattCertificates.composite_of_product (factor := 13) (cofactor := 1017241) rfl (by decide) (by decide)
    exact decide_eq_false composite
  second := false
  second_correct := by
    have composite : ¬ Nat.Prime (corner 1818 2) := by change ¬ Nat.Prime 13220497; exact PrattCertificates.composite_of_product (factor := 397) (cofactor := 33301) rfl (by decide) (by decide)
    exact decide_eq_false composite
  third := false
  third_correct := by
    have composite : ¬ Nat.Prime (corner 1818 3) := by change ¬ Nat.Prime 13216861; exact PrattCertificates.composite_of_product (factor := 7) (cofactor := 1888123) rfl (by decide) (by decide)
    exact decide_eq_false composite
theorem row_count_1818 : row 1818 = 0 := (row_1818.checked).trans rfl
theorem model_count_1818 : modelRow 1818 = 0 := by decide
theorem modelrow_agrees_1818 : modelRow 1818 = row 1818 := model_count_1818.trans row_count_1818.symm
def row_1819 : RowCertificate 1819 where
  first := true
  first_correct := by
    have prime : Nat.Prime (corner 1819 1) := by change Nat.Prime 13238683; exact ProjectEulerSolutions.P58PrimeNodes.p_13238683.prime
    exact decide_eq_true prime
  second := false
  second_correct := by
    have composite : ¬ Nat.Prime (corner 1819 2) := by change ¬ Nat.Prime 13235045; exact PrattCertificates.composite_of_product (factor := 5) (cofactor := 2647009) rfl (by decide) (by decide)
    exact decide_eq_false composite
  third := false
  third_correct := by
    have composite : ¬ Nat.Prime (corner 1819 3) := by change ¬ Nat.Prime 13231407; exact PrattCertificates.composite_of_product (factor := 3) (cofactor := 4410469) rfl (by decide) (by decide)
    exact decide_eq_false composite
theorem row_count_1819 : row 1819 = 1 := (row_1819.checked).trans rfl
theorem model_count_1819 : modelRow 1819 = 1 := by decide
theorem modelrow_agrees_1819 : modelRow 1819 = row 1819 := model_count_1819.trans row_count_1819.symm
def row_1820 : RowCertificate 1820 where
  first := false
  first_correct := by
    have composite : ¬ Nat.Prime (corner 1820 1) := by change ¬ Nat.Prime 13253241; exact PrattCertificates.composite_of_product (factor := 3) (cofactor := 4417747) rfl (by decide) (by decide)
    exact decide_eq_false composite
  second := false
  second_correct := by
    have composite : ¬ Nat.Prime (corner 1820 2) := by change ¬ Nat.Prime 13249601; exact PrattCertificates.composite_of_product (factor := 41) (cofactor := 323161) rfl (by decide) (by decide)
    exact decide_eq_false composite
  third := true
  third_correct := by
    have prime : Nat.Prime (corner 1820 3) := by change Nat.Prime 13245961; exact ProjectEulerSolutions.P58PrimeNodes.p_13245961.prime
    exact decide_eq_true prime
theorem row_count_1820 : row 1820 = 1 := (row_1820.checked).trans rfl
theorem model_count_1820 : modelRow 1820 = 1 := by decide
theorem modelrow_agrees_1820 : modelRow 1820 = row 1820 := model_count_1820.trans row_count_1820.symm
def row_1821 : RowCertificate 1821 where
  first := false
  first_correct := by
    have composite : ¬ Nat.Prime (corner 1821 1) := by change ¬ Nat.Prime 13267807; exact PrattCertificates.composite_of_product (factor := 7) (cofactor := 1895401) rfl (by decide) (by decide)
    exact decide_eq_false composite
  second := false
  second_correct := by
    have composite : ¬ Nat.Prime (corner 1821 2) := by change ¬ Nat.Prime 13264165; exact PrattCertificates.composite_of_product (factor := 5) (cofactor := 2652833) rfl (by decide) (by decide)
    exact decide_eq_false composite
  third := false
  third_correct := by
    have composite : ¬ Nat.Prime (corner 1821 3) := by change ¬ Nat.Prime 13260523; exact PrattCertificates.composite_of_product (factor := 73) (cofactor := 181651) rfl (by decide) (by decide)
    exact decide_eq_false composite
theorem row_count_1821 : row 1821 = 0 := (row_1821.checked).trans rfl
theorem model_count_1821 : modelRow 1821 = 0 := by decide
theorem modelrow_agrees_1821 : modelRow 1821 = row 1821 := model_count_1821.trans row_count_1821.symm
def row_1822 : RowCertificate 1822 where
  first := false
  first_correct := by
    have composite : ¬ Nat.Prime (corner 1822 1) := by change ¬ Nat.Prime 13282381; exact PrattCertificates.composite_of_product (factor := 7) (cofactor := 1897483) rfl (by decide) (by decide)
    exact decide_eq_false composite
  second := true
  second_correct := by
    have prime : Nat.Prime (corner 1822 2) := by change Nat.Prime 13278737; exact ProjectEulerSolutions.P58PrimeNodes.p_13278737.prime
    exact decide_eq_true prime
  third := false
  third_correct := by
    have composite : ¬ Nat.Prime (corner 1822 3) := by change ¬ Nat.Prime 13275093; exact PrattCertificates.composite_of_product (factor := 3) (cofactor := 4425031) rfl (by decide) (by decide)
    exact decide_eq_false composite
theorem row_count_1822 : row 1822 = 1 := (row_1822.checked).trans rfl
theorem model_count_1822 : modelRow 1822 = 1 := by decide
theorem modelrow_agrees_1822 : modelRow 1822 = row 1822 := model_count_1822.trans row_count_1822.symm
def row_1823 : RowCertificate 1823 where
  first := false
  first_correct := by
    have composite : ¬ Nat.Prime (corner 1823 1) := by change ¬ Nat.Prime 13296963; exact PrattCertificates.composite_of_product (factor := 3) (cofactor := 4432321) rfl (by decide) (by decide)
    exact decide_eq_false composite
  second := false
  second_correct := by
    have composite : ¬ Nat.Prime (corner 1823 2) := by change ¬ Nat.Prime 13293317; exact PrattCertificates.composite_of_product (factor := 101) (cofactor := 131617) rfl (by decide) (by decide)
    exact decide_eq_false composite
  third := true
  third_correct := by
    have prime : Nat.Prime (corner 1823 3) := by change Nat.Prime 13289671; exact ProjectEulerSolutions.P58PrimeNodes.p_13289671.prime
    exact decide_eq_true prime
theorem row_count_1823 : row 1823 = 1 := (row_1823.checked).trans rfl
theorem model_count_1823 : modelRow 1823 = 1 := by decide
theorem modelrow_agrees_1823 : modelRow 1823 = row 1823 := model_count_1823.trans row_count_1823.symm
def row_1824 : RowCertificate 1824 where
  first := false
  first_correct := by
    have composite : ¬ Nat.Prime (corner 1824 1) := by change ¬ Nat.Prime 13311553; exact PrattCertificates.composite_of_product (factor := 43) (cofactor := 309571) rfl (by decide) (by decide)
    exact decide_eq_false composite
  second := false
  second_correct := by
    have composite : ¬ Nat.Prime (corner 1824 2) := by change ¬ Nat.Prime 13307905; exact PrattCertificates.composite_of_product (factor := 5) (cofactor := 2661581) rfl (by decide) (by decide)
    exact decide_eq_false composite
  third := false
  third_correct := by
    have composite : ¬ Nat.Prime (corner 1824 3) := by change ¬ Nat.Prime 13304257; exact PrattCertificates.composite_of_product (factor := 67) (cofactor := 198571) rfl (by decide) (by decide)
    exact decide_eq_false composite
theorem row_count_1824 : row 1824 = 0 := (row_1824.checked).trans rfl
theorem model_count_1824 : modelRow 1824 = 0 := by decide
theorem modelrow_agrees_1824 : modelRow 1824 = row 1824 := model_count_1824.trans row_count_1824.symm
def row_1825 : RowCertificate 1825 where
  first := false
  first_correct := by
    have composite : ¬ Nat.Prime (corner 1825 1) := by change ¬ Nat.Prime 13326151; exact PrattCertificates.composite_of_product (factor := 97) (cofactor := 137383) rfl (by decide) (by decide)
    exact decide_eq_false composite
  second := true
  second_correct := by
    have prime : Nat.Prime (corner 1825 2) := by change Nat.Prime 13322501; exact ProjectEulerSolutions.P58PrimeNodes.p_13322501.prime
    exact decide_eq_true prime
  third := false
  third_correct := by
    have composite : ¬ Nat.Prime (corner 1825 3) := by change ¬ Nat.Prime 13318851; exact PrattCertificates.composite_of_product (factor := 3) (cofactor := 4439617) rfl (by decide) (by decide)
    exact decide_eq_false composite
theorem row_count_1825 : row 1825 = 1 := (row_1825.checked).trans rfl
theorem model_count_1825 : modelRow 1825 = 1 := by decide
theorem modelrow_agrees_1825 : modelRow 1825 = row 1825 := model_count_1825.trans row_count_1825.symm
def row_1826 : RowCertificate 1826 where
  first := false
  first_correct := by
    have composite : ¬ Nat.Prime (corner 1826 1) := by change ¬ Nat.Prime 13340757; exact PrattCertificates.composite_of_product (factor := 3) (cofactor := 4446919) rfl (by decide) (by decide)
    exact decide_eq_false composite
  second := false
  second_correct := by
    have composite : ¬ Nat.Prime (corner 1826 2) := by change ¬ Nat.Prime 13337105; exact PrattCertificates.composite_of_product (factor := 5) (cofactor := 2667421) rfl (by decide) (by decide)
    exact decide_eq_false composite
  third := false
  third_correct := by
    have composite : ¬ Nat.Prime (corner 1826 3) := by change ¬ Nat.Prime 13333453; exact PrattCertificates.composite_of_product (factor := 7) (cofactor := 1904779) rfl (by decide) (by decide)
    exact decide_eq_false composite
theorem row_count_1826 : row 1826 = 0 := (row_1826.checked).trans rfl
theorem model_count_1826 : modelRow 1826 = 0 := by decide
theorem modelrow_agrees_1826 : modelRow 1826 = row 1826 := model_count_1826.trans row_count_1826.symm
def row_1827 : RowCertificate 1827 where
  first := true
  first_correct := by
    have prime : Nat.Prime (corner 1827 1) := by change Nat.Prime 13355371; exact ProjectEulerSolutions.P58PrimeNodes.p_13355371.prime
    exact decide_eq_true prime
  second := false
  second_correct := by
    have composite : ¬ Nat.Prime (corner 1827 2) := by change ¬ Nat.Prime 13351717; exact PrattCertificates.composite_of_product (factor := 293) (cofactor := 45569) rfl (by decide) (by decide)
    exact decide_eq_false composite
  third := false
  third_correct := by
    have composite : ¬ Nat.Prime (corner 1827 3) := by change ¬ Nat.Prime 13348063; exact PrattCertificates.composite_of_product (factor := 1249) (cofactor := 10687) rfl (by decide) (by decide)
    exact decide_eq_false composite
theorem row_count_1827 : row 1827 = 1 := (row_1827.checked).trans rfl
theorem model_count_1827 : modelRow 1827 = 1 := by decide
theorem modelrow_agrees_1827 : modelRow 1827 = row 1827 := model_count_1827.trans row_count_1827.symm
def row_1828 : RowCertificate 1828 where
  first := false
  first_correct := by
    have composite : ¬ Nat.Prime (corner 1828 1) := by change ¬ Nat.Prime 13369993; exact PrattCertificates.composite_of_product (factor := 7) (cofactor := 1909999) rfl (by decide) (by decide)
    exact decide_eq_false composite
  second := false
  second_correct := by
    have composite : ¬ Nat.Prime (corner 1828 2) := by change ¬ Nat.Prime 13366337; exact PrattCertificates.composite_of_product (factor := 977) (cofactor := 13681) rfl (by decide) (by decide)
    exact decide_eq_false composite
  third := false
  third_correct := by
    have composite : ¬ Nat.Prime (corner 1828 3) := by change ¬ Nat.Prime 13362681; exact PrattCertificates.composite_of_product (factor := 3) (cofactor := 4454227) rfl (by decide) (by decide)
    exact decide_eq_false composite
theorem row_count_1828 : row 1828 = 0 := (row_1828.checked).trans rfl
theorem model_count_1828 : modelRow 1828 = 0 := by decide
theorem modelrow_agrees_1828 : modelRow 1828 = row 1828 := model_count_1828.trans row_count_1828.symm
def row_1829 : RowCertificate 1829 where
  first := false
  first_correct := by
    have composite : ¬ Nat.Prime (corner 1829 1) := by change ¬ Nat.Prime 13384623; exact PrattCertificates.composite_of_product (factor := 3) (cofactor := 4461541) rfl (by decide) (by decide)
    exact decide_eq_false composite
  second := false
  second_correct := by
    have composite : ¬ Nat.Prime (corner 1829 2) := by change ¬ Nat.Prime 13380965; exact PrattCertificates.composite_of_product (factor := 5) (cofactor := 2676193) rfl (by decide) (by decide)
    exact decide_eq_false composite
  third := false
  third_correct := by
    have composite : ¬ Nat.Prime (corner 1829 3) := by change ¬ Nat.Prime 13377307; exact PrattCertificates.composite_of_product (factor := 79) (cofactor := 169333) rfl (by decide) (by decide)
    exact decide_eq_false composite
theorem row_count_1829 : row 1829 = 0 := (row_1829.checked).trans rfl
theorem model_count_1829 : modelRow 1829 = 0 := by decide
theorem modelrow_agrees_1829 : modelRow 1829 = row 1829 := model_count_1829.trans row_count_1829.symm
def row_1830 : RowCertificate 1830 where
  first := false
  first_correct := by
    have composite : ¬ Nat.Prime (corner 1830 1) := by change ¬ Nat.Prime 13399261; exact PrattCertificates.composite_of_product (factor := 109) (cofactor := 122929) rfl (by decide) (by decide)
    exact decide_eq_false composite
  second := true
  second_correct := by
    have prime : Nat.Prime (corner 1830 2) := by change Nat.Prime 13395601; exact ProjectEulerSolutions.P58PrimeNodes.p_13395601.prime
    exact decide_eq_true prime
  third := false
  third_correct := by
    have composite : ¬ Nat.Prime (corner 1830 3) := by change ¬ Nat.Prime 13391941; exact PrattCertificates.composite_of_product (factor := 19) (cofactor := 704839) rfl (by decide) (by decide)
    exact decide_eq_false composite
theorem row_count_1830 : row 1830 = 1 := (row_1830.checked).trans rfl
theorem model_count_1830 : modelRow 1830 = 1 := by decide
theorem modelrow_agrees_1830 : modelRow 1830 = row 1830 := model_count_1830.trans row_count_1830.symm
def row_1831 : RowCertificate 1831 where
  first := false
  first_correct := by
    have composite : ¬ Nat.Prime (corner 1831 1) := by change ¬ Nat.Prime 13413907; exact PrattCertificates.composite_of_product (factor := 13) (cofactor := 1031839) rfl (by decide) (by decide)
    exact decide_eq_false composite
  second := false
  second_correct := by
    have composite : ¬ Nat.Prime (corner 1831 2) := by change ¬ Nat.Prime 13410245; exact PrattCertificates.composite_of_product (factor := 5) (cofactor := 2682049) rfl (by decide) (by decide)
    exact decide_eq_false composite
  third := false
  third_correct := by
    have composite : ¬ Nat.Prime (corner 1831 3) := by change ¬ Nat.Prime 13406583; exact PrattCertificates.composite_of_product (factor := 3) (cofactor := 4468861) rfl (by decide) (by decide)
    exact decide_eq_false composite
theorem row_count_1831 : row 1831 = 0 := (row_1831.checked).trans rfl
theorem model_count_1831 : modelRow 1831 = 0 := by decide
theorem modelrow_agrees_1831 : modelRow 1831 = row 1831 := model_count_1831.trans row_count_1831.symm
def row_1832 : RowCertificate 1832 where
  first := false
  first_correct := by
    have composite : ¬ Nat.Prime (corner 1832 1) := by change ¬ Nat.Prime 13428561; exact PrattCertificates.composite_of_product (factor := 3) (cofactor := 4476187) rfl (by decide) (by decide)
    exact decide_eq_false composite
  second := false
  second_correct := by
    have composite : ¬ Nat.Prime (corner 1832 2) := by change ¬ Nat.Prime 13424897; exact PrattCertificates.composite_of_product (factor := 97) (cofactor := 138401) rfl (by decide) (by decide)
    exact decide_eq_false composite
  third := false
  third_correct := by
    have composite : ¬ Nat.Prime (corner 1832 3) := by change ¬ Nat.Prime 13421233; exact PrattCertificates.composite_of_product (factor := 7) (cofactor := 1917319) rfl (by decide) (by decide)
    exact decide_eq_false composite
theorem row_count_1832 : row 1832 = 0 := (row_1832.checked).trans rfl
theorem model_count_1832 : modelRow 1832 = 0 := by decide
theorem modelrow_agrees_1832 : modelRow 1832 = row 1832 := model_count_1832.trans row_count_1832.symm
def row_1833 : RowCertificate 1833 where
  first := false
  first_correct := by
    have composite : ¬ Nat.Prime (corner 1833 1) := by change ¬ Nat.Prime 13443223; exact PrattCertificates.composite_of_product (factor := 307) (cofactor := 43789) rfl (by decide) (by decide)
    exact decide_eq_false composite
  second := false
  second_correct := by
    have composite : ¬ Nat.Prime (corner 1833 2) := by change ¬ Nat.Prime 13439557; exact PrattCertificates.composite_of_product (factor := 29) (cofactor := 463433) rfl (by decide) (by decide)
    exact decide_eq_false composite
  third := false
  third_correct := by
    have composite : ¬ Nat.Prime (corner 1833 3) := by change ¬ Nat.Prime 13435891; exact PrattCertificates.composite_of_product (factor := 7) (cofactor := 1919413) rfl (by decide) (by decide)
    exact decide_eq_false composite
theorem row_count_1833 : row 1833 = 0 := (row_1833.checked).trans rfl
theorem model_count_1833 : modelRow 1833 = 0 := by decide
theorem modelrow_agrees_1833 : modelRow 1833 = row 1833 := model_count_1833.trans row_count_1833.symm
def row_1834 : RowCertificate 1834 where
  first := false
  first_correct := by
    have composite : ¬ Nat.Prime (corner 1834 1) := by change ¬ Nat.Prime 13457893; exact PrattCertificates.composite_of_product (factor := 181) (cofactor := 74353) rfl (by decide) (by decide)
    exact decide_eq_false composite
  second := false
  second_correct := by
    have composite : ¬ Nat.Prime (corner 1834 2) := by change ¬ Nat.Prime 13454225; exact PrattCertificates.composite_of_product (factor := 5) (cofactor := 2690845) rfl (by decide) (by decide)
    exact decide_eq_false composite
  third := false
  third_correct := by
    have composite : ¬ Nat.Prime (corner 1834 3) := by change ¬ Nat.Prime 13450557; exact PrattCertificates.composite_of_product (factor := 3) (cofactor := 4483519) rfl (by decide) (by decide)
    exact decide_eq_false composite
theorem row_count_1834 : row 1834 = 0 := (row_1834.checked).trans rfl
theorem model_count_1834 : modelRow 1834 = 0 := by decide
theorem modelrow_agrees_1834 : modelRow 1834 = row 1834 := model_count_1834.trans row_count_1834.symm
def row_1835 : RowCertificate 1835 where
  first := false
  first_correct := by
    have composite : ¬ Nat.Prime (corner 1835 1) := by change ¬ Nat.Prime 13472571; exact PrattCertificates.composite_of_product (factor := 3) (cofactor := 4490857) rfl (by decide) (by decide)
    exact decide_eq_false composite
  second := true
  second_correct := by
    have prime : Nat.Prime (corner 1835 2) := by change Nat.Prime 13468901; exact ProjectEulerSolutions.P58PrimeNodes.p_13468901.prime
    exact decide_eq_true prime
  third := false
  third_correct := by
    have composite : ¬ Nat.Prime (corner 1835 3) := by change ¬ Nat.Prime 13465231; exact PrattCertificates.composite_of_product (factor := 13) (cofactor := 1035787) rfl (by decide) (by decide)
    exact decide_eq_false composite
theorem row_count_1835 : row 1835 = 1 := (row_1835.checked).trans rfl
theorem model_count_1835 : modelRow 1835 = 1 := by decide
theorem modelrow_agrees_1835 : modelRow 1835 = row 1835 := model_count_1835.trans row_count_1835.symm
def row_1836 : RowCertificate 1836 where
  first := false
  first_correct := by
    have composite : ¬ Nat.Prime (corner 1836 1) := by change ¬ Nat.Prime 13487257; exact PrattCertificates.composite_of_product (factor := 7) (cofactor := 1926751) rfl (by decide) (by decide)
    exact decide_eq_false composite
  second := false
  second_correct := by
    have composite : ¬ Nat.Prime (corner 1836 2) := by change ¬ Nat.Prime 13483585; exact PrattCertificates.composite_of_product (factor := 5) (cofactor := 2696717) rfl (by decide) (by decide)
    exact decide_eq_false composite
  third := true
  third_correct := by
    have prime : Nat.Prime (corner 1836 3) := by change Nat.Prime 13479913; exact ProjectEulerSolutions.P58PrimeNodes.p_13479913.prime
    exact decide_eq_true prime
theorem row_count_1836 : row 1836 = 1 := (row_1836.checked).trans rfl
theorem model_count_1836 : modelRow 1836 = 1 := by decide
theorem modelrow_agrees_1836 : modelRow 1836 = row 1836 := model_count_1836.trans row_count_1836.symm
def row_1837 : RowCertificate 1837 where
  first := false
  first_correct := by
    have composite : ¬ Nat.Prime (corner 1837 1) := by change ¬ Nat.Prime 13501951; exact PrattCertificates.composite_of_product (factor := 19) (cofactor := 710629) rfl (by decide) (by decide)
    exact decide_eq_false composite
  second := false
  second_correct := by
    have composite : ¬ Nat.Prime (corner 1837 2) := by change ¬ Nat.Prime 13498277; exact PrattCertificates.composite_of_product (factor := 13) (cofactor := 1038329) rfl (by decide) (by decide)
    exact decide_eq_false composite
  third := false
  third_correct := by
    have composite : ¬ Nat.Prime (corner 1837 3) := by change ¬ Nat.Prime 13494603; exact PrattCertificates.composite_of_product (factor := 3) (cofactor := 4498201) rfl (by decide) (by decide)
    exact decide_eq_false composite
theorem row_count_1837 : row 1837 = 0 := (row_1837.checked).trans rfl
theorem model_count_1837 : modelRow 1837 = 0 := by decide
theorem modelrow_agrees_1837 : modelRow 1837 = row 1837 := model_count_1837.trans row_count_1837.symm
def row_1838 : RowCertificate 1838 where
  first := false
  first_correct := by
    have composite : ¬ Nat.Prime (corner 1838 1) := by change ¬ Nat.Prime 13516653; exact PrattCertificates.composite_of_product (factor := 3) (cofactor := 4505551) rfl (by decide) (by decide)
    exact decide_eq_false composite
  second := false
  second_correct := by
    have composite : ¬ Nat.Prime (corner 1838 2) := by change ¬ Nat.Prime 13512977; exact PrattCertificates.composite_of_product (factor := 17) (cofactor := 794881) rfl (by decide) (by decide)
    exact decide_eq_false composite
  third := false
  third_correct := by
    have composite : ¬ Nat.Prime (corner 1838 3) := by change ¬ Nat.Prime 13509301; exact PrattCertificates.composite_of_product (factor := 13) (cofactor := 1039177) rfl (by decide) (by decide)
    exact decide_eq_false composite
theorem row_count_1838 : row 1838 = 0 := (row_1838.checked).trans rfl
theorem model_count_1838 : modelRow 1838 = 0 := by decide
theorem modelrow_agrees_1838 : modelRow 1838 = row 1838 := model_count_1838.trans row_count_1838.symm
def row_1839 : RowCertificate 1839 where
  first := false
  first_correct := by
    have composite : ¬ Nat.Prime (corner 1839 1) := by change ¬ Nat.Prime 13531363; exact PrattCertificates.composite_of_product (factor := 19) (cofactor := 712177) rfl (by decide) (by decide)
    exact decide_eq_false composite
  second := false
  second_correct := by
    have composite : ¬ Nat.Prime (corner 1839 2) := by change ¬ Nat.Prime 13527685; exact PrattCertificates.composite_of_product (factor := 5) (cofactor := 2705537) rfl (by decide) (by decide)
    exact decide_eq_false composite
  third := false
  third_correct := by
    have composite : ¬ Nat.Prime (corner 1839 3) := by change ¬ Nat.Prime 13524007; exact PrattCertificates.composite_of_product (factor := 7) (cofactor := 1932001) rfl (by decide) (by decide)
    exact decide_eq_false composite
theorem row_count_1839 : row 1839 = 0 := (row_1839.checked).trans rfl
theorem model_count_1839 : modelRow 1839 = 0 := by decide
theorem modelrow_agrees_1839 : modelRow 1839 = row 1839 := model_count_1839.trans row_count_1839.symm
def row_1840 : RowCertificate 1840 where
  first := false
  first_correct := by
    have composite : ¬ Nat.Prime (corner 1840 1) := by change ¬ Nat.Prime 13546081; exact PrattCertificates.composite_of_product (factor := 643) (cofactor := 21067) rfl (by decide) (by decide)
    exact decide_eq_false composite
  second := false
  second_correct := by
    have composite : ¬ Nat.Prime (corner 1840 2) := by change ¬ Nat.Prime 13542401; exact PrattCertificates.composite_of_product (factor := 53) (cofactor := 255517) rfl (by decide) (by decide)
    exact decide_eq_false composite
  third := false
  third_correct := by
    have composite : ¬ Nat.Prime (corner 1840 3) := by change ¬ Nat.Prime 13538721; exact PrattCertificates.composite_of_product (factor := 3) (cofactor := 4512907) rfl (by decide) (by decide)
    exact decide_eq_false composite
theorem row_count_1840 : row 1840 = 0 := (row_1840.checked).trans rfl
theorem model_count_1840 : modelRow 1840 = 0 := by decide
theorem modelrow_agrees_1840 : modelRow 1840 = row 1840 := model_count_1840.trans row_count_1840.symm
def row_1841 : RowCertificate 1841 where
  first := false
  first_correct := by
    have composite : ¬ Nat.Prime (corner 1841 1) := by change ¬ Nat.Prime 13560807; exact PrattCertificates.composite_of_product (factor := 3) (cofactor := 4520269) rfl (by decide) (by decide)
    exact decide_eq_false composite
  second := false
  second_correct := by
    have composite : ¬ Nat.Prime (corner 1841 2) := by change ¬ Nat.Prime 13557125; exact PrattCertificates.composite_of_product (factor := 5) (cofactor := 2711425) rfl (by decide) (by decide)
    exact decide_eq_false composite
  third := false
  third_correct := by
    have composite : ¬ Nat.Prime (corner 1841 3) := by change ¬ Nat.Prime 13553443; exact PrattCertificates.composite_of_product (factor := 2617) (cofactor := 5179) rfl (by decide) (by decide)
    exact decide_eq_false composite
theorem row_count_1841 : row 1841 = 0 := (row_1841.checked).trans rfl
theorem model_count_1841 : modelRow 1841 = 0 := by decide
theorem modelrow_agrees_1841 : modelRow 1841 = row 1841 := model_count_1841.trans row_count_1841.symm
def row_1842 : RowCertificate 1842 where
  first := false
  first_correct := by
    have composite : ¬ Nat.Prime (corner 1842 1) := by change ¬ Nat.Prime 13575541; exact PrattCertificates.composite_of_product (factor := 7) (cofactor := 1939363) rfl (by decide) (by decide)
    exact decide_eq_false composite
  second := false
  second_correct := by
    have composite : ¬ Nat.Prime (corner 1842 2) := by change ¬ Nat.Prime 13571857; exact PrattCertificates.composite_of_product (factor := 13) (cofactor := 1043989) rfl (by decide) (by decide)
    exact decide_eq_false composite
  third := false
  third_correct := by
    have composite : ¬ Nat.Prime (corner 1842 3) := by change ¬ Nat.Prime 13568173; exact PrattCertificates.composite_of_product (factor := 31) (cofactor := 437683) rfl (by decide) (by decide)
    exact decide_eq_false composite
theorem row_count_1842 : row 1842 = 0 := (row_1842.checked).trans rfl
theorem model_count_1842 : modelRow 1842 = 0 := by decide
theorem modelrow_agrees_1842 : modelRow 1842 = row 1842 := model_count_1842.trans row_count_1842.symm
def row_1843 : RowCertificate 1843 where
  first := false
  first_correct := by
    have composite : ¬ Nat.Prime (corner 1843 1) := by change ¬ Nat.Prime 13590283; exact PrattCertificates.composite_of_product (factor := 7) (cofactor := 1941469) rfl (by decide) (by decide)
    exact decide_eq_false composite
  second := true
  second_correct := by
    have prime : Nat.Prime (corner 1843 2) := by change Nat.Prime 13586597; exact ProjectEulerSolutions.P58PrimeNodes.p_13586597.prime
    exact decide_eq_true prime
  third := false
  third_correct := by
    have composite : ¬ Nat.Prime (corner 1843 3) := by change ¬ Nat.Prime 13582911; exact PrattCertificates.composite_of_product (factor := 3) (cofactor := 4527637) rfl (by decide) (by decide)
    exact decide_eq_false composite
theorem row_count_1843 : row 1843 = 1 := (row_1843.checked).trans rfl
theorem model_count_1843 : modelRow 1843 = 1 := by decide
theorem modelrow_agrees_1843 : modelRow 1843 = row 1843 := model_count_1843.trans row_count_1843.symm
def row_1844 : RowCertificate 1844 where
  first := false
  first_correct := by
    have composite : ¬ Nat.Prime (corner 1844 1) := by change ¬ Nat.Prime 13605033; exact PrattCertificates.composite_of_product (factor := 3) (cofactor := 4535011) rfl (by decide) (by decide)
    exact decide_eq_false composite
  second := false
  second_correct := by
    have composite : ¬ Nat.Prime (corner 1844 2) := by change ¬ Nat.Prime 13601345; exact PrattCertificates.composite_of_product (factor := 5) (cofactor := 2720269) rfl (by decide) (by decide)
    exact decide_eq_false composite
  third := true
  third_correct := by
    have prime : Nat.Prime (corner 1844 3) := by change Nat.Prime 13597657; exact ProjectEulerSolutions.P58PrimeNodes.p_13597657.prime
    exact decide_eq_true prime
theorem row_count_1844 : row 1844 = 1 := (row_1844.checked).trans rfl
theorem model_count_1844 : modelRow 1844 = 1 := by decide
theorem modelrow_agrees_1844 : modelRow 1844 = row 1844 := model_count_1844.trans row_count_1844.symm
def row_1845 : RowCertificate 1845 where
  first := false
  first_correct := by
    have composite : ¬ Nat.Prime (corner 1845 1) := by change ¬ Nat.Prime 13619791; exact PrattCertificates.composite_of_product (factor := 163) (cofactor := 83557) rfl (by decide) (by decide)
    exact decide_eq_false composite
  second := false
  second_correct := by
    have composite : ¬ Nat.Prime (corner 1845 2) := by change ¬ Nat.Prime 13616101; exact PrattCertificates.composite_of_product (factor := 317) (cofactor := 42953) rfl (by decide) (by decide)
    exact decide_eq_false composite
  third := false
  third_correct := by
    have composite : ¬ Nat.Prime (corner 1845 3) := by change ¬ Nat.Prime 13612411; exact PrattCertificates.composite_of_product (factor := 37) (cofactor := 367903) rfl (by decide) (by decide)
    exact decide_eq_false composite
theorem row_count_1845 : row 1845 = 0 := (row_1845.checked).trans rfl
theorem model_count_1845 : modelRow 1845 = 0 := by decide
theorem modelrow_agrees_1845 : modelRow 1845 = row 1845 := model_count_1845.trans row_count_1845.symm
def row_1846 : RowCertificate 1846 where
  first := true
  first_correct := by
    have prime : Nat.Prime (corner 1846 1) := by change Nat.Prime 13634557; exact ProjectEulerSolutions.P58PrimeNodes.p_13634557.prime
    exact decide_eq_true prime
  second := false
  second_correct := by
    have composite : ¬ Nat.Prime (corner 1846 2) := by change ¬ Nat.Prime 13630865; exact PrattCertificates.composite_of_product (factor := 5) (cofactor := 2726173) rfl (by decide) (by decide)
    exact decide_eq_false composite
  third := false
  third_correct := by
    have composite : ¬ Nat.Prime (corner 1846 3) := by change ¬ Nat.Prime 13627173; exact PrattCertificates.composite_of_product (factor := 3) (cofactor := 4542391) rfl (by decide) (by decide)
    exact decide_eq_false composite
theorem row_count_1846 : row 1846 = 1 := (row_1846.checked).trans rfl
theorem model_count_1846 : modelRow 1846 = 1 := by decide
theorem modelrow_agrees_1846 : modelRow 1846 = row 1846 := model_count_1846.trans row_count_1846.symm
def row_1847 : RowCertificate 1847 where
  first := false
  first_correct := by
    have composite : ¬ Nat.Prime (corner 1847 1) := by change ¬ Nat.Prime 13649331; exact PrattCertificates.composite_of_product (factor := 3) (cofactor := 4549777) rfl (by decide) (by decide)
    exact decide_eq_false composite
  second := false
  second_correct := by
    have composite : ¬ Nat.Prime (corner 1847 2) := by change ¬ Nat.Prime 13645637; exact PrattCertificates.composite_of_product (factor := 37) (cofactor := 368801) rfl (by decide) (by decide)
    exact decide_eq_false composite
  third := false
  third_correct := by
    have composite : ¬ Nat.Prime (corner 1847 3) := by change ¬ Nat.Prime 13641943; exact PrattCertificates.composite_of_product (factor := 7) (cofactor := 1948849) rfl (by decide) (by decide)
    exact decide_eq_false composite
theorem row_count_1847 : row 1847 = 0 := (row_1847.checked).trans rfl
theorem model_count_1847 : modelRow 1847 = 0 := by decide
theorem modelrow_agrees_1847 : modelRow 1847 = row 1847 := model_count_1847.trans row_count_1847.symm
def row_1848 : RowCertificate 1848 where
  first := false
  first_correct := by
    have composite : ¬ Nat.Prime (corner 1848 1) := by change ¬ Nat.Prime 13664113; exact PrattCertificates.composite_of_product (factor := 859) (cofactor := 15907) rfl (by decide) (by decide)
    exact decide_eq_false composite
  second := false
  second_correct := by
    have composite : ¬ Nat.Prime (corner 1848 2) := by change ¬ Nat.Prime 13660417; exact PrattCertificates.composite_of_product (factor := 73) (cofactor := 187129) rfl (by decide) (by decide)
    exact decide_eq_false composite
  third := false
  third_correct := by
    have composite : ¬ Nat.Prime (corner 1848 3) := by change ¬ Nat.Prime 13656721; exact PrattCertificates.composite_of_product (factor := 13) (cofactor := 1050517) rfl (by decide) (by decide)
    exact decide_eq_false composite
theorem row_count_1848 : row 1848 = 0 := (row_1848.checked).trans rfl
theorem model_count_1848 : modelRow 1848 = 0 := by decide
theorem modelrow_agrees_1848 : modelRow 1848 = row 1848 := model_count_1848.trans row_count_1848.symm
def row_1849 : RowCertificate 1849 where
  first := false
  first_correct := by
    have composite : ¬ Nat.Prime (corner 1849 1) := by change ¬ Nat.Prime 13678903; exact PrattCertificates.composite_of_product (factor := 7) (cofactor := 1954129) rfl (by decide) (by decide)
    exact decide_eq_false composite
  second := false
  second_correct := by
    have composite : ¬ Nat.Prime (corner 1849 2) := by change ¬ Nat.Prime 13675205; exact PrattCertificates.composite_of_product (factor := 5) (cofactor := 2735041) rfl (by decide) (by decide)
    exact decide_eq_false composite
  third := false
  third_correct := by
    have composite : ¬ Nat.Prime (corner 1849 3) := by change ¬ Nat.Prime 13671507; exact PrattCertificates.composite_of_product (factor := 3) (cofactor := 4557169) rfl (by decide) (by decide)
    exact decide_eq_false composite
theorem row_count_1849 : row 1849 = 0 := (row_1849.checked).trans rfl
theorem model_count_1849 : modelRow 1849 = 0 := by decide
theorem modelrow_agrees_1849 : modelRow 1849 = row 1849 := model_count_1849.trans row_count_1849.symm
def row_1850 : RowCertificate 1850 where
  first := false
  first_correct := by
    have composite : ¬ Nat.Prime (corner 1850 1) := by change ¬ Nat.Prime 13693701; exact PrattCertificates.composite_of_product (factor := 3) (cofactor := 4564567) rfl (by decide) (by decide)
    exact decide_eq_false composite
  second := false
  second_correct := by
    have composite : ¬ Nat.Prime (corner 1850 2) := by change ¬ Nat.Prime 13690001; exact PrattCertificates.composite_of_product (factor := 13) (cofactor := 1053077) rfl (by decide) (by decide)
    exact decide_eq_false composite
  third := false
  third_correct := by
    have composite : ¬ Nat.Prime (corner 1850 3) := by change ¬ Nat.Prime 13686301; exact PrattCertificates.composite_of_product (factor := 3163) (cofactor := 4327) rfl (by decide) (by decide)
    exact decide_eq_false composite
theorem row_count_1850 : row 1850 = 0 := (row_1850.checked).trans rfl
theorem model_count_1850 : modelRow 1850 = 0 := by decide
theorem modelrow_agrees_1850 : modelRow 1850 = row 1850 := model_count_1850.trans row_count_1850.symm
def row_1851 : RowCertificate 1851 where
  first := false
  first_correct := by
    have composite : ¬ Nat.Prime (corner 1851 1) := by change ¬ Nat.Prime 13708507; exact PrattCertificates.composite_of_product (factor := 127) (cofactor := 107941) rfl (by decide) (by decide)
    exact decide_eq_false composite
  second := false
  second_correct := by
    have composite : ¬ Nat.Prime (corner 1851 2) := by change ¬ Nat.Prime 13704805; exact PrattCertificates.composite_of_product (factor := 5) (cofactor := 2740961) rfl (by decide) (by decide)
    exact decide_eq_false composite
  third := false
  third_correct := by
    have composite : ¬ Nat.Prime (corner 1851 3) := by change ¬ Nat.Prime 13701103; exact PrattCertificates.composite_of_product (factor := 13) (cofactor := 1053931) rfl (by decide) (by decide)
    exact decide_eq_false composite
theorem row_count_1851 : row 1851 = 0 := (row_1851.checked).trans rfl
theorem model_count_1851 : modelRow 1851 = 0 := by decide
theorem modelrow_agrees_1851 : modelRow 1851 = row 1851 := model_count_1851.trans row_count_1851.symm
def row_1852 : RowCertificate 1852 where
  first := false
  first_correct := by
    have composite : ¬ Nat.Prime (corner 1852 1) := by change ¬ Nat.Prime 13723321; exact PrattCertificates.composite_of_product (factor := 43) (cofactor := 319147) rfl (by decide) (by decide)
    exact decide_eq_false composite
  second := false
  second_correct := by
    have composite : ¬ Nat.Prime (corner 1852 2) := by change ¬ Nat.Prime 13719617; exact PrattCertificates.composite_of_product (factor := 89) (cofactor := 154153) rfl (by decide) (by decide)
    exact decide_eq_false composite
  third := false
  third_correct := by
    have composite : ¬ Nat.Prime (corner 1852 3) := by change ¬ Nat.Prime 13715913; exact PrattCertificates.composite_of_product (factor := 3) (cofactor := 4571971) rfl (by decide) (by decide)
    exact decide_eq_false composite
theorem row_count_1852 : row 1852 = 0 := (row_1852.checked).trans rfl
theorem model_count_1852 : modelRow 1852 = 0 := by decide
theorem modelrow_agrees_1852 : modelRow 1852 = row 1852 := model_count_1852.trans row_count_1852.symm
def row_1853 : RowCertificate 1853 where
  first := false
  first_correct := by
    have composite : ¬ Nat.Prime (corner 1853 1) := by change ¬ Nat.Prime 13738143; exact PrattCertificates.composite_of_product (factor := 3) (cofactor := 4579381) rfl (by decide) (by decide)
    exact decide_eq_false composite
  second := false
  second_correct := by
    have composite : ¬ Nat.Prime (corner 1853 2) := by change ¬ Nat.Prime 13734437; exact PrattCertificates.composite_of_product (factor := 37) (cofactor := 371201) rfl (by decide) (by decide)
    exact decide_eq_false composite
  third := false
  third_correct := by
    have composite : ¬ Nat.Prime (corner 1853 3) := by change ¬ Nat.Prime 13730731; exact PrattCertificates.composite_of_product (factor := 7) (cofactor := 1961533) rfl (by decide) (by decide)
    exact decide_eq_false composite
theorem row_count_1853 : row 1853 = 0 := (row_1853.checked).trans rfl
theorem model_count_1853 : modelRow 1853 = 0 := by decide
theorem modelrow_agrees_1853 : modelRow 1853 = row 1853 := model_count_1853.trans row_count_1853.symm
def row_1854 : RowCertificate 1854 where
  first := false
  first_correct := by
    have composite : ¬ Nat.Prime (corner 1854 1) := by change ¬ Nat.Prime 13752973; exact PrattCertificates.composite_of_product (factor := 13) (cofactor := 1057921) rfl (by decide) (by decide)
    exact decide_eq_false composite
  second := false
  second_correct := by
    have composite : ¬ Nat.Prime (corner 1854 2) := by change ¬ Nat.Prime 13749265; exact PrattCertificates.composite_of_product (factor := 5) (cofactor := 2749853) rfl (by decide) (by decide)
    exact decide_eq_false composite
  third := false
  third_correct := by
    have composite : ¬ Nat.Prime (corner 1854 3) := by change ¬ Nat.Prime 13745557; exact PrattCertificates.composite_of_product (factor := 7) (cofactor := 1963651) rfl (by decide) (by decide)
    exact decide_eq_false composite
theorem row_count_1854 : row 1854 = 0 := (row_1854.checked).trans rfl
theorem model_count_1854 : modelRow 1854 = 0 := by decide
theorem modelrow_agrees_1854 : modelRow 1854 = row 1854 := model_count_1854.trans row_count_1854.symm
def row_1855 : RowCertificate 1855 where
  first := false
  first_correct := by
    have composite : ¬ Nat.Prime (corner 1855 1) := by change ¬ Nat.Prime 13767811; exact PrattCertificates.composite_of_product (factor := 37) (cofactor := 372103) rfl (by decide) (by decide)
    exact decide_eq_false composite
  second := false
  second_correct := by
    have composite : ¬ Nat.Prime (corner 1855 2) := by change ¬ Nat.Prime 13764101; exact PrattCertificates.composite_of_product (factor := 13) (cofactor := 1058777) rfl (by decide) (by decide)
    exact decide_eq_false composite
  third := false
  third_correct := by
    have composite : ¬ Nat.Prime (corner 1855 3) := by change ¬ Nat.Prime 13760391; exact PrattCertificates.composite_of_product (factor := 3) (cofactor := 4586797) rfl (by decide) (by decide)
    exact decide_eq_false composite
theorem row_count_1855 : row 1855 = 0 := (row_1855.checked).trans rfl
theorem model_count_1855 : modelRow 1855 = 0 := by decide
theorem modelrow_agrees_1855 : modelRow 1855 = row 1855 := model_count_1855.trans row_count_1855.symm
def row_1856 : RowCertificate 1856 where
  first := false
  first_correct := by
    have composite : ¬ Nat.Prime (corner 1856 1) := by change ¬ Nat.Prime 13782657; exact PrattCertificates.composite_of_product (factor := 3) (cofactor := 4594219) rfl (by decide) (by decide)
    exact decide_eq_false composite
  second := false
  second_correct := by
    have composite : ¬ Nat.Prime (corner 1856 2) := by change ¬ Nat.Prime 13778945; exact PrattCertificates.composite_of_product (factor := 5) (cofactor := 2755789) rfl (by decide) (by decide)
    exact decide_eq_false composite
  third := true
  third_correct := by
    have prime : Nat.Prime (corner 1856 3) := by change Nat.Prime 13775233; exact ProjectEulerSolutions.P58PrimeNodes.p_13775233.prime
    exact decide_eq_true prime
theorem row_count_1856 : row 1856 = 1 := (row_1856.checked).trans rfl
theorem model_count_1856 : modelRow 1856 = 1 := by decide
theorem modelrow_agrees_1856 : modelRow 1856 = row 1856 := model_count_1856.trans row_count_1856.symm
def row_1857 : RowCertificate 1857 where
  first := false
  first_correct := by
    have composite : ¬ Nat.Prime (corner 1857 1) := by change ¬ Nat.Prime 13797511; exact PrattCertificates.composite_of_product (factor := 7) (cofactor := 1971073) rfl (by decide) (by decide)
    exact decide_eq_false composite
  second := false
  second_correct := by
    have composite : ¬ Nat.Prime (corner 1857 2) := by change ¬ Nat.Prime 13793797; exact PrattCertificates.composite_of_product (factor := 113) (cofactor := 122069) rfl (by decide) (by decide)
    exact decide_eq_false composite
  third := true
  third_correct := by
    have prime : Nat.Prime (corner 1857 3) := by change Nat.Prime 13790083; exact ProjectEulerSolutions.P58PrimeNodes.p_13790083.prime
    exact decide_eq_true prime
theorem row_count_1857 : row 1857 = 1 := (row_1857.checked).trans rfl
theorem model_count_1857 : modelRow 1857 = 1 := by decide
theorem modelrow_agrees_1857 : modelRow 1857 = row 1857 := model_count_1857.trans row_count_1857.symm
def row_1858 : RowCertificate 1858 where
  first := false
  first_correct := by
    have composite : ¬ Nat.Prime (corner 1858 1) := by change ¬ Nat.Prime 13812373; exact PrattCertificates.composite_of_product (factor := 19) (cofactor := 726967) rfl (by decide) (by decide)
    exact decide_eq_false composite
  second := true
  second_correct := by
    have prime : Nat.Prime (corner 1858 2) := by change Nat.Prime 13808657; exact ProjectEulerSolutions.P58PrimeNodes.p_13808657.prime
    exact decide_eq_true prime
  third := false
  third_correct := by
    have composite : ¬ Nat.Prime (corner 1858 3) := by change ¬ Nat.Prime 13804941; exact PrattCertificates.composite_of_product (factor := 3) (cofactor := 4601647) rfl (by decide) (by decide)
    exact decide_eq_false composite
theorem row_count_1858 : row 1858 = 1 := (row_1858.checked).trans rfl
theorem model_count_1858 : modelRow 1858 = 1 := by decide
theorem modelrow_agrees_1858 : modelRow 1858 = row 1858 := model_count_1858.trans row_count_1858.symm
def row_1859 : RowCertificate 1859 where
  first := false
  first_correct := by
    have composite : ¬ Nat.Prime (corner 1859 1) := by change ¬ Nat.Prime 13827243; exact PrattCertificates.composite_of_product (factor := 3) (cofactor := 4609081) rfl (by decide) (by decide)
    exact decide_eq_false composite
  second := false
  second_correct := by
    have composite : ¬ Nat.Prime (corner 1859 2) := by change ¬ Nat.Prime 13823525; exact PrattCertificates.composite_of_product (factor := 5) (cofactor := 2764705) rfl (by decide) (by decide)
    exact decide_eq_false composite
  third := false
  third_correct := by
    have composite : ¬ Nat.Prime (corner 1859 3) := by change ¬ Nat.Prime 13819807; exact PrattCertificates.composite_of_product (factor := 277) (cofactor := 49891) rfl (by decide) (by decide)
    exact decide_eq_false composite
theorem row_count_1859 : row 1859 = 0 := (row_1859.checked).trans rfl
theorem model_count_1859 : modelRow 1859 = 0 := by decide
theorem modelrow_agrees_1859 : modelRow 1859 = row 1859 := model_count_1859.trans row_count_1859.symm
def row_1860 : RowCertificate 1860 where
  first := true
  first_correct := by
    have prime : Nat.Prime (corner 1860 1) := by change Nat.Prime 13842121; exact ProjectEulerSolutions.P58PrimeNodes.p_13842121.prime
    exact decide_eq_true prime
  second := false
  second_correct := by
    have composite : ¬ Nat.Prime (corner 1860 2) := by change ¬ Nat.Prime 13838401; exact PrattCertificates.composite_of_product (factor := 3313) (cofactor := 4177) rfl (by decide) (by decide)
    exact decide_eq_false composite
  third := false
  third_correct := by
    have composite : ¬ Nat.Prime (corner 1860 3) := by change ¬ Nat.Prime 13834681; exact PrattCertificates.composite_of_product (factor := 7) (cofactor := 1976383) rfl (by decide) (by decide)
    exact decide_eq_false composite
theorem row_count_1860 : row 1860 = 1 := (row_1860.checked).trans rfl
theorem model_count_1860 : modelRow 1860 = 1 := by decide
theorem modelrow_agrees_1860 : modelRow 1860 = row 1860 := model_count_1860.trans row_count_1860.symm
def row_1861 : RowCertificate 1861 where
  first := false
  first_correct := by
    have composite : ¬ Nat.Prime (corner 1861 1) := by change ¬ Nat.Prime 13857007; exact PrattCertificates.composite_of_product (factor := 67) (cofactor := 206821) rfl (by decide) (by decide)
    exact decide_eq_false composite
  second := false
  second_correct := by
    have composite : ¬ Nat.Prime (corner 1861 2) := by change ¬ Nat.Prime 13853285; exact PrattCertificates.composite_of_product (factor := 5) (cofactor := 2770657) rfl (by decide) (by decide)
    exact decide_eq_false composite
  third := false
  third_correct := by
    have composite : ¬ Nat.Prime (corner 1861 3) := by change ¬ Nat.Prime 13849563; exact PrattCertificates.composite_of_product (factor := 3) (cofactor := 4616521) rfl (by decide) (by decide)
    exact decide_eq_false composite
theorem row_count_1861 : row 1861 = 0 := (row_1861.checked).trans rfl
theorem model_count_1861 : modelRow 1861 = 0 := by decide
theorem modelrow_agrees_1861 : modelRow 1861 = row 1861 := model_count_1861.trans row_count_1861.symm
def row_1862 : RowCertificate 1862 where
  first := false
  first_correct := by
    have composite : ¬ Nat.Prime (corner 1862 1) := by change ¬ Nat.Prime 13871901; exact PrattCertificates.composite_of_product (factor := 3) (cofactor := 4623967) rfl (by decide) (by decide)
    exact decide_eq_false composite
  second := false
  second_correct := by
    have composite : ¬ Nat.Prime (corner 1862 2) := by change ¬ Nat.Prime 13868177; exact PrattCertificates.composite_of_product (factor := 29) (cofactor := 478213) rfl (by decide) (by decide)
    exact decide_eq_false composite
  third := false
  third_correct := by
    have composite : ¬ Nat.Prime (corner 1862 3) := by change ¬ Nat.Prime 13864453; exact PrattCertificates.composite_of_product (factor := 283) (cofactor := 48991) rfl (by decide) (by decide)
    exact decide_eq_false composite
theorem row_count_1862 : row 1862 = 0 := (row_1862.checked).trans rfl
theorem model_count_1862 : modelRow 1862 = 0 := by decide
theorem modelrow_agrees_1862 : modelRow 1862 = row 1862 := model_count_1862.trans row_count_1862.symm
def row_1863 : RowCertificate 1863 where
  first := false
  first_correct := by
    have composite : ¬ Nat.Prime (corner 1863 1) := by change ¬ Nat.Prime 13886803; exact PrattCertificates.composite_of_product (factor := 7) (cofactor := 1983829) rfl (by decide) (by decide)
    exact decide_eq_false composite
  second := false
  second_correct := by
    have composite : ¬ Nat.Prime (corner 1863 2) := by change ¬ Nat.Prime 13883077; exact PrattCertificates.composite_of_product (factor := 13) (cofactor := 1067929) rfl (by decide) (by decide)
    exact decide_eq_false composite
  third := false
  third_correct := by
    have composite : ¬ Nat.Prime (corner 1863 3) := by change ¬ Nat.Prime 13879351; exact PrattCertificates.composite_of_product (factor := 31) (cofactor := 447721) rfl (by decide) (by decide)
    exact decide_eq_false composite
theorem row_count_1863 : row 1863 = 0 := (row_1863.checked).trans rfl
theorem model_count_1863 : modelRow 1863 = 0 := by decide
theorem modelrow_agrees_1863 : modelRow 1863 = row 1863 := model_count_1863.trans row_count_1863.symm
def row_1864 : RowCertificate 1864 where
  first := false
  first_correct := by
    have composite : ¬ Nat.Prime (corner 1864 1) := by change ¬ Nat.Prime 13901713; exact PrattCertificates.composite_of_product (factor := 7) (cofactor := 1985959) rfl (by decide) (by decide)
    exact decide_eq_false composite
  second := false
  second_correct := by
    have composite : ¬ Nat.Prime (corner 1864 2) := by change ¬ Nat.Prime 13897985; exact PrattCertificates.composite_of_product (factor := 5) (cofactor := 2779597) rfl (by decide) (by decide)
    exact decide_eq_false composite
  third := false
  third_correct := by
    have composite : ¬ Nat.Prime (corner 1864 3) := by change ¬ Nat.Prime 13894257; exact PrattCertificates.composite_of_product (factor := 3) (cofactor := 4631419) rfl (by decide) (by decide)
    exact decide_eq_false composite
theorem row_count_1864 : row 1864 = 0 := (row_1864.checked).trans rfl
theorem model_count_1864 : modelRow 1864 = 0 := by decide
theorem modelrow_agrees_1864 : modelRow 1864 = row 1864 := model_count_1864.trans row_count_1864.symm
def row_1865 : RowCertificate 1865 where
  first := false
  first_correct := by
    have composite : ¬ Nat.Prime (corner 1865 1) := by change ¬ Nat.Prime 13916631; exact PrattCertificates.composite_of_product (factor := 3) (cofactor := 4638877) rfl (by decide) (by decide)
    exact decide_eq_false composite
  second := true
  second_correct := by
    have prime : Nat.Prime (corner 1865 2) := by change Nat.Prime 13912901; exact ProjectEulerSolutions.P58PrimeNodes.p_13912901.prime
    exact decide_eq_true prime
  third := false
  third_correct := by
    have composite : ¬ Nat.Prime (corner 1865 3) := by change ¬ Nat.Prime 13909171; exact PrattCertificates.composite_of_product (factor := 2503) (cofactor := 5557) rfl (by decide) (by decide)
    exact decide_eq_false composite
theorem row_count_1865 : row 1865 = 1 := (row_1865.checked).trans rfl
theorem model_count_1865 : modelRow 1865 = 1 := by decide
theorem modelrow_agrees_1865 : modelRow 1865 = row 1865 := model_count_1865.trans row_count_1865.symm
def row_1866 : RowCertificate 1866 where
  first := true
  first_correct := by
    have prime : Nat.Prime (corner 1866 1) := by change Nat.Prime 13931557; exact ProjectEulerSolutions.P58PrimeNodes.p_13931557.prime
    exact decide_eq_true prime
  second := false
  second_correct := by
    have composite : ¬ Nat.Prime (corner 1866 2) := by change ¬ Nat.Prime 13927825; exact PrattCertificates.composite_of_product (factor := 5) (cofactor := 2785565) rfl (by decide) (by decide)
    exact decide_eq_false composite
  third := false
  third_correct := by
    have composite : ¬ Nat.Prime (corner 1866 3) := by change ¬ Nat.Prime 13924093; exact PrattCertificates.composite_of_product (factor := 19) (cofactor := 732847) rfl (by decide) (by decide)
    exact decide_eq_false composite
theorem row_count_1866 : row 1866 = 1 := (row_1866.checked).trans rfl
theorem model_count_1866 : modelRow 1866 = 1 := by decide
theorem modelrow_agrees_1866 : modelRow 1866 = row 1866 := model_count_1866.trans row_count_1866.symm
def row_1867 : RowCertificate 1867 where
  first := false
  first_correct := by
    have composite : ¬ Nat.Prime (corner 1867 1) := by change ¬ Nat.Prime 13946491; exact PrattCertificates.composite_of_product (factor := 13) (cofactor := 1072807) rfl (by decide) (by decide)
    exact decide_eq_false composite
  second := true
  second_correct := by
    have prime : Nat.Prime (corner 1867 2) := by change Nat.Prime 13942757; exact ProjectEulerSolutions.P58PrimeNodes.p_13942757.prime
    exact decide_eq_true prime
  third := false
  third_correct := by
    have composite : ¬ Nat.Prime (corner 1867 3) := by change ¬ Nat.Prime 13939023; exact PrattCertificates.composite_of_product (factor := 3) (cofactor := 4646341) rfl (by decide) (by decide)
    exact decide_eq_false composite
theorem row_count_1867 : row 1867 = 1 := (row_1867.checked).trans rfl
theorem model_count_1867 : modelRow 1867 = 1 := by decide
theorem modelrow_agrees_1867 : modelRow 1867 = row 1867 := model_count_1867.trans row_count_1867.symm
def row_1868 : RowCertificate 1868 where
  first := false
  first_correct := by
    have composite : ¬ Nat.Prime (corner 1868 1) := by change ¬ Nat.Prime 13961433; exact PrattCertificates.composite_of_product (factor := 3) (cofactor := 4653811) rfl (by decide) (by decide)
    exact decide_eq_false composite
  second := false
  second_correct := by
    have composite : ¬ Nat.Prime (corner 1868 2) := by change ¬ Nat.Prime 13957697; exact PrattCertificates.composite_of_product (factor := 13) (cofactor := 1073669) rfl (by decide) (by decide)
    exact decide_eq_false composite
  third := false
  third_correct := by
    have composite : ¬ Nat.Prime (corner 1868 3) := by change ¬ Nat.Prime 13953961; exact PrattCertificates.composite_of_product (factor := 7) (cofactor := 1993423) rfl (by decide) (by decide)
    exact decide_eq_false composite
theorem row_count_1868 : row 1868 = 0 := (row_1868.checked).trans rfl
theorem model_count_1868 : modelRow 1868 = 0 := by decide
theorem modelrow_agrees_1868 : modelRow 1868 = row 1868 := model_count_1868.trans row_count_1868.symm
def row_1869 : RowCertificate 1869 where
  first := false
  first_correct := by
    have composite : ¬ Nat.Prime (corner 1869 1) := by change ¬ Nat.Prime 13976383; exact PrattCertificates.composite_of_product (factor := 379) (cofactor := 36877) rfl (by decide) (by decide)
    exact decide_eq_false composite
  second := false
  second_correct := by
    have composite : ¬ Nat.Prime (corner 1869 2) := by change ¬ Nat.Prime 13972645; exact PrattCertificates.composite_of_product (factor := 5) (cofactor := 2794529) rfl (by decide) (by decide)
    exact decide_eq_false composite
  third := true
  third_correct := by
    have prime : Nat.Prime (corner 1869 3) := by change Nat.Prime 13968907; exact ProjectEulerSolutions.P58PrimeNodes.p_13968907.prime
    exact decide_eq_true prime
theorem row_count_1869 : row 1869 = 1 := (row_1869.checked).trans rfl
theorem model_count_1869 : modelRow 1869 = 1 := by decide
theorem modelrow_agrees_1869 : modelRow 1869 = row 1869 := model_count_1869.trans row_count_1869.symm
def row_1870 : RowCertificate 1870 where
  first := false
  first_correct := by
    have composite : ¬ Nat.Prime (corner 1870 1) := by change ¬ Nat.Prime 13991341; exact PrattCertificates.composite_of_product (factor := 7) (cofactor := 1998763) rfl (by decide) (by decide)
    exact decide_eq_false composite
  second := false
  second_correct := by
    have composite : ¬ Nat.Prime (corner 1870 2) := by change ¬ Nat.Prime 13987601; exact PrattCertificates.composite_of_product (factor := 41) (cofactor := 341161) rfl (by decide) (by decide)
    exact decide_eq_false composite
  third := false
  third_correct := by
    have composite : ¬ Nat.Prime (corner 1870 3) := by change ¬ Nat.Prime 13983861; exact PrattCertificates.composite_of_product (factor := 3) (cofactor := 4661287) rfl (by decide) (by decide)
    exact decide_eq_false composite
theorem row_count_1870 : row 1870 = 0 := (row_1870.checked).trans rfl
theorem model_count_1870 : modelRow 1870 = 0 := by decide
theorem modelrow_agrees_1870 : modelRow 1870 = row 1870 := model_count_1870.trans row_count_1870.symm
def row_1871 : RowCertificate 1871 where
  first := false
  first_correct := by
    have composite : ¬ Nat.Prime (corner 1871 1) := by change ¬ Nat.Prime 14006307; exact PrattCertificates.composite_of_product (factor := 3) (cofactor := 4668769) rfl (by decide) (by decide)
    exact decide_eq_false composite
  second := false
  second_correct := by
    have composite : ¬ Nat.Prime (corner 1871 2) := by change ¬ Nat.Prime 14002565; exact PrattCertificates.composite_of_product (factor := 5) (cofactor := 2800513) rfl (by decide) (by decide)
    exact decide_eq_false composite
  third := true
  third_correct := by
    have prime : Nat.Prime (corner 1871 3) := by change Nat.Prime 13998823; exact ProjectEulerSolutions.P58PrimeNodes.p_13998823.prime
    exact decide_eq_true prime
theorem row_count_1871 : row 1871 = 1 := (row_1871.checked).trans rfl
theorem model_count_1871 : modelRow 1871 = 1 := by decide
theorem modelrow_agrees_1871 : modelRow 1871 = row 1871 := model_count_1871.trans row_count_1871.symm
def row_1872 : RowCertificate 1872 where
  first := true
  first_correct := by
    have prime : Nat.Prime (corner 1872 1) := by change Nat.Prime 14021281; exact ProjectEulerSolutions.P58PrimeNodes.p_14021281.prime
    exact decide_eq_true prime
  second := false
  second_correct := by
    have composite : ¬ Nat.Prime (corner 1872 2) := by change ¬ Nat.Prime 14017537; exact PrattCertificates.composite_of_product (factor := 17) (cofactor := 824561) rfl (by decide) (by decide)
    exact decide_eq_false composite
  third := false
  third_correct := by
    have composite : ¬ Nat.Prime (corner 1872 3) := by change ¬ Nat.Prime 14013793; exact PrattCertificates.composite_of_product (factor := 613) (cofactor := 22861) rfl (by decide) (by decide)
    exact decide_eq_false composite
theorem row_count_1872 : row 1872 = 1 := (row_1872.checked).trans rfl
theorem model_count_1872 : modelRow 1872 = 1 := by decide
theorem modelrow_agrees_1872 : modelRow 1872 = row 1872 := model_count_1872.trans row_count_1872.symm
def row_1873 : RowCertificate 1873 where
  first := true
  first_correct := by
    have prime : Nat.Prime (corner 1873 1) := by change Nat.Prime 14036263; exact ProjectEulerSolutions.P58PrimeNodes.p_14036263.prime
    exact decide_eq_true prime
  second := true
  second_correct := by
    have prime : Nat.Prime (corner 1873 2) := by change Nat.Prime 14032517; exact ProjectEulerSolutions.P58PrimeNodes.p_14032517.prime
    exact decide_eq_true prime
  third := false
  third_correct := by
    have composite : ¬ Nat.Prime (corner 1873 3) := by change ¬ Nat.Prime 14028771; exact PrattCertificates.composite_of_product (factor := 3) (cofactor := 4676257) rfl (by decide) (by decide)
    exact decide_eq_false composite
theorem row_count_1873 : row 1873 = 2 := (row_1873.checked).trans rfl
theorem model_count_1873 : modelRow 1873 = 2 := by decide
theorem modelrow_agrees_1873 : modelRow 1873 = row 1873 := model_count_1873.trans row_count_1873.symm
def row_1874 : RowCertificate 1874 where
  first := false
  first_correct := by
    have composite : ¬ Nat.Prime (corner 1874 1) := by change ¬ Nat.Prime 14051253; exact PrattCertificates.composite_of_product (factor := 3) (cofactor := 4683751) rfl (by decide) (by decide)
    exact decide_eq_false composite
  second := false
  second_correct := by
    have composite : ¬ Nat.Prime (corner 1874 2) := by change ¬ Nat.Prime 14047505; exact PrattCertificates.composite_of_product (factor := 5) (cofactor := 2809501) rfl (by decide) (by decide)
    exact decide_eq_false composite
  third := false
  third_correct := by
    have composite : ¬ Nat.Prime (corner 1874 3) := by change ¬ Nat.Prime 14043757; exact PrattCertificates.composite_of_product (factor := 7) (cofactor := 2006251) rfl (by decide) (by decide)
    exact decide_eq_false composite
theorem row_count_1874 : row 1874 = 0 := (row_1874.checked).trans rfl
theorem model_count_1874 : modelRow 1874 = 0 := by decide
theorem modelrow_agrees_1874 : modelRow 1874 = row 1874 := model_count_1874.trans row_count_1874.symm
def row_1875 : RowCertificate 1875 where
  first := false
  first_correct := by
    have composite : ¬ Nat.Prime (corner 1875 1) := by change ¬ Nat.Prime 14066251; exact PrattCertificates.composite_of_product (factor := 19) (cofactor := 740329) rfl (by decide) (by decide)
    exact decide_eq_false composite
  second := false
  second_correct := by
    have composite : ¬ Nat.Prime (corner 1875 2) := by change ¬ Nat.Prime 14062501; exact PrattCertificates.composite_of_product (factor := 73) (cofactor := 192637) rfl (by decide) (by decide)
    exact decide_eq_false composite
  third := false
  third_correct := by
    have composite : ¬ Nat.Prime (corner 1875 3) := by change ¬ Nat.Prime 14058751; exact PrattCertificates.composite_of_product (factor := 7) (cofactor := 2008393) rfl (by decide) (by decide)
    exact decide_eq_false composite
theorem row_count_1875 : row 1875 = 0 := (row_1875.checked).trans rfl
theorem model_count_1875 : modelRow 1875 = 0 := by decide
theorem modelrow_agrees_1875 : modelRow 1875 = row 1875 := model_count_1875.trans row_count_1875.symm
def row_1876 : RowCertificate 1876 where
  first := false
  first_correct := by
    have composite : ¬ Nat.Prime (corner 1876 1) := by change ¬ Nat.Prime 14081257; exact PrattCertificates.composite_of_product (factor := 181) (cofactor := 77797) rfl (by decide) (by decide)
    exact decide_eq_false composite
  second := false
  second_correct := by
    have composite : ¬ Nat.Prime (corner 1876 2) := by change ¬ Nat.Prime 14077505; exact PrattCertificates.composite_of_product (factor := 5) (cofactor := 2815501) rfl (by decide) (by decide)
    exact decide_eq_false composite
  third := false
  third_correct := by
    have composite : ¬ Nat.Prime (corner 1876 3) := by change ¬ Nat.Prime 14073753; exact PrattCertificates.composite_of_product (factor := 3) (cofactor := 4691251) rfl (by decide) (by decide)
    exact decide_eq_false composite
theorem row_count_1876 : row 1876 = 0 := (row_1876.checked).trans rfl
theorem model_count_1876 : modelRow 1876 = 0 := by decide
theorem modelrow_agrees_1876 : modelRow 1876 = row 1876 := model_count_1876.trans row_count_1876.symm
def row_1877 : RowCertificate 1877 where
  first := false
  first_correct := by
    have composite : ¬ Nat.Prime (corner 1877 1) := by change ¬ Nat.Prime 14096271; exact PrattCertificates.composite_of_product (factor := 3) (cofactor := 4698757) rfl (by decide) (by decide)
    exact decide_eq_false composite
  second := true
  second_correct := by
    have prime : Nat.Prime (corner 1877 2) := by change Nat.Prime 14092517; exact ProjectEulerSolutions.P58PrimeNodes.p_14092517.prime
    exact decide_eq_true prime
  third := false
  third_correct := by
    have composite : ¬ Nat.Prime (corner 1877 3) := by change ¬ Nat.Prime 14088763; exact PrattCertificates.composite_of_product (factor := 13) (cofactor := 1083751) rfl (by decide) (by decide)
    exact decide_eq_false composite
theorem row_count_1877 : row 1877 = 1 := (row_1877.checked).trans rfl
theorem model_count_1877 : modelRow 1877 = 1 := by decide
theorem modelrow_agrees_1877 : modelRow 1877 = row 1877 := model_count_1877.trans row_count_1877.symm
def row_1878 : RowCertificate 1878 where
  first := false
  first_correct := by
    have composite : ¬ Nat.Prime (corner 1878 1) := by change ¬ Nat.Prime 14111293; exact PrattCertificates.composite_of_product (factor := 7) (cofactor := 2015899) rfl (by decide) (by decide)
    exact decide_eq_false composite
  second := true
  second_correct := by
    have prime : Nat.Prime (corner 1878 2) := by change Nat.Prime 14107537; exact ProjectEulerSolutions.P58PrimeNodes.p_14107537.prime
    exact decide_eq_true prime
  third := false
  third_correct := by
    have composite : ¬ Nat.Prime (corner 1878 3) := by change ¬ Nat.Prime 14103781; exact PrattCertificates.composite_of_product (factor := 157) (cofactor := 89833) rfl (by decide) (by decide)
    exact decide_eq_false composite
theorem row_count_1878 : row 1878 = 1 := (row_1878.checked).trans rfl
theorem model_count_1878 : modelRow 1878 = 1 := by decide
theorem modelrow_agrees_1878 : modelRow 1878 = row 1878 := model_count_1878.trans row_count_1878.symm
def row_1879 : RowCertificate 1879 where
  first := false
  first_correct := by
    have composite : ¬ Nat.Prime (corner 1879 1) := by change ¬ Nat.Prime 14126323; exact PrattCertificates.composite_of_product (factor := 229) (cofactor := 61687) rfl (by decide) (by decide)
    exact decide_eq_false composite
  second := false
  second_correct := by
    have composite : ¬ Nat.Prime (corner 1879 2) := by change ¬ Nat.Prime 14122565; exact PrattCertificates.composite_of_product (factor := 5) (cofactor := 2824513) rfl (by decide) (by decide)
    exact decide_eq_false composite
  third := false
  third_correct := by
    have composite : ¬ Nat.Prime (corner 1879 3) := by change ¬ Nat.Prime 14118807; exact PrattCertificates.composite_of_product (factor := 3) (cofactor := 4706269) rfl (by decide) (by decide)
    exact decide_eq_false composite
theorem row_count_1879 : row 1879 = 0 := (row_1879.checked).trans rfl
theorem model_count_1879 : modelRow 1879 = 0 := by decide
theorem modelrow_agrees_1879 : modelRow 1879 = row 1879 := model_count_1879.trans row_count_1879.symm
def row_1880 : RowCertificate 1880 where
  first := false
  first_correct := by
    have composite : ¬ Nat.Prime (corner 1880 1) := by change ¬ Nat.Prime 14141361; exact PrattCertificates.composite_of_product (factor := 3) (cofactor := 4713787) rfl (by decide) (by decide)
    exact decide_eq_false composite
  second := false
  second_correct := by
    have composite : ¬ Nat.Prime (corner 1880 2) := by change ¬ Nat.Prime 14137601; exact PrattCertificates.composite_of_product (factor := 421) (cofactor := 33581) rfl (by decide) (by decide)
    exact decide_eq_false composite
  third := false
  third_correct := by
    have composite : ¬ Nat.Prime (corner 1880 3) := by change ¬ Nat.Prime 14133841; exact PrattCertificates.composite_of_product (factor := 1783) (cofactor := 7927) rfl (by decide) (by decide)
    exact decide_eq_false composite
theorem row_count_1880 : row 1880 = 0 := (row_1880.checked).trans rfl
theorem model_count_1880 : modelRow 1880 = 0 := by decide
theorem modelrow_agrees_1880 : modelRow 1880 = row 1880 := model_count_1880.trans row_count_1880.symm
def row_1881 : RowCertificate 1881 where
  first := true
  first_correct := by
    have prime : Nat.Prime (corner 1881 1) := by change Nat.Prime 14156407; exact ProjectEulerSolutions.P58PrimeNodes.p_14156407.prime
    exact decide_eq_true prime
  second := false
  second_correct := by
    have composite : ¬ Nat.Prime (corner 1881 2) := by change ¬ Nat.Prime 14152645; exact PrattCertificates.composite_of_product (factor := 5) (cofactor := 2830529) rfl (by decide) (by decide)
    exact decide_eq_false composite
  third := false
  third_correct := by
    have composite : ¬ Nat.Prime (corner 1881 3) := by change ¬ Nat.Prime 14148883; exact PrattCertificates.composite_of_product (factor := 7) (cofactor := 2021269) rfl (by decide) (by decide)
    exact decide_eq_false composite
theorem row_count_1881 : row 1881 = 1 := (row_1881.checked).trans rfl
theorem model_count_1881 : modelRow 1881 = 1 := by decide
theorem modelrow_agrees_1881 : modelRow 1881 = row 1881 := model_count_1881.trans row_count_1881.symm
def row_1882 : RowCertificate 1882 where
  first := false
  first_correct := by
    have composite : ¬ Nat.Prime (corner 1882 1) := by change ¬ Nat.Prime 14171461; exact PrattCertificates.composite_of_product (factor := 103) (cofactor := 137587) rfl (by decide) (by decide)
    exact decide_eq_false composite
  second := true
  second_correct := by
    have prime : Nat.Prime (corner 1882 2) := by change Nat.Prime 14167697; exact ProjectEulerSolutions.P58PrimeNodes.p_14167697.prime
    exact decide_eq_true prime
  third := false
  third_correct := by
    have composite : ¬ Nat.Prime (corner 1882 3) := by change ¬ Nat.Prime 14163933; exact PrattCertificates.composite_of_product (factor := 3) (cofactor := 4721311) rfl (by decide) (by decide)
    exact decide_eq_false composite
theorem row_count_1882 : row 1882 = 1 := (row_1882.checked).trans rfl
theorem model_count_1882 : modelRow 1882 = 1 := by decide
theorem modelrow_agrees_1882 : modelRow 1882 = row 1882 := model_count_1882.trans row_count_1882.symm
def row_1883 : RowCertificate 1883 where
  first := false
  first_correct := by
    have composite : ¬ Nat.Prime (corner 1883 1) := by change ¬ Nat.Prime 14186523; exact PrattCertificates.composite_of_product (factor := 3) (cofactor := 4728841) rfl (by decide) (by decide)
    exact decide_eq_false composite
  second := false
  second_correct := by
    have composite : ¬ Nat.Prime (corner 1883 2) := by change ¬ Nat.Prime 14182757; exact PrattCertificates.composite_of_product (factor := 761) (cofactor := 18637) rfl (by decide) (by decide)
    exact decide_eq_false composite
  third := false
  third_correct := by
    have composite : ¬ Nat.Prime (corner 1883 3) := by change ¬ Nat.Prime 14178991; exact PrattCertificates.composite_of_product (factor := 271) (cofactor := 52321) rfl (by decide) (by decide)
    exact decide_eq_false composite
theorem row_count_1883 : row 1883 = 0 := (row_1883.checked).trans rfl
theorem model_count_1883 : modelRow 1883 = 0 := by decide
theorem modelrow_agrees_1883 : modelRow 1883 = row 1883 := model_count_1883.trans row_count_1883.symm
def row_1884 : RowCertificate 1884 where
  first := false
  first_correct := by
    have composite : ¬ Nat.Prime (corner 1884 1) := by change ¬ Nat.Prime 14201593; exact PrattCertificates.composite_of_product (factor := 7) (cofactor := 2028799) rfl (by decide) (by decide)
    exact decide_eq_false composite
  second := false
  second_correct := by
    have composite : ¬ Nat.Prime (corner 1884 2) := by change ¬ Nat.Prime 14197825; exact PrattCertificates.composite_of_product (factor := 5) (cofactor := 2839565) rfl (by decide) (by decide)
    exact decide_eq_false composite
  third := false
  third_correct := by
    have composite : ¬ Nat.Prime (corner 1884 3) := by change ¬ Nat.Prime 14194057; exact PrattCertificates.composite_of_product (factor := 1543) (cofactor := 9199) rfl (by decide) (by decide)
    exact decide_eq_false composite
theorem row_count_1884 : row 1884 = 0 := (row_1884.checked).trans rfl
theorem model_count_1884 : modelRow 1884 = 0 := by decide
theorem modelrow_agrees_1884 : modelRow 1884 = row 1884 := model_count_1884.trans row_count_1884.symm
def row_1885 : RowCertificate 1885 where
  first := false
  first_correct := by
    have composite : ¬ Nat.Prime (corner 1885 1) := by change ¬ Nat.Prime 14216671; exact PrattCertificates.composite_of_product (factor := 7) (cofactor := 2030953) rfl (by decide) (by decide)
    exact decide_eq_false composite
  second := false
  second_correct := by
    have composite : ¬ Nat.Prime (corner 1885 2) := by change ¬ Nat.Prime 14212901; exact PrattCertificates.composite_of_product (factor := 17) (cofactor := 836053) rfl (by decide) (by decide)
    exact decide_eq_false composite
  third := false
  third_correct := by
    have composite : ¬ Nat.Prime (corner 1885 3) := by change ¬ Nat.Prime 14209131; exact PrattCertificates.composite_of_product (factor := 3) (cofactor := 4736377) rfl (by decide) (by decide)
    exact decide_eq_false composite
theorem row_count_1885 : row 1885 = 0 := (row_1885.checked).trans rfl
theorem model_count_1885 : modelRow 1885 = 0 := by decide
theorem modelrow_agrees_1885 : modelRow 1885 = row 1885 := model_count_1885.trans row_count_1885.symm
def row_1886 : RowCertificate 1886 where
  first := false
  first_correct := by
    have composite : ¬ Nat.Prime (corner 1886 1) := by change ¬ Nat.Prime 14231757; exact PrattCertificates.composite_of_product (factor := 3) (cofactor := 4743919) rfl (by decide) (by decide)
    exact decide_eq_false composite
  second := false
  second_correct := by
    have composite : ¬ Nat.Prime (corner 1886 2) := by change ¬ Nat.Prime 14227985; exact PrattCertificates.composite_of_product (factor := 5) (cofactor := 2845597) rfl (by decide) (by decide)
    exact decide_eq_false composite
  third := true
  third_correct := by
    have prime : Nat.Prime (corner 1886 3) := by change Nat.Prime 14224213; exact ProjectEulerSolutions.P58PrimeNodes.p_14224213.prime
    exact decide_eq_true prime
theorem row_count_1886 : row 1886 = 1 := (row_1886.checked).trans rfl
theorem model_count_1886 : modelRow 1886 = 1 := by decide
theorem modelrow_agrees_1886 : modelRow 1886 = row 1886 := model_count_1886.trans row_count_1886.symm
def row_1887 : RowCertificate 1887 where
  first := false
  first_correct := by
    have composite : ¬ Nat.Prime (corner 1887 1) := by change ¬ Nat.Prime 14246851; exact PrattCertificates.composite_of_product (factor := 1129) (cofactor := 12619) rfl (by decide) (by decide)
    exact decide_eq_false composite
  second := true
  second_correct := by
    have prime : Nat.Prime (corner 1887 2) := by change Nat.Prime 14243077; exact ProjectEulerSolutions.P58PrimeNodes.p_14243077.prime
    exact decide_eq_true prime
  third := false
  third_correct := by
    have composite : ¬ Nat.Prime (corner 1887 3) := by change ¬ Nat.Prime 14239303; exact PrattCertificates.composite_of_product (factor := 13) (cofactor := 1095331) rfl (by decide) (by decide)
    exact decide_eq_false composite
theorem row_count_1887 : row 1887 = 1 := (row_1887.checked).trans rfl
theorem model_count_1887 : modelRow 1887 = 1 := by decide
theorem modelrow_agrees_1887 : modelRow 1887 = row 1887 := model_count_1887.trans row_count_1887.symm
def row_1888 : RowCertificate 1888 where
  first := false
  first_correct := by
    have composite : ¬ Nat.Prime (corner 1888 1) := by change ¬ Nat.Prime 14261953; exact PrattCertificates.composite_of_product (factor := 31) (cofactor := 460063) rfl (by decide) (by decide)
    exact decide_eq_false composite
  second := true
  second_correct := by
    have prime : Nat.Prime (corner 1888 2) := by change Nat.Prime 14258177; exact ProjectEulerSolutions.P58PrimeNodes.p_14258177.prime
    exact decide_eq_true prime
  third := false
  third_correct := by
    have composite : ¬ Nat.Prime (corner 1888 3) := by change ¬ Nat.Prime 14254401; exact PrattCertificates.composite_of_product (factor := 3) (cofactor := 4751467) rfl (by decide) (by decide)
    exact decide_eq_false composite
theorem row_count_1888 : row 1888 = 1 := (row_1888.checked).trans rfl
theorem model_count_1888 : modelRow 1888 = 1 := by decide
theorem modelrow_agrees_1888 : modelRow 1888 = row 1888 := model_count_1888.trans row_count_1888.symm
def row_1889 : RowCertificate 1889 where
  first := false
  first_correct := by
    have composite : ¬ Nat.Prime (corner 1889 1) := by change ¬ Nat.Prime 14277063; exact PrattCertificates.composite_of_product (factor := 3) (cofactor := 4759021) rfl (by decide) (by decide)
    exact decide_eq_false composite
  second := false
  second_correct := by
    have composite : ¬ Nat.Prime (corner 1889 2) := by change ¬ Nat.Prime 14273285; exact PrattCertificates.composite_of_product (factor := 5) (cofactor := 2854657) rfl (by decide) (by decide)
    exact decide_eq_false composite
  third := false
  third_correct := by
    have composite : ¬ Nat.Prime (corner 1889 3) := by change ¬ Nat.Prime 14269507; exact PrattCertificates.composite_of_product (factor := 7) (cofactor := 2038501) rfl (by decide) (by decide)
    exact decide_eq_false composite
theorem row_count_1889 : row 1889 = 0 := (row_1889.checked).trans rfl
theorem model_count_1889 : modelRow 1889 = 0 := by decide
theorem modelrow_agrees_1889 : modelRow 1889 = row 1889 := model_count_1889.trans row_count_1889.symm
def row_1890 : RowCertificate 1890 where
  first := false
  first_correct := by
    have composite : ¬ Nat.Prime (corner 1890 1) := by change ¬ Nat.Prime 14292181; exact PrattCertificates.composite_of_product (factor := 157) (cofactor := 91033) rfl (by decide) (by decide)
    exact decide_eq_false composite
  second := false
  second_correct := by
    have composite : ¬ Nat.Prime (corner 1890 2) := by change ¬ Nat.Prime 14288401; exact PrattCertificates.composite_of_product (factor := 37) (cofactor := 386173) rfl (by decide) (by decide)
    exact decide_eq_false composite
  third := false
  third_correct := by
    have composite : ¬ Nat.Prime (corner 1890 3) := by change ¬ Nat.Prime 14284621; exact PrattCertificates.composite_of_product (factor := 13) (cofactor := 1098817) rfl (by decide) (by decide)
    exact decide_eq_false composite
theorem row_count_1890 : row 1890 = 0 := (row_1890.checked).trans rfl
theorem model_count_1890 : modelRow 1890 = 0 := by decide
theorem modelrow_agrees_1890 : modelRow 1890 = row 1890 := model_count_1890.trans row_count_1890.symm
def row_1891 : RowCertificate 1891 where
  first := false
  first_correct := by
    have composite : ¬ Nat.Prime (corner 1891 1) := by change ¬ Nat.Prime 14307307; exact PrattCertificates.composite_of_product (factor := 7) (cofactor := 2043901) rfl (by decide) (by decide)
    exact decide_eq_false composite
  second := false
  second_correct := by
    have composite : ¬ Nat.Prime (corner 1891 2) := by change ¬ Nat.Prime 14303525; exact PrattCertificates.composite_of_product (factor := 5) (cofactor := 2860705) rfl (by decide) (by decide)
    exact decide_eq_false composite
  third := false
  third_correct := by
    have composite : ¬ Nat.Prime (corner 1891 3) := by change ¬ Nat.Prime 14299743; exact PrattCertificates.composite_of_product (factor := 3) (cofactor := 4766581) rfl (by decide) (by decide)
    exact decide_eq_false composite
theorem row_count_1891 : row 1891 = 0 := (row_1891.checked).trans rfl
theorem model_count_1891 : modelRow 1891 = 0 := by decide
theorem modelrow_agrees_1891 : modelRow 1891 = row 1891 := model_count_1891.trans row_count_1891.symm
def row_1892 : RowCertificate 1892 where
  first := false
  first_correct := by
    have composite : ¬ Nat.Prime (corner 1892 1) := by change ¬ Nat.Prime 14322441; exact PrattCertificates.composite_of_product (factor := 3) (cofactor := 4774147) rfl (by decide) (by decide)
    exact decide_eq_false composite
  second := true
  second_correct := by
    have prime : Nat.Prime (corner 1892 2) := by change Nat.Prime 14318657; exact ProjectEulerSolutions.P58PrimeNodes.p_14318657.prime
    exact decide_eq_true prime
  third := false
  third_correct := by
    have composite : ¬ Nat.Prime (corner 1892 3) := by change ¬ Nat.Prime 14314873; exact PrattCertificates.composite_of_product (factor := 211) (cofactor := 67843) rfl (by decide) (by decide)
    exact decide_eq_false composite
theorem row_count_1892 : row 1892 = 1 := (row_1892.checked).trans rfl
theorem model_count_1892 : modelRow 1892 = 1 := by decide
theorem modelrow_agrees_1892 : modelRow 1892 = row 1892 := model_count_1892.trans row_count_1892.symm
def row_1893 : RowCertificate 1893 where
  first := false
  first_correct := by
    have composite : ¬ Nat.Prime (corner 1893 1) := by change ¬ Nat.Prime 14337583; exact PrattCertificates.composite_of_product (factor := 13) (cofactor := 1102891) rfl (by decide) (by decide)
    exact decide_eq_false composite
  second := false
  second_correct := by
    have composite : ¬ Nat.Prime (corner 1893 2) := by change ¬ Nat.Prime 14333797; exact PrattCertificates.composite_of_product (factor := 53) (cofactor := 270449) rfl (by decide) (by decide)
    exact decide_eq_false composite
  third := true
  third_correct := by
    have prime : Nat.Prime (corner 1893 3) := by change Nat.Prime 14330011; exact ProjectEulerSolutions.P58PrimeNodes.p_14330011.prime
    exact decide_eq_true prime
theorem row_count_1893 : row 1893 = 1 := (row_1893.checked).trans rfl
theorem model_count_1893 : modelRow 1893 = 1 := by decide
theorem modelrow_agrees_1893 : modelRow 1893 = row 1893 := model_count_1893.trans row_count_1893.symm
def row_1894 : RowCertificate 1894 where
  first := false
  first_correct := by
    have composite : ¬ Nat.Prime (corner 1894 1) := by change ¬ Nat.Prime 14352733; exact PrattCertificates.composite_of_product (factor := 19) (cofactor := 755407) rfl (by decide) (by decide)
    exact decide_eq_false composite
  second := false
  second_correct := by
    have composite : ¬ Nat.Prime (corner 1894 2) := by change ¬ Nat.Prime 14348945; exact PrattCertificates.composite_of_product (factor := 5) (cofactor := 2869789) rfl (by decide) (by decide)
    exact decide_eq_false composite
  third := false
  third_correct := by
    have composite : ¬ Nat.Prime (corner 1894 3) := by change ¬ Nat.Prime 14345157; exact PrattCertificates.composite_of_product (factor := 3) (cofactor := 4781719) rfl (by decide) (by decide)
    exact decide_eq_false composite
theorem row_count_1894 : row 1894 = 0 := (row_1894.checked).trans rfl
theorem model_count_1894 : modelRow 1894 = 0 := by decide
theorem modelrow_agrees_1894 : modelRow 1894 = row 1894 := model_count_1894.trans row_count_1894.symm
def row_1895 : RowCertificate 1895 where
  first := false
  first_correct := by
    have composite : ¬ Nat.Prime (corner 1895 1) := by change ¬ Nat.Prime 14367891; exact PrattCertificates.composite_of_product (factor := 3) (cofactor := 4789297) rfl (by decide) (by decide)
    exact decide_eq_false composite
  second := true
  second_correct := by
    have prime : Nat.Prime (corner 1895 2) := by change Nat.Prime 14364101; exact ProjectEulerSolutions.P58PrimeNodes.p_14364101.prime
    exact decide_eq_true prime
  third := false
  third_correct := by
    have composite : ¬ Nat.Prime (corner 1895 3) := by change ¬ Nat.Prime 14360311; exact PrattCertificates.composite_of_product (factor := 7) (cofactor := 2051473) rfl (by decide) (by decide)
    exact decide_eq_false composite
theorem row_count_1895 : row 1895 = 1 := (row_1895.checked).trans rfl
theorem model_count_1895 : modelRow 1895 = 1 := by decide
theorem modelrow_agrees_1895 : modelRow 1895 = row 1895 := model_count_1895.trans row_count_1895.symm
def row_1896 : RowCertificate 1896 where
  first := false
  first_correct := by
    have composite : ¬ Nat.Prime (corner 1896 1) := by change ¬ Nat.Prime 14383057; exact PrattCertificates.composite_of_product (factor := 13) (cofactor := 1106389) rfl (by decide) (by decide)
    exact decide_eq_false composite
  second := false
  second_correct := by
    have composite : ¬ Nat.Prime (corner 1896 2) := by change ¬ Nat.Prime 14379265; exact PrattCertificates.composite_of_product (factor := 5) (cofactor := 2875853) rfl (by decide) (by decide)
    exact decide_eq_false composite
  third := false
  third_correct := by
    have composite : ¬ Nat.Prime (corner 1896 3) := by change ¬ Nat.Prime 14375473; exact PrattCertificates.composite_of_product (factor := 7) (cofactor := 2053639) rfl (by decide) (by decide)
    exact decide_eq_false composite
theorem row_count_1896 : row 1896 = 0 := (row_1896.checked).trans rfl
theorem model_count_1896 : modelRow 1896 = 0 := by decide
theorem modelrow_agrees_1896 : modelRow 1896 = row 1896 := model_count_1896.trans row_count_1896.symm
def row_1897 : RowCertificate 1897 where
  first := true
  first_correct := by
    have prime : Nat.Prime (corner 1897 1) := by change Nat.Prime 14398231; exact ProjectEulerSolutions.P58PrimeNodes.p_14398231.prime
    exact decide_eq_true prime
  second := true
  second_correct := by
    have prime : Nat.Prime (corner 1897 2) := by change Nat.Prime 14394437; exact ProjectEulerSolutions.P58PrimeNodes.p_14394437.prime
    exact decide_eq_true prime
  third := false
  third_correct := by
    have composite : ¬ Nat.Prime (corner 1897 3) := by change ¬ Nat.Prime 14390643; exact PrattCertificates.composite_of_product (factor := 3) (cofactor := 4796881) rfl (by decide) (by decide)
    exact decide_eq_false composite
theorem row_count_1897 : row 1897 = 2 := (row_1897.checked).trans rfl
theorem model_count_1897 : modelRow 1897 = 2 := by decide
theorem modelrow_agrees_1897 : modelRow 1897 = row 1897 := model_count_1897.trans row_count_1897.symm
def row_1898 : RowCertificate 1898 where
  first := false
  first_correct := by
    have composite : ¬ Nat.Prime (corner 1898 1) := by change ¬ Nat.Prime 14413413; exact PrattCertificates.composite_of_product (factor := 3) (cofactor := 4804471) rfl (by decide) (by decide)
    exact decide_eq_false composite
  second := false
  second_correct := by
    have composite : ¬ Nat.Prime (corner 1898 2) := by change ¬ Nat.Prime 14409617; exact PrattCertificates.composite_of_product (factor := 157) (cofactor := 91781) rfl (by decide) (by decide)
    exact decide_eq_false composite
  third := false
  third_correct := by
    have composite : ¬ Nat.Prime (corner 1898 3) := by change ¬ Nat.Prime 14405821; exact PrattCertificates.composite_of_product (factor := 61) (cofactor := 236161) rfl (by decide) (by decide)
    exact decide_eq_false composite
theorem row_count_1898 : row 1898 = 0 := (row_1898.checked).trans rfl
theorem model_count_1898 : modelRow 1898 = 0 := by decide
theorem modelrow_agrees_1898 : modelRow 1898 = row 1898 := model_count_1898.trans row_count_1898.symm
def row_1899 : RowCertificate 1899 where
  first := false
  first_correct := by
    have composite : ¬ Nat.Prime (corner 1899 1) := by change ¬ Nat.Prime 14428603; exact PrattCertificates.composite_of_product (factor := 7) (cofactor := 2061229) rfl (by decide) (by decide)
    exact decide_eq_false composite
  second := false
  second_correct := by
    have composite : ¬ Nat.Prime (corner 1899 2) := by change ¬ Nat.Prime 14424805; exact PrattCertificates.composite_of_product (factor := 5) (cofactor := 2884961) rfl (by decide) (by decide)
    exact decide_eq_false composite
  third := false
  third_correct := by
    have composite : ¬ Nat.Prime (corner 1899 3) := by change ¬ Nat.Prime 14421007; exact PrattCertificates.composite_of_product (factor := 1579) (cofactor := 9133) rfl (by decide) (by decide)
    exact decide_eq_false composite
theorem row_count_1899 : row 1899 = 0 := (row_1899.checked).trans rfl
theorem model_count_1899 : modelRow 1899 = 0 := by decide
theorem modelrow_agrees_1899 : modelRow 1899 = row 1899 := model_count_1899.trans row_count_1899.symm
def row_1900 : RowCertificate 1900 where
  first := false
  first_correct := by
    have composite : ¬ Nat.Prime (corner 1900 1) := by change ¬ Nat.Prime 14443801; exact PrattCertificates.composite_of_product (factor := 37) (cofactor := 390373) rfl (by decide) (by decide)
    exact decide_eq_false composite
  second := true
  second_correct := by
    have prime : Nat.Prime (corner 1900 2) := by change Nat.Prime 14440001; exact ProjectEulerSolutions.P58PrimeNodes.p_14440001.prime
    exact decide_eq_true prime
  third := false
  third_correct := by
    have composite : ¬ Nat.Prime (corner 1900 3) := by change ¬ Nat.Prime 14436201; exact PrattCertificates.composite_of_product (factor := 3) (cofactor := 4812067) rfl (by decide) (by decide)
    exact decide_eq_false composite
theorem row_count_1900 : row 1900 = 1 := (row_1900.checked).trans rfl
theorem model_count_1900 : modelRow 1900 = 1 := by decide
theorem modelrow_agrees_1900 : modelRow 1900 = row 1900 := model_count_1900.trans row_count_1900.symm
def row_1901 : RowCertificate 1901 where
  first := false
  first_correct := by
    have composite : ¬ Nat.Prime (corner 1901 1) := by change ¬ Nat.Prime 14459007; exact PrattCertificates.composite_of_product (factor := 3) (cofactor := 4819669) rfl (by decide) (by decide)
    exact decide_eq_false composite
  second := false
  second_correct := by
    have composite : ¬ Nat.Prime (corner 1901 2) := by change ¬ Nat.Prime 14455205; exact PrattCertificates.composite_of_product (factor := 5) (cofactor := 2891041) rfl (by decide) (by decide)
    exact decide_eq_false composite
  third := true
  third_correct := by
    have prime : Nat.Prime (corner 1901 3) := by change Nat.Prime 14451403; exact ProjectEulerSolutions.P58PrimeNodes.p_14451403.prime
    exact decide_eq_true prime
theorem row_count_1901 : row 1901 = 1 := (row_1901.checked).trans rfl
theorem model_count_1901 : modelRow 1901 = 1 := by decide
theorem modelrow_agrees_1901 : modelRow 1901 = row 1901 := model_count_1901.trans row_count_1901.symm
def row_1902 : RowCertificate 1902 where
  first := false
  first_correct := by
    have composite : ¬ Nat.Prime (corner 1902 1) := by change ¬ Nat.Prime 14474221; exact PrattCertificates.composite_of_product (factor := 73) (cofactor := 198277) rfl (by decide) (by decide)
    exact decide_eq_false composite
  second := false
  second_correct := by
    have composite : ¬ Nat.Prime (corner 1902 2) := by change ¬ Nat.Prime 14470417; exact PrattCertificates.composite_of_product (factor := 13) (cofactor := 1113109) rfl (by decide) (by decide)
    exact decide_eq_false composite
  third := false
  third_correct := by
    have composite : ¬ Nat.Prime (corner 1902 3) := by change ¬ Nat.Prime 14466613; exact PrattCertificates.composite_of_product (factor := 7) (cofactor := 2066659) rfl (by decide) (by decide)
    exact decide_eq_false composite
theorem row_count_1902 : row 1902 = 0 := (row_1902.checked).trans rfl
theorem model_count_1902 : modelRow 1902 = 0 := by decide
theorem modelrow_agrees_1902 : modelRow 1902 = row 1902 := model_count_1902.trans row_count_1902.symm
def row_1903 : RowCertificate 1903 where
  first := false
  first_correct := by
    have composite : ¬ Nat.Prime (corner 1903 1) := by change ¬ Nat.Prime 14489443; exact PrattCertificates.composite_of_product (factor := 1399) (cofactor := 10357) rfl (by decide) (by decide)
    exact decide_eq_false composite
  second := true
  second_correct := by
    have prime : Nat.Prime (corner 1903 2) := by change Nat.Prime 14485637; exact ProjectEulerSolutions.P58PrimeNodes.p_14485637.prime
    exact decide_eq_true prime
  third := false
  third_correct := by
    have composite : ¬ Nat.Prime (corner 1903 3) := by change ¬ Nat.Prime 14481831; exact PrattCertificates.composite_of_product (factor := 3) (cofactor := 4827277) rfl (by decide) (by decide)
    exact decide_eq_false composite
theorem row_count_1903 : row 1903 = 1 := (row_1903.checked).trans rfl
theorem model_count_1903 : modelRow 1903 = 1 := by decide
theorem modelrow_agrees_1903 : modelRow 1903 = row 1903 := model_count_1903.trans row_count_1903.symm
def row_1904 : RowCertificate 1904 where
  first := false
  first_correct := by
    have composite : ¬ Nat.Prime (corner 1904 1) := by change ¬ Nat.Prime 14504673; exact PrattCertificates.composite_of_product (factor := 3) (cofactor := 4834891) rfl (by decide) (by decide)
    exact decide_eq_false composite
  second := false
  second_correct := by
    have composite : ¬ Nat.Prime (corner 1904 2) := by change ¬ Nat.Prime 14500865; exact PrattCertificates.composite_of_product (factor := 5) (cofactor := 2900173) rfl (by decide) (by decide)
    exact decide_eq_false composite
  third := false
  third_correct := by
    have composite : ¬ Nat.Prime (corner 1904 3) := by change ¬ Nat.Prime 14497057; exact PrattCertificates.composite_of_product (factor := 19) (cofactor := 763003) rfl (by decide) (by decide)
    exact decide_eq_false composite
theorem row_count_1904 : row 1904 = 0 := (row_1904.checked).trans rfl
theorem model_count_1904 : modelRow 1904 = 0 := by decide
theorem modelrow_agrees_1904 : modelRow 1904 = row 1904 := model_count_1904.trans row_count_1904.symm
def row_1905 : RowCertificate 1905 where
  first := false
  first_correct := by
    have composite : ¬ Nat.Prime (corner 1905 1) := by change ¬ Nat.Prime 14519911; exact PrattCertificates.composite_of_product (factor := 7) (cofactor := 2074273) rfl (by decide) (by decide)
    exact decide_eq_false composite
  second := false
  second_correct := by
    have composite : ¬ Nat.Prime (corner 1905 2) := by change ¬ Nat.Prime 14516101; exact PrattCertificates.composite_of_product (factor := 821) (cofactor := 17681) rfl (by decide) (by decide)
    exact decide_eq_false composite
  third := false
  third_correct := by
    have composite : ¬ Nat.Prime (corner 1905 3) := by change ¬ Nat.Prime 14512291; exact PrattCertificates.composite_of_product (factor := 421) (cofactor := 34471) rfl (by decide) (by decide)
    exact decide_eq_false composite
theorem row_count_1905 : row 1905 = 0 := (row_1905.checked).trans rfl
theorem model_count_1905 : modelRow 1905 = 0 := by decide
theorem modelrow_agrees_1905 : modelRow 1905 = row 1905 := model_count_1905.trans row_count_1905.symm
def row_1906 : RowCertificate 1906 where
  first := false
  first_correct := by
    have composite : ¬ Nat.Prime (corner 1906 1) := by change ¬ Nat.Prime 14535157; exact PrattCertificates.composite_of_product (factor := 7) (cofactor := 2076451) rfl (by decide) (by decide)
    exact decide_eq_false composite
  second := false
  second_correct := by
    have composite : ¬ Nat.Prime (corner 1906 2) := by change ¬ Nat.Prime 14531345; exact PrattCertificates.composite_of_product (factor := 5) (cofactor := 2906269) rfl (by decide) (by decide)
    exact decide_eq_false composite
  third := false
  third_correct := by
    have composite : ¬ Nat.Prime (corner 1906 3) := by change ¬ Nat.Prime 14527533; exact PrattCertificates.composite_of_product (factor := 3) (cofactor := 4842511) rfl (by decide) (by decide)
    exact decide_eq_false composite
theorem row_count_1906 : row 1906 = 0 := (row_1906.checked).trans rfl
theorem model_count_1906 : modelRow 1906 = 0 := by decide
theorem modelrow_agrees_1906 : modelRow 1906 = row 1906 := model_count_1906.trans row_count_1906.symm
def row_1907 : RowCertificate 1907 where
  first := false
  first_correct := by
    have composite : ¬ Nat.Prime (corner 1907 1) := by change ¬ Nat.Prime 14550411; exact PrattCertificates.composite_of_product (factor := 3) (cofactor := 4850137) rfl (by decide) (by decide)
    exact decide_eq_false composite
  second := false
  second_correct := by
    have composite : ¬ Nat.Prime (corner 1907 2) := by change ¬ Nat.Prime 14546597; exact PrattCertificates.composite_of_product (factor := 13) (cofactor := 1118969) rfl (by decide) (by decide)
    exact decide_eq_false composite
  third := false
  third_correct := by
    have composite : ¬ Nat.Prime (corner 1907 3) := by change ¬ Nat.Prime 14542783; exact PrattCertificates.composite_of_product (factor := 1303) (cofactor := 11161) rfl (by decide) (by decide)
    exact decide_eq_false composite
theorem row_count_1907 : row 1907 = 0 := (row_1907.checked).trans rfl
theorem model_count_1907 : modelRow 1907 = 0 := by decide
theorem modelrow_agrees_1907 : modelRow 1907 = row 1907 := model_count_1907.trans row_count_1907.symm
def row_1908 : RowCertificate 1908 where
  first := false
  first_correct := by
    have composite : ¬ Nat.Prime (corner 1908 1) := by change ¬ Nat.Prime 14565673; exact PrattCertificates.composite_of_product (factor := 2011) (cofactor := 7243) rfl (by decide) (by decide)
    exact decide_eq_false composite
  second := false
  second_correct := by
    have composite : ¬ Nat.Prime (corner 1908 2) := by change ¬ Nat.Prime 14561857; exact PrattCertificates.composite_of_product (factor := 29) (cofactor := 502133) rfl (by decide) (by decide)
    exact decide_eq_false composite
  third := false
  third_correct := by
    have composite : ¬ Nat.Prime (corner 1908 3) := by change ¬ Nat.Prime 14558041; exact PrattCertificates.composite_of_product (factor := 79) (cofactor := 184279) rfl (by decide) (by decide)
    exact decide_eq_false composite
theorem row_count_1908 : row 1908 = 0 := (row_1908.checked).trans rfl
theorem model_count_1908 : modelRow 1908 = 0 := by decide
theorem modelrow_agrees_1908 : modelRow 1908 = row 1908 := model_count_1908.trans row_count_1908.symm
def row_1909 : RowCertificate 1909 where
  first := false
  first_correct := by
    have composite : ¬ Nat.Prime (corner 1909 1) := by change ¬ Nat.Prime 14580943; exact PrattCertificates.composite_of_product (factor := 13) (cofactor := 1121611) rfl (by decide) (by decide)
    exact decide_eq_false composite
  second := false
  second_correct := by
    have composite : ¬ Nat.Prime (corner 1909 2) := by change ¬ Nat.Prime 14577125; exact PrattCertificates.composite_of_product (factor := 5) (cofactor := 2915425) rfl (by decide) (by decide)
    exact decide_eq_false composite
  third := false
  third_correct := by
    have composite : ¬ Nat.Prime (corner 1909 3) := by change ¬ Nat.Prime 14573307; exact PrattCertificates.composite_of_product (factor := 3) (cofactor := 4857769) rfl (by decide) (by decide)
    exact decide_eq_false composite
theorem row_count_1909 : row 1909 = 0 := (row_1909.checked).trans rfl
theorem model_count_1909 : modelRow 1909 = 0 := by decide
theorem modelrow_agrees_1909 : modelRow 1909 = row 1909 := model_count_1909.trans row_count_1909.symm
def row_1910 : RowCertificate 1910 where
  first := false
  first_correct := by
    have composite : ¬ Nat.Prime (corner 1910 1) := by change ¬ Nat.Prime 14596221; exact PrattCertificates.composite_of_product (factor := 3) (cofactor := 4865407) rfl (by decide) (by decide)
    exact decide_eq_false composite
  second := false
  second_correct := by
    have composite : ¬ Nat.Prime (corner 1910 2) := by change ¬ Nat.Prime 14592401; exact PrattCertificates.composite_of_product (factor := 181) (cofactor := 80621) rfl (by decide) (by decide)
    exact decide_eq_false composite
  third := false
  third_correct := by
    have composite : ¬ Nat.Prime (corner 1910 3) := by change ¬ Nat.Prime 14588581; exact PrattCertificates.composite_of_product (factor := 7) (cofactor := 2084083) rfl (by decide) (by decide)
    exact decide_eq_false composite
theorem row_count_1910 : row 1910 = 0 := (row_1910.checked).trans rfl
theorem model_count_1910 : modelRow 1910 = 0 := by decide
theorem modelrow_agrees_1910 : modelRow 1910 = row 1910 := model_count_1910.trans row_count_1910.symm
def row_1911 : RowCertificate 1911 where
  first := false
  first_correct := by
    have composite : ¬ Nat.Prime (corner 1911 1) := by change ¬ Nat.Prime 14611507; exact PrattCertificates.composite_of_product (factor := 271) (cofactor := 53917) rfl (by decide) (by decide)
    exact decide_eq_false composite
  second := false
  second_correct := by
    have composite : ¬ Nat.Prime (corner 1911 2) := by change ¬ Nat.Prime 14607685; exact PrattCertificates.composite_of_product (factor := 5) (cofactor := 2921537) rfl (by decide) (by decide)
    exact decide_eq_false composite
  third := false
  third_correct := by
    have composite : ¬ Nat.Prime (corner 1911 3) := by change ¬ Nat.Prime 14603863; exact PrattCertificates.composite_of_product (factor := 37) (cofactor := 394699) rfl (by decide) (by decide)
    exact decide_eq_false composite
theorem row_count_1911 : row 1911 = 0 := (row_1911.checked).trans rfl
theorem model_count_1911 : modelRow 1911 = 0 := by decide
theorem modelrow_agrees_1911 : modelRow 1911 = row 1911 := model_count_1911.trans row_count_1911.symm
def row_1912 : RowCertificate 1912 where
  first := false
  first_correct := by
    have composite : ¬ Nat.Prime (corner 1912 1) := by change ¬ Nat.Prime 14626801; exact PrattCertificates.composite_of_product (factor := 7) (cofactor := 2089543) rfl (by decide) (by decide)
    exact decide_eq_false composite
  second := false
  second_correct := by
    have composite : ¬ Nat.Prime (corner 1912 2) := by change ¬ Nat.Prime 14622977; exact PrattCertificates.composite_of_product (factor := 409) (cofactor := 35753) rfl (by decide) (by decide)
    exact decide_eq_false composite
  third := false
  third_correct := by
    have composite : ¬ Nat.Prime (corner 1912 3) := by change ¬ Nat.Prime 14619153; exact PrattCertificates.composite_of_product (factor := 3) (cofactor := 4873051) rfl (by decide) (by decide)
    exact decide_eq_false composite
theorem row_count_1912 : row 1912 = 0 := (row_1912.checked).trans rfl
theorem model_count_1912 : modelRow 1912 = 0 := by decide
theorem modelrow_agrees_1912 : modelRow 1912 = row 1912 := model_count_1912.trans row_count_1912.symm
def row_1913 : RowCertificate 1913 where
  first := false
  first_correct := by
    have composite : ¬ Nat.Prime (corner 1913 1) := by change ¬ Nat.Prime 14642103; exact PrattCertificates.composite_of_product (factor := 3) (cofactor := 4880701) rfl (by decide) (by decide)
    exact decide_eq_false composite
  second := true
  second_correct := by
    have prime : Nat.Prime (corner 1913 2) := by change Nat.Prime 14638277; exact ProjectEulerSolutions.P58PrimeNodes.p_14638277.prime
    exact decide_eq_true prime
  third := false
  third_correct := by
    have composite : ¬ Nat.Prime (corner 1913 3) := by change ¬ Nat.Prime 14634451; exact PrattCertificates.composite_of_product (factor := 13) (cofactor := 1125727) rfl (by decide) (by decide)
    exact decide_eq_false composite
theorem row_count_1913 : row 1913 = 1 := (row_1913.checked).trans rfl
theorem model_count_1913 : modelRow 1913 = 1 := by decide
theorem modelrow_agrees_1913 : modelRow 1913 = row 1913 := model_count_1913.trans row_count_1913.symm
def row_1914 : RowCertificate 1914 where
  first := true
  first_correct := by
    have prime : Nat.Prime (corner 1914 1) := by change Nat.Prime 14657413; exact ProjectEulerSolutions.P58PrimeNodes.p_14657413.prime
    exact decide_eq_true prime
  second := false
  second_correct := by
    have composite : ¬ Nat.Prime (corner 1914 2) := by change ¬ Nat.Prime 14653585; exact PrattCertificates.composite_of_product (factor := 5) (cofactor := 2930717) rfl (by decide) (by decide)
    exact decide_eq_false composite
  third := false
  third_correct := by
    have composite : ¬ Nat.Prime (corner 1914 3) := by change ¬ Nat.Prime 14649757; exact PrattCertificates.composite_of_product (factor := 751) (cofactor := 19507) rfl (by decide) (by decide)
    exact decide_eq_false composite
theorem row_count_1914 : row 1914 = 1 := (row_1914.checked).trans rfl
theorem model_count_1914 : modelRow 1914 = 1 := by decide
theorem modelrow_agrees_1914 : modelRow 1914 = row 1914 := model_count_1914.trans row_count_1914.symm
def row_1915 : RowCertificate 1915 where
  first := false
  first_correct := by
    have composite : ¬ Nat.Prime (corner 1915 1) := by change ¬ Nat.Prime 14672731; exact PrattCertificates.composite_of_product (factor := 19) (cofactor := 772249) rfl (by decide) (by decide)
    exact decide_eq_false composite
  second := false
  second_correct := by
    have composite : ¬ Nat.Prime (corner 1915 2) := by change ¬ Nat.Prime 14668901; exact PrattCertificates.composite_of_product (factor := 13) (cofactor := 1128377) rfl (by decide) (by decide)
    exact decide_eq_false composite
  third := false
  third_correct := by
    have composite : ¬ Nat.Prime (corner 1915 3) := by change ¬ Nat.Prime 14665071; exact PrattCertificates.composite_of_product (factor := 3) (cofactor := 4888357) rfl (by decide) (by decide)
    exact decide_eq_false composite
theorem row_count_1915 : row 1915 = 0 := (row_1915.checked).trans rfl
theorem model_count_1915 : modelRow 1915 = 0 := by decide
theorem modelrow_agrees_1915 : modelRow 1915 = row 1915 := model_count_1915.trans row_count_1915.symm
def row_1916 : RowCertificate 1916 where
  first := false
  first_correct := by
    have composite : ¬ Nat.Prime (corner 1916 1) := by change ¬ Nat.Prime 14688057; exact PrattCertificates.composite_of_product (factor := 3) (cofactor := 4896019) rfl (by decide) (by decide)
    exact decide_eq_false composite
  second := false
  second_correct := by
    have composite : ¬ Nat.Prime (corner 1916 2) := by change ¬ Nat.Prime 14684225; exact PrattCertificates.composite_of_product (factor := 5) (cofactor := 2936845) rfl (by decide) (by decide)
    exact decide_eq_false composite
  third := false
  third_correct := by
    have composite : ¬ Nat.Prime (corner 1916 3) := by change ¬ Nat.Prime 14680393; exact PrattCertificates.composite_of_product (factor := 7) (cofactor := 2097199) rfl (by decide) (by decide)
    exact decide_eq_false composite
theorem row_count_1916 : row 1916 = 0 := (row_1916.checked).trans rfl
theorem model_count_1916 : modelRow 1916 = 0 := by decide
theorem modelrow_agrees_1916 : modelRow 1916 = row 1916 := model_count_1916.trans row_count_1916.symm
def row_1917 : RowCertificate 1917 where
  first := true
  first_correct := by
    have prime : Nat.Prime (corner 1917 1) := by change Nat.Prime 14703391; exact ProjectEulerSolutions.P58PrimeNodes.p_14703391.prime
    exact decide_eq_true prime
  second := false
  second_correct := by
    have composite : ¬ Nat.Prime (corner 1917 2) := by change ¬ Nat.Prime 14699557; exact PrattCertificates.composite_of_product (factor := 373) (cofactor := 39409) rfl (by decide) (by decide)
    exact decide_eq_false composite
  third := false
  third_correct := by
    have composite : ¬ Nat.Prime (corner 1917 3) := by change ¬ Nat.Prime 14695723; exact PrattCertificates.composite_of_product (factor := 7) (cofactor := 2099389) rfl (by decide) (by decide)
    exact decide_eq_false composite
theorem row_count_1917 : row 1917 = 1 := (row_1917.checked).trans rfl
theorem model_count_1917 : modelRow 1917 = 1 := by decide
theorem modelrow_agrees_1917 : modelRow 1917 = row 1917 := model_count_1917.trans row_count_1917.symm
def row_1918 : RowCertificate 1918 where
  first := true
  first_correct := by
    have prime : Nat.Prime (corner 1918 1) := by change Nat.Prime 14718733; exact ProjectEulerSolutions.P58PrimeNodes.p_14718733.prime
    exact decide_eq_true prime
  second := false
  second_correct := by
    have composite : ¬ Nat.Prime (corner 1918 2) := by change ¬ Nat.Prime 14714897; exact PrattCertificates.composite_of_product (factor := 3109) (cofactor := 4733) rfl (by decide) (by decide)
    exact decide_eq_false composite
  third := false
  third_correct := by
    have composite : ¬ Nat.Prime (corner 1918 3) := by change ¬ Nat.Prime 14711061; exact PrattCertificates.composite_of_product (factor := 3) (cofactor := 4903687) rfl (by decide) (by decide)
    exact decide_eq_false composite
theorem row_count_1918 : row 1918 = 1 := (row_1918.checked).trans rfl
theorem model_count_1918 : modelRow 1918 = 1 := by decide
theorem modelrow_agrees_1918 : modelRow 1918 = row 1918 := model_count_1918.trans row_count_1918.symm
def row_1919 : RowCertificate 1919 where
  first := false
  first_correct := by
    have composite : ¬ Nat.Prime (corner 1919 1) := by change ¬ Nat.Prime 14734083; exact PrattCertificates.composite_of_product (factor := 3) (cofactor := 4911361) rfl (by decide) (by decide)
    exact decide_eq_false composite
  second := false
  second_correct := by
    have composite : ¬ Nat.Prime (corner 1919 2) := by change ¬ Nat.Prime 14730245; exact PrattCertificates.composite_of_product (factor := 5) (cofactor := 2946049) rfl (by decide) (by decide)
    exact decide_eq_false composite
  third := false
  third_correct := by
    have composite : ¬ Nat.Prime (corner 1919 3) := by change ¬ Nat.Prime 14726407; exact PrattCertificates.composite_of_product (factor := 37) (cofactor := 398011) rfl (by decide) (by decide)
    exact decide_eq_false composite
theorem row_count_1919 : row 1919 = 0 := (row_1919.checked).trans rfl
theorem model_count_1919 : modelRow 1919 = 0 := by decide
theorem modelrow_agrees_1919 : modelRow 1919 = row 1919 := model_count_1919.trans row_count_1919.symm
def row_1920 : RowCertificate 1920 where
  first := false
  first_correct := by
    have composite : ¬ Nat.Prime (corner 1920 1) := by change ¬ Nat.Prime 14749441; exact PrattCertificates.composite_of_product (factor := 7) (cofactor := 2107063) rfl (by decide) (by decide)
    exact decide_eq_false composite
  second := false
  second_correct := by
    have composite : ¬ Nat.Prime (corner 1920 2) := by change ¬ Nat.Prime 14745601; exact PrattCertificates.composite_of_product (factor := 13) (cofactor := 1134277) rfl (by decide) (by decide)
    exact decide_eq_false composite
  third := true
  third_correct := by
    have prime : Nat.Prime (corner 1920 3) := by change Nat.Prime 14741761; exact ProjectEulerSolutions.P58PrimeNodes.p_14741761.prime
    exact decide_eq_true prime
theorem row_count_1920 : row 1920 = 1 := (row_1920.checked).trans rfl
theorem model_count_1920 : modelRow 1920 = 1 := by decide
theorem modelrow_agrees_1920 : modelRow 1920 = row 1920 := model_count_1920.trans row_count_1920.symm
private def atomic_1793 : CountSegment 1 10 13120 1792 1793 933 933 := CountSegment.single (first := 1792) (initial := 933) row_count_1793 (by decide)
private theorem atomic_1793_agreement : RowAgreement 1792 1793 := RowAgreement.single (first := 1792) modelrow_agrees_1793
private def atomic_1794 : CountSegment 1 10 13120 1793 1794 933 934 := CountSegment.single (first := 1793) (initial := 933) row_count_1794 (by decide)
private theorem atomic_1794_agreement : RowAgreement 1793 1794 := RowAgreement.single (first := 1793) modelrow_agrees_1794
private def atomic_1795 : CountSegment 1 10 13120 1794 1795 934 934 := CountSegment.single (first := 1794) (initial := 934) row_count_1795 (by decide)
private theorem atomic_1795_agreement : RowAgreement 1794 1795 := RowAgreement.single (first := 1794) modelrow_agrees_1795
private def atomic_1796 : CountSegment 1 10 13120 1795 1796 934 934 := CountSegment.single (first := 1795) (initial := 934) row_count_1796 (by decide)
private theorem atomic_1796_agreement : RowAgreement 1795 1796 := RowAgreement.single (first := 1795) modelrow_agrees_1796
private def atomic_1797 : CountSegment 1 10 13120 1796 1797 934 935 := CountSegment.single (first := 1796) (initial := 934) row_count_1797 (by decide)
private theorem atomic_1797_agreement : RowAgreement 1796 1797 := RowAgreement.single (first := 1796) modelrow_agrees_1797
private def atomic_1798 : CountSegment 1 10 13120 1797 1798 935 935 := CountSegment.single (first := 1797) (initial := 935) row_count_1798 (by decide)
private theorem atomic_1798_agreement : RowAgreement 1797 1798 := RowAgreement.single (first := 1797) modelrow_agrees_1798
private def atomic_1799 : CountSegment 1 10 13120 1798 1799 935 935 := CountSegment.single (first := 1798) (initial := 935) row_count_1799 (by decide)
private theorem atomic_1799_agreement : RowAgreement 1798 1799 := RowAgreement.single (first := 1798) modelrow_agrees_1799
private def atomic_1800 : CountSegment 1 10 13120 1799 1800 935 935 := CountSegment.single (first := 1799) (initial := 935) row_count_1800 (by decide)
private theorem atomic_1800_agreement : RowAgreement 1799 1800 := RowAgreement.single (first := 1799) modelrow_agrees_1800
private def atomic_1801 : CountSegment 1 10 13120 1800 1801 935 935 := CountSegment.single (first := 1800) (initial := 935) row_count_1801 (by decide)
private theorem atomic_1801_agreement : RowAgreement 1800 1801 := RowAgreement.single (first := 1800) modelrow_agrees_1801
private def atomic_1802 : CountSegment 1 10 13120 1801 1802 935 936 := CountSegment.single (first := 1801) (initial := 935) row_count_1802 (by decide)
private theorem atomic_1802_agreement : RowAgreement 1801 1802 := RowAgreement.single (first := 1801) modelrow_agrees_1802
private def atomic_1803 : CountSegment 1 10 13120 1802 1803 936 937 := CountSegment.single (first := 1802) (initial := 936) row_count_1803 (by decide)
private theorem atomic_1803_agreement : RowAgreement 1802 1803 := RowAgreement.single (first := 1802) modelrow_agrees_1803
private def atomic_1804 : CountSegment 1 10 13120 1803 1804 937 938 := CountSegment.single (first := 1803) (initial := 937) row_count_1804 (by decide)
private theorem atomic_1804_agreement : RowAgreement 1803 1804 := RowAgreement.single (first := 1803) modelrow_agrees_1804
private def atomic_1805 : CountSegment 1 10 13120 1804 1805 938 938 := CountSegment.single (first := 1804) (initial := 938) row_count_1805 (by decide)
private theorem atomic_1805_agreement : RowAgreement 1804 1805 := RowAgreement.single (first := 1804) modelrow_agrees_1805
private def atomic_1806 : CountSegment 1 10 13120 1805 1806 938 939 := CountSegment.single (first := 1805) (initial := 938) row_count_1806 (by decide)
private theorem atomic_1806_agreement : RowAgreement 1805 1806 := RowAgreement.single (first := 1805) modelrow_agrees_1806
private def atomic_1807 : CountSegment 1 10 13120 1806 1807 939 939 := CountSegment.single (first := 1806) (initial := 939) row_count_1807 (by decide)
private theorem atomic_1807_agreement : RowAgreement 1806 1807 := RowAgreement.single (first := 1806) modelrow_agrees_1807
private def atomic_1808 : CountSegment 1 10 13120 1807 1808 939 939 := CountSegment.single (first := 1807) (initial := 939) row_count_1808 (by decide)
private theorem atomic_1808_agreement : RowAgreement 1807 1808 := RowAgreement.single (first := 1807) modelrow_agrees_1808
private def atomic_1809 : CountSegment 1 10 13120 1808 1809 939 939 := CountSegment.single (first := 1808) (initial := 939) row_count_1809 (by decide)
private theorem atomic_1809_agreement : RowAgreement 1808 1809 := RowAgreement.single (first := 1808) modelrow_agrees_1809
private def atomic_1810 : CountSegment 1 10 13120 1809 1810 939 939 := CountSegment.single (first := 1809) (initial := 939) row_count_1810 (by decide)
private theorem atomic_1810_agreement : RowAgreement 1809 1810 := RowAgreement.single (first := 1809) modelrow_agrees_1810
private def atomic_1811 : CountSegment 1 10 13120 1810 1811 939 939 := CountSegment.single (first := 1810) (initial := 939) row_count_1811 (by decide)
private theorem atomic_1811_agreement : RowAgreement 1810 1811 := RowAgreement.single (first := 1810) modelrow_agrees_1811
private def atomic_1812 : CountSegment 1 10 13120 1811 1812 939 940 := CountSegment.single (first := 1811) (initial := 939) row_count_1812 (by decide)
private theorem atomic_1812_agreement : RowAgreement 1811 1812 := RowAgreement.single (first := 1811) modelrow_agrees_1812
private def atomic_1813 : CountSegment 1 10 13120 1812 1813 940 941 := CountSegment.single (first := 1812) (initial := 940) row_count_1813 (by decide)
private theorem atomic_1813_agreement : RowAgreement 1812 1813 := RowAgreement.single (first := 1812) modelrow_agrees_1813
private def atomic_1814 : CountSegment 1 10 13120 1813 1814 941 941 := CountSegment.single (first := 1813) (initial := 941) row_count_1814 (by decide)
private theorem atomic_1814_agreement : RowAgreement 1813 1814 := RowAgreement.single (first := 1813) modelrow_agrees_1814
private def atomic_1815 : CountSegment 1 10 13120 1814 1815 941 942 := CountSegment.single (first := 1814) (initial := 941) row_count_1815 (by decide)
private theorem atomic_1815_agreement : RowAgreement 1814 1815 := RowAgreement.single (first := 1814) modelrow_agrees_1815
private def atomic_1816 : CountSegment 1 10 13120 1815 1816 942 942 := CountSegment.single (first := 1815) (initial := 942) row_count_1816 (by decide)
private theorem atomic_1816_agreement : RowAgreement 1815 1816 := RowAgreement.single (first := 1815) modelrow_agrees_1816
private def atomic_1817 : CountSegment 1 10 13120 1816 1817 942 942 := CountSegment.single (first := 1816) (initial := 942) row_count_1817 (by decide)
private theorem atomic_1817_agreement : RowAgreement 1816 1817 := RowAgreement.single (first := 1816) modelrow_agrees_1817
private def atomic_1818 : CountSegment 1 10 13120 1817 1818 942 942 := CountSegment.single (first := 1817) (initial := 942) row_count_1818 (by decide)
private theorem atomic_1818_agreement : RowAgreement 1817 1818 := RowAgreement.single (first := 1817) modelrow_agrees_1818
private def atomic_1819 : CountSegment 1 10 13120 1818 1819 942 943 := CountSegment.single (first := 1818) (initial := 942) row_count_1819 (by decide)
private theorem atomic_1819_agreement : RowAgreement 1818 1819 := RowAgreement.single (first := 1818) modelrow_agrees_1819
private def atomic_1820 : CountSegment 1 10 13120 1819 1820 943 944 := CountSegment.single (first := 1819) (initial := 943) row_count_1820 (by decide)
private theorem atomic_1820_agreement : RowAgreement 1819 1820 := RowAgreement.single (first := 1819) modelrow_agrees_1820
private def atomic_1821 : CountSegment 1 10 13120 1820 1821 944 944 := CountSegment.single (first := 1820) (initial := 944) row_count_1821 (by decide)
private theorem atomic_1821_agreement : RowAgreement 1820 1821 := RowAgreement.single (first := 1820) modelrow_agrees_1821
private def atomic_1822 : CountSegment 1 10 13120 1821 1822 944 945 := CountSegment.single (first := 1821) (initial := 944) row_count_1822 (by decide)
private theorem atomic_1822_agreement : RowAgreement 1821 1822 := RowAgreement.single (first := 1821) modelrow_agrees_1822
private def atomic_1823 : CountSegment 1 10 13120 1822 1823 945 946 := CountSegment.single (first := 1822) (initial := 945) row_count_1823 (by decide)
private theorem atomic_1823_agreement : RowAgreement 1822 1823 := RowAgreement.single (first := 1822) modelrow_agrees_1823
private def atomic_1824 : CountSegment 1 10 13120 1823 1824 946 946 := CountSegment.single (first := 1823) (initial := 946) row_count_1824 (by decide)
private theorem atomic_1824_agreement : RowAgreement 1823 1824 := RowAgreement.single (first := 1823) modelrow_agrees_1824
private def atomic_1825 : CountSegment 1 10 13120 1824 1825 946 947 := CountSegment.single (first := 1824) (initial := 946) row_count_1825 (by decide)
private theorem atomic_1825_agreement : RowAgreement 1824 1825 := RowAgreement.single (first := 1824) modelrow_agrees_1825
private def atomic_1826 : CountSegment 1 10 13120 1825 1826 947 947 := CountSegment.single (first := 1825) (initial := 947) row_count_1826 (by decide)
private theorem atomic_1826_agreement : RowAgreement 1825 1826 := RowAgreement.single (first := 1825) modelrow_agrees_1826
private def atomic_1827 : CountSegment 1 10 13120 1826 1827 947 948 := CountSegment.single (first := 1826) (initial := 947) row_count_1827 (by decide)
private theorem atomic_1827_agreement : RowAgreement 1826 1827 := RowAgreement.single (first := 1826) modelrow_agrees_1827
private def atomic_1828 : CountSegment 1 10 13120 1827 1828 948 948 := CountSegment.single (first := 1827) (initial := 948) row_count_1828 (by decide)
private theorem atomic_1828_agreement : RowAgreement 1827 1828 := RowAgreement.single (first := 1827) modelrow_agrees_1828
private def atomic_1829 : CountSegment 1 10 13120 1828 1829 948 948 := CountSegment.single (first := 1828) (initial := 948) row_count_1829 (by decide)
private theorem atomic_1829_agreement : RowAgreement 1828 1829 := RowAgreement.single (first := 1828) modelrow_agrees_1829
private def atomic_1830 : CountSegment 1 10 13120 1829 1830 948 949 := CountSegment.single (first := 1829) (initial := 948) row_count_1830 (by decide)
private theorem atomic_1830_agreement : RowAgreement 1829 1830 := RowAgreement.single (first := 1829) modelrow_agrees_1830
private def atomic_1831 : CountSegment 1 10 13120 1830 1831 949 949 := CountSegment.single (first := 1830) (initial := 949) row_count_1831 (by decide)
private theorem atomic_1831_agreement : RowAgreement 1830 1831 := RowAgreement.single (first := 1830) modelrow_agrees_1831
private def atomic_1832 : CountSegment 1 10 13120 1831 1832 949 949 := CountSegment.single (first := 1831) (initial := 949) row_count_1832 (by decide)
private theorem atomic_1832_agreement : RowAgreement 1831 1832 := RowAgreement.single (first := 1831) modelrow_agrees_1832
private def atomic_1833 : CountSegment 1 10 13120 1832 1833 949 949 := CountSegment.single (first := 1832) (initial := 949) row_count_1833 (by decide)
private theorem atomic_1833_agreement : RowAgreement 1832 1833 := RowAgreement.single (first := 1832) modelrow_agrees_1833
private def atomic_1834 : CountSegment 1 10 13120 1833 1834 949 949 := CountSegment.single (first := 1833) (initial := 949) row_count_1834 (by decide)
private theorem atomic_1834_agreement : RowAgreement 1833 1834 := RowAgreement.single (first := 1833) modelrow_agrees_1834
private def atomic_1835 : CountSegment 1 10 13120 1834 1835 949 950 := CountSegment.single (first := 1834) (initial := 949) row_count_1835 (by decide)
private theorem atomic_1835_agreement : RowAgreement 1834 1835 := RowAgreement.single (first := 1834) modelrow_agrees_1835
private def atomic_1836 : CountSegment 1 10 13120 1835 1836 950 951 := CountSegment.single (first := 1835) (initial := 950) row_count_1836 (by decide)
private theorem atomic_1836_agreement : RowAgreement 1835 1836 := RowAgreement.single (first := 1835) modelrow_agrees_1836
private def atomic_1837 : CountSegment 1 10 13120 1836 1837 951 951 := CountSegment.single (first := 1836) (initial := 951) row_count_1837 (by decide)
private theorem atomic_1837_agreement : RowAgreement 1836 1837 := RowAgreement.single (first := 1836) modelrow_agrees_1837
private def atomic_1838 : CountSegment 1 10 13120 1837 1838 951 951 := CountSegment.single (first := 1837) (initial := 951) row_count_1838 (by decide)
private theorem atomic_1838_agreement : RowAgreement 1837 1838 := RowAgreement.single (first := 1837) modelrow_agrees_1838
private def atomic_1839 : CountSegment 1 10 13120 1838 1839 951 951 := CountSegment.single (first := 1838) (initial := 951) row_count_1839 (by decide)
private theorem atomic_1839_agreement : RowAgreement 1838 1839 := RowAgreement.single (first := 1838) modelrow_agrees_1839
private def atomic_1840 : CountSegment 1 10 13120 1839 1840 951 951 := CountSegment.single (first := 1839) (initial := 951) row_count_1840 (by decide)
private theorem atomic_1840_agreement : RowAgreement 1839 1840 := RowAgreement.single (first := 1839) modelrow_agrees_1840
private def atomic_1841 : CountSegment 1 10 13120 1840 1841 951 951 := CountSegment.single (first := 1840) (initial := 951) row_count_1841 (by decide)
private theorem atomic_1841_agreement : RowAgreement 1840 1841 := RowAgreement.single (first := 1840) modelrow_agrees_1841
private def atomic_1842 : CountSegment 1 10 13120 1841 1842 951 951 := CountSegment.single (first := 1841) (initial := 951) row_count_1842 (by decide)
private theorem atomic_1842_agreement : RowAgreement 1841 1842 := RowAgreement.single (first := 1841) modelrow_agrees_1842
private def atomic_1843 : CountSegment 1 10 13120 1842 1843 951 952 := CountSegment.single (first := 1842) (initial := 951) row_count_1843 (by decide)
private theorem atomic_1843_agreement : RowAgreement 1842 1843 := RowAgreement.single (first := 1842) modelrow_agrees_1843
private def atomic_1844 : CountSegment 1 10 13120 1843 1844 952 953 := CountSegment.single (first := 1843) (initial := 952) row_count_1844 (by decide)
private theorem atomic_1844_agreement : RowAgreement 1843 1844 := RowAgreement.single (first := 1843) modelrow_agrees_1844
private def atomic_1845 : CountSegment 1 10 13120 1844 1845 953 953 := CountSegment.single (first := 1844) (initial := 953) row_count_1845 (by decide)
private theorem atomic_1845_agreement : RowAgreement 1844 1845 := RowAgreement.single (first := 1844) modelrow_agrees_1845
private def atomic_1846 : CountSegment 1 10 13120 1845 1846 953 954 := CountSegment.single (first := 1845) (initial := 953) row_count_1846 (by decide)
private theorem atomic_1846_agreement : RowAgreement 1845 1846 := RowAgreement.single (first := 1845) modelrow_agrees_1846
private def atomic_1847 : CountSegment 1 10 13120 1846 1847 954 954 := CountSegment.single (first := 1846) (initial := 954) row_count_1847 (by decide)
private theorem atomic_1847_agreement : RowAgreement 1846 1847 := RowAgreement.single (first := 1846) modelrow_agrees_1847
private def atomic_1848 : CountSegment 1 10 13120 1847 1848 954 954 := CountSegment.single (first := 1847) (initial := 954) row_count_1848 (by decide)
private theorem atomic_1848_agreement : RowAgreement 1847 1848 := RowAgreement.single (first := 1847) modelrow_agrees_1848
private def atomic_1849 : CountSegment 1 10 13120 1848 1849 954 954 := CountSegment.single (first := 1848) (initial := 954) row_count_1849 (by decide)
private theorem atomic_1849_agreement : RowAgreement 1848 1849 := RowAgreement.single (first := 1848) modelrow_agrees_1849
private def atomic_1850 : CountSegment 1 10 13120 1849 1850 954 954 := CountSegment.single (first := 1849) (initial := 954) row_count_1850 (by decide)
private theorem atomic_1850_agreement : RowAgreement 1849 1850 := RowAgreement.single (first := 1849) modelrow_agrees_1850
private def atomic_1851 : CountSegment 1 10 13120 1850 1851 954 954 := CountSegment.single (first := 1850) (initial := 954) row_count_1851 (by decide)
private theorem atomic_1851_agreement : RowAgreement 1850 1851 := RowAgreement.single (first := 1850) modelrow_agrees_1851
private def atomic_1852 : CountSegment 1 10 13120 1851 1852 954 954 := CountSegment.single (first := 1851) (initial := 954) row_count_1852 (by decide)
private theorem atomic_1852_agreement : RowAgreement 1851 1852 := RowAgreement.single (first := 1851) modelrow_agrees_1852
private def atomic_1853 : CountSegment 1 10 13120 1852 1853 954 954 := CountSegment.single (first := 1852) (initial := 954) row_count_1853 (by decide)
private theorem atomic_1853_agreement : RowAgreement 1852 1853 := RowAgreement.single (first := 1852) modelrow_agrees_1853
private def atomic_1854 : CountSegment 1 10 13120 1853 1854 954 954 := CountSegment.single (first := 1853) (initial := 954) row_count_1854 (by decide)
private theorem atomic_1854_agreement : RowAgreement 1853 1854 := RowAgreement.single (first := 1853) modelrow_agrees_1854
private def atomic_1855 : CountSegment 1 10 13120 1854 1855 954 954 := CountSegment.single (first := 1854) (initial := 954) row_count_1855 (by decide)
private theorem atomic_1855_agreement : RowAgreement 1854 1855 := RowAgreement.single (first := 1854) modelrow_agrees_1855
private def atomic_1856 : CountSegment 1 10 13120 1855 1856 954 955 := CountSegment.single (first := 1855) (initial := 954) row_count_1856 (by decide)
private theorem atomic_1856_agreement : RowAgreement 1855 1856 := RowAgreement.single (first := 1855) modelrow_agrees_1856
private def atomic_1857 : CountSegment 1 10 13120 1856 1857 955 956 := CountSegment.single (first := 1856) (initial := 955) row_count_1857 (by decide)
private theorem atomic_1857_agreement : RowAgreement 1856 1857 := RowAgreement.single (first := 1856) modelrow_agrees_1857
private def atomic_1858 : CountSegment 1 10 13120 1857 1858 956 957 := CountSegment.single (first := 1857) (initial := 956) row_count_1858 (by decide)
private theorem atomic_1858_agreement : RowAgreement 1857 1858 := RowAgreement.single (first := 1857) modelrow_agrees_1858
private def atomic_1859 : CountSegment 1 10 13120 1858 1859 957 957 := CountSegment.single (first := 1858) (initial := 957) row_count_1859 (by decide)
private theorem atomic_1859_agreement : RowAgreement 1858 1859 := RowAgreement.single (first := 1858) modelrow_agrees_1859
private def atomic_1860 : CountSegment 1 10 13120 1859 1860 957 958 := CountSegment.single (first := 1859) (initial := 957) row_count_1860 (by decide)
private theorem atomic_1860_agreement : RowAgreement 1859 1860 := RowAgreement.single (first := 1859) modelrow_agrees_1860
private def atomic_1861 : CountSegment 1 10 13120 1860 1861 958 958 := CountSegment.single (first := 1860) (initial := 958) row_count_1861 (by decide)
private theorem atomic_1861_agreement : RowAgreement 1860 1861 := RowAgreement.single (first := 1860) modelrow_agrees_1861
private def atomic_1862 : CountSegment 1 10 13120 1861 1862 958 958 := CountSegment.single (first := 1861) (initial := 958) row_count_1862 (by decide)
private theorem atomic_1862_agreement : RowAgreement 1861 1862 := RowAgreement.single (first := 1861) modelrow_agrees_1862
private def atomic_1863 : CountSegment 1 10 13120 1862 1863 958 958 := CountSegment.single (first := 1862) (initial := 958) row_count_1863 (by decide)
private theorem atomic_1863_agreement : RowAgreement 1862 1863 := RowAgreement.single (first := 1862) modelrow_agrees_1863
private def atomic_1864 : CountSegment 1 10 13120 1863 1864 958 958 := CountSegment.single (first := 1863) (initial := 958) row_count_1864 (by decide)
private theorem atomic_1864_agreement : RowAgreement 1863 1864 := RowAgreement.single (first := 1863) modelrow_agrees_1864
private def atomic_1865 : CountSegment 1 10 13120 1864 1865 958 959 := CountSegment.single (first := 1864) (initial := 958) row_count_1865 (by decide)
private theorem atomic_1865_agreement : RowAgreement 1864 1865 := RowAgreement.single (first := 1864) modelrow_agrees_1865
private def atomic_1866 : CountSegment 1 10 13120 1865 1866 959 960 := CountSegment.single (first := 1865) (initial := 959) row_count_1866 (by decide)
private theorem atomic_1866_agreement : RowAgreement 1865 1866 := RowAgreement.single (first := 1865) modelrow_agrees_1866
private def atomic_1867 : CountSegment 1 10 13120 1866 1867 960 961 := CountSegment.single (first := 1866) (initial := 960) row_count_1867 (by decide)
private theorem atomic_1867_agreement : RowAgreement 1866 1867 := RowAgreement.single (first := 1866) modelrow_agrees_1867
private def atomic_1868 : CountSegment 1 10 13120 1867 1868 961 961 := CountSegment.single (first := 1867) (initial := 961) row_count_1868 (by decide)
private theorem atomic_1868_agreement : RowAgreement 1867 1868 := RowAgreement.single (first := 1867) modelrow_agrees_1868
private def atomic_1869 : CountSegment 1 10 13120 1868 1869 961 962 := CountSegment.single (first := 1868) (initial := 961) row_count_1869 (by decide)
private theorem atomic_1869_agreement : RowAgreement 1868 1869 := RowAgreement.single (first := 1868) modelrow_agrees_1869
private def atomic_1870 : CountSegment 1 10 13120 1869 1870 962 962 := CountSegment.single (first := 1869) (initial := 962) row_count_1870 (by decide)
private theorem atomic_1870_agreement : RowAgreement 1869 1870 := RowAgreement.single (first := 1869) modelrow_agrees_1870
private def atomic_1871 : CountSegment 1 10 13120 1870 1871 962 963 := CountSegment.single (first := 1870) (initial := 962) row_count_1871 (by decide)
private theorem atomic_1871_agreement : RowAgreement 1870 1871 := RowAgreement.single (first := 1870) modelrow_agrees_1871
private def atomic_1872 : CountSegment 1 10 13120 1871 1872 963 964 := CountSegment.single (first := 1871) (initial := 963) row_count_1872 (by decide)
private theorem atomic_1872_agreement : RowAgreement 1871 1872 := RowAgreement.single (first := 1871) modelrow_agrees_1872
private def atomic_1873 : CountSegment 1 10 13120 1872 1873 964 966 := CountSegment.single (first := 1872) (initial := 964) row_count_1873 (by decide)
private theorem atomic_1873_agreement : RowAgreement 1872 1873 := RowAgreement.single (first := 1872) modelrow_agrees_1873
private def atomic_1874 : CountSegment 1 10 13120 1873 1874 966 966 := CountSegment.single (first := 1873) (initial := 966) row_count_1874 (by decide)
private theorem atomic_1874_agreement : RowAgreement 1873 1874 := RowAgreement.single (first := 1873) modelrow_agrees_1874
private def atomic_1875 : CountSegment 1 10 13120 1874 1875 966 966 := CountSegment.single (first := 1874) (initial := 966) row_count_1875 (by decide)
private theorem atomic_1875_agreement : RowAgreement 1874 1875 := RowAgreement.single (first := 1874) modelrow_agrees_1875
private def atomic_1876 : CountSegment 1 10 13120 1875 1876 966 966 := CountSegment.single (first := 1875) (initial := 966) row_count_1876 (by decide)
private theorem atomic_1876_agreement : RowAgreement 1875 1876 := RowAgreement.single (first := 1875) modelrow_agrees_1876
private def atomic_1877 : CountSegment 1 10 13120 1876 1877 966 967 := CountSegment.single (first := 1876) (initial := 966) row_count_1877 (by decide)
private theorem atomic_1877_agreement : RowAgreement 1876 1877 := RowAgreement.single (first := 1876) modelrow_agrees_1877
private def atomic_1878 : CountSegment 1 10 13120 1877 1878 967 968 := CountSegment.single (first := 1877) (initial := 967) row_count_1878 (by decide)
private theorem atomic_1878_agreement : RowAgreement 1877 1878 := RowAgreement.single (first := 1877) modelrow_agrees_1878
private def atomic_1879 : CountSegment 1 10 13120 1878 1879 968 968 := CountSegment.single (first := 1878) (initial := 968) row_count_1879 (by decide)
private theorem atomic_1879_agreement : RowAgreement 1878 1879 := RowAgreement.single (first := 1878) modelrow_agrees_1879
private def atomic_1880 : CountSegment 1 10 13120 1879 1880 968 968 := CountSegment.single (first := 1879) (initial := 968) row_count_1880 (by decide)
private theorem atomic_1880_agreement : RowAgreement 1879 1880 := RowAgreement.single (first := 1879) modelrow_agrees_1880
private def atomic_1881 : CountSegment 1 10 13120 1880 1881 968 969 := CountSegment.single (first := 1880) (initial := 968) row_count_1881 (by decide)
private theorem atomic_1881_agreement : RowAgreement 1880 1881 := RowAgreement.single (first := 1880) modelrow_agrees_1881
private def atomic_1882 : CountSegment 1 10 13120 1881 1882 969 970 := CountSegment.single (first := 1881) (initial := 969) row_count_1882 (by decide)
private theorem atomic_1882_agreement : RowAgreement 1881 1882 := RowAgreement.single (first := 1881) modelrow_agrees_1882
private def atomic_1883 : CountSegment 1 10 13120 1882 1883 970 970 := CountSegment.single (first := 1882) (initial := 970) row_count_1883 (by decide)
private theorem atomic_1883_agreement : RowAgreement 1882 1883 := RowAgreement.single (first := 1882) modelrow_agrees_1883
private def atomic_1884 : CountSegment 1 10 13120 1883 1884 970 970 := CountSegment.single (first := 1883) (initial := 970) row_count_1884 (by decide)
private theorem atomic_1884_agreement : RowAgreement 1883 1884 := RowAgreement.single (first := 1883) modelrow_agrees_1884
private def atomic_1885 : CountSegment 1 10 13120 1884 1885 970 970 := CountSegment.single (first := 1884) (initial := 970) row_count_1885 (by decide)
private theorem atomic_1885_agreement : RowAgreement 1884 1885 := RowAgreement.single (first := 1884) modelrow_agrees_1885
private def atomic_1886 : CountSegment 1 10 13120 1885 1886 970 971 := CountSegment.single (first := 1885) (initial := 970) row_count_1886 (by decide)
private theorem atomic_1886_agreement : RowAgreement 1885 1886 := RowAgreement.single (first := 1885) modelrow_agrees_1886
private def atomic_1887 : CountSegment 1 10 13120 1886 1887 971 972 := CountSegment.single (first := 1886) (initial := 971) row_count_1887 (by decide)
private theorem atomic_1887_agreement : RowAgreement 1886 1887 := RowAgreement.single (first := 1886) modelrow_agrees_1887
private def atomic_1888 : CountSegment 1 10 13120 1887 1888 972 973 := CountSegment.single (first := 1887) (initial := 972) row_count_1888 (by decide)
private theorem atomic_1888_agreement : RowAgreement 1887 1888 := RowAgreement.single (first := 1887) modelrow_agrees_1888
private def atomic_1889 : CountSegment 1 10 13120 1888 1889 973 973 := CountSegment.single (first := 1888) (initial := 973) row_count_1889 (by decide)
private theorem atomic_1889_agreement : RowAgreement 1888 1889 := RowAgreement.single (first := 1888) modelrow_agrees_1889
private def atomic_1890 : CountSegment 1 10 13120 1889 1890 973 973 := CountSegment.single (first := 1889) (initial := 973) row_count_1890 (by decide)
private theorem atomic_1890_agreement : RowAgreement 1889 1890 := RowAgreement.single (first := 1889) modelrow_agrees_1890
private def atomic_1891 : CountSegment 1 10 13120 1890 1891 973 973 := CountSegment.single (first := 1890) (initial := 973) row_count_1891 (by decide)
private theorem atomic_1891_agreement : RowAgreement 1890 1891 := RowAgreement.single (first := 1890) modelrow_agrees_1891
private def atomic_1892 : CountSegment 1 10 13120 1891 1892 973 974 := CountSegment.single (first := 1891) (initial := 973) row_count_1892 (by decide)
private theorem atomic_1892_agreement : RowAgreement 1891 1892 := RowAgreement.single (first := 1891) modelrow_agrees_1892
private def atomic_1893 : CountSegment 1 10 13120 1892 1893 974 975 := CountSegment.single (first := 1892) (initial := 974) row_count_1893 (by decide)
private theorem atomic_1893_agreement : RowAgreement 1892 1893 := RowAgreement.single (first := 1892) modelrow_agrees_1893
private def atomic_1894 : CountSegment 1 10 13120 1893 1894 975 975 := CountSegment.single (first := 1893) (initial := 975) row_count_1894 (by decide)
private theorem atomic_1894_agreement : RowAgreement 1893 1894 := RowAgreement.single (first := 1893) modelrow_agrees_1894
private def atomic_1895 : CountSegment 1 10 13120 1894 1895 975 976 := CountSegment.single (first := 1894) (initial := 975) row_count_1895 (by decide)
private theorem atomic_1895_agreement : RowAgreement 1894 1895 := RowAgreement.single (first := 1894) modelrow_agrees_1895
private def atomic_1896 : CountSegment 1 10 13120 1895 1896 976 976 := CountSegment.single (first := 1895) (initial := 976) row_count_1896 (by decide)
private theorem atomic_1896_agreement : RowAgreement 1895 1896 := RowAgreement.single (first := 1895) modelrow_agrees_1896
private def atomic_1897 : CountSegment 1 10 13120 1896 1897 976 978 := CountSegment.single (first := 1896) (initial := 976) row_count_1897 (by decide)
private theorem atomic_1897_agreement : RowAgreement 1896 1897 := RowAgreement.single (first := 1896) modelrow_agrees_1897
private def atomic_1898 : CountSegment 1 10 13120 1897 1898 978 978 := CountSegment.single (first := 1897) (initial := 978) row_count_1898 (by decide)
private theorem atomic_1898_agreement : RowAgreement 1897 1898 := RowAgreement.single (first := 1897) modelrow_agrees_1898
private def atomic_1899 : CountSegment 1 10 13120 1898 1899 978 978 := CountSegment.single (first := 1898) (initial := 978) row_count_1899 (by decide)
private theorem atomic_1899_agreement : RowAgreement 1898 1899 := RowAgreement.single (first := 1898) modelrow_agrees_1899
private def atomic_1900 : CountSegment 1 10 13120 1899 1900 978 979 := CountSegment.single (first := 1899) (initial := 978) row_count_1900 (by decide)
private theorem atomic_1900_agreement : RowAgreement 1899 1900 := RowAgreement.single (first := 1899) modelrow_agrees_1900
private def atomic_1901 : CountSegment 1 10 13120 1900 1901 979 980 := CountSegment.single (first := 1900) (initial := 979) row_count_1901 (by decide)
private theorem atomic_1901_agreement : RowAgreement 1900 1901 := RowAgreement.single (first := 1900) modelrow_agrees_1901
private def atomic_1902 : CountSegment 1 10 13120 1901 1902 980 980 := CountSegment.single (first := 1901) (initial := 980) row_count_1902 (by decide)
private theorem atomic_1902_agreement : RowAgreement 1901 1902 := RowAgreement.single (first := 1901) modelrow_agrees_1902
private def atomic_1903 : CountSegment 1 10 13120 1902 1903 980 981 := CountSegment.single (first := 1902) (initial := 980) row_count_1903 (by decide)
private theorem atomic_1903_agreement : RowAgreement 1902 1903 := RowAgreement.single (first := 1902) modelrow_agrees_1903
private def atomic_1904 : CountSegment 1 10 13120 1903 1904 981 981 := CountSegment.single (first := 1903) (initial := 981) row_count_1904 (by decide)
private theorem atomic_1904_agreement : RowAgreement 1903 1904 := RowAgreement.single (first := 1903) modelrow_agrees_1904
private def atomic_1905 : CountSegment 1 10 13120 1904 1905 981 981 := CountSegment.single (first := 1904) (initial := 981) row_count_1905 (by decide)
private theorem atomic_1905_agreement : RowAgreement 1904 1905 := RowAgreement.single (first := 1904) modelrow_agrees_1905
private def atomic_1906 : CountSegment 1 10 13120 1905 1906 981 981 := CountSegment.single (first := 1905) (initial := 981) row_count_1906 (by decide)
private theorem atomic_1906_agreement : RowAgreement 1905 1906 := RowAgreement.single (first := 1905) modelrow_agrees_1906
private def atomic_1907 : CountSegment 1 10 13120 1906 1907 981 981 := CountSegment.single (first := 1906) (initial := 981) row_count_1907 (by decide)
private theorem atomic_1907_agreement : RowAgreement 1906 1907 := RowAgreement.single (first := 1906) modelrow_agrees_1907
private def atomic_1908 : CountSegment 1 10 13120 1907 1908 981 981 := CountSegment.single (first := 1907) (initial := 981) row_count_1908 (by decide)
private theorem atomic_1908_agreement : RowAgreement 1907 1908 := RowAgreement.single (first := 1907) modelrow_agrees_1908
private def atomic_1909 : CountSegment 1 10 13120 1908 1909 981 981 := CountSegment.single (first := 1908) (initial := 981) row_count_1909 (by decide)
private theorem atomic_1909_agreement : RowAgreement 1908 1909 := RowAgreement.single (first := 1908) modelrow_agrees_1909
private def atomic_1910 : CountSegment 1 10 13120 1909 1910 981 981 := CountSegment.single (first := 1909) (initial := 981) row_count_1910 (by decide)
private theorem atomic_1910_agreement : RowAgreement 1909 1910 := RowAgreement.single (first := 1909) modelrow_agrees_1910
private def atomic_1911 : CountSegment 1 10 13120 1910 1911 981 981 := CountSegment.single (first := 1910) (initial := 981) row_count_1911 (by decide)
private theorem atomic_1911_agreement : RowAgreement 1910 1911 := RowAgreement.single (first := 1910) modelrow_agrees_1911
private def atomic_1912 : CountSegment 1 10 13120 1911 1912 981 981 := CountSegment.single (first := 1911) (initial := 981) row_count_1912 (by decide)
private theorem atomic_1912_agreement : RowAgreement 1911 1912 := RowAgreement.single (first := 1911) modelrow_agrees_1912
private def atomic_1913 : CountSegment 1 10 13120 1912 1913 981 982 := CountSegment.single (first := 1912) (initial := 981) row_count_1913 (by decide)
private theorem atomic_1913_agreement : RowAgreement 1912 1913 := RowAgreement.single (first := 1912) modelrow_agrees_1913
private def atomic_1914 : CountSegment 1 10 13120 1913 1914 982 983 := CountSegment.single (first := 1913) (initial := 982) row_count_1914 (by decide)
private theorem atomic_1914_agreement : RowAgreement 1913 1914 := RowAgreement.single (first := 1913) modelrow_agrees_1914
private def atomic_1915 : CountSegment 1 10 13120 1914 1915 983 983 := CountSegment.single (first := 1914) (initial := 983) row_count_1915 (by decide)
private theorem atomic_1915_agreement : RowAgreement 1914 1915 := RowAgreement.single (first := 1914) modelrow_agrees_1915
private def atomic_1916 : CountSegment 1 10 13120 1915 1916 983 983 := CountSegment.single (first := 1915) (initial := 983) row_count_1916 (by decide)
private theorem atomic_1916_agreement : RowAgreement 1915 1916 := RowAgreement.single (first := 1915) modelrow_agrees_1916
private def atomic_1917 : CountSegment 1 10 13120 1916 1917 983 984 := CountSegment.single (first := 1916) (initial := 983) row_count_1917 (by decide)
private theorem atomic_1917_agreement : RowAgreement 1916 1917 := RowAgreement.single (first := 1916) modelrow_agrees_1917
private def atomic_1918 : CountSegment 1 10 13120 1917 1918 984 985 := CountSegment.single (first := 1917) (initial := 984) row_count_1918 (by decide)
private theorem atomic_1918_agreement : RowAgreement 1917 1918 := RowAgreement.single (first := 1917) modelrow_agrees_1918
private def atomic_1919 : CountSegment 1 10 13120 1918 1919 985 985 := CountSegment.single (first := 1918) (initial := 985) row_count_1919 (by decide)
private theorem atomic_1919_agreement : RowAgreement 1918 1919 := RowAgreement.single (first := 1918) modelrow_agrees_1919
private def atomic_1920 : CountSegment 1 10 13120 1919 1920 985 986 := CountSegment.single (first := 1919) (initial := 985) row_count_1920 (by decide)
private theorem atomic_1920_agreement : RowAgreement 1919 1920 := RowAgreement.single (first := 1919) modelrow_agrees_1920
private def within_014_000 : CountSegment 1 10 13120 1792 1794 933 934 := CountSegment.join atomic_1793 atomic_1794
private theorem within_014_000_agreement : RowAgreement 1792 1794 := RowAgreement.join atomic_1793_agreement atomic_1794_agreement
private def within_014_001 : CountSegment 1 10 13120 1794 1796 934 934 := CountSegment.join atomic_1795 atomic_1796
private theorem within_014_001_agreement : RowAgreement 1794 1796 := RowAgreement.join atomic_1795_agreement atomic_1796_agreement
private def within_014_002 : CountSegment 1 10 13120 1792 1796 933 934 := CountSegment.join within_014_000 within_014_001
private theorem within_014_002_agreement : RowAgreement 1792 1796 := RowAgreement.join within_014_000_agreement within_014_001_agreement
private def within_014_003 : CountSegment 1 10 13120 1796 1798 934 935 := CountSegment.join atomic_1797 atomic_1798
private theorem within_014_003_agreement : RowAgreement 1796 1798 := RowAgreement.join atomic_1797_agreement atomic_1798_agreement
private def within_014_004 : CountSegment 1 10 13120 1798 1800 935 935 := CountSegment.join atomic_1799 atomic_1800
private theorem within_014_004_agreement : RowAgreement 1798 1800 := RowAgreement.join atomic_1799_agreement atomic_1800_agreement
private def within_014_005 : CountSegment 1 10 13120 1796 1800 934 935 := CountSegment.join within_014_003 within_014_004
private theorem within_014_005_agreement : RowAgreement 1796 1800 := RowAgreement.join within_014_003_agreement within_014_004_agreement
private def within_014_006 : CountSegment 1 10 13120 1792 1800 933 935 := CountSegment.join within_014_002 within_014_005
private theorem within_014_006_agreement : RowAgreement 1792 1800 := RowAgreement.join within_014_002_agreement within_014_005_agreement
private def within_014_007 : CountSegment 1 10 13120 1800 1802 935 936 := CountSegment.join atomic_1801 atomic_1802
private theorem within_014_007_agreement : RowAgreement 1800 1802 := RowAgreement.join atomic_1801_agreement atomic_1802_agreement
private def within_014_008 : CountSegment 1 10 13120 1802 1804 936 938 := CountSegment.join atomic_1803 atomic_1804
private theorem within_014_008_agreement : RowAgreement 1802 1804 := RowAgreement.join atomic_1803_agreement atomic_1804_agreement
private def within_014_009 : CountSegment 1 10 13120 1800 1804 935 938 := CountSegment.join within_014_007 within_014_008
private theorem within_014_009_agreement : RowAgreement 1800 1804 := RowAgreement.join within_014_007_agreement within_014_008_agreement
private def within_014_010 : CountSegment 1 10 13120 1804 1806 938 939 := CountSegment.join atomic_1805 atomic_1806
private theorem within_014_010_agreement : RowAgreement 1804 1806 := RowAgreement.join atomic_1805_agreement atomic_1806_agreement
private def within_014_011 : CountSegment 1 10 13120 1806 1808 939 939 := CountSegment.join atomic_1807 atomic_1808
private theorem within_014_011_agreement : RowAgreement 1806 1808 := RowAgreement.join atomic_1807_agreement atomic_1808_agreement
private def within_014_012 : CountSegment 1 10 13120 1804 1808 938 939 := CountSegment.join within_014_010 within_014_011
private theorem within_014_012_agreement : RowAgreement 1804 1808 := RowAgreement.join within_014_010_agreement within_014_011_agreement
private def within_014_013 : CountSegment 1 10 13120 1800 1808 935 939 := CountSegment.join within_014_009 within_014_012
private theorem within_014_013_agreement : RowAgreement 1800 1808 := RowAgreement.join within_014_009_agreement within_014_012_agreement
private def within_014_014 : CountSegment 1 10 13120 1792 1808 933 939 := CountSegment.join within_014_006 within_014_013
private theorem within_014_014_agreement : RowAgreement 1792 1808 := RowAgreement.join within_014_006_agreement within_014_013_agreement
private def within_014_015 : CountSegment 1 10 13120 1808 1810 939 939 := CountSegment.join atomic_1809 atomic_1810
private theorem within_014_015_agreement : RowAgreement 1808 1810 := RowAgreement.join atomic_1809_agreement atomic_1810_agreement
private def within_014_016 : CountSegment 1 10 13120 1810 1812 939 940 := CountSegment.join atomic_1811 atomic_1812
private theorem within_014_016_agreement : RowAgreement 1810 1812 := RowAgreement.join atomic_1811_agreement atomic_1812_agreement
private def within_014_017 : CountSegment 1 10 13120 1808 1812 939 940 := CountSegment.join within_014_015 within_014_016
private theorem within_014_017_agreement : RowAgreement 1808 1812 := RowAgreement.join within_014_015_agreement within_014_016_agreement
private def within_014_018 : CountSegment 1 10 13120 1812 1814 940 941 := CountSegment.join atomic_1813 atomic_1814
private theorem within_014_018_agreement : RowAgreement 1812 1814 := RowAgreement.join atomic_1813_agreement atomic_1814_agreement
private def within_014_019 : CountSegment 1 10 13120 1814 1816 941 942 := CountSegment.join atomic_1815 atomic_1816
private theorem within_014_019_agreement : RowAgreement 1814 1816 := RowAgreement.join atomic_1815_agreement atomic_1816_agreement
private def within_014_020 : CountSegment 1 10 13120 1812 1816 940 942 := CountSegment.join within_014_018 within_014_019
private theorem within_014_020_agreement : RowAgreement 1812 1816 := RowAgreement.join within_014_018_agreement within_014_019_agreement
private def within_014_021 : CountSegment 1 10 13120 1808 1816 939 942 := CountSegment.join within_014_017 within_014_020
private theorem within_014_021_agreement : RowAgreement 1808 1816 := RowAgreement.join within_014_017_agreement within_014_020_agreement
private def within_014_022 : CountSegment 1 10 13120 1816 1818 942 942 := CountSegment.join atomic_1817 atomic_1818
private theorem within_014_022_agreement : RowAgreement 1816 1818 := RowAgreement.join atomic_1817_agreement atomic_1818_agreement
private def within_014_023 : CountSegment 1 10 13120 1818 1820 942 944 := CountSegment.join atomic_1819 atomic_1820
private theorem within_014_023_agreement : RowAgreement 1818 1820 := RowAgreement.join atomic_1819_agreement atomic_1820_agreement
private def within_014_024 : CountSegment 1 10 13120 1816 1820 942 944 := CountSegment.join within_014_022 within_014_023
private theorem within_014_024_agreement : RowAgreement 1816 1820 := RowAgreement.join within_014_022_agreement within_014_023_agreement
private def within_014_025 : CountSegment 1 10 13120 1820 1822 944 945 := CountSegment.join atomic_1821 atomic_1822
private theorem within_014_025_agreement : RowAgreement 1820 1822 := RowAgreement.join atomic_1821_agreement atomic_1822_agreement
private def within_014_026 : CountSegment 1 10 13120 1822 1824 945 946 := CountSegment.join atomic_1823 atomic_1824
private theorem within_014_026_agreement : RowAgreement 1822 1824 := RowAgreement.join atomic_1823_agreement atomic_1824_agreement
private def within_014_027 : CountSegment 1 10 13120 1820 1824 944 946 := CountSegment.join within_014_025 within_014_026
private theorem within_014_027_agreement : RowAgreement 1820 1824 := RowAgreement.join within_014_025_agreement within_014_026_agreement
private def within_014_028 : CountSegment 1 10 13120 1816 1824 942 946 := CountSegment.join within_014_024 within_014_027
private theorem within_014_028_agreement : RowAgreement 1816 1824 := RowAgreement.join within_014_024_agreement within_014_027_agreement
private def within_014_029 : CountSegment 1 10 13120 1808 1824 939 946 := CountSegment.join within_014_021 within_014_028
private theorem within_014_029_agreement : RowAgreement 1808 1824 := RowAgreement.join within_014_021_agreement within_014_028_agreement
private def within_014_030 : CountSegment 1 10 13120 1792 1824 933 946 := CountSegment.join within_014_014 within_014_029
private theorem within_014_030_agreement : RowAgreement 1792 1824 := RowAgreement.join within_014_014_agreement within_014_029_agreement
private def within_014_031 : CountSegment 1 10 13120 1824 1826 946 947 := CountSegment.join atomic_1825 atomic_1826
private theorem within_014_031_agreement : RowAgreement 1824 1826 := RowAgreement.join atomic_1825_agreement atomic_1826_agreement
private def within_014_032 : CountSegment 1 10 13120 1826 1828 947 948 := CountSegment.join atomic_1827 atomic_1828
private theorem within_014_032_agreement : RowAgreement 1826 1828 := RowAgreement.join atomic_1827_agreement atomic_1828_agreement
private def within_014_033 : CountSegment 1 10 13120 1824 1828 946 948 := CountSegment.join within_014_031 within_014_032
private theorem within_014_033_agreement : RowAgreement 1824 1828 := RowAgreement.join within_014_031_agreement within_014_032_agreement
private def within_014_034 : CountSegment 1 10 13120 1828 1830 948 949 := CountSegment.join atomic_1829 atomic_1830
private theorem within_014_034_agreement : RowAgreement 1828 1830 := RowAgreement.join atomic_1829_agreement atomic_1830_agreement
private def within_014_035 : CountSegment 1 10 13120 1830 1832 949 949 := CountSegment.join atomic_1831 atomic_1832
private theorem within_014_035_agreement : RowAgreement 1830 1832 := RowAgreement.join atomic_1831_agreement atomic_1832_agreement
private def within_014_036 : CountSegment 1 10 13120 1828 1832 948 949 := CountSegment.join within_014_034 within_014_035
private theorem within_014_036_agreement : RowAgreement 1828 1832 := RowAgreement.join within_014_034_agreement within_014_035_agreement
private def within_014_037 : CountSegment 1 10 13120 1824 1832 946 949 := CountSegment.join within_014_033 within_014_036
private theorem within_014_037_agreement : RowAgreement 1824 1832 := RowAgreement.join within_014_033_agreement within_014_036_agreement
private def within_014_038 : CountSegment 1 10 13120 1832 1834 949 949 := CountSegment.join atomic_1833 atomic_1834
private theorem within_014_038_agreement : RowAgreement 1832 1834 := RowAgreement.join atomic_1833_agreement atomic_1834_agreement
private def within_014_039 : CountSegment 1 10 13120 1834 1836 949 951 := CountSegment.join atomic_1835 atomic_1836
private theorem within_014_039_agreement : RowAgreement 1834 1836 := RowAgreement.join atomic_1835_agreement atomic_1836_agreement
private def within_014_040 : CountSegment 1 10 13120 1832 1836 949 951 := CountSegment.join within_014_038 within_014_039
private theorem within_014_040_agreement : RowAgreement 1832 1836 := RowAgreement.join within_014_038_agreement within_014_039_agreement
private def within_014_041 : CountSegment 1 10 13120 1836 1838 951 951 := CountSegment.join atomic_1837 atomic_1838
private theorem within_014_041_agreement : RowAgreement 1836 1838 := RowAgreement.join atomic_1837_agreement atomic_1838_agreement
private def within_014_042 : CountSegment 1 10 13120 1838 1840 951 951 := CountSegment.join atomic_1839 atomic_1840
private theorem within_014_042_agreement : RowAgreement 1838 1840 := RowAgreement.join atomic_1839_agreement atomic_1840_agreement
private def within_014_043 : CountSegment 1 10 13120 1836 1840 951 951 := CountSegment.join within_014_041 within_014_042
private theorem within_014_043_agreement : RowAgreement 1836 1840 := RowAgreement.join within_014_041_agreement within_014_042_agreement
private def within_014_044 : CountSegment 1 10 13120 1832 1840 949 951 := CountSegment.join within_014_040 within_014_043
private theorem within_014_044_agreement : RowAgreement 1832 1840 := RowAgreement.join within_014_040_agreement within_014_043_agreement
private def within_014_045 : CountSegment 1 10 13120 1824 1840 946 951 := CountSegment.join within_014_037 within_014_044
private theorem within_014_045_agreement : RowAgreement 1824 1840 := RowAgreement.join within_014_037_agreement within_014_044_agreement
private def within_014_046 : CountSegment 1 10 13120 1840 1842 951 951 := CountSegment.join atomic_1841 atomic_1842
private theorem within_014_046_agreement : RowAgreement 1840 1842 := RowAgreement.join atomic_1841_agreement atomic_1842_agreement
private def within_014_047 : CountSegment 1 10 13120 1842 1844 951 953 := CountSegment.join atomic_1843 atomic_1844
private theorem within_014_047_agreement : RowAgreement 1842 1844 := RowAgreement.join atomic_1843_agreement atomic_1844_agreement
private def within_014_048 : CountSegment 1 10 13120 1840 1844 951 953 := CountSegment.join within_014_046 within_014_047
private theorem within_014_048_agreement : RowAgreement 1840 1844 := RowAgreement.join within_014_046_agreement within_014_047_agreement
private def within_014_049 : CountSegment 1 10 13120 1844 1846 953 954 := CountSegment.join atomic_1845 atomic_1846
private theorem within_014_049_agreement : RowAgreement 1844 1846 := RowAgreement.join atomic_1845_agreement atomic_1846_agreement
private def within_014_050 : CountSegment 1 10 13120 1846 1848 954 954 := CountSegment.join atomic_1847 atomic_1848
private theorem within_014_050_agreement : RowAgreement 1846 1848 := RowAgreement.join atomic_1847_agreement atomic_1848_agreement
private def within_014_051 : CountSegment 1 10 13120 1844 1848 953 954 := CountSegment.join within_014_049 within_014_050
private theorem within_014_051_agreement : RowAgreement 1844 1848 := RowAgreement.join within_014_049_agreement within_014_050_agreement
private def within_014_052 : CountSegment 1 10 13120 1840 1848 951 954 := CountSegment.join within_014_048 within_014_051
private theorem within_014_052_agreement : RowAgreement 1840 1848 := RowAgreement.join within_014_048_agreement within_014_051_agreement
private def within_014_053 : CountSegment 1 10 13120 1848 1850 954 954 := CountSegment.join atomic_1849 atomic_1850
private theorem within_014_053_agreement : RowAgreement 1848 1850 := RowAgreement.join atomic_1849_agreement atomic_1850_agreement
private def within_014_054 : CountSegment 1 10 13120 1850 1852 954 954 := CountSegment.join atomic_1851 atomic_1852
private theorem within_014_054_agreement : RowAgreement 1850 1852 := RowAgreement.join atomic_1851_agreement atomic_1852_agreement
private def within_014_055 : CountSegment 1 10 13120 1848 1852 954 954 := CountSegment.join within_014_053 within_014_054
private theorem within_014_055_agreement : RowAgreement 1848 1852 := RowAgreement.join within_014_053_agreement within_014_054_agreement
private def within_014_056 : CountSegment 1 10 13120 1852 1854 954 954 := CountSegment.join atomic_1853 atomic_1854
private theorem within_014_056_agreement : RowAgreement 1852 1854 := RowAgreement.join atomic_1853_agreement atomic_1854_agreement
private def within_014_057 : CountSegment 1 10 13120 1854 1856 954 955 := CountSegment.join atomic_1855 atomic_1856
private theorem within_014_057_agreement : RowAgreement 1854 1856 := RowAgreement.join atomic_1855_agreement atomic_1856_agreement
private def within_014_058 : CountSegment 1 10 13120 1852 1856 954 955 := CountSegment.join within_014_056 within_014_057
private theorem within_014_058_agreement : RowAgreement 1852 1856 := RowAgreement.join within_014_056_agreement within_014_057_agreement
private def within_014_059 : CountSegment 1 10 13120 1848 1856 954 955 := CountSegment.join within_014_055 within_014_058
private theorem within_014_059_agreement : RowAgreement 1848 1856 := RowAgreement.join within_014_055_agreement within_014_058_agreement
private def within_014_060 : CountSegment 1 10 13120 1840 1856 951 955 := CountSegment.join within_014_052 within_014_059
private theorem within_014_060_agreement : RowAgreement 1840 1856 := RowAgreement.join within_014_052_agreement within_014_059_agreement
private def within_014_061 : CountSegment 1 10 13120 1824 1856 946 955 := CountSegment.join within_014_045 within_014_060
private theorem within_014_061_agreement : RowAgreement 1824 1856 := RowAgreement.join within_014_045_agreement within_014_060_agreement
private def within_014_062 : CountSegment 1 10 13120 1792 1856 933 955 := CountSegment.join within_014_030 within_014_061
private theorem within_014_062_agreement : RowAgreement 1792 1856 := RowAgreement.join within_014_030_agreement within_014_061_agreement
private def within_014_063 : CountSegment 1 10 13120 1856 1858 955 957 := CountSegment.join atomic_1857 atomic_1858
private theorem within_014_063_agreement : RowAgreement 1856 1858 := RowAgreement.join atomic_1857_agreement atomic_1858_agreement
private def within_014_064 : CountSegment 1 10 13120 1858 1860 957 958 := CountSegment.join atomic_1859 atomic_1860
private theorem within_014_064_agreement : RowAgreement 1858 1860 := RowAgreement.join atomic_1859_agreement atomic_1860_agreement
private def within_014_065 : CountSegment 1 10 13120 1856 1860 955 958 := CountSegment.join within_014_063 within_014_064
private theorem within_014_065_agreement : RowAgreement 1856 1860 := RowAgreement.join within_014_063_agreement within_014_064_agreement
private def within_014_066 : CountSegment 1 10 13120 1860 1862 958 958 := CountSegment.join atomic_1861 atomic_1862
private theorem within_014_066_agreement : RowAgreement 1860 1862 := RowAgreement.join atomic_1861_agreement atomic_1862_agreement
private def within_014_067 : CountSegment 1 10 13120 1862 1864 958 958 := CountSegment.join atomic_1863 atomic_1864
private theorem within_014_067_agreement : RowAgreement 1862 1864 := RowAgreement.join atomic_1863_agreement atomic_1864_agreement
private def within_014_068 : CountSegment 1 10 13120 1860 1864 958 958 := CountSegment.join within_014_066 within_014_067
private theorem within_014_068_agreement : RowAgreement 1860 1864 := RowAgreement.join within_014_066_agreement within_014_067_agreement
private def within_014_069 : CountSegment 1 10 13120 1856 1864 955 958 := CountSegment.join within_014_065 within_014_068
private theorem within_014_069_agreement : RowAgreement 1856 1864 := RowAgreement.join within_014_065_agreement within_014_068_agreement
private def within_014_070 : CountSegment 1 10 13120 1864 1866 958 960 := CountSegment.join atomic_1865 atomic_1866
private theorem within_014_070_agreement : RowAgreement 1864 1866 := RowAgreement.join atomic_1865_agreement atomic_1866_agreement
private def within_014_071 : CountSegment 1 10 13120 1866 1868 960 961 := CountSegment.join atomic_1867 atomic_1868
private theorem within_014_071_agreement : RowAgreement 1866 1868 := RowAgreement.join atomic_1867_agreement atomic_1868_agreement
private def within_014_072 : CountSegment 1 10 13120 1864 1868 958 961 := CountSegment.join within_014_070 within_014_071
private theorem within_014_072_agreement : RowAgreement 1864 1868 := RowAgreement.join within_014_070_agreement within_014_071_agreement
private def within_014_073 : CountSegment 1 10 13120 1868 1870 961 962 := CountSegment.join atomic_1869 atomic_1870
private theorem within_014_073_agreement : RowAgreement 1868 1870 := RowAgreement.join atomic_1869_agreement atomic_1870_agreement
private def within_014_074 : CountSegment 1 10 13120 1870 1872 962 964 := CountSegment.join atomic_1871 atomic_1872
private theorem within_014_074_agreement : RowAgreement 1870 1872 := RowAgreement.join atomic_1871_agreement atomic_1872_agreement
private def within_014_075 : CountSegment 1 10 13120 1868 1872 961 964 := CountSegment.join within_014_073 within_014_074
private theorem within_014_075_agreement : RowAgreement 1868 1872 := RowAgreement.join within_014_073_agreement within_014_074_agreement
private def within_014_076 : CountSegment 1 10 13120 1864 1872 958 964 := CountSegment.join within_014_072 within_014_075
private theorem within_014_076_agreement : RowAgreement 1864 1872 := RowAgreement.join within_014_072_agreement within_014_075_agreement
private def within_014_077 : CountSegment 1 10 13120 1856 1872 955 964 := CountSegment.join within_014_069 within_014_076
private theorem within_014_077_agreement : RowAgreement 1856 1872 := RowAgreement.join within_014_069_agreement within_014_076_agreement
private def within_014_078 : CountSegment 1 10 13120 1872 1874 964 966 := CountSegment.join atomic_1873 atomic_1874
private theorem within_014_078_agreement : RowAgreement 1872 1874 := RowAgreement.join atomic_1873_agreement atomic_1874_agreement
private def within_014_079 : CountSegment 1 10 13120 1874 1876 966 966 := CountSegment.join atomic_1875 atomic_1876
private theorem within_014_079_agreement : RowAgreement 1874 1876 := RowAgreement.join atomic_1875_agreement atomic_1876_agreement
private def within_014_080 : CountSegment 1 10 13120 1872 1876 964 966 := CountSegment.join within_014_078 within_014_079
private theorem within_014_080_agreement : RowAgreement 1872 1876 := RowAgreement.join within_014_078_agreement within_014_079_agreement
private def within_014_081 : CountSegment 1 10 13120 1876 1878 966 968 := CountSegment.join atomic_1877 atomic_1878
private theorem within_014_081_agreement : RowAgreement 1876 1878 := RowAgreement.join atomic_1877_agreement atomic_1878_agreement
private def within_014_082 : CountSegment 1 10 13120 1878 1880 968 968 := CountSegment.join atomic_1879 atomic_1880
private theorem within_014_082_agreement : RowAgreement 1878 1880 := RowAgreement.join atomic_1879_agreement atomic_1880_agreement
private def within_014_083 : CountSegment 1 10 13120 1876 1880 966 968 := CountSegment.join within_014_081 within_014_082
private theorem within_014_083_agreement : RowAgreement 1876 1880 := RowAgreement.join within_014_081_agreement within_014_082_agreement
private def within_014_084 : CountSegment 1 10 13120 1872 1880 964 968 := CountSegment.join within_014_080 within_014_083
private theorem within_014_084_agreement : RowAgreement 1872 1880 := RowAgreement.join within_014_080_agreement within_014_083_agreement
private def within_014_085 : CountSegment 1 10 13120 1880 1882 968 970 := CountSegment.join atomic_1881 atomic_1882
private theorem within_014_085_agreement : RowAgreement 1880 1882 := RowAgreement.join atomic_1881_agreement atomic_1882_agreement
private def within_014_086 : CountSegment 1 10 13120 1882 1884 970 970 := CountSegment.join atomic_1883 atomic_1884
private theorem within_014_086_agreement : RowAgreement 1882 1884 := RowAgreement.join atomic_1883_agreement atomic_1884_agreement
private def within_014_087 : CountSegment 1 10 13120 1880 1884 968 970 := CountSegment.join within_014_085 within_014_086
private theorem within_014_087_agreement : RowAgreement 1880 1884 := RowAgreement.join within_014_085_agreement within_014_086_agreement
private def within_014_088 : CountSegment 1 10 13120 1884 1886 970 971 := CountSegment.join atomic_1885 atomic_1886
private theorem within_014_088_agreement : RowAgreement 1884 1886 := RowAgreement.join atomic_1885_agreement atomic_1886_agreement
private def within_014_089 : CountSegment 1 10 13120 1886 1888 971 973 := CountSegment.join atomic_1887 atomic_1888
private theorem within_014_089_agreement : RowAgreement 1886 1888 := RowAgreement.join atomic_1887_agreement atomic_1888_agreement
private def within_014_090 : CountSegment 1 10 13120 1884 1888 970 973 := CountSegment.join within_014_088 within_014_089
private theorem within_014_090_agreement : RowAgreement 1884 1888 := RowAgreement.join within_014_088_agreement within_014_089_agreement
private def within_014_091 : CountSegment 1 10 13120 1880 1888 968 973 := CountSegment.join within_014_087 within_014_090
private theorem within_014_091_agreement : RowAgreement 1880 1888 := RowAgreement.join within_014_087_agreement within_014_090_agreement
private def within_014_092 : CountSegment 1 10 13120 1872 1888 964 973 := CountSegment.join within_014_084 within_014_091
private theorem within_014_092_agreement : RowAgreement 1872 1888 := RowAgreement.join within_014_084_agreement within_014_091_agreement
private def within_014_093 : CountSegment 1 10 13120 1856 1888 955 973 := CountSegment.join within_014_077 within_014_092
private theorem within_014_093_agreement : RowAgreement 1856 1888 := RowAgreement.join within_014_077_agreement within_014_092_agreement
private def within_014_094 : CountSegment 1 10 13120 1888 1890 973 973 := CountSegment.join atomic_1889 atomic_1890
private theorem within_014_094_agreement : RowAgreement 1888 1890 := RowAgreement.join atomic_1889_agreement atomic_1890_agreement
private def within_014_095 : CountSegment 1 10 13120 1890 1892 973 974 := CountSegment.join atomic_1891 atomic_1892
private theorem within_014_095_agreement : RowAgreement 1890 1892 := RowAgreement.join atomic_1891_agreement atomic_1892_agreement
private def within_014_096 : CountSegment 1 10 13120 1888 1892 973 974 := CountSegment.join within_014_094 within_014_095
private theorem within_014_096_agreement : RowAgreement 1888 1892 := RowAgreement.join within_014_094_agreement within_014_095_agreement
private def within_014_097 : CountSegment 1 10 13120 1892 1894 974 975 := CountSegment.join atomic_1893 atomic_1894
private theorem within_014_097_agreement : RowAgreement 1892 1894 := RowAgreement.join atomic_1893_agreement atomic_1894_agreement
private def within_014_098 : CountSegment 1 10 13120 1894 1896 975 976 := CountSegment.join atomic_1895 atomic_1896
private theorem within_014_098_agreement : RowAgreement 1894 1896 := RowAgreement.join atomic_1895_agreement atomic_1896_agreement
private def within_014_099 : CountSegment 1 10 13120 1892 1896 974 976 := CountSegment.join within_014_097 within_014_098
private theorem within_014_099_agreement : RowAgreement 1892 1896 := RowAgreement.join within_014_097_agreement within_014_098_agreement
private def within_014_100 : CountSegment 1 10 13120 1888 1896 973 976 := CountSegment.join within_014_096 within_014_099
private theorem within_014_100_agreement : RowAgreement 1888 1896 := RowAgreement.join within_014_096_agreement within_014_099_agreement
private def within_014_101 : CountSegment 1 10 13120 1896 1898 976 978 := CountSegment.join atomic_1897 atomic_1898
private theorem within_014_101_agreement : RowAgreement 1896 1898 := RowAgreement.join atomic_1897_agreement atomic_1898_agreement
private def within_014_102 : CountSegment 1 10 13120 1898 1900 978 979 := CountSegment.join atomic_1899 atomic_1900
private theorem within_014_102_agreement : RowAgreement 1898 1900 := RowAgreement.join atomic_1899_agreement atomic_1900_agreement
private def within_014_103 : CountSegment 1 10 13120 1896 1900 976 979 := CountSegment.join within_014_101 within_014_102
private theorem within_014_103_agreement : RowAgreement 1896 1900 := RowAgreement.join within_014_101_agreement within_014_102_agreement
private def within_014_104 : CountSegment 1 10 13120 1900 1902 979 980 := CountSegment.join atomic_1901 atomic_1902
private theorem within_014_104_agreement : RowAgreement 1900 1902 := RowAgreement.join atomic_1901_agreement atomic_1902_agreement
private def within_014_105 : CountSegment 1 10 13120 1902 1904 980 981 := CountSegment.join atomic_1903 atomic_1904
private theorem within_014_105_agreement : RowAgreement 1902 1904 := RowAgreement.join atomic_1903_agreement atomic_1904_agreement
private def within_014_106 : CountSegment 1 10 13120 1900 1904 979 981 := CountSegment.join within_014_104 within_014_105
private theorem within_014_106_agreement : RowAgreement 1900 1904 := RowAgreement.join within_014_104_agreement within_014_105_agreement
private def within_014_107 : CountSegment 1 10 13120 1896 1904 976 981 := CountSegment.join within_014_103 within_014_106
private theorem within_014_107_agreement : RowAgreement 1896 1904 := RowAgreement.join within_014_103_agreement within_014_106_agreement
private def within_014_108 : CountSegment 1 10 13120 1888 1904 973 981 := CountSegment.join within_014_100 within_014_107
private theorem within_014_108_agreement : RowAgreement 1888 1904 := RowAgreement.join within_014_100_agreement within_014_107_agreement
private def within_014_109 : CountSegment 1 10 13120 1904 1906 981 981 := CountSegment.join atomic_1905 atomic_1906
private theorem within_014_109_agreement : RowAgreement 1904 1906 := RowAgreement.join atomic_1905_agreement atomic_1906_agreement
private def within_014_110 : CountSegment 1 10 13120 1906 1908 981 981 := CountSegment.join atomic_1907 atomic_1908
private theorem within_014_110_agreement : RowAgreement 1906 1908 := RowAgreement.join atomic_1907_agreement atomic_1908_agreement
private def within_014_111 : CountSegment 1 10 13120 1904 1908 981 981 := CountSegment.join within_014_109 within_014_110
private theorem within_014_111_agreement : RowAgreement 1904 1908 := RowAgreement.join within_014_109_agreement within_014_110_agreement
private def within_014_112 : CountSegment 1 10 13120 1908 1910 981 981 := CountSegment.join atomic_1909 atomic_1910
private theorem within_014_112_agreement : RowAgreement 1908 1910 := RowAgreement.join atomic_1909_agreement atomic_1910_agreement
private def within_014_113 : CountSegment 1 10 13120 1910 1912 981 981 := CountSegment.join atomic_1911 atomic_1912
private theorem within_014_113_agreement : RowAgreement 1910 1912 := RowAgreement.join atomic_1911_agreement atomic_1912_agreement
private def within_014_114 : CountSegment 1 10 13120 1908 1912 981 981 := CountSegment.join within_014_112 within_014_113
private theorem within_014_114_agreement : RowAgreement 1908 1912 := RowAgreement.join within_014_112_agreement within_014_113_agreement
private def within_014_115 : CountSegment 1 10 13120 1904 1912 981 981 := CountSegment.join within_014_111 within_014_114
private theorem within_014_115_agreement : RowAgreement 1904 1912 := RowAgreement.join within_014_111_agreement within_014_114_agreement
private def within_014_116 : CountSegment 1 10 13120 1912 1914 981 983 := CountSegment.join atomic_1913 atomic_1914
private theorem within_014_116_agreement : RowAgreement 1912 1914 := RowAgreement.join atomic_1913_agreement atomic_1914_agreement
private def within_014_117 : CountSegment 1 10 13120 1914 1916 983 983 := CountSegment.join atomic_1915 atomic_1916
private theorem within_014_117_agreement : RowAgreement 1914 1916 := RowAgreement.join atomic_1915_agreement atomic_1916_agreement
private def within_014_118 : CountSegment 1 10 13120 1912 1916 981 983 := CountSegment.join within_014_116 within_014_117
private theorem within_014_118_agreement : RowAgreement 1912 1916 := RowAgreement.join within_014_116_agreement within_014_117_agreement
private def within_014_119 : CountSegment 1 10 13120 1916 1918 983 985 := CountSegment.join atomic_1917 atomic_1918
private theorem within_014_119_agreement : RowAgreement 1916 1918 := RowAgreement.join atomic_1917_agreement atomic_1918_agreement
private def within_014_120 : CountSegment 1 10 13120 1918 1920 985 986 := CountSegment.join atomic_1919 atomic_1920
private theorem within_014_120_agreement : RowAgreement 1918 1920 := RowAgreement.join atomic_1919_agreement atomic_1920_agreement
private def within_014_121 : CountSegment 1 10 13120 1916 1920 983 986 := CountSegment.join within_014_119 within_014_120
private theorem within_014_121_agreement : RowAgreement 1916 1920 := RowAgreement.join within_014_119_agreement within_014_120_agreement
private def within_014_122 : CountSegment 1 10 13120 1912 1920 981 986 := CountSegment.join within_014_118 within_014_121
private theorem within_014_122_agreement : RowAgreement 1912 1920 := RowAgreement.join within_014_118_agreement within_014_121_agreement
private def within_014_123 : CountSegment 1 10 13120 1904 1920 981 986 := CountSegment.join within_014_115 within_014_122
private theorem within_014_123_agreement : RowAgreement 1904 1920 := RowAgreement.join within_014_115_agreement within_014_122_agreement
private def within_014_124 : CountSegment 1 10 13120 1888 1920 973 986 := CountSegment.join within_014_108 within_014_123
private theorem within_014_124_agreement : RowAgreement 1888 1920 := RowAgreement.join within_014_108_agreement within_014_123_agreement
private def within_014_125 : CountSegment 1 10 13120 1856 1920 955 986 := CountSegment.join within_014_093 within_014_124
private theorem within_014_125_agreement : RowAgreement 1856 1920 := RowAgreement.join within_014_093_agreement within_014_124_agreement
private def within_014_126 : CountSegment 1 10 13120 1792 1920 933 986 := CountSegment.join within_014_062 within_014_125
private theorem within_014_126_agreement : RowAgreement 1792 1920 := RowAgreement.join within_014_062_agreement within_014_125_agreement
def segment_014 : CountSegment 1 10 13120 1792 1920 933 986 := within_014_126
theorem agreement_014 : RowAgreement 1792 1920 := within_014_126_agreement
end ProjectEulerSolutions.P58Numerical
