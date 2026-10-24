import ProjectEulerSolutions.Lib.SpiralPrimeSegments
import ProjectEulerSolutions.Lib.SpiralPrimeCertificates.All
import ProjectEulerSolutions.Lib.MillerRabin
import ProjectEulerSolutions.Lib.SpiralMillerAgreement
/-! Finite spiral row and counter certificates. Python execution is checked separately. -/
namespace ProjectEulerSolutions.P58Numerical
open SpiralPrimes
set_option Elab.async false
set_option maxRecDepth 4096
def row_12161 : RowCertificate 12161 where
  first := false
  first_correct := by
    have composite : ¬ Nat.Prime (corner 12161 1) := by change ¬ Nat.Prime 591584007; exact PrattCertificates.composite_of_product (factor := 3) (cofactor := 197194669) rfl (by decide) (by decide)
    exact decide_eq_false composite
  second := false
  second_correct := by
    have composite : ¬ Nat.Prime (corner 12161 2) := by change ¬ Nat.Prime 591559685; exact PrattCertificates.composite_of_product (factor := 5) (cofactor := 118311937) rfl (by decide) (by decide)
    exact decide_eq_false composite
  third := true
  third_correct := by
    have prime : Nat.Prime (corner 12161 3) := by change Nat.Prime 591535363; exact ProjectEulerSolutions.P58PrimeNodes.p_591535363.prime
    exact decide_eq_true prime
theorem row_count_12161 : row 12161 = 1 := (row_12161.checked).trans rfl
theorem model_count_12161 : modelRow 12161 = 1 := by decide
theorem modelrow_agrees_12161 : modelRow 12161 = row 12161 := model_count_12161.trans row_count_12161.symm
def row_12162 : RowCertificate 12162 where
  first := false
  first_correct := by
    have composite : ¬ Nat.Prime (corner 12162 1) := by change ¬ Nat.Prime 591681301; exact PrattCertificates.composite_of_product (factor := 1579) (cofactor := 374719) rfl (by decide) (by decide)
    exact decide_eq_false composite
  second := true
  second_correct := by
    have prime : Nat.Prime (corner 12162 2) := by change Nat.Prime 591656977; exact ProjectEulerSolutions.P58PrimeNodes.p_591656977.prime
    exact decide_eq_true prime
  third := false
  third_correct := by
    have composite : ¬ Nat.Prime (corner 12162 3) := by change ¬ Nat.Prime 591632653; exact PrattCertificates.composite_of_product (factor := 8623) (cofactor := 68611) rfl (by decide) (by decide)
    exact decide_eq_false composite
theorem row_count_12162 : row 12162 = 1 := (row_12162.checked).trans rfl
theorem model_count_12162 : modelRow 12162 = 1 := by decide
theorem modelrow_agrees_12162 : modelRow 12162 = row 12162 := model_count_12162.trans row_count_12162.symm
def row_12163 : RowCertificate 12163 where
  first := false
  first_correct := by
    have composite : ¬ Nat.Prime (corner 12163 1) := by change ¬ Nat.Prime 591778603; exact PrattCertificates.composite_of_product (factor := 13) (cofactor := 45521431) rfl (by decide) (by decide)
    exact decide_eq_false composite
  second := true
  second_correct := by
    have prime : Nat.Prime (corner 12163 2) := by change Nat.Prime 591754277; exact ProjectEulerSolutions.P58PrimeNodes.p_591754277.prime
    exact decide_eq_true prime
  third := false
  third_correct := by
    have composite : ¬ Nat.Prime (corner 12163 3) := by change ¬ Nat.Prime 591729951; exact PrattCertificates.composite_of_product (factor := 3) (cofactor := 197243317) rfl (by decide) (by decide)
    exact decide_eq_false composite
theorem row_count_12163 : row 12163 = 1 := (row_12163.checked).trans rfl
theorem model_count_12163 : modelRow 12163 = 1 := by decide
theorem modelrow_agrees_12163 : modelRow 12163 = row 12163 := model_count_12163.trans row_count_12163.symm
def row_12164 : RowCertificate 12164 where
  first := false
  first_correct := by
    have composite : ¬ Nat.Prime (corner 12164 1) := by change ¬ Nat.Prime 591875913; exact PrattCertificates.composite_of_product (factor := 3) (cofactor := 197291971) rfl (by decide) (by decide)
    exact decide_eq_false composite
  second := false
  second_correct := by
    have composite : ¬ Nat.Prime (corner 12164 2) := by change ¬ Nat.Prime 591851585; exact PrattCertificates.composite_of_product (factor := 5) (cofactor := 118370317) rfl (by decide) (by decide)
    exact decide_eq_false composite
  third := false
  third_correct := by
    have composite : ¬ Nat.Prime (corner 12164 3) := by change ¬ Nat.Prime 591827257; exact PrattCertificates.composite_of_product (factor := 7) (cofactor := 84546751) rfl (by decide) (by decide)
    exact decide_eq_false composite
theorem row_count_12164 : row 12164 = 0 := (row_12164.checked).trans rfl
theorem model_count_12164 : modelRow 12164 = 0 := by decide
theorem modelrow_agrees_12164 : modelRow 12164 = row 12164 := model_count_12164.trans row_count_12164.symm
def row_12165 : RowCertificate 12165 where
  first := true
  first_correct := by
    have prime : Nat.Prime (corner 12165 1) := by change Nat.Prime 591973231; exact ProjectEulerSolutions.P58PrimeNodes.p_591973231.prime
    exact decide_eq_true prime
  second := false
  second_correct := by
    have composite : ¬ Nat.Prime (corner 12165 2) := by change ¬ Nat.Prime 591948901; exact PrattCertificates.composite_of_product (factor := 457) (cofactor := 1295293) rfl (by decide) (by decide)
    exact decide_eq_false composite
  third := false
  third_correct := by
    have composite : ¬ Nat.Prime (corner 12165 3) := by change ¬ Nat.Prime 591924571; exact PrattCertificates.composite_of_product (factor := 7) (cofactor := 84560653) rfl (by decide) (by decide)
    exact decide_eq_false composite
theorem row_count_12165 : row 12165 = 1 := (row_12165.checked).trans rfl
theorem model_count_12165 : modelRow 12165 = 1 := by decide
theorem modelrow_agrees_12165 : modelRow 12165 = row 12165 := model_count_12165.trans row_count_12165.symm
def row_12166 : RowCertificate 12166 where
  first := false
  first_correct := by
    have composite : ¬ Nat.Prime (corner 12166 1) := by change ¬ Nat.Prime 592070557; exact PrattCertificates.composite_of_product (factor := 13) (cofactor := 45543889) rfl (by decide) (by decide)
    exact decide_eq_false composite
  second := false
  second_correct := by
    have composite : ¬ Nat.Prime (corner 12166 2) := by change ¬ Nat.Prime 592046225; exact PrattCertificates.composite_of_product (factor := 5) (cofactor := 118409245) rfl (by decide) (by decide)
    exact decide_eq_false composite
  third := false
  third_correct := by
    have composite : ¬ Nat.Prime (corner 12166 3) := by change ¬ Nat.Prime 592021893; exact PrattCertificates.composite_of_product (factor := 3) (cofactor := 197340631) rfl (by decide) (by decide)
    exact decide_eq_false composite
theorem row_count_12166 : row 12166 = 0 := (row_12166.checked).trans rfl
theorem model_count_12166 : modelRow 12166 = 0 := by decide
theorem modelrow_agrees_12166 : modelRow 12166 = row 12166 := model_count_12166.trans row_count_12166.symm
def row_12167 : RowCertificate 12167 where
  first := false
  first_correct := by
    have composite : ¬ Nat.Prime (corner 12167 1) := by change ¬ Nat.Prime 592167891; exact PrattCertificates.composite_of_product (factor := 3) (cofactor := 197389297) rfl (by decide) (by decide)
    exact decide_eq_false composite
  second := false
  second_correct := by
    have composite : ¬ Nat.Prime (corner 12167 2) := by change ¬ Nat.Prime 592143557; exact PrattCertificates.composite_of_product (factor := 4373) (cofactor := 135409) rfl (by decide) (by decide)
    exact decide_eq_false composite
  third := false
  third_correct := by
    have composite : ¬ Nat.Prime (corner 12167 3) := by change ¬ Nat.Prime 592119223; exact PrattCertificates.composite_of_product (factor := 373) (cofactor := 1587451) rfl (by decide) (by decide)
    exact decide_eq_false composite
theorem row_count_12167 : row 12167 = 0 := (row_12167.checked).trans rfl
theorem model_count_12167 : modelRow 12167 = 0 := by decide
theorem modelrow_agrees_12167 : modelRow 12167 = row 12167 := model_count_12167.trans row_count_12167.symm
def row_12168 : RowCertificate 12168 where
  first := false
  first_correct := by
    have composite : ¬ Nat.Prime (corner 12168 1) := by change ¬ Nat.Prime 592265233; exact PrattCertificates.composite_of_product (factor := 7) (cofactor := 84609319) rfl (by decide) (by decide)
    exact decide_eq_false composite
  second := false
  second_correct := by
    have composite : ¬ Nat.Prime (corner 12168 2) := by change ¬ Nat.Prime 592240897; exact PrattCertificates.composite_of_product (factor := 73) (cofactor := 8112889) rfl (by decide) (by decide)
    exact decide_eq_false composite
  third := false
  third_correct := by
    have composite : ¬ Nat.Prime (corner 12168 3) := by change ¬ Nat.Prime 592216561; exact PrattCertificates.composite_of_product (factor := 37) (cofactor := 16005853) rfl (by decide) (by decide)
    exact decide_eq_false composite
theorem row_count_12168 : row 12168 = 0 := (row_12168.checked).trans rfl
theorem model_count_12168 : modelRow 12168 = 0 := by decide
theorem modelrow_agrees_12168 : modelRow 12168 = row 12168 := model_count_12168.trans row_count_12168.symm
def row_12169 : RowCertificate 12169 where
  first := true
  first_correct := by
    have prime : Nat.Prime (corner 12169 1) := by change Nat.Prime 592362583; exact ProjectEulerSolutions.P58PrimeNodes.p_592362583.prime
    exact decide_eq_true prime
  second := false
  second_correct := by
    have composite : ¬ Nat.Prime (corner 12169 2) := by change ¬ Nat.Prime 592338245; exact PrattCertificates.composite_of_product (factor := 5) (cofactor := 118467649) rfl (by decide) (by decide)
    exact decide_eq_false composite
  third := false
  third_correct := by
    have composite : ¬ Nat.Prime (corner 12169 3) := by change ¬ Nat.Prime 592313907; exact PrattCertificates.composite_of_product (factor := 3) (cofactor := 197437969) rfl (by decide) (by decide)
    exact decide_eq_false composite
theorem row_count_12169 : row 12169 = 1 := (row_12169.checked).trans rfl
theorem model_count_12169 : modelRow 12169 = 1 := by decide
theorem modelrow_agrees_12169 : modelRow 12169 = row 12169 := model_count_12169.trans row_count_12169.symm
def row_12170 : RowCertificate 12170 where
  first := false
  first_correct := by
    have composite : ¬ Nat.Prime (corner 12170 1) := by change ¬ Nat.Prime 592459941; exact PrattCertificates.composite_of_product (factor := 3) (cofactor := 197486647) rfl (by decide) (by decide)
    exact decide_eq_false composite
  second := false
  second_correct := by
    have composite : ¬ Nat.Prime (corner 12170 2) := by change ¬ Nat.Prime 592435601; exact PrattCertificates.composite_of_product (factor := 17) (cofactor := 34849153) rfl (by decide) (by decide)
    exact decide_eq_false composite
  third := false
  third_correct := by
    have composite : ¬ Nat.Prime (corner 12170 3) := by change ¬ Nat.Prime 592411261; exact PrattCertificates.composite_of_product (factor := 13) (cofactor := 45570097) rfl (by decide) (by decide)
    exact decide_eq_false composite
theorem row_count_12170 : row 12170 = 0 := (row_12170.checked).trans rfl
theorem model_count_12170 : modelRow 12170 = 0 := by decide
theorem modelrow_agrees_12170 : modelRow 12170 = row 12170 := model_count_12170.trans row_count_12170.symm
def row_12171 : RowCertificate 12171 where
  first := true
  first_correct := by
    have prime : Nat.Prime (corner 12171 1) := by change Nat.Prime 592557307; exact ProjectEulerSolutions.P58PrimeNodes.p_592557307.prime
    exact decide_eq_true prime
  second := false
  second_correct := by
    have composite : ¬ Nat.Prime (corner 12171 2) := by change ¬ Nat.Prime 592532965; exact PrattCertificates.composite_of_product (factor := 5) (cofactor := 118506593) rfl (by decide) (by decide)
    exact decide_eq_false composite
  third := false
  third_correct := by
    have composite : ¬ Nat.Prime (corner 12171 3) := by change ¬ Nat.Prime 592508623; exact PrattCertificates.composite_of_product (factor := 7) (cofactor := 84644089) rfl (by decide) (by decide)
    exact decide_eq_false composite
theorem row_count_12171 : row 12171 = 1 := (row_12171.checked).trans rfl
theorem model_count_12171 : modelRow 12171 = 1 := by decide
theorem modelrow_agrees_12171 : modelRow 12171 = row 12171 := model_count_12171.trans row_count_12171.symm
def row_12172 : RowCertificate 12172 where
  first := false
  first_correct := by
    have composite : ¬ Nat.Prime (corner 12172 1) := by change ¬ Nat.Prime 592654681; exact PrattCertificates.composite_of_product (factor := 43) (cofactor := 13782667) rfl (by decide) (by decide)
    exact decide_eq_false composite
  second := false
  second_correct := by
    have composite : ¬ Nat.Prime (corner 12172 2) := by change ¬ Nat.Prime 592630337; exact PrattCertificates.composite_of_product (factor := 13) (cofactor := 45586949) rfl (by decide) (by decide)
    exact decide_eq_false composite
  third := false
  third_correct := by
    have composite : ¬ Nat.Prime (corner 12172 3) := by change ¬ Nat.Prime 592605993; exact PrattCertificates.composite_of_product (factor := 3) (cofactor := 197535331) rfl (by decide) (by decide)
    exact decide_eq_false composite
theorem row_count_12172 : row 12172 = 0 := (row_12172.checked).trans rfl
theorem model_count_12172 : modelRow 12172 = 0 := by decide
theorem modelrow_agrees_12172 : modelRow 12172 = row 12172 := model_count_12172.trans row_count_12172.symm
def row_12173 : RowCertificate 12173 where
  first := false
  first_correct := by
    have composite : ¬ Nat.Prime (corner 12173 1) := by change ¬ Nat.Prime 592752063; exact PrattCertificates.composite_of_product (factor := 3) (cofactor := 197584021) rfl (by decide) (by decide)
    exact decide_eq_false composite
  second := false
  second_correct := by
    have composite : ¬ Nat.Prime (corner 12173 2) := by change ¬ Nat.Prime 592727717; exact PrattCertificates.composite_of_product (factor := 677) (cofactor := 875521) rfl (by decide) (by decide)
    exact decide_eq_false composite
  third := false
  third_correct := by
    have composite : ¬ Nat.Prime (corner 12173 3) := by change ¬ Nat.Prime 592703371; exact PrattCertificates.composite_of_product (factor := 13) (cofactor := 45592567) rfl (by decide) (by decide)
    exact decide_eq_false composite
theorem row_count_12173 : row 12173 = 0 := (row_12173.checked).trans rfl
theorem model_count_12173 : modelRow 12173 = 0 := by decide
theorem modelrow_agrees_12173 : modelRow 12173 = row 12173 := model_count_12173.trans row_count_12173.symm
def row_12174 : RowCertificate 12174 where
  first := false
  first_correct := by
    have composite : ¬ Nat.Prime (corner 12174 1) := by change ¬ Nat.Prime 592849453; exact PrattCertificates.composite_of_product (factor := 7) (cofactor := 84692779) rfl (by decide) (by decide)
    exact decide_eq_false composite
  second := false
  second_correct := by
    have composite : ¬ Nat.Prime (corner 12174 2) := by change ¬ Nat.Prime 592825105; exact PrattCertificates.composite_of_product (factor := 5) (cofactor := 118565021) rfl (by decide) (by decide)
    exact decide_eq_false composite
  third := false
  third_correct := by
    have composite : ¬ Nat.Prime (corner 12174 3) := by change ¬ Nat.Prime 592800757; exact PrattCertificates.composite_of_product (factor := 157) (cofactor := 3775801) rfl (by decide) (by decide)
    exact decide_eq_false composite
theorem row_count_12174 : row 12174 = 0 := (row_12174.checked).trans rfl
theorem model_count_12174 : modelRow 12174 = 0 := by decide
theorem modelrow_agrees_12174 : modelRow 12174 = row 12174 := model_count_12174.trans row_count_12174.symm
def row_12175 : RowCertificate 12175 where
  first := false
  first_correct := by
    have composite : ¬ Nat.Prime (corner 12175 1) := by change ¬ Nat.Prime 592946851; exact PrattCertificates.composite_of_product (factor := 7) (cofactor := 84706693) rfl (by decide) (by decide)
    exact decide_eq_false composite
  second := false
  second_correct := by
    have composite : ¬ Nat.Prime (corner 12175 2) := by change ¬ Nat.Prime 592922501; exact PrattCertificates.composite_of_product (factor := 53) (cofactor := 11187217) rfl (by decide) (by decide)
    exact decide_eq_false composite
  third := false
  third_correct := by
    have composite : ¬ Nat.Prime (corner 12175 3) := by change ¬ Nat.Prime 592898151; exact PrattCertificates.composite_of_product (factor := 3) (cofactor := 197632717) rfl (by decide) (by decide)
    exact decide_eq_false composite
theorem row_count_12175 : row 12175 = 0 := (row_12175.checked).trans rfl
theorem model_count_12175 : modelRow 12175 = 0 := by decide
theorem modelrow_agrees_12175 : modelRow 12175 = row 12175 := model_count_12175.trans row_count_12175.symm
def row_12176 : RowCertificate 12176 where
  first := false
  first_correct := by
    have composite : ¬ Nat.Prime (corner 12176 1) := by change ¬ Nat.Prime 593044257; exact PrattCertificates.composite_of_product (factor := 3) (cofactor := 197681419) rfl (by decide) (by decide)
    exact decide_eq_false composite
  second := false
  second_correct := by
    have composite : ¬ Nat.Prime (corner 12176 2) := by change ¬ Nat.Prime 593019905; exact PrattCertificates.composite_of_product (factor := 5) (cofactor := 118603981) rfl (by decide) (by decide)
    exact decide_eq_false composite
  third := true
  third_correct := by
    have prime : Nat.Prime (corner 12176 3) := by change Nat.Prime 592995553; exact ProjectEulerSolutions.P58PrimeNodes.p_592995553.prime
    exact decide_eq_true prime
theorem row_count_12176 : row 12176 = 1 := (row_12176.checked).trans rfl
theorem model_count_12176 : modelRow 12176 = 1 := by decide
theorem modelrow_agrees_12176 : modelRow 12176 = row 12176 := model_count_12176.trans row_count_12176.symm
def row_12177 : RowCertificate 12177 where
  first := false
  first_correct := by
    have composite : ¬ Nat.Prime (corner 12177 1) := by change ¬ Nat.Prime 593141671; exact PrattCertificates.composite_of_product (factor := 103) (cofactor := 5758657) rfl (by decide) (by decide)
    exact decide_eq_false composite
  second := false
  second_correct := by
    have composite : ¬ Nat.Prime (corner 12177 2) := by change ¬ Nat.Prime 593117317; exact PrattCertificates.composite_of_product (factor := 13) (cofactor := 45624409) rfl (by decide) (by decide)
    exact decide_eq_false composite
  third := false
  third_correct := by
    have composite : ¬ Nat.Prime (corner 12177 3) := by change ¬ Nat.Prime 593092963; exact PrattCertificates.composite_of_product (factor := 1051) (cofactor := 564313) rfl (by decide) (by decide)
    exact decide_eq_false composite
theorem row_count_12177 : row 12177 = 0 := (row_12177.checked).trans rfl
theorem model_count_12177 : modelRow 12177 = 0 := by decide
theorem modelrow_agrees_12177 : modelRow 12177 = row 12177 := model_count_12177.trans row_count_12177.symm
def row_12178 : RowCertificate 12178 where
  first := false
  first_correct := by
    have composite : ¬ Nat.Prime (corner 12178 1) := by change ¬ Nat.Prime 593239093; exact PrattCertificates.composite_of_product (factor := 37) (cofactor := 16033489) rfl (by decide) (by decide)
    exact decide_eq_false composite
  second := true
  second_correct := by
    have prime : Nat.Prime (corner 12178 2) := by change Nat.Prime 593214737; exact ProjectEulerSolutions.P58PrimeNodes.p_593214737.prime
    exact decide_eq_true prime
  third := false
  third_correct := by
    have composite : ¬ Nat.Prime (corner 12178 3) := by change ¬ Nat.Prime 593190381; exact PrattCertificates.composite_of_product (factor := 3) (cofactor := 197730127) rfl (by decide) (by decide)
    exact decide_eq_false composite
theorem row_count_12178 : row 12178 = 1 := (row_12178.checked).trans rfl
theorem model_count_12178 : modelRow 12178 = 1 := by decide
theorem modelrow_agrees_12178 : modelRow 12178 = row 12178 := model_count_12178.trans row_count_12178.symm
def row_12179 : RowCertificate 12179 where
  first := false
  first_correct := by
    have composite : ¬ Nat.Prime (corner 12179 1) := by change ¬ Nat.Prime 593336523; exact PrattCertificates.composite_of_product (factor := 3) (cofactor := 197778841) rfl (by decide) (by decide)
    exact decide_eq_false composite
  second := false
  second_correct := by
    have composite : ¬ Nat.Prime (corner 12179 2) := by change ¬ Nat.Prime 593312165; exact PrattCertificates.composite_of_product (factor := 5) (cofactor := 118662433) rfl (by decide) (by decide)
    exact decide_eq_false composite
  third := false
  third_correct := by
    have composite : ¬ Nat.Prime (corner 12179 3) := by change ¬ Nat.Prime 593287807; exact PrattCertificates.composite_of_product (factor := 7) (cofactor := 84755401) rfl (by decide) (by decide)
    exact decide_eq_false composite
theorem row_count_12179 : row 12179 = 0 := (row_12179.checked).trans rfl
theorem model_count_12179 : modelRow 12179 = 0 := by decide
theorem modelrow_agrees_12179 : modelRow 12179 = row 12179 := model_count_12179.trans row_count_12179.symm
def row_12180 : RowCertificate 12180 where
  first := false
  first_correct := by
    have composite : ¬ Nat.Prime (corner 12180 1) := by change ¬ Nat.Prime 593433961; exact PrattCertificates.composite_of_product (factor := 31) (cofactor := 19143031) rfl (by decide) (by decide)
    exact decide_eq_false composite
  second := false
  second_correct := by
    have composite : ¬ Nat.Prime (corner 12180 2) := by change ¬ Nat.Prime 593409601; exact PrattCertificates.composite_of_product (factor := 1117) (cofactor := 531253) rfl (by decide) (by decide)
    exact decide_eq_false composite
  third := false
  third_correct := by
    have composite : ¬ Nat.Prime (corner 12180 3) := by change ¬ Nat.Prime 593385241; exact PrattCertificates.composite_of_product (factor := 1753) (cofactor := 338497) rfl (by decide) (by decide)
    exact decide_eq_false composite
theorem row_count_12180 : row 12180 = 0 := (row_12180.checked).trans rfl
theorem model_count_12180 : modelRow 12180 = 0 := by decide
theorem modelrow_agrees_12180 : modelRow 12180 = row 12180 := model_count_12180.trans row_count_12180.symm
def row_12181 : RowCertificate 12181 where
  first := false
  first_correct := by
    have composite : ¬ Nat.Prime (corner 12181 1) := by change ¬ Nat.Prime 593531407; exact PrattCertificates.composite_of_product (factor := 7) (cofactor := 84790201) rfl (by decide) (by decide)
    exact decide_eq_false composite
  second := false
  second_correct := by
    have composite : ¬ Nat.Prime (corner 12181 2) := by change ¬ Nat.Prime 593507045; exact PrattCertificates.composite_of_product (factor := 5) (cofactor := 118701409) rfl (by decide) (by decide)
    exact decide_eq_false composite
  third := false
  third_correct := by
    have composite : ¬ Nat.Prime (corner 12181 3) := by change ¬ Nat.Prime 593482683; exact PrattCertificates.composite_of_product (factor := 3) (cofactor := 197827561) rfl (by decide) (by decide)
    exact decide_eq_false composite
theorem row_count_12181 : row 12181 = 0 := (row_12181.checked).trans rfl
theorem model_count_12181 : modelRow 12181 = 0 := by decide
theorem modelrow_agrees_12181 : modelRow 12181 = row 12181 := model_count_12181.trans row_count_12181.symm
def row_12182 : RowCertificate 12182 where
  first := false
  first_correct := by
    have composite : ¬ Nat.Prime (corner 12182 1) := by change ¬ Nat.Prime 593628861; exact PrattCertificates.composite_of_product (factor := 3) (cofactor := 197876287) rfl (by decide) (by decide)
    exact decide_eq_false composite
  second := true
  second_correct := by
    have prime : Nat.Prime (corner 12182 2) := by change Nat.Prime 593604497; exact ProjectEulerSolutions.P58PrimeNodes.p_593604497.prime
    exact decide_eq_true prime
  third := true
  third_correct := by
    have prime : Nat.Prime (corner 12182 3) := by change Nat.Prime 593580133; exact ProjectEulerSolutions.P58PrimeNodes.p_593580133.prime
    exact decide_eq_true prime
theorem row_count_12182 : row 12182 = 2 := (row_12182.checked).trans rfl
theorem model_count_12182 : modelRow 12182 = 2 := by decide
theorem modelrow_agrees_12182 : modelRow 12182 = row 12182 := model_count_12182.trans row_count_12182.symm
def row_12183 : RowCertificate 12183 where
  first := true
  first_correct := by
    have prime : Nat.Prime (corner 12183 1) := by change Nat.Prime 593726323; exact ProjectEulerSolutions.P58PrimeNodes.p_593726323.prime
    exact decide_eq_true prime
  second := true
  second_correct := by
    have prime : Nat.Prime (corner 12183 2) := by change Nat.Prime 593701957; exact ProjectEulerSolutions.P58PrimeNodes.p_593701957.prime
    exact decide_eq_true prime
  third := false
  third_correct := by
    have composite : ¬ Nat.Prime (corner 12183 3) := by change ¬ Nat.Prime 593677591; exact PrattCertificates.composite_of_product (factor := 13) (cofactor := 45667507) rfl (by decide) (by decide)
    exact decide_eq_false composite
theorem row_count_12183 : row 12183 = 2 := (row_12183.checked).trans rfl
theorem model_count_12183 : modelRow 12183 = 2 := by decide
theorem modelrow_agrees_12183 : modelRow 12183 = row 12183 := model_count_12183.trans row_count_12183.symm
def row_12184 : RowCertificate 12184 where
  first := false
  first_correct := by
    have composite : ¬ Nat.Prime (corner 12184 1) := by change ¬ Nat.Prime 593823793; exact PrattCertificates.composite_of_product (factor := 229) (cofactor := 2593117) rfl (by decide) (by decide)
    exact decide_eq_false composite
  second := false
  second_correct := by
    have composite : ¬ Nat.Prime (corner 12184 2) := by change ¬ Nat.Prime 593799425; exact PrattCertificates.composite_of_product (factor := 5) (cofactor := 118759885) rfl (by decide) (by decide)
    exact decide_eq_false composite
  third := false
  third_correct := by
    have composite : ¬ Nat.Prime (corner 12184 3) := by change ¬ Nat.Prime 593775057; exact PrattCertificates.composite_of_product (factor := 3) (cofactor := 197925019) rfl (by decide) (by decide)
    exact decide_eq_false composite
theorem row_count_12184 : row 12184 = 0 := (row_12184.checked).trans rfl
theorem model_count_12184 : modelRow 12184 = 0 := by decide
theorem modelrow_agrees_12184 : modelRow 12184 = row 12184 := model_count_12184.trans row_count_12184.symm
def row_12185 : RowCertificate 12185 where
  first := false
  first_correct := by
    have composite : ¬ Nat.Prime (corner 12185 1) := by change ¬ Nat.Prime 593921271; exact PrattCertificates.composite_of_product (factor := 3) (cofactor := 197973757) rfl (by decide) (by decide)
    exact decide_eq_false composite
  second := false
  second_correct := by
    have composite : ¬ Nat.Prime (corner 12185 2) := by change ¬ Nat.Prime 593896901; exact PrattCertificates.composite_of_product (factor := 13) (cofactor := 45684377) rfl (by decide) (by decide)
    exact decide_eq_false composite
  third := false
  third_correct := by
    have composite : ¬ Nat.Prime (corner 12185 3) := by change ¬ Nat.Prime 593872531; exact PrattCertificates.composite_of_product (factor := 7) (cofactor := 84838933) rfl (by decide) (by decide)
    exact decide_eq_false composite
theorem row_count_12185 : row 12185 = 0 := (row_12185.checked).trans rfl
theorem model_count_12185 : modelRow 12185 = 0 := by decide
theorem modelrow_agrees_12185 : modelRow 12185 = row 12185 := model_count_12185.trans row_count_12185.symm
def row_12186 : RowCertificate 12186 where
  first := false
  first_correct := by
    have composite : ¬ Nat.Prime (corner 12186 1) := by change ¬ Nat.Prime 594018757; exact PrattCertificates.composite_of_product (factor := 37) (cofactor := 16054561) rfl (by decide) (by decide)
    exact decide_eq_false composite
  second := false
  second_correct := by
    have composite : ¬ Nat.Prime (corner 12186 2) := by change ¬ Nat.Prime 593994385; exact PrattCertificates.composite_of_product (factor := 5) (cofactor := 118798877) rfl (by decide) (by decide)
    exact decide_eq_false composite
  third := false
  third_correct := by
    have composite : ¬ Nat.Prime (corner 12186 3) := by change ¬ Nat.Prime 593970013; exact PrattCertificates.composite_of_product (factor := 7) (cofactor := 84852859) rfl (by decide) (by decide)
    exact decide_eq_false composite
theorem row_count_12186 : row 12186 = 0 := (row_12186.checked).trans rfl
theorem model_count_12186 : modelRow 12186 = 0 := by decide
theorem modelrow_agrees_12186 : modelRow 12186 = row 12186 := model_count_12186.trans row_count_12186.symm
def row_12187 : RowCertificate 12187 where
  first := false
  first_correct := by
    have composite : ¬ Nat.Prime (corner 12187 1) := by change ¬ Nat.Prime 594116251; exact PrattCertificates.composite_of_product (factor := 43) (cofactor := 13816657) rfl (by decide) (by decide)
    exact decide_eq_false composite
  second := false
  second_correct := by
    have composite : ¬ Nat.Prime (corner 12187 2) := by change ¬ Nat.Prime 594091877; exact PrattCertificates.composite_of_product (factor := 17) (cofactor := 34946581) rfl (by decide) (by decide)
    exact decide_eq_false composite
  third := false
  third_correct := by
    have composite : ¬ Nat.Prime (corner 12187 3) := by change ¬ Nat.Prime 594067503; exact PrattCertificates.composite_of_product (factor := 3) (cofactor := 198022501) rfl (by decide) (by decide)
    exact decide_eq_false composite
theorem row_count_12187 : row 12187 = 0 := (row_12187.checked).trans rfl
theorem model_count_12187 : modelRow 12187 = 0 := by decide
theorem modelrow_agrees_12187 : modelRow 12187 = row 12187 := model_count_12187.trans row_count_12187.symm
def row_12188 : RowCertificate 12188 where
  first := false
  first_correct := by
    have composite : ¬ Nat.Prime (corner 12188 1) := by change ¬ Nat.Prime 594213753; exact PrattCertificates.composite_of_product (factor := 3) (cofactor := 198071251) rfl (by decide) (by decide)
    exact decide_eq_false composite
  second := false
  second_correct := by
    have composite : ¬ Nat.Prime (corner 12188 2) := by change ¬ Nat.Prime 594189377; exact PrattCertificates.composite_of_product (factor := 233) (cofactor := 2550169) rfl (by decide) (by decide)
    exact decide_eq_false composite
  third := false
  third_correct := by
    have composite : ¬ Nat.Prime (corner 12188 3) := by change ¬ Nat.Prime 594165001; exact PrattCertificates.composite_of_product (factor := 22531) (cofactor := 26371) rfl (by decide) (by decide)
    exact decide_eq_false composite
theorem row_count_12188 : row 12188 = 0 := (row_12188.checked).trans rfl
theorem model_count_12188 : modelRow 12188 = 0 := by decide
theorem modelrow_agrees_12188 : modelRow 12188 = row 12188 := model_count_12188.trans row_count_12188.symm
def row_12189 : RowCertificate 12189 where
  first := false
  first_correct := by
    have composite : ¬ Nat.Prime (corner 12189 1) := by change ¬ Nat.Prime 594311263; exact PrattCertificates.composite_of_product (factor := 7) (cofactor := 84901609) rfl (by decide) (by decide)
    exact decide_eq_false composite
  second := false
  second_correct := by
    have composite : ¬ Nat.Prime (corner 12189 2) := by change ¬ Nat.Prime 594286885; exact PrattCertificates.composite_of_product (factor := 5) (cofactor := 118857377) rfl (by decide) (by decide)
    exact decide_eq_false composite
  third := false
  third_correct := by
    have composite : ¬ Nat.Prime (corner 12189 3) := by change ¬ Nat.Prime 594262507; exact PrattCertificates.composite_of_product (factor := 13693) (cofactor := 43399) rfl (by decide) (by decide)
    exact decide_eq_false composite
theorem row_count_12189 : row 12189 = 0 := (row_12189.checked).trans rfl
theorem model_count_12189 : modelRow 12189 = 0 := by decide
theorem modelrow_agrees_12189 : modelRow 12189 = row 12189 := model_count_12189.trans row_count_12189.symm
def row_12190 : RowCertificate 12190 where
  first := true
  first_correct := by
    have prime : Nat.Prime (corner 12190 1) := by change Nat.Prime 594408781; exact ProjectEulerSolutions.P58PrimeNodes.p_594408781.prime
    exact decide_eq_true prime
  second := false
  second_correct := by
    have composite : ¬ Nat.Prime (corner 12190 2) := by change ¬ Nat.Prime 594384401; exact PrattCertificates.composite_of_product (factor := 13) (cofactor := 45721877) rfl (by decide) (by decide)
    exact decide_eq_false composite
  third := false
  third_correct := by
    have composite : ¬ Nat.Prime (corner 12190 3) := by change ¬ Nat.Prime 594360021; exact PrattCertificates.composite_of_product (factor := 3) (cofactor := 198120007) rfl (by decide) (by decide)
    exact decide_eq_false composite
theorem row_count_12190 : row 12190 = 1 := (row_12190.checked).trans rfl
theorem model_count_12190 : modelRow 12190 = 1 := by decide
theorem modelrow_agrees_12190 : modelRow 12190 = row 12190 := model_count_12190.trans row_count_12190.symm
def row_12191 : RowCertificate 12191 where
  first := false
  first_correct := by
    have composite : ¬ Nat.Prime (corner 12191 1) := by change ¬ Nat.Prime 594506307; exact PrattCertificates.composite_of_product (factor := 3) (cofactor := 198168769) rfl (by decide) (by decide)
    exact decide_eq_false composite
  second := false
  second_correct := by
    have composite : ¬ Nat.Prime (corner 12191 2) := by change ¬ Nat.Prime 594481925; exact PrattCertificates.composite_of_product (factor := 5) (cofactor := 118896385) rfl (by decide) (by decide)
    exact decide_eq_false composite
  third := true
  third_correct := by
    have prime : Nat.Prime (corner 12191 3) := by change Nat.Prime 594457543; exact ProjectEulerSolutions.P58PrimeNodes.p_594457543.prime
    exact decide_eq_true prime
theorem row_count_12191 : row 12191 = 1 := (row_12191.checked).trans rfl
theorem model_count_12191 : modelRow 12191 = 1 := by decide
theorem modelrow_agrees_12191 : modelRow 12191 = row 12191 := model_count_12191.trans row_count_12191.symm
def row_12192 : RowCertificate 12192 where
  first := false
  first_correct := by
    have composite : ¬ Nat.Prime (corner 12192 1) := by change ¬ Nat.Prime 594603841; exact PrattCertificates.composite_of_product (factor := 13) (cofactor := 45738757) rfl (by decide) (by decide)
    exact decide_eq_false composite
  second := true
  second_correct := by
    have prime : Nat.Prime (corner 12192 2) := by change Nat.Prime 594579457; exact ProjectEulerSolutions.P58PrimeNodes.p_594579457.prime
    exact decide_eq_true prime
  third := false
  third_correct := by
    have composite : ¬ Nat.Prime (corner 12192 3) := by change ¬ Nat.Prime 594555073; exact PrattCertificates.composite_of_product (factor := 7) (cofactor := 84936439) rfl (by decide) (by decide)
    exact decide_eq_false composite
theorem row_count_12192 : row 12192 = 1 := (row_12192.checked).trans rfl
theorem model_count_12192 : modelRow 12192 = 1 := by decide
theorem modelrow_agrees_12192 : modelRow 12192 = row 12192 := model_count_12192.trans row_count_12192.symm
def row_12193 : RowCertificate 12193 where
  first := false
  first_correct := by
    have composite : ¬ Nat.Prime (corner 12193 1) := by change ¬ Nat.Prime 594701383; exact PrattCertificates.composite_of_product (factor := 61) (cofactor := 9749203) rfl (by decide) (by decide)
    exact decide_eq_false composite
  second := false
  second_correct := by
    have composite : ¬ Nat.Prime (corner 12193 2) := by change ¬ Nat.Prime 594676997; exact PrattCertificates.composite_of_product (factor := 41) (cofactor := 14504317) rfl (by decide) (by decide)
    exact decide_eq_false composite
  third := false
  third_correct := by
    have composite : ¬ Nat.Prime (corner 12193 3) := by change ¬ Nat.Prime 594652611; exact PrattCertificates.composite_of_product (factor := 3) (cofactor := 198217537) rfl (by decide) (by decide)
    exact decide_eq_false composite
theorem row_count_12193 : row 12193 = 0 := (row_12193.checked).trans rfl
theorem model_count_12193 : modelRow 12193 = 0 := by decide
theorem modelrow_agrees_12193 : modelRow 12193 = row 12193 := model_count_12193.trans row_count_12193.symm
def row_12194 : RowCertificate 12194 where
  first := false
  first_correct := by
    have composite : ¬ Nat.Prime (corner 12194 1) := by change ¬ Nat.Prime 594798933; exact PrattCertificates.composite_of_product (factor := 3) (cofactor := 198266311) rfl (by decide) (by decide)
    exact decide_eq_false composite
  second := false
  second_correct := by
    have composite : ¬ Nat.Prime (corner 12194 2) := by change ¬ Nat.Prime 594774545; exact PrattCertificates.composite_of_product (factor := 5) (cofactor := 118954909) rfl (by decide) (by decide)
    exact decide_eq_false composite
  third := false
  third_correct := by
    have composite : ¬ Nat.Prime (corner 12194 3) := by change ¬ Nat.Prime 594750157; exact PrattCertificates.composite_of_product (factor := 43) (cofactor := 13831399) rfl (by decide) (by decide)
    exact decide_eq_false composite
theorem row_count_12194 : row 12194 = 0 := (row_12194.checked).trans rfl
theorem model_count_12194 : modelRow 12194 = 0 := by decide
theorem modelrow_agrees_12194 : modelRow 12194 = row 12194 := model_count_12194.trans row_count_12194.symm
def row_12195 : RowCertificate 12195 where
  first := false
  first_correct := by
    have composite : ¬ Nat.Prime (corner 12195 1) := by change ¬ Nat.Prime 594896491; exact PrattCertificates.composite_of_product (factor := 7) (cofactor := 84985213) rfl (by decide) (by decide)
    exact decide_eq_false composite
  second := false
  second_correct := by
    have composite : ¬ Nat.Prime (corner 12195 2) := by change ¬ Nat.Prime 594872101; exact PrattCertificates.composite_of_product (factor := 557) (cofactor := 1067993) rfl (by decide) (by decide)
    exact decide_eq_false composite
  third := false
  third_correct := by
    have composite : ¬ Nat.Prime (corner 12195 3) := by change ¬ Nat.Prime 594847711; exact PrattCertificates.composite_of_product (factor := 14437) (cofactor := 41203) rfl (by decide) (by decide)
    exact decide_eq_false composite
theorem row_count_12195 : row 12195 = 0 := (row_12195.checked).trans rfl
theorem model_count_12195 : modelRow 12195 = 0 := by decide
theorem modelrow_agrees_12195 : modelRow 12195 = row 12195 := model_count_12195.trans row_count_12195.symm
def row_12196 : RowCertificate 12196 where
  first := false
  first_correct := by
    have composite : ¬ Nat.Prime (corner 12196 1) := by change ¬ Nat.Prime 594994057; exact PrattCertificates.composite_of_product (factor := 7) (cofactor := 84999151) rfl (by decide) (by decide)
    exact decide_eq_false composite
  second := false
  second_correct := by
    have composite : ¬ Nat.Prime (corner 12196 2) := by change ¬ Nat.Prime 594969665; exact PrattCertificates.composite_of_product (factor := 5) (cofactor := 118993933) rfl (by decide) (by decide)
    exact decide_eq_false composite
  third := false
  third_correct := by
    have composite : ¬ Nat.Prime (corner 12196 3) := by change ¬ Nat.Prime 594945273; exact PrattCertificates.composite_of_product (factor := 3) (cofactor := 198315091) rfl (by decide) (by decide)
    exact decide_eq_false composite
theorem row_count_12196 : row 12196 = 0 := (row_12196.checked).trans rfl
theorem model_count_12196 : modelRow 12196 = 0 := by decide
theorem modelrow_agrees_12196 : modelRow 12196 = row 12196 := model_count_12196.trans row_count_12196.symm
def row_12197 : RowCertificate 12197 where
  first := false
  first_correct := by
    have composite : ¬ Nat.Prime (corner 12197 1) := by change ¬ Nat.Prime 595091631; exact PrattCertificates.composite_of_product (factor := 3) (cofactor := 198363877) rfl (by decide) (by decide)
    exact decide_eq_false composite
  second := false
  second_correct := by
    have composite : ¬ Nat.Prime (corner 12197 2) := by change ¬ Nat.Prime 595067237; exact PrattCertificates.composite_of_product (factor := 281) (cofactor := 2117677) rfl (by decide) (by decide)
    exact decide_eq_false composite
  third := false
  third_correct := by
    have composite : ¬ Nat.Prime (corner 12197 3) := by change ¬ Nat.Prime 595042843; exact PrattCertificates.composite_of_product (factor := 37) (cofactor := 16082239) rfl (by decide) (by decide)
    exact decide_eq_false composite
theorem row_count_12197 : row 12197 = 0 := (row_12197.checked).trans rfl
theorem model_count_12197 : modelRow 12197 = 0 := by decide
theorem modelrow_agrees_12197 : modelRow 12197 = row 12197 := model_count_12197.trans row_count_12197.symm
def row_12198 : RowCertificate 12198 where
  first := false
  first_correct := by
    have composite : ¬ Nat.Prime (corner 12198 1) := by change ¬ Nat.Prime 595189213; exact PrattCertificates.composite_of_product (factor := 6151) (cofactor := 96763) rfl (by decide) (by decide)
    exact decide_eq_false composite
  second := false
  second_correct := by
    have composite : ¬ Nat.Prime (corner 12198 2) := by change ¬ Nat.Prime 595164817; exact PrattCertificates.composite_of_product (factor := 13) (cofactor := 45781909) rfl (by decide) (by decide)
    exact decide_eq_false composite
  third := true
  third_correct := by
    have prime : Nat.Prime (corner 12198 3) := by change Nat.Prime 595140421; exact ProjectEulerSolutions.P58PrimeNodes.p_595140421.prime
    exact decide_eq_true prime
theorem row_count_12198 : row 12198 = 1 := (row_12198.checked).trans rfl
theorem model_count_12198 : modelRow 12198 = 1 := by decide
theorem modelrow_agrees_12198 : modelRow 12198 = row 12198 := model_count_12198.trans row_count_12198.symm
def row_12199 : RowCertificate 12199 where
  first := true
  first_correct := by
    have prime : Nat.Prime (corner 12199 1) := by change Nat.Prime 595286803; exact ProjectEulerSolutions.P58PrimeNodes.p_595286803.prime
    exact decide_eq_true prime
  second := false
  second_correct := by
    have composite : ¬ Nat.Prime (corner 12199 2) := by change ¬ Nat.Prime 595262405; exact PrattCertificates.composite_of_product (factor := 5) (cofactor := 119052481) rfl (by decide) (by decide)
    exact decide_eq_false composite
  third := false
  third_correct := by
    have composite : ¬ Nat.Prime (corner 12199 3) := by change ¬ Nat.Prime 595238007; exact PrattCertificates.composite_of_product (factor := 3) (cofactor := 198412669) rfl (by decide) (by decide)
    exact decide_eq_false composite
theorem row_count_12199 : row 12199 = 1 := (row_12199.checked).trans rfl
theorem model_count_12199 : modelRow 12199 = 1 := by decide
theorem modelrow_agrees_12199 : modelRow 12199 = row 12199 := model_count_12199.trans row_count_12199.symm
def row_12200 : RowCertificate 12200 where
  first := false
  first_correct := by
    have composite : ¬ Nat.Prime (corner 12200 1) := by change ¬ Nat.Prime 595384401; exact PrattCertificates.composite_of_product (factor := 3) (cofactor := 198461467) rfl (by decide) (by decide)
    exact decide_eq_false composite
  second := false
  second_correct := by
    have composite : ¬ Nat.Prime (corner 12200 2) := by change ¬ Nat.Prime 595360001; exact PrattCertificates.composite_of_product (factor := 701) (cofactor := 849301) rfl (by decide) (by decide)
    exact decide_eq_false composite
  third := false
  third_correct := by
    have composite : ¬ Nat.Prime (corner 12200 3) := by change ¬ Nat.Prime 595335601; exact PrattCertificates.composite_of_product (factor := 7) (cofactor := 85047943) rfl (by decide) (by decide)
    exact decide_eq_false composite
theorem row_count_12200 : row 12200 = 0 := (row_12200.checked).trans rfl
theorem model_count_12200 : modelRow 12200 = 0 := by decide
theorem modelrow_agrees_12200 : modelRow 12200 = row 12200 := model_count_12200.trans row_count_12200.symm
def row_12201 : RowCertificate 12201 where
  first := false
  first_correct := by
    have composite : ¬ Nat.Prime (corner 12201 1) := by change ¬ Nat.Prime 595482007; exact PrattCertificates.composite_of_product (factor := 31) (cofactor := 19209097) rfl (by decide) (by decide)
    exact decide_eq_false composite
  second := false
  second_correct := by
    have composite : ¬ Nat.Prime (corner 12201 2) := by change ¬ Nat.Prime 595457605; exact PrattCertificates.composite_of_product (factor := 5) (cofactor := 119091521) rfl (by decide) (by decide)
    exact decide_eq_false composite
  third := true
  third_correct := by
    have prime : Nat.Prime (corner 12201 3) := by change Nat.Prime 595433203; exact ProjectEulerSolutions.P58PrimeNodes.p_595433203.prime
    exact decide_eq_true prime
theorem row_count_12201 : row 12201 = 1 := (row_12201.checked).trans rfl
theorem model_count_12201 : modelRow 12201 = 1 := by decide
theorem modelrow_agrees_12201 : modelRow 12201 = row 12201 := model_count_12201.trans row_count_12201.symm
def row_12202 : RowCertificate 12202 where
  first := false
  first_correct := by
    have composite : ¬ Nat.Prime (corner 12202 1) := by change ¬ Nat.Prime 595579621; exact PrattCertificates.composite_of_product (factor := 7) (cofactor := 85082803) rfl (by decide) (by decide)
    exact decide_eq_false composite
  second := false
  second_correct := by
    have composite : ¬ Nat.Prime (corner 12202 2) := by change ¬ Nat.Prime 595555217; exact PrattCertificates.composite_of_product (factor := 41) (cofactor := 14525737) rfl (by decide) (by decide)
    exact decide_eq_false composite
  third := false
  third_correct := by
    have composite : ¬ Nat.Prime (corner 12202 3) := by change ¬ Nat.Prime 595530813; exact PrattCertificates.composite_of_product (factor := 3) (cofactor := 198510271) rfl (by decide) (by decide)
    exact decide_eq_false composite
theorem row_count_12202 : row 12202 = 0 := (row_12202.checked).trans rfl
theorem model_count_12202 : modelRow 12202 = 0 := by decide
theorem modelrow_agrees_12202 : modelRow 12202 = row 12202 := model_count_12202.trans row_count_12202.symm
def row_12203 : RowCertificate 12203 where
  first := false
  first_correct := by
    have composite : ¬ Nat.Prime (corner 12203 1) := by change ¬ Nat.Prime 595677243; exact PrattCertificates.composite_of_product (factor := 3) (cofactor := 198559081) rfl (by decide) (by decide)
    exact decide_eq_false composite
  second := false
  second_correct := by
    have composite : ¬ Nat.Prime (corner 12203 2) := by change ¬ Nat.Prime 595652837; exact PrattCertificates.composite_of_product (factor := 13) (cofactor := 45819449) rfl (by decide) (by decide)
    exact decide_eq_false composite
  third := true
  third_correct := by
    have prime : Nat.Prime (corner 12203 3) := by change Nat.Prime 595628431; exact ProjectEulerSolutions.P58PrimeNodes.p_595628431.prime
    exact decide_eq_true prime
theorem row_count_12203 : row 12203 = 1 := (row_12203.checked).trans rfl
theorem model_count_12203 : modelRow 12203 = 1 := by decide
theorem modelrow_agrees_12203 : modelRow 12203 = row 12203 := model_count_12203.trans row_count_12203.symm
def row_12204 : RowCertificate 12204 where
  first := false
  first_correct := by
    have composite : ¬ Nat.Prime (corner 12204 1) := by change ¬ Nat.Prime 595774873; exact PrattCertificates.composite_of_product (factor := 97) (cofactor := 6142009) rfl (by decide) (by decide)
    exact decide_eq_false composite
  second := false
  second_correct := by
    have composite : ¬ Nat.Prime (corner 12204 2) := by change ¬ Nat.Prime 595750465; exact PrattCertificates.composite_of_product (factor := 5) (cofactor := 119150093) rfl (by decide) (by decide)
    exact decide_eq_false composite
  third := false
  third_correct := by
    have composite : ¬ Nat.Prime (corner 12204 3) := by change ¬ Nat.Prime 595726057; exact PrattCertificates.composite_of_product (factor := 19) (cofactor := 31354003) rfl (by decide) (by decide)
    exact decide_eq_false composite
theorem row_count_12204 : row 12204 = 0 := (row_12204.checked).trans rfl
theorem model_count_12204 : modelRow 12204 = 0 := by decide
theorem modelrow_agrees_12204 : modelRow 12204 = row 12204 := model_count_12204.trans row_count_12204.symm
def row_12205 : RowCertificate 12205 where
  first := false
  first_correct := by
    have composite : ¬ Nat.Prime (corner 12205 1) := by change ¬ Nat.Prime 595872511; exact PrattCertificates.composite_of_product (factor := 13) (cofactor := 45836347) rfl (by decide) (by decide)
    exact decide_eq_false composite
  second := false
  second_correct := by
    have composite : ¬ Nat.Prime (corner 12205 2) := by change ¬ Nat.Prime 595848101; exact PrattCertificates.composite_of_product (factor := 53) (cofactor := 11242417) rfl (by decide) (by decide)
    exact decide_eq_false composite
  third := false
  third_correct := by
    have composite : ¬ Nat.Prime (corner 12205 3) := by change ¬ Nat.Prime 595823691; exact PrattCertificates.composite_of_product (factor := 3) (cofactor := 198607897) rfl (by decide) (by decide)
    exact decide_eq_false composite
theorem row_count_12205 : row 12205 = 0 := (row_12205.checked).trans rfl
theorem model_count_12205 : modelRow 12205 = 0 := by decide
theorem modelrow_agrees_12205 : modelRow 12205 = row 12205 := model_count_12205.trans row_count_12205.symm
def row_12206 : RowCertificate 12206 where
  first := false
  first_correct := by
    have composite : ¬ Nat.Prime (corner 12206 1) := by change ¬ Nat.Prime 595970157; exact PrattCertificates.composite_of_product (factor := 3) (cofactor := 198656719) rfl (by decide) (by decide)
    exact decide_eq_false composite
  second := false
  second_correct := by
    have composite : ¬ Nat.Prime (corner 12206 2) := by change ¬ Nat.Prime 595945745; exact PrattCertificates.composite_of_product (factor := 5) (cofactor := 119189149) rfl (by decide) (by decide)
    exact decide_eq_false composite
  third := false
  third_correct := by
    have composite : ¬ Nat.Prime (corner 12206 3) := by change ¬ Nat.Prime 595921333; exact PrattCertificates.composite_of_product (factor := 7) (cofactor := 85131619) rfl (by decide) (by decide)
    exact decide_eq_false composite
theorem row_count_12206 : row 12206 = 0 := (row_12206.checked).trans rfl
theorem model_count_12206 : modelRow 12206 = 0 := by decide
theorem modelrow_agrees_12206 : modelRow 12206 = row 12206 := model_count_12206.trans row_count_12206.symm
def row_12207 : RowCertificate 12207 where
  first := true
  first_correct := by
    have prime : Nat.Prime (corner 12207 1) := by change Nat.Prime 596067811; exact ProjectEulerSolutions.P58PrimeNodes.p_596067811.prime
    exact decide_eq_true prime
  second := false
  second_correct := by
    have composite : ¬ Nat.Prime (corner 12207 2) := by change ¬ Nat.Prime 596043397; exact PrattCertificates.composite_of_product (factor := 37) (cofactor := 16109281) rfl (by decide) (by decide)
    exact decide_eq_false composite
  third := false
  third_correct := by
    have composite : ¬ Nat.Prime (corner 12207 3) := by change ¬ Nat.Prime 596018983; exact PrattCertificates.composite_of_product (factor := 7) (cofactor := 85145569) rfl (by decide) (by decide)
    exact decide_eq_false composite
theorem row_count_12207 : row 12207 = 1 := (row_12207.checked).trans rfl
theorem model_count_12207 : modelRow 12207 = 1 := by decide
theorem modelrow_agrees_12207 : modelRow 12207 = row 12207 := model_count_12207.trans row_count_12207.symm
def row_12208 : RowCertificate 12208 where
  first := false
  first_correct := by
    have composite : ¬ Nat.Prime (corner 12208 1) := by change ¬ Nat.Prime 596165473; exact PrattCertificates.composite_of_product (factor := 15877) (cofactor := 37549) rfl (by decide) (by decide)
    exact decide_eq_false composite
  second := false
  second_correct := by
    have composite : ¬ Nat.Prime (corner 12208 2) := by change ¬ Nat.Prime 596141057; exact PrattCertificates.composite_of_product (factor := 17) (cofactor := 35067121) rfl (by decide) (by decide)
    exact decide_eq_false composite
  third := false
  third_correct := by
    have composite : ¬ Nat.Prime (corner 12208 3) := by change ¬ Nat.Prime 596116641; exact PrattCertificates.composite_of_product (factor := 3) (cofactor := 198705547) rfl (by decide) (by decide)
    exact decide_eq_false composite
theorem row_count_12208 : row 12208 = 0 := (row_12208.checked).trans rfl
theorem model_count_12208 : modelRow 12208 = 0 := by decide
theorem modelrow_agrees_12208 : modelRow 12208 = row 12208 := model_count_12208.trans row_count_12208.symm
def row_12209 : RowCertificate 12209 where
  first := false
  first_correct := by
    have composite : ¬ Nat.Prime (corner 12209 1) := by change ¬ Nat.Prime 596263143; exact PrattCertificates.composite_of_product (factor := 3) (cofactor := 198754381) rfl (by decide) (by decide)
    exact decide_eq_false composite
  second := false
  second_correct := by
    have composite : ¬ Nat.Prime (corner 12209 2) := by change ¬ Nat.Prime 596238725; exact PrattCertificates.composite_of_product (factor := 5) (cofactor := 119247745) rfl (by decide) (by decide)
    exact decide_eq_false composite
  third := false
  third_correct := by
    have composite : ¬ Nat.Prime (corner 12209 3) := by change ¬ Nat.Prime 596214307; exact PrattCertificates.composite_of_product (factor := 13) (cofactor := 45862639) rfl (by decide) (by decide)
    exact decide_eq_false composite
theorem row_count_12209 : row 12209 = 0 := (row_12209.checked).trans rfl
theorem model_count_12209 : modelRow 12209 = 0 := by decide
theorem modelrow_agrees_12209 : modelRow 12209 = row 12209 := model_count_12209.trans row_count_12209.symm
def row_12210 : RowCertificate 12210 where
  first := false
  first_correct := by
    have composite : ¬ Nat.Prime (corner 12210 1) := by change ¬ Nat.Prime 596360821; exact PrattCertificates.composite_of_product (factor := 7) (cofactor := 85194403) rfl (by decide) (by decide)
    exact decide_eq_false composite
  second := false
  second_correct := by
    have composite : ¬ Nat.Prime (corner 12210 2) := by change ¬ Nat.Prime 596336401; exact PrattCertificates.composite_of_product (factor := 89) (cofactor := 6700409) rfl (by decide) (by decide)
    exact decide_eq_false composite
  third := false
  third_correct := by
    have composite : ¬ Nat.Prime (corner 12210 3) := by change ¬ Nat.Prime 596311981; exact PrattCertificates.composite_of_product (factor := 3361) (cofactor := 177421) rfl (by decide) (by decide)
    exact decide_eq_false composite
theorem row_count_12210 : row 12210 = 0 := (row_12210.checked).trans rfl
theorem model_count_12210 : modelRow 12210 = 0 := by decide
theorem modelrow_agrees_12210 : modelRow 12210 = row 12210 := model_count_12210.trans row_count_12210.symm
def row_12211 : RowCertificate 12211 where
  first := false
  first_correct := by
    have composite : ¬ Nat.Prime (corner 12211 1) := by change ¬ Nat.Prime 596458507; exact PrattCertificates.composite_of_product (factor := 19) (cofactor := 31392553) rfl (by decide) (by decide)
    exact decide_eq_false composite
  second := false
  second_correct := by
    have composite : ¬ Nat.Prime (corner 12211 2) := by change ¬ Nat.Prime 596434085; exact PrattCertificates.composite_of_product (factor := 5) (cofactor := 119286817) rfl (by decide) (by decide)
    exact decide_eq_false composite
  third := false
  third_correct := by
    have composite : ¬ Nat.Prime (corner 12211 3) := by change ¬ Nat.Prime 596409663; exact PrattCertificates.composite_of_product (factor := 3) (cofactor := 198803221) rfl (by decide) (by decide)
    exact decide_eq_false composite
theorem row_count_12211 : row 12211 = 0 := (row_12211.checked).trans rfl
theorem model_count_12211 : modelRow 12211 = 0 := by decide
theorem modelrow_agrees_12211 : modelRow 12211 = row 12211 := model_count_12211.trans row_count_12211.symm
def row_12212 : RowCertificate 12212 where
  first := false
  first_correct := by
    have composite : ¬ Nat.Prime (corner 12212 1) := by change ¬ Nat.Prime 596556201; exact PrattCertificates.composite_of_product (factor := 3) (cofactor := 198852067) rfl (by decide) (by decide)
    exact decide_eq_false composite
  second := false
  second_correct := by
    have composite : ¬ Nat.Prime (corner 12212 2) := by change ¬ Nat.Prime 596531777; exact PrattCertificates.composite_of_product (factor := 8237) (cofactor := 72421) rfl (by decide) (by decide)
    exact decide_eq_false composite
  third := false
  third_correct := by
    have composite : ¬ Nat.Prime (corner 12212 3) := by change ¬ Nat.Prime 596507353; exact PrattCertificates.composite_of_product (factor := 13) (cofactor := 45885181) rfl (by decide) (by decide)
    exact decide_eq_false composite
theorem row_count_12212 : row 12212 = 0 := (row_12212.checked).trans rfl
theorem model_count_12212 : modelRow 12212 = 0 := by decide
theorem modelrow_agrees_12212 : modelRow 12212 = row 12212 := model_count_12212.trans row_count_12212.symm
def row_12213 : RowCertificate 12213 where
  first := false
  first_correct := by
    have composite : ¬ Nat.Prime (corner 12213 1) := by change ¬ Nat.Prime 596653903; exact PrattCertificates.composite_of_product (factor := 19) (cofactor := 31402837) rfl (by decide) (by decide)
    exact decide_eq_false composite
  second := false
  second_correct := by
    have composite : ¬ Nat.Prime (corner 12213 2) := by change ¬ Nat.Prime 596629477; exact PrattCertificates.composite_of_product (factor := 37) (cofactor := 16125121) rfl (by decide) (by decide)
    exact decide_eq_false composite
  third := false
  third_correct := by
    have composite : ¬ Nat.Prime (corner 12213 3) := by change ¬ Nat.Prime 596605051; exact PrattCertificates.composite_of_product (factor := 7) (cofactor := 85229293) rfl (by decide) (by decide)
    exact decide_eq_false composite
theorem row_count_12213 : row 12213 = 0 := (row_12213.checked).trans rfl
theorem model_count_12213 : modelRow 12213 = 0 := by decide
theorem modelrow_agrees_12213 : modelRow 12213 = row 12213 := model_count_12213.trans row_count_12213.symm
def row_12214 : RowCertificate 12214 where
  first := false
  first_correct := by
    have composite : ¬ Nat.Prime (corner 12214 1) := by change ¬ Nat.Prime 596751613; exact PrattCertificates.composite_of_product (factor := 20443) (cofactor := 29191) rfl (by decide) (by decide)
    exact decide_eq_false composite
  second := false
  second_correct := by
    have composite : ¬ Nat.Prime (corner 12214 2) := by change ¬ Nat.Prime 596727185; exact PrattCertificates.composite_of_product (factor := 5) (cofactor := 119345437) rfl (by decide) (by decide)
    exact decide_eq_false composite
  third := false
  third_correct := by
    have composite : ¬ Nat.Prime (corner 12214 3) := by change ¬ Nat.Prime 596702757; exact PrattCertificates.composite_of_product (factor := 3) (cofactor := 198900919) rfl (by decide) (by decide)
    exact decide_eq_false composite
theorem row_count_12214 : row 12214 = 0 := (row_12214.checked).trans rfl
theorem model_count_12214 : modelRow 12214 = 0 := by decide
theorem modelrow_agrees_12214 : modelRow 12214 = row 12214 := model_count_12214.trans row_count_12214.symm
def row_12215 : RowCertificate 12215 where
  first := false
  first_correct := by
    have composite : ¬ Nat.Prime (corner 12215 1) := by change ¬ Nat.Prime 596849331; exact PrattCertificates.composite_of_product (factor := 3) (cofactor := 198949777) rfl (by decide) (by decide)
    exact decide_eq_false composite
  second := false
  second_correct := by
    have composite : ¬ Nat.Prime (corner 12215 2) := by change ¬ Nat.Prime 596824901; exact PrattCertificates.composite_of_product (factor := 29) (cofactor := 20580169) rfl (by decide) (by decide)
    exact decide_eq_false composite
  third := false
  third_correct := by
    have composite : ¬ Nat.Prime (corner 12215 3) := by change ¬ Nat.Prime 596800471; exact PrattCertificates.composite_of_product (factor := 151) (cofactor := 3952321) rfl (by decide) (by decide)
    exact decide_eq_false composite
theorem row_count_12215 : row 12215 = 0 := (row_12215.checked).trans rfl
theorem model_count_12215 : modelRow 12215 = 0 := by decide
theorem modelrow_agrees_12215 : modelRow 12215 = row 12215 := model_count_12215.trans row_count_12215.symm
def row_12216 : RowCertificate 12216 where
  first := false
  first_correct := by
    have composite : ¬ Nat.Prime (corner 12216 1) := by change ¬ Nat.Prime 596947057; exact PrattCertificates.composite_of_product (factor := 7) (cofactor := 85278151) rfl (by decide) (by decide)
    exact decide_eq_false composite
  second := false
  second_correct := by
    have composite : ¬ Nat.Prime (corner 12216 2) := by change ¬ Nat.Prime 596922625; exact PrattCertificates.composite_of_product (factor := 5) (cofactor := 119384525) rfl (by decide) (by decide)
    exact decide_eq_false composite
  third := true
  third_correct := by
    have prime : Nat.Prime (corner 12216 3) := by change Nat.Prime 596898193; exact ProjectEulerSolutions.P58PrimeNodes.p_596898193.prime
    exact decide_eq_true prime
theorem row_count_12216 : row 12216 = 1 := (row_12216.checked).trans rfl
theorem model_count_12216 : modelRow 12216 = 1 := by decide
theorem modelrow_agrees_12216 : modelRow 12216 = row 12216 := model_count_12216.trans row_count_12216.symm
def row_12217 : RowCertificate 12217 where
  first := false
  first_correct := by
    have composite : ¬ Nat.Prime (corner 12217 1) := by change ¬ Nat.Prime 597044791; exact PrattCertificates.composite_of_product (factor := 7) (cofactor := 85292113) rfl (by decide) (by decide)
    exact decide_eq_false composite
  second := true
  second_correct := by
    have prime : Nat.Prime (corner 12217 2) := by change Nat.Prime 597020357; exact ProjectEulerSolutions.P58PrimeNodes.p_597020357.prime
    exact decide_eq_true prime
  third := false
  third_correct := by
    have composite : ¬ Nat.Prime (corner 12217 3) := by change ¬ Nat.Prime 596995923; exact PrattCertificates.composite_of_product (factor := 3) (cofactor := 198998641) rfl (by decide) (by decide)
    exact decide_eq_false composite
theorem row_count_12217 : row 12217 = 1 := (row_12217.checked).trans rfl
theorem model_count_12217 : modelRow 12217 = 1 := by decide
theorem modelrow_agrees_12217 : modelRow 12217 = row 12217 := model_count_12217.trans row_count_12217.symm
def row_12218 : RowCertificate 12218 where
  first := false
  first_correct := by
    have composite : ¬ Nat.Prime (corner 12218 1) := by change ¬ Nat.Prime 597142533; exact PrattCertificates.composite_of_product (factor := 3) (cofactor := 199047511) rfl (by decide) (by decide)
    exact decide_eq_false composite
  second := false
  second_correct := by
    have composite : ¬ Nat.Prime (corner 12218 2) := by change ¬ Nat.Prime 597118097; exact PrattCertificates.composite_of_product (factor := 277) (cofactor := 2155661) rfl (by decide) (by decide)
    exact decide_eq_false composite
  third := true
  third_correct := by
    have prime : Nat.Prime (corner 12218 3) := by change Nat.Prime 597093661; exact ProjectEulerSolutions.P58PrimeNodes.p_597093661.prime
    exact decide_eq_true prime
theorem row_count_12218 : row 12218 = 1 := (row_12218.checked).trans rfl
theorem model_count_12218 : modelRow 12218 = 1 := by decide
theorem modelrow_agrees_12218 : modelRow 12218 = row 12218 := model_count_12218.trans row_count_12218.symm
def row_12219 : RowCertificate 12219 where
  first := false
  first_correct := by
    have composite : ¬ Nat.Prime (corner 12219 1) := by change ¬ Nat.Prime 597240283; exact PrattCertificates.composite_of_product (factor := 421) (cofactor := 1418623) rfl (by decide) (by decide)
    exact decide_eq_false composite
  second := false
  second_correct := by
    have composite : ¬ Nat.Prime (corner 12219 2) := by change ¬ Nat.Prime 597215845; exact PrattCertificates.composite_of_product (factor := 5) (cofactor := 119443169) rfl (by decide) (by decide)
    exact decide_eq_false composite
  third := true
  third_correct := by
    have prime : Nat.Prime (corner 12219 3) := by change Nat.Prime 597191407; exact ProjectEulerSolutions.P58PrimeNodes.p_597191407.prime
    exact decide_eq_true prime
theorem row_count_12219 : row 12219 = 1 := (row_12219.checked).trans rfl
theorem model_count_12219 : modelRow 12219 = 1 := by decide
theorem modelrow_agrees_12219 : modelRow 12219 = row 12219 := model_count_12219.trans row_count_12219.symm
def row_12220 : RowCertificate 12220 where
  first := false
  first_correct := by
    have composite : ¬ Nat.Prime (corner 12220 1) := by change ¬ Nat.Prime 597338041; exact PrattCertificates.composite_of_product (factor := 13591) (cofactor := 43951) rfl (by decide) (by decide)
    exact decide_eq_false composite
  second := true
  second_correct := by
    have prime : Nat.Prime (corner 12220 2) := by change Nat.Prime 597313601; exact ProjectEulerSolutions.P58PrimeNodes.p_597313601.prime
    exact decide_eq_true prime
  third := false
  third_correct := by
    have composite : ¬ Nat.Prime (corner 12220 3) := by change ¬ Nat.Prime 597289161; exact PrattCertificates.composite_of_product (factor := 3) (cofactor := 199096387) rfl (by decide) (by decide)
    exact decide_eq_false composite
theorem row_count_12220 : row 12220 = 1 := (row_12220.checked).trans rfl
theorem model_count_12220 : modelRow 12220 = 1 := by decide
theorem modelrow_agrees_12220 : modelRow 12220 = row 12220 := model_count_12220.trans row_count_12220.symm
def row_12221 : RowCertificate 12221 where
  first := false
  first_correct := by
    have composite : ¬ Nat.Prime (corner 12221 1) := by change ¬ Nat.Prime 597435807; exact PrattCertificates.composite_of_product (factor := 3) (cofactor := 199145269) rfl (by decide) (by decide)
    exact decide_eq_false composite
  second := false
  second_correct := by
    have composite : ¬ Nat.Prime (corner 12221 2) := by change ¬ Nat.Prime 597411365; exact PrattCertificates.composite_of_product (factor := 5) (cofactor := 119482273) rfl (by decide) (by decide)
    exact decide_eq_false composite
  third := false
  third_correct := by
    have composite : ¬ Nat.Prime (corner 12221 3) := by change ¬ Nat.Prime 597386923; exact PrattCertificates.composite_of_product (factor := 7) (cofactor := 85340989) rfl (by decide) (by decide)
    exact decide_eq_false composite
theorem row_count_12221 : row 12221 = 0 := (row_12221.checked).trans rfl
theorem model_count_12221 : modelRow 12221 = 0 := by decide
theorem modelrow_agrees_12221 : modelRow 12221 = row 12221 := model_count_12221.trans row_count_12221.symm
def row_12222 : RowCertificate 12222 where
  first := false
  first_correct := by
    have composite : ¬ Nat.Prime (corner 12222 1) := by change ¬ Nat.Prime 597533581; exact PrattCertificates.composite_of_product (factor := 11497) (cofactor := 51973) rfl (by decide) (by decide)
    exact decide_eq_false composite
  second := true
  second_correct := by
    have prime : Nat.Prime (corner 12222 2) := by change Nat.Prime 597509137; exact ProjectEulerSolutions.P58PrimeNodes.p_597509137.prime
    exact decide_eq_true prime
  third := false
  third_correct := by
    have composite : ¬ Nat.Prime (corner 12222 3) := by change ¬ Nat.Prime 597484693; exact PrattCertificates.composite_of_product (factor := 13) (cofactor := 45960361) rfl (by decide) (by decide)
    exact decide_eq_false composite
theorem row_count_12222 : row 12222 = 1 := (row_12222.checked).trans rfl
theorem model_count_12222 : modelRow 12222 = 1 := by decide
theorem modelrow_agrees_12222 : modelRow 12222 = row 12222 := model_count_12222.trans row_count_12222.symm
def row_12223 : RowCertificate 12223 where
  first := false
  first_correct := by
    have composite : ¬ Nat.Prime (corner 12223 1) := by change ¬ Nat.Prime 597631363; exact PrattCertificates.composite_of_product (factor := 7) (cofactor := 85375909) rfl (by decide) (by decide)
    exact decide_eq_false composite
  second := true
  second_correct := by
    have prime : Nat.Prime (corner 12223 2) := by change Nat.Prime 597606917; exact ProjectEulerSolutions.P58PrimeNodes.p_597606917.prime
    exact decide_eq_true prime
  third := false
  third_correct := by
    have composite : ¬ Nat.Prime (corner 12223 3) := by change ¬ Nat.Prime 597582471; exact PrattCertificates.composite_of_product (factor := 3) (cofactor := 199194157) rfl (by decide) (by decide)
    exact decide_eq_false composite
theorem row_count_12223 : row 12223 = 1 := (row_12223.checked).trans rfl
theorem model_count_12223 : modelRow 12223 = 1 := by decide
theorem modelrow_agrees_12223 : modelRow 12223 = row 12223 := model_count_12223.trans row_count_12223.symm
def row_12224 : RowCertificate 12224 where
  first := false
  first_correct := by
    have composite : ¬ Nat.Prime (corner 12224 1) := by change ¬ Nat.Prime 597729153; exact PrattCertificates.composite_of_product (factor := 3) (cofactor := 199243051) rfl (by decide) (by decide)
    exact decide_eq_false composite
  second := false
  second_correct := by
    have composite : ¬ Nat.Prime (corner 12224 2) := by change ¬ Nat.Prime 597704705; exact PrattCertificates.composite_of_product (factor := 5) (cofactor := 119540941) rfl (by decide) (by decide)
    exact decide_eq_false composite
  third := false
  third_correct := by
    have composite : ¬ Nat.Prime (corner 12224 3) := by change ¬ Nat.Prime 597680257; exact PrattCertificates.composite_of_product (factor := 61) (cofactor := 9798037) rfl (by decide) (by decide)
    exact decide_eq_false composite
theorem row_count_12224 : row 12224 = 0 := (row_12224.checked).trans rfl
theorem model_count_12224 : modelRow 12224 = 0 := by decide
theorem modelrow_agrees_12224 : modelRow 12224 = row 12224 := model_count_12224.trans row_count_12224.symm
def row_12225 : RowCertificate 12225 where
  first := false
  first_correct := by
    have composite : ¬ Nat.Prime (corner 12225 1) := by change ¬ Nat.Prime 597826951; exact PrattCertificates.composite_of_product (factor := 1129) (cofactor := 529519) rfl (by decide) (by decide)
    exact decide_eq_false composite
  second := false
  second_correct := by
    have composite : ¬ Nat.Prime (corner 12225 2) := by change ¬ Nat.Prime 597802501; exact PrattCertificates.composite_of_product (factor := 17) (cofactor := 35164853) rfl (by decide) (by decide)
    exact decide_eq_false composite
  third := false
  third_correct := by
    have composite : ¬ Nat.Prime (corner 12225 3) := by change ¬ Nat.Prime 597778051; exact PrattCertificates.composite_of_product (factor := 13) (cofactor := 45982927) rfl (by decide) (by decide)
    exact decide_eq_false composite
theorem row_count_12225 : row 12225 = 0 := (row_12225.checked).trans rfl
theorem model_count_12225 : modelRow 12225 = 0 := by decide
theorem modelrow_agrees_12225 : modelRow 12225 = row 12225 := model_count_12225.trans row_count_12225.symm
def row_12226 : RowCertificate 12226 where
  first := true
  first_correct := by
    have prime : Nat.Prime (corner 12226 1) := by change Nat.Prime 597924757; exact ProjectEulerSolutions.P58PrimeNodes.p_597924757.prime
    exact decide_eq_true prime
  second := false
  second_correct := by
    have composite : ¬ Nat.Prime (corner 12226 2) := by change ¬ Nat.Prime 597900305; exact PrattCertificates.composite_of_product (factor := 5) (cofactor := 119580061) rfl (by decide) (by decide)
    exact decide_eq_false composite
  third := false
  third_correct := by
    have composite : ¬ Nat.Prime (corner 12226 3) := by change ¬ Nat.Prime 597875853; exact PrattCertificates.composite_of_product (factor := 3) (cofactor := 199291951) rfl (by decide) (by decide)
    exact decide_eq_false composite
theorem row_count_12226 : row 12226 = 1 := (row_12226.checked).trans rfl
theorem model_count_12226 : modelRow 12226 = 1 := by decide
theorem modelrow_agrees_12226 : modelRow 12226 = row 12226 := model_count_12226.trans row_count_12226.symm
def row_12227 : RowCertificate 12227 where
  first := false
  first_correct := by
    have composite : ¬ Nat.Prime (corner 12227 1) := by change ¬ Nat.Prime 598022571; exact PrattCertificates.composite_of_product (factor := 3) (cofactor := 199340857) rfl (by decide) (by decide)
    exact decide_eq_false composite
  second := false
  second_correct := by
    have composite : ¬ Nat.Prime (corner 12227 2) := by change ¬ Nat.Prime 597998117; exact PrattCertificates.composite_of_product (factor := 181) (cofactor := 3303857) rfl (by decide) (by decide)
    exact decide_eq_false composite
  third := false
  third_correct := by
    have composite : ¬ Nat.Prime (corner 12227 3) := by change ¬ Nat.Prime 597973663; exact PrattCertificates.composite_of_product (factor := 7) (cofactor := 85424809) rfl (by decide) (by decide)
    exact decide_eq_false composite
theorem row_count_12227 : row 12227 = 0 := (row_12227.checked).trans rfl
theorem model_count_12227 : modelRow 12227 = 0 := by decide
theorem modelrow_agrees_12227 : modelRow 12227 = row 12227 := model_count_12227.trans row_count_12227.symm
def row_12228 : RowCertificate 12228 where
  first := false
  first_correct := by
    have composite : ¬ Nat.Prime (corner 12228 1) := by change ¬ Nat.Prime 598120393; exact PrattCertificates.composite_of_product (factor := 13) (cofactor := 46009261) rfl (by decide) (by decide)
    exact decide_eq_false composite
  second := false
  second_correct := by
    have composite : ¬ Nat.Prime (corner 12228 2) := by change ¬ Nat.Prime 598095937; exact PrattCertificates.composite_of_product (factor := 53) (cofactor := 11284829) rfl (by decide) (by decide)
    exact decide_eq_false composite
  third := false
  third_correct := by
    have composite : ¬ Nat.Prime (corner 12228 3) := by change ¬ Nat.Prime 598071481; exact PrattCertificates.composite_of_product (factor := 7) (cofactor := 85438783) rfl (by decide) (by decide)
    exact decide_eq_false composite
theorem row_count_12228 : row 12228 = 0 := (row_12228.checked).trans rfl
theorem model_count_12228 : modelRow 12228 = 0 := by decide
theorem modelrow_agrees_12228 : modelRow 12228 = row 12228 := model_count_12228.trans row_count_12228.symm
def row_12229 : RowCertificate 12229 where
  first := true
  first_correct := by
    have prime : Nat.Prime (corner 12229 1) := by change Nat.Prime 598218223; exact ProjectEulerSolutions.P58PrimeNodes.p_598218223.prime
    exact decide_eq_true prime
  second := false
  second_correct := by
    have composite : ¬ Nat.Prime (corner 12229 2) := by change ¬ Nat.Prime 598193765; exact PrattCertificates.composite_of_product (factor := 5) (cofactor := 119638753) rfl (by decide) (by decide)
    exact decide_eq_false composite
  third := false
  third_correct := by
    have composite : ¬ Nat.Prime (corner 12229 3) := by change ¬ Nat.Prime 598169307; exact PrattCertificates.composite_of_product (factor := 3) (cofactor := 199389769) rfl (by decide) (by decide)
    exact decide_eq_false composite
theorem row_count_12229 : row 12229 = 1 := (row_12229.checked).trans rfl
theorem model_count_12229 : modelRow 12229 = 1 := by decide
theorem modelrow_agrees_12229 : modelRow 12229 = row 12229 := model_count_12229.trans row_count_12229.symm
def row_12230 : RowCertificate 12230 where
  first := false
  first_correct := by
    have composite : ¬ Nat.Prime (corner 12230 1) := by change ¬ Nat.Prime 598316061; exact PrattCertificates.composite_of_product (factor := 3) (cofactor := 199438687) rfl (by decide) (by decide)
    exact decide_eq_false composite
  second := true
  second_correct := by
    have prime : Nat.Prime (corner 12230 2) := by change Nat.Prime 598291601; exact ProjectEulerSolutions.P58PrimeNodes.p_598291601.prime
    exact decide_eq_true prime
  third := true
  third_correct := by
    have prime : Nat.Prime (corner 12230 3) := by change Nat.Prime 598267141; exact ProjectEulerSolutions.P58PrimeNodes.p_598267141.prime
    exact decide_eq_true prime
theorem row_count_12230 : row 12230 = 2 := (row_12230.checked).trans rfl
theorem model_count_12230 : modelRow 12230 = 2 := by decide
theorem modelrow_agrees_12230 : modelRow 12230 = row 12230 := model_count_12230.trans row_count_12230.symm
def row_12231 : RowCertificate 12231 where
  first := false
  first_correct := by
    have composite : ¬ Nat.Prime (corner 12231 1) := by change ¬ Nat.Prime 598413907; exact PrattCertificates.composite_of_product (factor := 7) (cofactor := 85487701) rfl (by decide) (by decide)
    exact decide_eq_false composite
  second := false
  second_correct := by
    have composite : ¬ Nat.Prime (corner 12231 2) := by change ¬ Nat.Prime 598389445; exact PrattCertificates.composite_of_product (factor := 5) (cofactor := 119677889) rfl (by decide) (by decide)
    exact decide_eq_false composite
  third := false
  third_correct := by
    have composite : ¬ Nat.Prime (corner 12231 3) := by change ¬ Nat.Prime 598364983; exact PrattCertificates.composite_of_product (factor := 109) (cofactor := 5489587) rfl (by decide) (by decide)
    exact decide_eq_false composite
theorem row_count_12231 : row 12231 = 0 := (row_12231.checked).trans rfl
theorem model_count_12231 : modelRow 12231 = 0 := by decide
theorem modelrow_agrees_12231 : modelRow 12231 = row 12231 := model_count_12231.trans row_count_12231.symm
def row_12232 : RowCertificate 12232 where
  first := false
  first_correct := by
    have composite : ¬ Nat.Prime (corner 12232 1) := by change ¬ Nat.Prime 598511761; exact PrattCertificates.composite_of_product (factor := 19) (cofactor := 31500619) rfl (by decide) (by decide)
    exact decide_eq_false composite
  second := false
  second_correct := by
    have composite : ¬ Nat.Prime (corner 12232 2) := by change ¬ Nat.Prime 598487297; exact PrattCertificates.composite_of_product (factor := 29) (cofactor := 20637493) rfl (by decide) (by decide)
    exact decide_eq_false composite
  third := false
  third_correct := by
    have composite : ¬ Nat.Prime (corner 12232 3) := by change ¬ Nat.Prime 598462833; exact PrattCertificates.composite_of_product (factor := 3) (cofactor := 199487611) rfl (by decide) (by decide)
    exact decide_eq_false composite
theorem row_count_12232 : row 12232 = 0 := (row_12232.checked).trans rfl
theorem model_count_12232 : modelRow 12232 = 0 := by decide
theorem modelrow_agrees_12232 : modelRow 12232 = row 12232 := model_count_12232.trans row_count_12232.symm
def row_12233 : RowCertificate 12233 where
  first := false
  first_correct := by
    have composite : ¬ Nat.Prime (corner 12233 1) := by change ¬ Nat.Prime 598609623; exact PrattCertificates.composite_of_product (factor := 3) (cofactor := 199536541) rfl (by decide) (by decide)
    exact decide_eq_false composite
  second := false
  second_correct := by
    have composite : ¬ Nat.Prime (corner 12233 2) := by change ¬ Nat.Prime 598585157; exact PrattCertificates.composite_of_product (factor := 97) (cofactor := 6170981) rfl (by decide) (by decide)
    exact decide_eq_false composite
  third := true
  third_correct := by
    have prime : Nat.Prime (corner 12233 3) := by change Nat.Prime 598560691; exact ProjectEulerSolutions.P58PrimeNodes.p_598560691.prime
    exact decide_eq_true prime
theorem row_count_12233 : row 12233 = 1 := (row_12233.checked).trans rfl
theorem model_count_12233 : modelRow 12233 = 1 := by decide
theorem modelrow_agrees_12233 : modelRow 12233 = row 12233 := model_count_12233.trans row_count_12233.symm
def row_12234 : RowCertificate 12234 where
  first := false
  first_correct := by
    have composite : ¬ Nat.Prime (corner 12234 1) := by change ¬ Nat.Prime 598707493; exact PrattCertificates.composite_of_product (factor := 9859) (cofactor := 60727) rfl (by decide) (by decide)
    exact decide_eq_false composite
  second := false
  second_correct := by
    have composite : ¬ Nat.Prime (corner 12234 2) := by change ¬ Nat.Prime 598683025; exact PrattCertificates.composite_of_product (factor := 5) (cofactor := 119736605) rfl (by decide) (by decide)
    exact decide_eq_false composite
  third := false
  third_correct := by
    have composite : ¬ Nat.Prime (corner 12234 3) := by change ¬ Nat.Prime 598658557; exact PrattCertificates.composite_of_product (factor := 7) (cofactor := 85522651) rfl (by decide) (by decide)
    exact decide_eq_false composite
theorem row_count_12234 : row 12234 = 0 := (row_12234.checked).trans rfl
theorem model_count_12234 : modelRow 12234 = 0 := by decide
theorem modelrow_agrees_12234 : modelRow 12234 = row 12234 := model_count_12234.trans row_count_12234.symm
def row_12235 : RowCertificate 12235 where
  first := true
  first_correct := by
    have prime : Nat.Prime (corner 12235 1) := by change Nat.Prime 598805371; exact ProjectEulerSolutions.P58PrimeNodes.p_598805371.prime
    exact decide_eq_true prime
  second := true
  second_correct := by
    have prime : Nat.Prime (corner 12235 2) := by change Nat.Prime 598780901; exact ProjectEulerSolutions.P58PrimeNodes.p_598780901.prime
    exact decide_eq_true prime
  third := false
  third_correct := by
    have composite : ¬ Nat.Prime (corner 12235 3) := by change ¬ Nat.Prime 598756431; exact PrattCertificates.composite_of_product (factor := 3) (cofactor := 199585477) rfl (by decide) (by decide)
    exact decide_eq_false composite
theorem row_count_12235 : row 12235 = 2 := (row_12235.checked).trans rfl
theorem model_count_12235 : modelRow 12235 = 2 := by decide
theorem modelrow_agrees_12235 : modelRow 12235 = row 12235 := model_count_12235.trans row_count_12235.symm
def row_12236 : RowCertificate 12236 where
  first := false
  first_correct := by
    have composite : ¬ Nat.Prime (corner 12236 1) := by change ¬ Nat.Prime 598903257; exact PrattCertificates.composite_of_product (factor := 3) (cofactor := 199634419) rfl (by decide) (by decide)
    exact decide_eq_false composite
  second := false
  second_correct := by
    have composite : ¬ Nat.Prime (corner 12236 2) := by change ¬ Nat.Prime 598878785; exact PrattCertificates.composite_of_product (factor := 5) (cofactor := 119775757) rfl (by decide) (by decide)
    exact decide_eq_false composite
  third := true
  third_correct := by
    have prime : Nat.Prime (corner 12236 3) := by change Nat.Prime 598854313; exact ProjectEulerSolutions.P58PrimeNodes.p_598854313.prime
    exact decide_eq_true prime
theorem row_count_12236 : row 12236 = 1 := (row_12236.checked).trans rfl
theorem model_count_12236 : modelRow 12236 = 1 := by decide
theorem modelrow_agrees_12236 : modelRow 12236 = row 12236 := model_count_12236.trans row_count_12236.symm
def row_12237 : RowCertificate 12237 where
  first := false
  first_correct := by
    have composite : ¬ Nat.Prime (corner 12237 1) := by change ¬ Nat.Prime 599001151; exact PrattCertificates.composite_of_product (factor := 7) (cofactor := 85571593) rfl (by decide) (by decide)
    exact decide_eq_false composite
  second := false
  second_correct := by
    have composite : ¬ Nat.Prime (corner 12237 2) := by change ¬ Nat.Prime 598976677; exact PrattCertificates.composite_of_product (factor := 13) (cofactor := 46075129) rfl (by decide) (by decide)
    exact decide_eq_false composite
  third := false
  third_correct := by
    have composite : ¬ Nat.Prime (corner 12237 3) := by change ¬ Nat.Prime 598952203; exact PrattCertificates.composite_of_product (factor := 43) (cofactor := 13929121) rfl (by decide) (by decide)
    exact decide_eq_false composite
theorem row_count_12237 : row 12237 = 0 := (row_12237.checked).trans rfl
theorem model_count_12237 : modelRow 12237 = 0 := by decide
theorem modelrow_agrees_12237 : modelRow 12237 = row 12237 := model_count_12237.trans row_count_12237.symm
def row_12238 : RowCertificate 12238 where
  first := false
  first_correct := by
    have composite : ¬ Nat.Prime (corner 12238 1) := by change ¬ Nat.Prime 599099053; exact PrattCertificates.composite_of_product (factor := 7) (cofactor := 85585579) rfl (by decide) (by decide)
    exact decide_eq_false composite
  second := false
  second_correct := by
    have composite : ¬ Nat.Prime (corner 12238 2) := by change ¬ Nat.Prime 599074577; exact PrattCertificates.composite_of_product (factor := 17) (cofactor := 35239681) rfl (by decide) (by decide)
    exact decide_eq_false composite
  third := false
  third_correct := by
    have composite : ¬ Nat.Prime (corner 12238 3) := by change ¬ Nat.Prime 599050101; exact PrattCertificates.composite_of_product (factor := 3) (cofactor := 199683367) rfl (by decide) (by decide)
    exact decide_eq_false composite
theorem row_count_12238 : row 12238 = 0 := (row_12238.checked).trans rfl
theorem model_count_12238 : modelRow 12238 = 0 := by decide
theorem modelrow_agrees_12238 : modelRow 12238 = row 12238 := model_count_12238.trans row_count_12238.symm
def row_12239 : RowCertificate 12239 where
  first := false
  first_correct := by
    have composite : ¬ Nat.Prime (corner 12239 1) := by change ¬ Nat.Prime 599196963; exact PrattCertificates.composite_of_product (factor := 3) (cofactor := 199732321) rfl (by decide) (by decide)
    exact decide_eq_false composite
  second := false
  second_correct := by
    have composite : ¬ Nat.Prime (corner 12239 2) := by change ¬ Nat.Prime 599172485; exact PrattCertificates.composite_of_product (factor := 5) (cofactor := 119834497) rfl (by decide) (by decide)
    exact decide_eq_false composite
  third := false
  third_correct := by
    have composite : ¬ Nat.Prime (corner 12239 3) := by change ¬ Nat.Prime 599148007; exact PrattCertificates.composite_of_product (factor := 1213) (cofactor := 493939) rfl (by decide) (by decide)
    exact decide_eq_false composite
theorem row_count_12239 : row 12239 = 0 := (row_12239.checked).trans rfl
theorem model_count_12239 : modelRow 12239 = 0 := by decide
theorem modelrow_agrees_12239 : modelRow 12239 = row 12239 := model_count_12239.trans row_count_12239.symm
def row_12240 : RowCertificate 12240 where
  first := true
  first_correct := by
    have prime : Nat.Prime (corner 12240 1) := by change Nat.Prime 599294881; exact ProjectEulerSolutions.P58PrimeNodes.p_599294881.prime
    exact decide_eq_true prime
  second := false
  second_correct := by
    have composite : ¬ Nat.Prime (corner 12240 2) := by change ¬ Nat.Prime 599270401; exact PrattCertificates.composite_of_product (factor := 149) (cofactor := 4021949) rfl (by decide) (by decide)
    exact decide_eq_false composite
  third := false
  third_correct := by
    have composite : ¬ Nat.Prime (corner 12240 3) := by change ¬ Nat.Prime 599245921; exact PrattCertificates.composite_of_product (factor := 19) (cofactor := 31539259) rfl (by decide) (by decide)
    exact decide_eq_false composite
theorem row_count_12240 : row 12240 = 1 := (row_12240.checked).trans rfl
theorem model_count_12240 : modelRow 12240 = 1 := by decide
theorem modelrow_agrees_12240 : modelRow 12240 = row 12240 := model_count_12240.trans row_count_12240.symm
def row_12241 : RowCertificate 12241 where
  first := false
  first_correct := by
    have composite : ¬ Nat.Prime (corner 12241 1) := by change ¬ Nat.Prime 599392807; exact PrattCertificates.composite_of_product (factor := 13) (cofactor := 46107139) rfl (by decide) (by decide)
    exact decide_eq_false composite
  second := false
  second_correct := by
    have composite : ¬ Nat.Prime (corner 12241 2) := by change ¬ Nat.Prime 599368325; exact PrattCertificates.composite_of_product (factor := 5) (cofactor := 119873665) rfl (by decide) (by decide)
    exact decide_eq_false composite
  third := false
  third_correct := by
    have composite : ¬ Nat.Prime (corner 12241 3) := by change ¬ Nat.Prime 599343843; exact PrattCertificates.composite_of_product (factor := 3) (cofactor := 199781281) rfl (by decide) (by decide)
    exact decide_eq_false composite
theorem row_count_12241 : row 12241 = 0 := (row_12241.checked).trans rfl
theorem model_count_12241 : modelRow 12241 = 0 := by decide
theorem modelrow_agrees_12241 : modelRow 12241 = row 12241 := model_count_12241.trans row_count_12241.symm
def row_12242 : RowCertificate 12242 where
  first := false
  first_correct := by
    have composite : ¬ Nat.Prime (corner 12242 1) := by change ¬ Nat.Prime 599490741; exact PrattCertificates.composite_of_product (factor := 3) (cofactor := 199830247) rfl (by decide) (by decide)
    exact decide_eq_false composite
  second := false
  second_correct := by
    have composite : ¬ Nat.Prime (corner 12242 2) := by change ¬ Nat.Prime 599466257; exact PrattCertificates.composite_of_product (factor := 13) (cofactor := 46112789) rfl (by decide) (by decide)
    exact decide_eq_false composite
  third := false
  third_correct := by
    have composite : ¬ Nat.Prime (corner 12242 3) := by change ¬ Nat.Prime 599441773; exact PrattCertificates.composite_of_product (factor := 7) (cofactor := 85634539) rfl (by decide) (by decide)
    exact decide_eq_false composite
theorem row_count_12242 : row 12242 = 0 := (row_12242.checked).trans rfl
theorem model_count_12242 : modelRow 12242 = 0 := by decide
theorem modelrow_agrees_12242 : modelRow 12242 = row 12242 := model_count_12242.trans row_count_12242.symm
def row_12243 : RowCertificate 12243 where
  first := false
  first_correct := by
    have composite : ¬ Nat.Prime (corner 12243 1) := by change ¬ Nat.Prime 599588683; exact PrattCertificates.composite_of_product (factor := 409) (cofactor := 1465987) rfl (by decide) (by decide)
    exact decide_eq_false composite
  second := false
  second_correct := by
    have composite : ¬ Nat.Prime (corner 12243 2) := by change ¬ Nat.Prime 599564197; exact PrattCertificates.composite_of_product (factor := 41) (cofactor := 14623517) rfl (by decide) (by decide)
    exact decide_eq_false composite
  third := true
  third_correct := by
    have prime : Nat.Prime (corner 12243 3) := by change Nat.Prime 599539711; exact ProjectEulerSolutions.P58PrimeNodes.p_599539711.prime
    exact decide_eq_true prime
theorem row_count_12243 : row 12243 = 1 := (row_12243.checked).trans rfl
theorem model_count_12243 : modelRow 12243 = 1 := by decide
theorem modelrow_agrees_12243 : modelRow 12243 = row 12243 := model_count_12243.trans row_count_12243.symm
def row_12244 : RowCertificate 12244 where
  first := false
  first_correct := by
    have composite : ¬ Nat.Prime (corner 12244 1) := by change ¬ Nat.Prime 599686633; exact PrattCertificates.composite_of_product (factor := 7) (cofactor := 85669519) rfl (by decide) (by decide)
    exact decide_eq_false composite
  second := false
  second_correct := by
    have composite : ¬ Nat.Prime (corner 12244 2) := by change ¬ Nat.Prime 599662145; exact PrattCertificates.composite_of_product (factor := 5) (cofactor := 119932429) rfl (by decide) (by decide)
    exact decide_eq_false composite
  third := false
  third_correct := by
    have composite : ¬ Nat.Prime (corner 12244 3) := by change ¬ Nat.Prime 599637657; exact PrattCertificates.composite_of_product (factor := 3) (cofactor := 199879219) rfl (by decide) (by decide)
    exact decide_eq_false composite
theorem row_count_12244 : row 12244 = 0 := (row_12244.checked).trans rfl
theorem model_count_12244 : modelRow 12244 = 0 := by decide
theorem modelrow_agrees_12244 : modelRow 12244 = row 12244 := model_count_12244.trans row_count_12244.symm
def row_12245 : RowCertificate 12245 where
  first := false
  first_correct := by
    have composite : ¬ Nat.Prime (corner 12245 1) := by change ¬ Nat.Prime 599784591; exact PrattCertificates.composite_of_product (factor := 3) (cofactor := 199928197) rfl (by decide) (by decide)
    exact decide_eq_false composite
  second := true
  second_correct := by
    have prime : Nat.Prime (corner 12245 2) := by change Nat.Prime 599760101; exact ProjectEulerSolutions.P58PrimeNodes.p_599760101.prime
    exact decide_eq_true prime
  third := false
  third_correct := by
    have composite : ¬ Nat.Prime (corner 12245 3) := by change ¬ Nat.Prime 599735611; exact PrattCertificates.composite_of_product (factor := 4549) (cofactor := 131839) rfl (by decide) (by decide)
    exact decide_eq_false composite
theorem row_count_12245 : row 12245 = 1 := (row_12245.checked).trans rfl
theorem model_count_12245 : modelRow 12245 = 1 := by decide
theorem modelrow_agrees_12245 : modelRow 12245 = row 12245 := model_count_12245.trans row_count_12245.symm
def row_12246 : RowCertificate 12246 where
  first := false
  first_correct := by
    have composite : ¬ Nat.Prime (corner 12246 1) := by change ¬ Nat.Prime 599882557; exact PrattCertificates.composite_of_product (factor := 67) (cofactor := 8953471) rfl (by decide) (by decide)
    exact decide_eq_false composite
  second := false
  second_correct := by
    have composite : ¬ Nat.Prime (corner 12246 2) := by change ¬ Nat.Prime 599858065; exact PrattCertificates.composite_of_product (factor := 5) (cofactor := 119971613) rfl (by decide) (by decide)
    exact decide_eq_false composite
  third := false
  third_correct := by
    have composite : ¬ Nat.Prime (corner 12246 3) := by change ¬ Nat.Prime 599833573; exact PrattCertificates.composite_of_product (factor := 127) (cofactor := 4723099) rfl (by decide) (by decide)
    exact decide_eq_false composite
theorem row_count_12246 : row 12246 = 0 := (row_12246.checked).trans rfl
theorem model_count_12246 : modelRow 12246 = 0 := by decide
theorem modelrow_agrees_12246 : modelRow 12246 = row 12246 := model_count_12246.trans row_count_12246.symm
def row_12247 : RowCertificate 12247 where
  first := false
  first_correct := by
    have composite : ¬ Nat.Prime (corner 12247 1) := by change ¬ Nat.Prime 599980531; exact PrattCertificates.composite_of_product (factor := 151) (cofactor := 3973381) rfl (by decide) (by decide)
    exact decide_eq_false composite
  second := true
  second_correct := by
    have prime : Nat.Prime (corner 12247 2) := by change Nat.Prime 599956037; exact ProjectEulerSolutions.P58PrimeNodes.p_599956037.prime
    exact decide_eq_true prime
  third := false
  third_correct := by
    have composite : ¬ Nat.Prime (corner 12247 3) := by change ¬ Nat.Prime 599931543; exact PrattCertificates.composite_of_product (factor := 3) (cofactor := 199977181) rfl (by decide) (by decide)
    exact decide_eq_false composite
theorem row_count_12247 : row 12247 = 1 := (row_12247.checked).trans rfl
theorem model_count_12247 : modelRow 12247 = 1 := by decide
theorem modelrow_agrees_12247 : modelRow 12247 = row 12247 := model_count_12247.trans row_count_12247.symm
def row_12248 : RowCertificate 12248 where
  first := false
  first_correct := by
    have composite : ¬ Nat.Prime (corner 12248 1) := by change ¬ Nat.Prime 600078513; exact PrattCertificates.composite_of_product (factor := 3) (cofactor := 200026171) rfl (by decide) (by decide)
    exact decide_eq_false composite
  second := true
  second_correct := by
    have prime : Nat.Prime (corner 12248 2) := by change Nat.Prime 600054017; exact ProjectEulerSolutions.P58PrimeNodes.p_600054017.prime
    exact decide_eq_true prime
  third := false
  third_correct := by
    have composite : ¬ Nat.Prime (corner 12248 3) := by change ¬ Nat.Prime 600029521; exact PrattCertificates.composite_of_product (factor := 7) (cofactor := 85718503) rfl (by decide) (by decide)
    exact decide_eq_false composite
theorem row_count_12248 : row 12248 = 1 := (row_12248.checked).trans rfl
theorem model_count_12248 : modelRow 12248 = 1 := by decide
theorem modelrow_agrees_12248 : modelRow 12248 = row 12248 := model_count_12248.trans row_count_12248.symm
def row_12249 : RowCertificate 12249 where
  first := false
  first_correct := by
    have composite : ¬ Nat.Prime (corner 12249 1) := by change ¬ Nat.Prime 600176503; exact PrattCertificates.composite_of_product (factor := 19) (cofactor := 31588237) rfl (by decide) (by decide)
    exact decide_eq_false composite
  second := false
  second_correct := by
    have composite : ¬ Nat.Prime (corner 12249 2) := by change ¬ Nat.Prime 600152005; exact PrattCertificates.composite_of_product (factor := 5) (cofactor := 120030401) rfl (by decide) (by decide)
    exact decide_eq_false composite
  third := false
  third_correct := by
    have composite : ¬ Nat.Prime (corner 12249 3) := by change ¬ Nat.Prime 600127507; exact PrattCertificates.composite_of_product (factor := 7) (cofactor := 85732501) rfl (by decide) (by decide)
    exact decide_eq_false composite
theorem row_count_12249 : row 12249 = 0 := (row_12249.checked).trans rfl
theorem model_count_12249 : modelRow 12249 = 0 := by decide
theorem modelrow_agrees_12249 : modelRow 12249 = row 12249 := model_count_12249.trans row_count_12249.symm
def row_12250 : RowCertificate 12250 where
  first := false
  first_correct := by
    have composite : ¬ Nat.Prime (corner 12250 1) := by change ¬ Nat.Prime 600274501; exact PrattCertificates.composite_of_product (factor := 1069) (cofactor := 561529) rfl (by decide) (by decide)
    exact decide_eq_false composite
  second := false
  second_correct := by
    have composite : ¬ Nat.Prime (corner 12250 2) := by change ¬ Nat.Prime 600250001; exact PrattCertificates.composite_of_product (factor := 13) (cofactor := 46173077) rfl (by decide) (by decide)
    exact decide_eq_false composite
  third := false
  third_correct := by
    have composite : ¬ Nat.Prime (corner 12250 3) := by change ¬ Nat.Prime 600225501; exact PrattCertificates.composite_of_product (factor := 3) (cofactor := 200075167) rfl (by decide) (by decide)
    exact decide_eq_false composite
theorem row_count_12250 : row 12250 = 0 := (row_12250.checked).trans rfl
theorem model_count_12250 : modelRow 12250 = 0 := by decide
theorem modelrow_agrees_12250 : modelRow 12250 = row 12250 := model_count_12250.trans row_count_12250.symm
def row_12251 : RowCertificate 12251 where
  first := false
  first_correct := by
    have composite : ¬ Nat.Prime (corner 12251 1) := by change ¬ Nat.Prime 600372507; exact PrattCertificates.composite_of_product (factor := 3) (cofactor := 200124169) rfl (by decide) (by decide)
    exact decide_eq_false composite
  second := false
  second_correct := by
    have composite : ¬ Nat.Prime (corner 12251 2) := by change ¬ Nat.Prime 600348005; exact PrattCertificates.composite_of_product (factor := 5) (cofactor := 120069601) rfl (by decide) (by decide)
    exact decide_eq_false composite
  third := false
  third_correct := by
    have composite : ¬ Nat.Prime (corner 12251 3) := by change ¬ Nat.Prime 600323503; exact PrattCertificates.composite_of_product (factor := 13) (cofactor := 46178731) rfl (by decide) (by decide)
    exact decide_eq_false composite
theorem row_count_12251 : row 12251 = 0 := (row_12251.checked).trans rfl
theorem model_count_12251 : modelRow 12251 = 0 := by decide
theorem modelrow_agrees_12251 : modelRow 12251 = row 12251 := model_count_12251.trans row_count_12251.symm
def row_12252 : RowCertificate 12252 where
  first := false
  first_correct := by
    have composite : ¬ Nat.Prime (corner 12252 1) := by change ¬ Nat.Prime 600470521; exact PrattCertificates.composite_of_product (factor := 7) (cofactor := 85781503) rfl (by decide) (by decide)
    exact decide_eq_false composite
  second := false
  second_correct := by
    have composite : ¬ Nat.Prime (corner 12252 2) := by change ¬ Nat.Prime 600446017; exact PrattCertificates.composite_of_product (factor := 19037) (cofactor := 31541) rfl (by decide) (by decide)
    exact decide_eq_false composite
  third := false
  third_correct := by
    have composite : ¬ Nat.Prime (corner 12252 3) := by change ¬ Nat.Prime 600421513; exact PrattCertificates.composite_of_product (factor := 43) (cofactor := 13963291) rfl (by decide) (by decide)
    exact decide_eq_false composite
theorem row_count_12252 : row 12252 = 0 := (row_12252.checked).trans rfl
theorem model_count_12252 : modelRow 12252 = 0 := by decide
theorem modelrow_agrees_12252 : modelRow 12252 = row 12252 := model_count_12252.trans row_count_12252.symm
def row_12253 : RowCertificate 12253 where
  first := false
  first_correct := by
    have composite : ¬ Nat.Prime (corner 12253 1) := by change ¬ Nat.Prime 600568543; exact PrattCertificates.composite_of_product (factor := 139) (cofactor := 4320637) rfl (by decide) (by decide)
    exact decide_eq_false composite
  second := false
  second_correct := by
    have composite : ¬ Nat.Prime (corner 12253 2) := by change ¬ Nat.Prime 600544037; exact PrattCertificates.composite_of_product (factor := 113) (cofactor := 5314549) rfl (by decide) (by decide)
    exact decide_eq_false composite
  third := false
  third_correct := by
    have composite : ¬ Nat.Prime (corner 12253 3) := by change ¬ Nat.Prime 600519531; exact PrattCertificates.composite_of_product (factor := 3) (cofactor := 200173177) rfl (by decide) (by decide)
    exact decide_eq_false composite
theorem row_count_12253 : row 12253 = 0 := (row_12253.checked).trans rfl
theorem model_count_12253 : modelRow 12253 = 0 := by decide
theorem modelrow_agrees_12253 : modelRow 12253 = row 12253 := model_count_12253.trans row_count_12253.symm
def row_12254 : RowCertificate 12254 where
  first := false
  first_correct := by
    have composite : ¬ Nat.Prime (corner 12254 1) := by change ¬ Nat.Prime 600666573; exact PrattCertificates.composite_of_product (factor := 3) (cofactor := 200222191) rfl (by decide) (by decide)
    exact decide_eq_false composite
  second := false
  second_correct := by
    have composite : ¬ Nat.Prime (corner 12254 2) := by change ¬ Nat.Prime 600642065; exact PrattCertificates.composite_of_product (factor := 5) (cofactor := 120128413) rfl (by decide) (by decide)
    exact decide_eq_false composite
  third := false
  third_correct := by
    have composite : ¬ Nat.Prime (corner 12254 3) := by change ¬ Nat.Prime 600617557; exact PrattCertificates.composite_of_product (factor := 619) (cofactor := 970303) rfl (by decide) (by decide)
    exact decide_eq_false composite
theorem row_count_12254 : row 12254 = 0 := (row_12254.checked).trans rfl
theorem model_count_12254 : modelRow 12254 = 0 := by decide
theorem modelrow_agrees_12254 : modelRow 12254 = row 12254 := model_count_12254.trans row_count_12254.symm
def row_12255 : RowCertificate 12255 where
  first := true
  first_correct := by
    have prime : Nat.Prime (corner 12255 1) := by change Nat.Prime 600764611; exact ProjectEulerSolutions.P58PrimeNodes.p_600764611.prime
    exact decide_eq_true prime
  second := false
  second_correct := by
    have composite : ¬ Nat.Prime (corner 12255 2) := by change ¬ Nat.Prime 600740101; exact PrattCertificates.composite_of_product (factor := 13) (cofactor := 46210777) rfl (by decide) (by decide)
    exact decide_eq_false composite
  third := false
  third_correct := by
    have composite : ¬ Nat.Prime (corner 12255 3) := by change ¬ Nat.Prime 600715591; exact PrattCertificates.composite_of_product (factor := 7) (cofactor := 85816513) rfl (by decide) (by decide)
    exact decide_eq_false composite
theorem row_count_12255 : row 12255 = 1 := (row_12255.checked).trans rfl
theorem model_count_12255 : modelRow 12255 = 1 := by decide
theorem modelrow_agrees_12255 : modelRow 12255 = row 12255 := model_count_12255.trans row_count_12255.symm
def row_12256 : RowCertificate 12256 where
  first := false
  first_correct := by
    have composite : ¬ Nat.Prime (corner 12256 1) := by change ¬ Nat.Prime 600862657; exact PrattCertificates.composite_of_product (factor := 313) (cofactor := 1919689) rfl (by decide) (by decide)
    exact decide_eq_false composite
  second := false
  second_correct := by
    have composite : ¬ Nat.Prime (corner 12256 2) := by change ¬ Nat.Prime 600838145; exact PrattCertificates.composite_of_product (factor := 5) (cofactor := 120167629) rfl (by decide) (by decide)
    exact decide_eq_false composite
  third := false
  third_correct := by
    have composite : ¬ Nat.Prime (corner 12256 3) := by change ¬ Nat.Prime 600813633; exact PrattCertificates.composite_of_product (factor := 3) (cofactor := 200271211) rfl (by decide) (by decide)
    exact decide_eq_false composite
theorem row_count_12256 : row 12256 = 0 := (row_12256.checked).trans rfl
theorem model_count_12256 : modelRow 12256 = 0 := by decide
theorem modelrow_agrees_12256 : modelRow 12256 = row 12256 := model_count_12256.trans row_count_12256.symm
def row_12257 : RowCertificate 12257 where
  first := false
  first_correct := by
    have composite : ¬ Nat.Prime (corner 12257 1) := by change ¬ Nat.Prime 600960711; exact PrattCertificates.composite_of_product (factor := 3) (cofactor := 200320237) rfl (by decide) (by decide)
    exact decide_eq_false composite
  second := false
  second_correct := by
    have composite : ¬ Nat.Prime (corner 12257 2) := by change ¬ Nat.Prime 600936197; exact PrattCertificates.composite_of_product (factor := 373) (cofactor := 1611089) rfl (by decide) (by decide)
    exact decide_eq_false composite
  third := false
  third_correct := by
    have composite : ¬ Nat.Prime (corner 12257 3) := by change ¬ Nat.Prime 600911683; exact PrattCertificates.composite_of_product (factor := 79) (cofactor := 7606477) rfl (by decide) (by decide)
    exact decide_eq_false composite
theorem row_count_12257 : row 12257 = 0 := (row_12257.checked).trans rfl
theorem model_count_12257 : modelRow 12257 = 0 := by decide
theorem modelrow_agrees_12257 : modelRow 12257 = row 12257 := model_count_12257.trans row_count_12257.symm
def row_12258 : RowCertificate 12258 where
  first := false
  first_correct := by
    have composite : ¬ Nat.Prime (corner 12258 1) := by change ¬ Nat.Prime 601058773; exact PrattCertificates.composite_of_product (factor := 7) (cofactor := 85865539) rfl (by decide) (by decide)
    exact decide_eq_false composite
  second := false
  second_correct := by
    have composite : ¬ Nat.Prime (corner 12258 2) := by change ¬ Nat.Prime 601034257; exact PrattCertificates.composite_of_product (factor := 53) (cofactor := 11340269) rfl (by decide) (by decide)
    exact decide_eq_false composite
  third := false
  third_correct := by
    have composite : ¬ Nat.Prime (corner 12258 3) := by change ¬ Nat.Prime 601009741; exact PrattCertificates.composite_of_product (factor := 31) (cofactor := 19387411) rfl (by decide) (by decide)
    exact decide_eq_false composite
theorem row_count_12258 : row 12258 = 0 := (row_12258.checked).trans rfl
theorem model_count_12258 : modelRow 12258 = 0 := by decide
theorem modelrow_agrees_12258 : modelRow 12258 = row 12258 := model_count_12258.trans row_count_12258.symm
def row_12259 : RowCertificate 12259 where
  first := false
  first_correct := by
    have composite : ¬ Nat.Prime (corner 12259 1) := by change ¬ Nat.Prime 601156843; exact PrattCertificates.composite_of_product (factor := 7) (cofactor := 85879549) rfl (by decide) (by decide)
    exact decide_eq_false composite
  second := false
  second_correct := by
    have composite : ¬ Nat.Prime (corner 12259 2) := by change ¬ Nat.Prime 601132325; exact PrattCertificates.composite_of_product (factor := 5) (cofactor := 120226465) rfl (by decide) (by decide)
    exact decide_eq_false composite
  third := false
  third_correct := by
    have composite : ¬ Nat.Prime (corner 12259 3) := by change ¬ Nat.Prime 601107807; exact PrattCertificates.composite_of_product (factor := 3) (cofactor := 200369269) rfl (by decide) (by decide)
    exact decide_eq_false composite
theorem row_count_12259 : row 12259 = 0 := (row_12259.checked).trans rfl
theorem model_count_12259 : modelRow 12259 = 0 := by decide
theorem modelrow_agrees_12259 : modelRow 12259 = row 12259 := model_count_12259.trans row_count_12259.symm
def row_12260 : RowCertificate 12260 where
  first := false
  first_correct := by
    have composite : ¬ Nat.Prime (corner 12260 1) := by change ¬ Nat.Prime 601254921; exact PrattCertificates.composite_of_product (factor := 3) (cofactor := 200418307) rfl (by decide) (by decide)
    exact decide_eq_false composite
  second := false
  second_correct := by
    have composite : ¬ Nat.Prime (corner 12260 2) := by change ¬ Nat.Prime 601230401; exact PrattCertificates.composite_of_product (factor := 157) (cofactor := 3829493) rfl (by decide) (by decide)
    exact decide_eq_false composite
  third := false
  third_correct := by
    have composite : ¬ Nat.Prime (corner 12260 3) := by change ¬ Nat.Prime 601205881; exact PrattCertificates.composite_of_product (factor := 73) (cofactor := 8235697) rfl (by decide) (by decide)
    exact decide_eq_false composite
theorem row_count_12260 : row 12260 = 0 := (row_12260.checked).trans rfl
theorem model_count_12260 : modelRow 12260 = 0 := by decide
theorem modelrow_agrees_12260 : modelRow 12260 = row 12260 := model_count_12260.trans row_count_12260.symm
def row_12261 : RowCertificate 12261 where
  first := true
  first_correct := by
    have prime : Nat.Prime (corner 12261 1) := by change Nat.Prime 601353007; exact ProjectEulerSolutions.P58PrimeNodes.p_601353007.prime
    exact decide_eq_true prime
  second := false
  second_correct := by
    have composite : ¬ Nat.Prime (corner 12261 2) := by change ¬ Nat.Prime 601328485; exact PrattCertificates.composite_of_product (factor := 5) (cofactor := 120265697) rfl (by decide) (by decide)
    exact decide_eq_false composite
  third := false
  third_correct := by
    have composite : ¬ Nat.Prime (corner 12261 3) := by change ¬ Nat.Prime 601303963; exact PrattCertificates.composite_of_product (factor := 13) (cofactor := 46254151) rfl (by decide) (by decide)
    exact decide_eq_false composite
theorem row_count_12261 : row 12261 = 1 := (row_12261.checked).trans rfl
theorem model_count_12261 : modelRow 12261 = 1 := by decide
theorem modelrow_agrees_12261 : modelRow 12261 = row 12261 := model_count_12261.trans row_count_12261.symm
def row_12262 : RowCertificate 12262 where
  first := true
  first_correct := by
    have prime : Nat.Prime (corner 12262 1) := by change Nat.Prime 601451101; exact ProjectEulerSolutions.P58PrimeNodes.p_601451101.prime
    exact decide_eq_true prime
  second := true
  second_correct := by
    have prime : Nat.Prime (corner 12262 2) := by change Nat.Prime 601426577; exact ProjectEulerSolutions.P58PrimeNodes.p_601426577.prime
    exact decide_eq_true prime
  third := false
  third_correct := by
    have composite : ¬ Nat.Prime (corner 12262 3) := by change ¬ Nat.Prime 601402053; exact PrattCertificates.composite_of_product (factor := 3) (cofactor := 200467351) rfl (by decide) (by decide)
    exact decide_eq_false composite
theorem row_count_12262 : row 12262 = 2 := (row_12262.checked).trans rfl
theorem model_count_12262 : modelRow 12262 = 2 := by decide
theorem modelrow_agrees_12262 : modelRow 12262 = row 12262 := model_count_12262.trans row_count_12262.symm
def row_12263 : RowCertificate 12263 where
  first := false
  first_correct := by
    have composite : ¬ Nat.Prime (corner 12263 1) := by change ¬ Nat.Prime 601549203; exact PrattCertificates.composite_of_product (factor := 3) (cofactor := 200516401) rfl (by decide) (by decide)
    exact decide_eq_false composite
  second := false
  second_correct := by
    have composite : ¬ Nat.Prime (corner 12263 2) := by change ¬ Nat.Prime 601524677; exact PrattCertificates.composite_of_product (factor := 13) (cofactor := 46271129) rfl (by decide) (by decide)
    exact decide_eq_false composite
  third := false
  third_correct := by
    have composite : ¬ Nat.Prime (corner 12263 3) := by change ¬ Nat.Prime 601500151; exact PrattCertificates.composite_of_product (factor := 7) (cofactor := 85928593) rfl (by decide) (by decide)
    exact decide_eq_false composite
theorem row_count_12263 : row 12263 = 0 := (row_12263.checked).trans rfl
theorem model_count_12263 : modelRow 12263 = 0 := by decide
theorem modelrow_agrees_12263 : modelRow 12263 = row 12263 := model_count_12263.trans row_count_12263.symm
def row_12264 : RowCertificate 12264 where
  first := true
  first_correct := by
    have prime : Nat.Prime (corner 12264 1) := by change Nat.Prime 601647313; exact ProjectEulerSolutions.P58PrimeNodes.p_601647313.prime
    exact decide_eq_true prime
  second := false
  second_correct := by
    have composite : ¬ Nat.Prime (corner 12264 2) := by change ¬ Nat.Prime 601622785; exact PrattCertificates.composite_of_product (factor := 5) (cofactor := 120324557) rfl (by decide) (by decide)
    exact decide_eq_false composite
  third := false
  third_correct := by
    have composite : ¬ Nat.Prime (corner 12264 3) := by change ¬ Nat.Prime 601598257; exact PrattCertificates.composite_of_product (factor := 13) (cofactor := 46276789) rfl (by decide) (by decide)
    exact decide_eq_false composite
theorem row_count_12264 : row 12264 = 1 := (row_12264.checked).trans rfl
theorem model_count_12264 : modelRow 12264 = 1 := by decide
theorem modelrow_agrees_12264 : modelRow 12264 = row 12264 := model_count_12264.trans row_count_12264.symm
def row_12265 : RowCertificate 12265 where
  first := false
  first_correct := by
    have composite : ¬ Nat.Prime (corner 12265 1) := by change ¬ Nat.Prime 601745431; exact PrattCertificates.composite_of_product (factor := 7) (cofactor := 85963633) rfl (by decide) (by decide)
    exact decide_eq_false composite
  second := false
  second_correct := by
    have composite : ¬ Nat.Prime (corner 12265 2) := by change ¬ Nat.Prime 601720901; exact PrattCertificates.composite_of_product (factor := 89) (cofactor := 6760909) rfl (by decide) (by decide)
    exact decide_eq_false composite
  third := false
  third_correct := by
    have composite : ¬ Nat.Prime (corner 12265 3) := by change ¬ Nat.Prime 601696371; exact PrattCertificates.composite_of_product (factor := 3) (cofactor := 200565457) rfl (by decide) (by decide)
    exact decide_eq_false composite
theorem row_count_12265 : row 12265 = 0 := (row_12265.checked).trans rfl
theorem model_count_12265 : modelRow 12265 = 0 := by decide
theorem modelrow_agrees_12265 : modelRow 12265 = row 12265 := model_count_12265.trans row_count_12265.symm
def row_12266 : RowCertificate 12266 where
  first := false
  first_correct := by
    have composite : ¬ Nat.Prime (corner 12266 1) := by change ¬ Nat.Prime 601843557; exact PrattCertificates.composite_of_product (factor := 3) (cofactor := 200614519) rfl (by decide) (by decide)
    exact decide_eq_false composite
  second := false
  second_correct := by
    have composite : ¬ Nat.Prime (corner 12266 2) := by change ¬ Nat.Prime 601819025; exact PrattCertificates.composite_of_product (factor := 5) (cofactor := 120363805) rfl (by decide) (by decide)
    exact decide_eq_false composite
  third := true
  third_correct := by
    have prime : Nat.Prime (corner 12266 3) := by change Nat.Prime 601794493; exact ProjectEulerSolutions.P58PrimeNodes.p_601794493.prime
    exact decide_eq_true prime
theorem row_count_12266 : row 12266 = 1 := (row_12266.checked).trans rfl
theorem model_count_12266 : modelRow 12266 = 1 := by decide
theorem modelrow_agrees_12266 : modelRow 12266 = row 12266 := model_count_12266.trans row_count_12266.symm
def row_12267 : RowCertificate 12267 where
  first := false
  first_correct := by
    have composite : ¬ Nat.Prime (corner 12267 1) := by change ¬ Nat.Prime 601941691; exact PrattCertificates.composite_of_product (factor := 13) (cofactor := 46303207) rfl (by decide) (by decide)
    exact decide_eq_false composite
  second := true
  second_correct := by
    have prime : Nat.Prime (corner 12267 2) := by change Nat.Prime 601917157; exact ProjectEulerSolutions.P58PrimeNodes.p_601917157.prime
    exact decide_eq_true prime
  third := true
  third_correct := by
    have prime : Nat.Prime (corner 12267 3) := by change Nat.Prime 601892623; exact ProjectEulerSolutions.P58PrimeNodes.p_601892623.prime
    exact decide_eq_true prime
theorem row_count_12267 : row 12267 = 2 := (row_12267.checked).trans rfl
theorem model_count_12267 : modelRow 12267 = 2 := by decide
theorem modelrow_agrees_12267 : modelRow 12267 = row 12267 := model_count_12267.trans row_count_12267.symm
def row_12268 : RowCertificate 12268 where
  first := false
  first_correct := by
    have composite : ¬ Nat.Prime (corner 12268 1) := by change ¬ Nat.Prime 602039833; exact PrattCertificates.composite_of_product (factor := 19) (cofactor := 31686307) rfl (by decide) (by decide)
    exact decide_eq_false composite
  second := false
  second_correct := by
    have composite : ¬ Nat.Prime (corner 12268 2) := by change ¬ Nat.Prime 602015297; exact PrattCertificates.composite_of_product (factor := 13) (cofactor := 46308869) rfl (by decide) (by decide)
    exact decide_eq_false composite
  third := false
  third_correct := by
    have composite : ¬ Nat.Prime (corner 12268 3) := by change ¬ Nat.Prime 601990761; exact PrattCertificates.composite_of_product (factor := 3) (cofactor := 200663587) rfl (by decide) (by decide)
    exact decide_eq_false composite
theorem row_count_12268 : row 12268 = 0 := (row_12268.checked).trans rfl
theorem model_count_12268 : modelRow 12268 = 0 := by decide
theorem modelrow_agrees_12268 : modelRow 12268 = row 12268 := model_count_12268.trans row_count_12268.symm
def row_12269 : RowCertificate 12269 where
  first := false
  first_correct := by
    have composite : ¬ Nat.Prime (corner 12269 1) := by change ¬ Nat.Prime 602137983; exact PrattCertificates.composite_of_product (factor := 3) (cofactor := 200712661) rfl (by decide) (by decide)
    exact decide_eq_false composite
  second := false
  second_correct := by
    have composite : ¬ Nat.Prime (corner 12269 2) := by change ¬ Nat.Prime 602113445; exact PrattCertificates.composite_of_product (factor := 5) (cofactor := 120422689) rfl (by decide) (by decide)
    exact decide_eq_false composite
  third := false
  third_correct := by
    have composite : ¬ Nat.Prime (corner 12269 3) := by change ¬ Nat.Prime 602088907; exact PrattCertificates.composite_of_product (factor := 7) (cofactor := 86012701) rfl (by decide) (by decide)
    exact decide_eq_false composite
theorem row_count_12269 : row 12269 = 0 := (row_12269.checked).trans rfl
theorem model_count_12269 : modelRow 12269 = 0 := by decide
theorem modelrow_agrees_12269 : modelRow 12269 = row 12269 := model_count_12269.trans row_count_12269.symm
def row_12270 : RowCertificate 12270 where
  first := false
  first_correct := by
    have composite : ¬ Nat.Prime (corner 12270 1) := by change ¬ Nat.Prime 602236141; exact PrattCertificates.composite_of_product (factor := 13) (cofactor := 46325857) rfl (by decide) (by decide)
    exact decide_eq_false composite
  second := false
  second_correct := by
    have composite : ¬ Nat.Prime (corner 12270 2) := by change ¬ Nat.Prime 602211601; exact PrattCertificates.composite_of_product (factor := 7393) (cofactor := 81457) rfl (by decide) (by decide)
    exact decide_eq_false composite
  third := false
  third_correct := by
    have composite : ¬ Nat.Prime (corner 12270 3) := by change ¬ Nat.Prime 602187061; exact PrattCertificates.composite_of_product (factor := 7) (cofactor := 86026723) rfl (by decide) (by decide)
    exact decide_eq_false composite
theorem row_count_12270 : row 12270 = 0 := (row_12270.checked).trans rfl
theorem model_count_12270 : modelRow 12270 = 0 := by decide
theorem modelrow_agrees_12270 : modelRow 12270 = row 12270 := model_count_12270.trans row_count_12270.symm
def row_12271 : RowCertificate 12271 where
  first := false
  first_correct := by
    have composite : ¬ Nat.Prime (corner 12271 1) := by change ¬ Nat.Prime 602334307; exact PrattCertificates.composite_of_product (factor := 307) (cofactor := 1962001) rfl (by decide) (by decide)
    exact decide_eq_false composite
  second := false
  second_correct := by
    have composite : ¬ Nat.Prime (corner 12271 2) := by change ¬ Nat.Prime 602309765; exact PrattCertificates.composite_of_product (factor := 5) (cofactor := 120461953) rfl (by decide) (by decide)
    exact decide_eq_false composite
  third := false
  third_correct := by
    have composite : ¬ Nat.Prime (corner 12271 3) := by change ¬ Nat.Prime 602285223; exact PrattCertificates.composite_of_product (factor := 3) (cofactor := 200761741) rfl (by decide) (by decide)
    exact decide_eq_false composite
theorem row_count_12271 : row 12271 = 0 := (row_12271.checked).trans rfl
theorem model_count_12271 : modelRow 12271 = 0 := by decide
theorem modelrow_agrees_12271 : modelRow 12271 = row 12271 := model_count_12271.trans row_count_12271.symm
def row_12272 : RowCertificate 12272 where
  first := false
  first_correct := by
    have composite : ¬ Nat.Prime (corner 12272 1) := by change ¬ Nat.Prime 602432481; exact PrattCertificates.composite_of_product (factor := 3) (cofactor := 200810827) rfl (by decide) (by decide)
    exact decide_eq_false composite
  second := false
  second_correct := by
    have composite : ¬ Nat.Prime (corner 12272 2) := by change ¬ Nat.Prime 602407937; exact PrattCertificates.composite_of_product (factor := 17) (cofactor := 35435761) rfl (by decide) (by decide)
    exact decide_eq_false composite
  third := true
  third_correct := by
    have prime : Nat.Prime (corner 12272 3) := by change Nat.Prime 602383393; exact ProjectEulerSolutions.P58PrimeNodes.p_602383393.prime
    exact decide_eq_true prime
theorem row_count_12272 : row 12272 = 1 := (row_12272.checked).trans rfl
theorem model_count_12272 : modelRow 12272 = 1 := by decide
theorem modelrow_agrees_12272 : modelRow 12272 = row 12272 := model_count_12272.trans row_count_12272.symm
def row_12273 : RowCertificate 12273 where
  first := false
  first_correct := by
    have composite : ¬ Nat.Prime (corner 12273 1) := by change ¬ Nat.Prime 602530663; exact PrattCertificates.composite_of_product (factor := 7) (cofactor := 86075809) rfl (by decide) (by decide)
    exact decide_eq_false composite
  second := false
  second_correct := by
    have composite : ¬ Nat.Prime (corner 12273 2) := by change ¬ Nat.Prime 602506117; exact PrattCertificates.composite_of_product (factor := 29) (cofactor := 20776073) rfl (by decide) (by decide)
    exact decide_eq_false composite
  third := false
  third_correct := by
    have composite : ¬ Nat.Prime (corner 12273 3) := by change ¬ Nat.Prime 602481571; exact PrattCertificates.composite_of_product (factor := 79) (cofactor := 7626349) rfl (by decide) (by decide)
    exact decide_eq_false composite
theorem row_count_12273 : row 12273 = 0 := (row_12273.checked).trans rfl
theorem model_count_12273 : modelRow 12273 = 0 := by decide
theorem modelrow_agrees_12273 : modelRow 12273 = row 12273 := model_count_12273.trans row_count_12273.symm
def row_12274 : RowCertificate 12274 where
  first := false
  first_correct := by
    have composite : ¬ Nat.Prime (corner 12274 1) := by change ¬ Nat.Prime 602628853; exact PrattCertificates.composite_of_product (factor := 8263) (cofactor := 72931) rfl (by decide) (by decide)
    exact decide_eq_false composite
  second := false
  second_correct := by
    have composite : ¬ Nat.Prime (corner 12274 2) := by change ¬ Nat.Prime 602604305; exact PrattCertificates.composite_of_product (factor := 5) (cofactor := 120520861) rfl (by decide) (by decide)
    exact decide_eq_false composite
  third := false
  third_correct := by
    have composite : ¬ Nat.Prime (corner 12274 3) := by change ¬ Nat.Prime 602579757; exact PrattCertificates.composite_of_product (factor := 3) (cofactor := 200859919) rfl (by decide) (by decide)
    exact decide_eq_false composite
theorem row_count_12274 : row 12274 = 0 := (row_12274.checked).trans rfl
theorem model_count_12274 : modelRow 12274 = 0 := by decide
theorem modelrow_agrees_12274 : modelRow 12274 = row 12274 := model_count_12274.trans row_count_12274.symm
def row_12275 : RowCertificate 12275 where
  first := false
  first_correct := by
    have composite : ¬ Nat.Prime (corner 12275 1) := by change ¬ Nat.Prime 602727051; exact PrattCertificates.composite_of_product (factor := 3) (cofactor := 200909017) rfl (by decide) (by decide)
    exact decide_eq_false composite
  second := false
  second_correct := by
    have composite : ¬ Nat.Prime (corner 12275 2) := by change ¬ Nat.Prime 602702501; exact PrattCertificates.composite_of_product (factor := 41) (cofactor := 14700061) rfl (by decide) (by decide)
    exact decide_eq_false composite
  third := false
  third_correct := by
    have composite : ¬ Nat.Prime (corner 12275 3) := by change ¬ Nat.Prime 602677951; exact PrattCertificates.composite_of_product (factor := 8761) (cofactor := 68791) rfl (by decide) (by decide)
    exact decide_eq_false composite
theorem row_count_12275 : row 12275 = 0 := (row_12275.checked).trans rfl
theorem model_count_12275 : modelRow 12275 = 0 := by decide
theorem modelrow_agrees_12275 : modelRow 12275 = row 12275 := model_count_12275.trans row_count_12275.symm
def row_12276 : RowCertificate 12276 where
  first := false
  first_correct := by
    have composite : ¬ Nat.Prime (corner 12276 1) := by change ¬ Nat.Prime 602825257; exact PrattCertificates.composite_of_product (factor := 349) (cofactor := 1727293) rfl (by decide) (by decide)
    exact decide_eq_false composite
  second := false
  second_correct := by
    have composite : ¬ Nat.Prime (corner 12276 2) := by change ¬ Nat.Prime 602800705; exact PrattCertificates.composite_of_product (factor := 5) (cofactor := 120560141) rfl (by decide) (by decide)
    exact decide_eq_false composite
  third := false
  third_correct := by
    have composite : ¬ Nat.Prime (corner 12276 3) := by change ¬ Nat.Prime 602776153; exact PrattCertificates.composite_of_product (factor := 7) (cofactor := 86110879) rfl (by decide) (by decide)
    exact decide_eq_false composite
theorem row_count_12276 : row 12276 = 0 := (row_12276.checked).trans rfl
theorem model_count_12276 : modelRow 12276 = 0 := by decide
theorem modelrow_agrees_12276 : modelRow 12276 = row 12276 := model_count_12276.trans row_count_12276.symm
def row_12277 : RowCertificate 12277 where
  first := false
  first_correct := by
    have composite : ¬ Nat.Prime (corner 12277 1) := by change ¬ Nat.Prime 602923471; exact PrattCertificates.composite_of_product (factor := 163) (cofactor := 3698917) rfl (by decide) (by decide)
    exact decide_eq_false composite
  second := false
  second_correct := by
    have composite : ¬ Nat.Prime (corner 12277 2) := by change ¬ Nat.Prime 602898917; exact PrattCertificates.composite_of_product (factor := 233) (cofactor := 2587549) rfl (by decide) (by decide)
    exact decide_eq_false composite
  third := false
  third_correct := by
    have composite : ¬ Nat.Prime (corner 12277 3) := by change ¬ Nat.Prime 602874363; exact PrattCertificates.composite_of_product (factor := 3) (cofactor := 200958121) rfl (by decide) (by decide)
    exact decide_eq_false composite
theorem row_count_12277 : row 12277 = 0 := (row_12277.checked).trans rfl
theorem model_count_12277 : modelRow 12277 = 0 := by decide
theorem modelrow_agrees_12277 : modelRow 12277 = row 12277 := model_count_12277.trans row_count_12277.symm
def row_12278 : RowCertificate 12278 where
  first := false
  first_correct := by
    have composite : ¬ Nat.Prime (corner 12278 1) := by change ¬ Nat.Prime 603021693; exact PrattCertificates.composite_of_product (factor := 3) (cofactor := 201007231) rfl (by decide) (by decide)
    exact decide_eq_false composite
  second := true
  second_correct := by
    have prime : Nat.Prime (corner 12278 2) := by change Nat.Prime 602997137; exact ProjectEulerSolutions.P58PrimeNodes.p_602997137.prime
    exact decide_eq_true prime
  third := false
  third_correct := by
    have composite : ¬ Nat.Prime (corner 12278 3) := by change ¬ Nat.Prime 602972581; exact PrattCertificates.composite_of_product (factor := 19) (cofactor := 31735399) rfl (by decide) (by decide)
    exact decide_eq_false composite
theorem row_count_12278 : row 12278 = 1 := (row_12278.checked).trans rfl
theorem model_count_12278 : modelRow 12278 = 1 := by decide
theorem modelrow_agrees_12278 : modelRow 12278 = row 12278 := model_count_12278.trans row_count_12278.symm
def row_12279 : RowCertificate 12279 where
  first := false
  first_correct := by
    have composite : ¬ Nat.Prime (corner 12279 1) := by change ¬ Nat.Prime 603119923; exact PrattCertificates.composite_of_product (factor := 7) (cofactor := 86159989) rfl (by decide) (by decide)
    exact decide_eq_false composite
  second := false
  second_correct := by
    have composite : ¬ Nat.Prime (corner 12279 2) := by change ¬ Nat.Prime 603095365; exact PrattCertificates.composite_of_product (factor := 5) (cofactor := 120619073) rfl (by decide) (by decide)
    exact decide_eq_false composite
  third := false
  third_correct := by
    have composite : ¬ Nat.Prime (corner 12279 3) := by change ¬ Nat.Prime 603070807; exact PrattCertificates.composite_of_product (factor := 31) (cofactor := 19453897) rfl (by decide) (by decide)
    exact decide_eq_false composite
theorem row_count_12279 : row 12279 = 0 := (row_12279.checked).trans rfl
theorem model_count_12279 : modelRow 12279 = 0 := by decide
theorem modelrow_agrees_12279 : modelRow 12279 = row 12279 := model_count_12279.trans row_count_12279.symm
def row_12280 : RowCertificate 12280 where
  first := false
  first_correct := by
    have composite : ¬ Nat.Prime (corner 12280 1) := by change ¬ Nat.Prime 603218161; exact PrattCertificates.composite_of_product (factor := 7) (cofactor := 86174023) rfl (by decide) (by decide)
    exact decide_eq_false composite
  second := false
  second_correct := by
    have composite : ¬ Nat.Prime (corner 12280 2) := by change ¬ Nat.Prime 603193601; exact PrattCertificates.composite_of_product (factor := 137) (cofactor := 4402873) rfl (by decide) (by decide)
    exact decide_eq_false composite
  third := false
  third_correct := by
    have composite : ¬ Nat.Prime (corner 12280 3) := by change ¬ Nat.Prime 603169041; exact PrattCertificates.composite_of_product (factor := 3) (cofactor := 201056347) rfl (by decide) (by decide)
    exact decide_eq_false composite
theorem row_count_12280 : row 12280 = 0 := (row_12280.checked).trans rfl
theorem model_count_12280 : modelRow 12280 = 0 := by decide
theorem modelrow_agrees_12280 : modelRow 12280 = row 12280 := model_count_12280.trans row_count_12280.symm
def row_12281 : RowCertificate 12281 where
  first := false
  first_correct := by
    have composite : ¬ Nat.Prime (corner 12281 1) := by change ¬ Nat.Prime 603316407; exact PrattCertificates.composite_of_product (factor := 3) (cofactor := 201105469) rfl (by decide) (by decide)
    exact decide_eq_false composite
  second := false
  second_correct := by
    have composite : ¬ Nat.Prime (corner 12281 2) := by change ¬ Nat.Prime 603291845; exact PrattCertificates.composite_of_product (factor := 5) (cofactor := 120658369) rfl (by decide) (by decide)
    exact decide_eq_false composite
  third := true
  third_correct := by
    have prime : Nat.Prime (corner 12281 3) := by change Nat.Prime 603267283; exact ProjectEulerSolutions.P58PrimeNodes.p_603267283.prime
    exact decide_eq_true prime
theorem row_count_12281 : row 12281 = 1 := (row_12281.checked).trans rfl
theorem model_count_12281 : modelRow 12281 = 1 := by decide
theorem modelrow_agrees_12281 : modelRow 12281 = row 12281 := model_count_12281.trans row_count_12281.symm
def row_12282 : RowCertificate 12282 where
  first := false
  first_correct := by
    have composite : ¬ Nat.Prime (corner 12282 1) := by change ¬ Nat.Prime 603414661; exact PrattCertificates.composite_of_product (factor := 10513) (cofactor := 57397) rfl (by decide) (by decide)
    exact decide_eq_false composite
  second := true
  second_correct := by
    have prime : Nat.Prime (corner 12282 2) := by change Nat.Prime 603390097; exact ProjectEulerSolutions.P58PrimeNodes.p_603390097.prime
    exact decide_eq_true prime
  third := false
  third_correct := by
    have composite : ¬ Nat.Prime (corner 12282 3) := by change ¬ Nat.Prime 603365533; exact PrattCertificates.composite_of_product (factor := 7417) (cofactor := 81349) rfl (by decide) (by decide)
    exact decide_eq_false composite
theorem row_count_12282 : row 12282 = 1 := (row_12282.checked).trans rfl
theorem model_count_12282 : modelRow 12282 = 1 := by decide
theorem modelrow_agrees_12282 : modelRow 12282 = row 12282 := model_count_12282.trans row_count_12282.symm
def row_12283 : RowCertificate 12283 where
  first := false
  first_correct := by
    have composite : ¬ Nat.Prime (corner 12283 1) := by change ¬ Nat.Prime 603512923; exact PrattCertificates.composite_of_product (factor := 13) (cofactor := 46424071) rfl (by decide) (by decide)
    exact decide_eq_false composite
  second := false
  second_correct := by
    have composite : ¬ Nat.Prime (corner 12283 2) := by change ¬ Nat.Prime 603488357; exact PrattCertificates.composite_of_product (factor := 349) (cofactor := 1729193) rfl (by decide) (by decide)
    exact decide_eq_false composite
  third := false
  third_correct := by
    have composite : ¬ Nat.Prime (corner 12283 3) := by change ¬ Nat.Prime 603463791; exact PrattCertificates.composite_of_product (factor := 3) (cofactor := 201154597) rfl (by decide) (by decide)
    exact decide_eq_false composite
theorem row_count_12283 : row 12283 = 0 := (row_12283.checked).trans rfl
theorem model_count_12283 : modelRow 12283 = 0 := by decide
theorem modelrow_agrees_12283 : modelRow 12283 = row 12283 := model_count_12283.trans row_count_12283.symm
def row_12284 : RowCertificate 12284 where
  first := false
  first_correct := by
    have composite : ¬ Nat.Prime (corner 12284 1) := by change ¬ Nat.Prime 603611193; exact PrattCertificates.composite_of_product (factor := 3) (cofactor := 201203731) rfl (by decide) (by decide)
    exact decide_eq_false composite
  second := false
  second_correct := by
    have composite : ¬ Nat.Prime (corner 12284 2) := by change ¬ Nat.Prime 603586625; exact PrattCertificates.composite_of_product (factor := 5) (cofactor := 120717325) rfl (by decide) (by decide)
    exact decide_eq_false composite
  third := false
  third_correct := by
    have composite : ¬ Nat.Prime (corner 12284 3) := by change ¬ Nat.Prime 603562057; exact PrattCertificates.composite_of_product (factor := 7) (cofactor := 86223151) rfl (by decide) (by decide)
    exact decide_eq_false composite
theorem row_count_12284 : row 12284 = 0 := (row_12284.checked).trans rfl
theorem model_count_12284 : modelRow 12284 = 0 := by decide
theorem modelrow_agrees_12284 : modelRow 12284 = row 12284 := model_count_12284.trans row_count_12284.symm
def row_12285 : RowCertificate 12285 where
  first := false
  first_correct := by
    have composite : ¬ Nat.Prime (corner 12285 1) := by change ¬ Nat.Prime 603709471; exact PrattCertificates.composite_of_product (factor := 103) (cofactor := 5861257) rfl (by decide) (by decide)
    exact decide_eq_false composite
  second := false
  second_correct := by
    have composite : ¬ Nat.Prime (corner 12285 2) := by change ¬ Nat.Prime 603684901; exact PrattCertificates.composite_of_product (factor := 2797) (cofactor := 215833) rfl (by decide) (by decide)
    exact decide_eq_false composite
  third := false
  third_correct := by
    have composite : ¬ Nat.Prime (corner 12285 3) := by change ¬ Nat.Prime 603660331; exact PrattCertificates.composite_of_product (factor := 61) (cofactor := 9896071) rfl (by decide) (by decide)
    exact decide_eq_false composite
theorem row_count_12285 : row 12285 = 0 := (row_12285.checked).trans rfl
theorem model_count_12285 : modelRow 12285 = 0 := by decide
theorem modelrow_agrees_12285 : modelRow 12285 = row 12285 := model_count_12285.trans row_count_12285.symm
def row_12286 : RowCertificate 12286 where
  first := false
  first_correct := by
    have composite : ¬ Nat.Prime (corner 12286 1) := by change ¬ Nat.Prime 603807757; exact PrattCertificates.composite_of_product (factor := 7) (cofactor := 86258251) rfl (by decide) (by decide)
    exact decide_eq_false composite
  second := false
  second_correct := by
    have composite : ¬ Nat.Prime (corner 12286 2) := by change ¬ Nat.Prime 603783185; exact PrattCertificates.composite_of_product (factor := 5) (cofactor := 120756637) rfl (by decide) (by decide)
    exact decide_eq_false composite
  third := false
  third_correct := by
    have composite : ¬ Nat.Prime (corner 12286 3) := by change ¬ Nat.Prime 603758613; exact PrattCertificates.composite_of_product (factor := 3) (cofactor := 201252871) rfl (by decide) (by decide)
    exact decide_eq_false composite
theorem row_count_12286 : row 12286 = 0 := (row_12286.checked).trans rfl
theorem model_count_12286 : modelRow 12286 = 0 := by decide
theorem modelrow_agrees_12286 : modelRow 12286 = row 12286 := model_count_12286.trans row_count_12286.symm
def row_12287 : RowCertificate 12287 where
  first := false
  first_correct := by
    have composite : ¬ Nat.Prime (corner 12287 1) := by change ¬ Nat.Prime 603906051; exact PrattCertificates.composite_of_product (factor := 3) (cofactor := 201302017) rfl (by decide) (by decide)
    exact decide_eq_false composite
  second := false
  second_correct := by
    have composite : ¬ Nat.Prime (corner 12287 2) := by change ¬ Nat.Prime 603881477; exact PrattCertificates.composite_of_product (factor := 37) (cofactor := 16321121) rfl (by decide) (by decide)
    exact decide_eq_false composite
  third := false
  third_correct := by
    have composite : ¬ Nat.Prime (corner 12287 3) := by change ¬ Nat.Prime 603856903; exact PrattCertificates.composite_of_product (factor := 13) (cofactor := 46450531) rfl (by decide) (by decide)
    exact decide_eq_false composite
theorem row_count_12287 : row 12287 = 0 := (row_12287.checked).trans rfl
theorem model_count_12287 : modelRow 12287 = 0 := by decide
theorem modelrow_agrees_12287 : modelRow 12287 = row 12287 := model_count_12287.trans row_count_12287.symm
def row_12288 : RowCertificate 12288 where
  first := false
  first_correct := by
    have composite : ¬ Nat.Prime (corner 12288 1) := by change ¬ Nat.Prime 604004353; exact PrattCertificates.composite_of_product (factor := 97) (cofactor := 6226849) rfl (by decide) (by decide)
    exact decide_eq_false composite
  second := false
  second_correct := by
    have composite : ¬ Nat.Prime (corner 12288 2) := by change ¬ Nat.Prime 603979777; exact PrattCertificates.composite_of_product (factor := 1093) (cofactor := 552589) rfl (by decide) (by decide)
    exact decide_eq_false composite
  third := true
  third_correct := by
    have prime : Nat.Prime (corner 12288 3) := by change Nat.Prime 603955201; exact ProjectEulerSolutions.P58PrimeNodes.p_603955201.prime
    exact decide_eq_true prime
theorem row_count_12288 : row 12288 = 1 := (row_12288.checked).trans rfl
theorem model_count_12288 : modelRow 12288 = 1 := by decide
theorem modelrow_agrees_12288 : modelRow 12288 = row 12288 := model_count_12288.trans row_count_12288.symm
private def atomic_12161 : CountSegment 1 10 13120 12160 12161 4919 4920 := CountSegment.single (first := 12160) (initial := 4919) row_count_12161 (by decide)
private theorem atomic_12161_agreement : RowAgreement 12160 12161 := RowAgreement.single (first := 12160) modelrow_agrees_12161
private def atomic_12162 : CountSegment 1 10 13120 12161 12162 4920 4921 := CountSegment.single (first := 12161) (initial := 4920) row_count_12162 (by decide)
private theorem atomic_12162_agreement : RowAgreement 12161 12162 := RowAgreement.single (first := 12161) modelrow_agrees_12162
private def atomic_12163 : CountSegment 1 10 13120 12162 12163 4921 4922 := CountSegment.single (first := 12162) (initial := 4921) row_count_12163 (by decide)
private theorem atomic_12163_agreement : RowAgreement 12162 12163 := RowAgreement.single (first := 12162) modelrow_agrees_12163
private def atomic_12164 : CountSegment 1 10 13120 12163 12164 4922 4922 := CountSegment.single (first := 12163) (initial := 4922) row_count_12164 (by decide)
private theorem atomic_12164_agreement : RowAgreement 12163 12164 := RowAgreement.single (first := 12163) modelrow_agrees_12164
private def atomic_12165 : CountSegment 1 10 13120 12164 12165 4922 4923 := CountSegment.single (first := 12164) (initial := 4922) row_count_12165 (by decide)
private theorem atomic_12165_agreement : RowAgreement 12164 12165 := RowAgreement.single (first := 12164) modelrow_agrees_12165
private def atomic_12166 : CountSegment 1 10 13120 12165 12166 4923 4923 := CountSegment.single (first := 12165) (initial := 4923) row_count_12166 (by decide)
private theorem atomic_12166_agreement : RowAgreement 12165 12166 := RowAgreement.single (first := 12165) modelrow_agrees_12166
private def atomic_12167 : CountSegment 1 10 13120 12166 12167 4923 4923 := CountSegment.single (first := 12166) (initial := 4923) row_count_12167 (by decide)
private theorem atomic_12167_agreement : RowAgreement 12166 12167 := RowAgreement.single (first := 12166) modelrow_agrees_12167
private def atomic_12168 : CountSegment 1 10 13120 12167 12168 4923 4923 := CountSegment.single (first := 12167) (initial := 4923) row_count_12168 (by decide)
private theorem atomic_12168_agreement : RowAgreement 12167 12168 := RowAgreement.single (first := 12167) modelrow_agrees_12168
private def atomic_12169 : CountSegment 1 10 13120 12168 12169 4923 4924 := CountSegment.single (first := 12168) (initial := 4923) row_count_12169 (by decide)
private theorem atomic_12169_agreement : RowAgreement 12168 12169 := RowAgreement.single (first := 12168) modelrow_agrees_12169
private def atomic_12170 : CountSegment 1 10 13120 12169 12170 4924 4924 := CountSegment.single (first := 12169) (initial := 4924) row_count_12170 (by decide)
private theorem atomic_12170_agreement : RowAgreement 12169 12170 := RowAgreement.single (first := 12169) modelrow_agrees_12170
private def atomic_12171 : CountSegment 1 10 13120 12170 12171 4924 4925 := CountSegment.single (first := 12170) (initial := 4924) row_count_12171 (by decide)
private theorem atomic_12171_agreement : RowAgreement 12170 12171 := RowAgreement.single (first := 12170) modelrow_agrees_12171
private def atomic_12172 : CountSegment 1 10 13120 12171 12172 4925 4925 := CountSegment.single (first := 12171) (initial := 4925) row_count_12172 (by decide)
private theorem atomic_12172_agreement : RowAgreement 12171 12172 := RowAgreement.single (first := 12171) modelrow_agrees_12172
private def atomic_12173 : CountSegment 1 10 13120 12172 12173 4925 4925 := CountSegment.single (first := 12172) (initial := 4925) row_count_12173 (by decide)
private theorem atomic_12173_agreement : RowAgreement 12172 12173 := RowAgreement.single (first := 12172) modelrow_agrees_12173
private def atomic_12174 : CountSegment 1 10 13120 12173 12174 4925 4925 := CountSegment.single (first := 12173) (initial := 4925) row_count_12174 (by decide)
private theorem atomic_12174_agreement : RowAgreement 12173 12174 := RowAgreement.single (first := 12173) modelrow_agrees_12174
private def atomic_12175 : CountSegment 1 10 13120 12174 12175 4925 4925 := CountSegment.single (first := 12174) (initial := 4925) row_count_12175 (by decide)
private theorem atomic_12175_agreement : RowAgreement 12174 12175 := RowAgreement.single (first := 12174) modelrow_agrees_12175
private def atomic_12176 : CountSegment 1 10 13120 12175 12176 4925 4926 := CountSegment.single (first := 12175) (initial := 4925) row_count_12176 (by decide)
private theorem atomic_12176_agreement : RowAgreement 12175 12176 := RowAgreement.single (first := 12175) modelrow_agrees_12176
private def atomic_12177 : CountSegment 1 10 13120 12176 12177 4926 4926 := CountSegment.single (first := 12176) (initial := 4926) row_count_12177 (by decide)
private theorem atomic_12177_agreement : RowAgreement 12176 12177 := RowAgreement.single (first := 12176) modelrow_agrees_12177
private def atomic_12178 : CountSegment 1 10 13120 12177 12178 4926 4927 := CountSegment.single (first := 12177) (initial := 4926) row_count_12178 (by decide)
private theorem atomic_12178_agreement : RowAgreement 12177 12178 := RowAgreement.single (first := 12177) modelrow_agrees_12178
private def atomic_12179 : CountSegment 1 10 13120 12178 12179 4927 4927 := CountSegment.single (first := 12178) (initial := 4927) row_count_12179 (by decide)
private theorem atomic_12179_agreement : RowAgreement 12178 12179 := RowAgreement.single (first := 12178) modelrow_agrees_12179
private def atomic_12180 : CountSegment 1 10 13120 12179 12180 4927 4927 := CountSegment.single (first := 12179) (initial := 4927) row_count_12180 (by decide)
private theorem atomic_12180_agreement : RowAgreement 12179 12180 := RowAgreement.single (first := 12179) modelrow_agrees_12180
private def atomic_12181 : CountSegment 1 10 13120 12180 12181 4927 4927 := CountSegment.single (first := 12180) (initial := 4927) row_count_12181 (by decide)
private theorem atomic_12181_agreement : RowAgreement 12180 12181 := RowAgreement.single (first := 12180) modelrow_agrees_12181
private def atomic_12182 : CountSegment 1 10 13120 12181 12182 4927 4929 := CountSegment.single (first := 12181) (initial := 4927) row_count_12182 (by decide)
private theorem atomic_12182_agreement : RowAgreement 12181 12182 := RowAgreement.single (first := 12181) modelrow_agrees_12182
private def atomic_12183 : CountSegment 1 10 13120 12182 12183 4929 4931 := CountSegment.single (first := 12182) (initial := 4929) row_count_12183 (by decide)
private theorem atomic_12183_agreement : RowAgreement 12182 12183 := RowAgreement.single (first := 12182) modelrow_agrees_12183
private def atomic_12184 : CountSegment 1 10 13120 12183 12184 4931 4931 := CountSegment.single (first := 12183) (initial := 4931) row_count_12184 (by decide)
private theorem atomic_12184_agreement : RowAgreement 12183 12184 := RowAgreement.single (first := 12183) modelrow_agrees_12184
private def atomic_12185 : CountSegment 1 10 13120 12184 12185 4931 4931 := CountSegment.single (first := 12184) (initial := 4931) row_count_12185 (by decide)
private theorem atomic_12185_agreement : RowAgreement 12184 12185 := RowAgreement.single (first := 12184) modelrow_agrees_12185
private def atomic_12186 : CountSegment 1 10 13120 12185 12186 4931 4931 := CountSegment.single (first := 12185) (initial := 4931) row_count_12186 (by decide)
private theorem atomic_12186_agreement : RowAgreement 12185 12186 := RowAgreement.single (first := 12185) modelrow_agrees_12186
private def atomic_12187 : CountSegment 1 10 13120 12186 12187 4931 4931 := CountSegment.single (first := 12186) (initial := 4931) row_count_12187 (by decide)
private theorem atomic_12187_agreement : RowAgreement 12186 12187 := RowAgreement.single (first := 12186) modelrow_agrees_12187
private def atomic_12188 : CountSegment 1 10 13120 12187 12188 4931 4931 := CountSegment.single (first := 12187) (initial := 4931) row_count_12188 (by decide)
private theorem atomic_12188_agreement : RowAgreement 12187 12188 := RowAgreement.single (first := 12187) modelrow_agrees_12188
private def atomic_12189 : CountSegment 1 10 13120 12188 12189 4931 4931 := CountSegment.single (first := 12188) (initial := 4931) row_count_12189 (by decide)
private theorem atomic_12189_agreement : RowAgreement 12188 12189 := RowAgreement.single (first := 12188) modelrow_agrees_12189
private def atomic_12190 : CountSegment 1 10 13120 12189 12190 4931 4932 := CountSegment.single (first := 12189) (initial := 4931) row_count_12190 (by decide)
private theorem atomic_12190_agreement : RowAgreement 12189 12190 := RowAgreement.single (first := 12189) modelrow_agrees_12190
private def atomic_12191 : CountSegment 1 10 13120 12190 12191 4932 4933 := CountSegment.single (first := 12190) (initial := 4932) row_count_12191 (by decide)
private theorem atomic_12191_agreement : RowAgreement 12190 12191 := RowAgreement.single (first := 12190) modelrow_agrees_12191
private def atomic_12192 : CountSegment 1 10 13120 12191 12192 4933 4934 := CountSegment.single (first := 12191) (initial := 4933) row_count_12192 (by decide)
private theorem atomic_12192_agreement : RowAgreement 12191 12192 := RowAgreement.single (first := 12191) modelrow_agrees_12192
private def atomic_12193 : CountSegment 1 10 13120 12192 12193 4934 4934 := CountSegment.single (first := 12192) (initial := 4934) row_count_12193 (by decide)
private theorem atomic_12193_agreement : RowAgreement 12192 12193 := RowAgreement.single (first := 12192) modelrow_agrees_12193
private def atomic_12194 : CountSegment 1 10 13120 12193 12194 4934 4934 := CountSegment.single (first := 12193) (initial := 4934) row_count_12194 (by decide)
private theorem atomic_12194_agreement : RowAgreement 12193 12194 := RowAgreement.single (first := 12193) modelrow_agrees_12194
private def atomic_12195 : CountSegment 1 10 13120 12194 12195 4934 4934 := CountSegment.single (first := 12194) (initial := 4934) row_count_12195 (by decide)
private theorem atomic_12195_agreement : RowAgreement 12194 12195 := RowAgreement.single (first := 12194) modelrow_agrees_12195
private def atomic_12196 : CountSegment 1 10 13120 12195 12196 4934 4934 := CountSegment.single (first := 12195) (initial := 4934) row_count_12196 (by decide)
private theorem atomic_12196_agreement : RowAgreement 12195 12196 := RowAgreement.single (first := 12195) modelrow_agrees_12196
private def atomic_12197 : CountSegment 1 10 13120 12196 12197 4934 4934 := CountSegment.single (first := 12196) (initial := 4934) row_count_12197 (by decide)
private theorem atomic_12197_agreement : RowAgreement 12196 12197 := RowAgreement.single (first := 12196) modelrow_agrees_12197
private def atomic_12198 : CountSegment 1 10 13120 12197 12198 4934 4935 := CountSegment.single (first := 12197) (initial := 4934) row_count_12198 (by decide)
private theorem atomic_12198_agreement : RowAgreement 12197 12198 := RowAgreement.single (first := 12197) modelrow_agrees_12198
private def atomic_12199 : CountSegment 1 10 13120 12198 12199 4935 4936 := CountSegment.single (first := 12198) (initial := 4935) row_count_12199 (by decide)
private theorem atomic_12199_agreement : RowAgreement 12198 12199 := RowAgreement.single (first := 12198) modelrow_agrees_12199
private def atomic_12200 : CountSegment 1 10 13120 12199 12200 4936 4936 := CountSegment.single (first := 12199) (initial := 4936) row_count_12200 (by decide)
private theorem atomic_12200_agreement : RowAgreement 12199 12200 := RowAgreement.single (first := 12199) modelrow_agrees_12200
private def atomic_12201 : CountSegment 1 10 13120 12200 12201 4936 4937 := CountSegment.single (first := 12200) (initial := 4936) row_count_12201 (by decide)
private theorem atomic_12201_agreement : RowAgreement 12200 12201 := RowAgreement.single (first := 12200) modelrow_agrees_12201
private def atomic_12202 : CountSegment 1 10 13120 12201 12202 4937 4937 := CountSegment.single (first := 12201) (initial := 4937) row_count_12202 (by decide)
private theorem atomic_12202_agreement : RowAgreement 12201 12202 := RowAgreement.single (first := 12201) modelrow_agrees_12202
private def atomic_12203 : CountSegment 1 10 13120 12202 12203 4937 4938 := CountSegment.single (first := 12202) (initial := 4937) row_count_12203 (by decide)
private theorem atomic_12203_agreement : RowAgreement 12202 12203 := RowAgreement.single (first := 12202) modelrow_agrees_12203
private def atomic_12204 : CountSegment 1 10 13120 12203 12204 4938 4938 := CountSegment.single (first := 12203) (initial := 4938) row_count_12204 (by decide)
private theorem atomic_12204_agreement : RowAgreement 12203 12204 := RowAgreement.single (first := 12203) modelrow_agrees_12204
private def atomic_12205 : CountSegment 1 10 13120 12204 12205 4938 4938 := CountSegment.single (first := 12204) (initial := 4938) row_count_12205 (by decide)
private theorem atomic_12205_agreement : RowAgreement 12204 12205 := RowAgreement.single (first := 12204) modelrow_agrees_12205
private def atomic_12206 : CountSegment 1 10 13120 12205 12206 4938 4938 := CountSegment.single (first := 12205) (initial := 4938) row_count_12206 (by decide)
private theorem atomic_12206_agreement : RowAgreement 12205 12206 := RowAgreement.single (first := 12205) modelrow_agrees_12206
private def atomic_12207 : CountSegment 1 10 13120 12206 12207 4938 4939 := CountSegment.single (first := 12206) (initial := 4938) row_count_12207 (by decide)
private theorem atomic_12207_agreement : RowAgreement 12206 12207 := RowAgreement.single (first := 12206) modelrow_agrees_12207
private def atomic_12208 : CountSegment 1 10 13120 12207 12208 4939 4939 := CountSegment.single (first := 12207) (initial := 4939) row_count_12208 (by decide)
private theorem atomic_12208_agreement : RowAgreement 12207 12208 := RowAgreement.single (first := 12207) modelrow_agrees_12208
private def atomic_12209 : CountSegment 1 10 13120 12208 12209 4939 4939 := CountSegment.single (first := 12208) (initial := 4939) row_count_12209 (by decide)
private theorem atomic_12209_agreement : RowAgreement 12208 12209 := RowAgreement.single (first := 12208) modelrow_agrees_12209
private def atomic_12210 : CountSegment 1 10 13120 12209 12210 4939 4939 := CountSegment.single (first := 12209) (initial := 4939) row_count_12210 (by decide)
private theorem atomic_12210_agreement : RowAgreement 12209 12210 := RowAgreement.single (first := 12209) modelrow_agrees_12210
private def atomic_12211 : CountSegment 1 10 13120 12210 12211 4939 4939 := CountSegment.single (first := 12210) (initial := 4939) row_count_12211 (by decide)
private theorem atomic_12211_agreement : RowAgreement 12210 12211 := RowAgreement.single (first := 12210) modelrow_agrees_12211
private def atomic_12212 : CountSegment 1 10 13120 12211 12212 4939 4939 := CountSegment.single (first := 12211) (initial := 4939) row_count_12212 (by decide)
private theorem atomic_12212_agreement : RowAgreement 12211 12212 := RowAgreement.single (first := 12211) modelrow_agrees_12212
private def atomic_12213 : CountSegment 1 10 13120 12212 12213 4939 4939 := CountSegment.single (first := 12212) (initial := 4939) row_count_12213 (by decide)
private theorem atomic_12213_agreement : RowAgreement 12212 12213 := RowAgreement.single (first := 12212) modelrow_agrees_12213
private def atomic_12214 : CountSegment 1 10 13120 12213 12214 4939 4939 := CountSegment.single (first := 12213) (initial := 4939) row_count_12214 (by decide)
private theorem atomic_12214_agreement : RowAgreement 12213 12214 := RowAgreement.single (first := 12213) modelrow_agrees_12214
private def atomic_12215 : CountSegment 1 10 13120 12214 12215 4939 4939 := CountSegment.single (first := 12214) (initial := 4939) row_count_12215 (by decide)
private theorem atomic_12215_agreement : RowAgreement 12214 12215 := RowAgreement.single (first := 12214) modelrow_agrees_12215
private def atomic_12216 : CountSegment 1 10 13120 12215 12216 4939 4940 := CountSegment.single (first := 12215) (initial := 4939) row_count_12216 (by decide)
private theorem atomic_12216_agreement : RowAgreement 12215 12216 := RowAgreement.single (first := 12215) modelrow_agrees_12216
private def atomic_12217 : CountSegment 1 10 13120 12216 12217 4940 4941 := CountSegment.single (first := 12216) (initial := 4940) row_count_12217 (by decide)
private theorem atomic_12217_agreement : RowAgreement 12216 12217 := RowAgreement.single (first := 12216) modelrow_agrees_12217
private def atomic_12218 : CountSegment 1 10 13120 12217 12218 4941 4942 := CountSegment.single (first := 12217) (initial := 4941) row_count_12218 (by decide)
private theorem atomic_12218_agreement : RowAgreement 12217 12218 := RowAgreement.single (first := 12217) modelrow_agrees_12218
private def atomic_12219 : CountSegment 1 10 13120 12218 12219 4942 4943 := CountSegment.single (first := 12218) (initial := 4942) row_count_12219 (by decide)
private theorem atomic_12219_agreement : RowAgreement 12218 12219 := RowAgreement.single (first := 12218) modelrow_agrees_12219
private def atomic_12220 : CountSegment 1 10 13120 12219 12220 4943 4944 := CountSegment.single (first := 12219) (initial := 4943) row_count_12220 (by decide)
private theorem atomic_12220_agreement : RowAgreement 12219 12220 := RowAgreement.single (first := 12219) modelrow_agrees_12220
private def atomic_12221 : CountSegment 1 10 13120 12220 12221 4944 4944 := CountSegment.single (first := 12220) (initial := 4944) row_count_12221 (by decide)
private theorem atomic_12221_agreement : RowAgreement 12220 12221 := RowAgreement.single (first := 12220) modelrow_agrees_12221
private def atomic_12222 : CountSegment 1 10 13120 12221 12222 4944 4945 := CountSegment.single (first := 12221) (initial := 4944) row_count_12222 (by decide)
private theorem atomic_12222_agreement : RowAgreement 12221 12222 := RowAgreement.single (first := 12221) modelrow_agrees_12222
private def atomic_12223 : CountSegment 1 10 13120 12222 12223 4945 4946 := CountSegment.single (first := 12222) (initial := 4945) row_count_12223 (by decide)
private theorem atomic_12223_agreement : RowAgreement 12222 12223 := RowAgreement.single (first := 12222) modelrow_agrees_12223
private def atomic_12224 : CountSegment 1 10 13120 12223 12224 4946 4946 := CountSegment.single (first := 12223) (initial := 4946) row_count_12224 (by decide)
private theorem atomic_12224_agreement : RowAgreement 12223 12224 := RowAgreement.single (first := 12223) modelrow_agrees_12224
private def atomic_12225 : CountSegment 1 10 13120 12224 12225 4946 4946 := CountSegment.single (first := 12224) (initial := 4946) row_count_12225 (by decide)
private theorem atomic_12225_agreement : RowAgreement 12224 12225 := RowAgreement.single (first := 12224) modelrow_agrees_12225
private def atomic_12226 : CountSegment 1 10 13120 12225 12226 4946 4947 := CountSegment.single (first := 12225) (initial := 4946) row_count_12226 (by decide)
private theorem atomic_12226_agreement : RowAgreement 12225 12226 := RowAgreement.single (first := 12225) modelrow_agrees_12226
private def atomic_12227 : CountSegment 1 10 13120 12226 12227 4947 4947 := CountSegment.single (first := 12226) (initial := 4947) row_count_12227 (by decide)
private theorem atomic_12227_agreement : RowAgreement 12226 12227 := RowAgreement.single (first := 12226) modelrow_agrees_12227
private def atomic_12228 : CountSegment 1 10 13120 12227 12228 4947 4947 := CountSegment.single (first := 12227) (initial := 4947) row_count_12228 (by decide)
private theorem atomic_12228_agreement : RowAgreement 12227 12228 := RowAgreement.single (first := 12227) modelrow_agrees_12228
private def atomic_12229 : CountSegment 1 10 13120 12228 12229 4947 4948 := CountSegment.single (first := 12228) (initial := 4947) row_count_12229 (by decide)
private theorem atomic_12229_agreement : RowAgreement 12228 12229 := RowAgreement.single (first := 12228) modelrow_agrees_12229
private def atomic_12230 : CountSegment 1 10 13120 12229 12230 4948 4950 := CountSegment.single (first := 12229) (initial := 4948) row_count_12230 (by decide)
private theorem atomic_12230_agreement : RowAgreement 12229 12230 := RowAgreement.single (first := 12229) modelrow_agrees_12230
private def atomic_12231 : CountSegment 1 10 13120 12230 12231 4950 4950 := CountSegment.single (first := 12230) (initial := 4950) row_count_12231 (by decide)
private theorem atomic_12231_agreement : RowAgreement 12230 12231 := RowAgreement.single (first := 12230) modelrow_agrees_12231
private def atomic_12232 : CountSegment 1 10 13120 12231 12232 4950 4950 := CountSegment.single (first := 12231) (initial := 4950) row_count_12232 (by decide)
private theorem atomic_12232_agreement : RowAgreement 12231 12232 := RowAgreement.single (first := 12231) modelrow_agrees_12232
private def atomic_12233 : CountSegment 1 10 13120 12232 12233 4950 4951 := CountSegment.single (first := 12232) (initial := 4950) row_count_12233 (by decide)
private theorem atomic_12233_agreement : RowAgreement 12232 12233 := RowAgreement.single (first := 12232) modelrow_agrees_12233
private def atomic_12234 : CountSegment 1 10 13120 12233 12234 4951 4951 := CountSegment.single (first := 12233) (initial := 4951) row_count_12234 (by decide)
private theorem atomic_12234_agreement : RowAgreement 12233 12234 := RowAgreement.single (first := 12233) modelrow_agrees_12234
private def atomic_12235 : CountSegment 1 10 13120 12234 12235 4951 4953 := CountSegment.single (first := 12234) (initial := 4951) row_count_12235 (by decide)
private theorem atomic_12235_agreement : RowAgreement 12234 12235 := RowAgreement.single (first := 12234) modelrow_agrees_12235
private def atomic_12236 : CountSegment 1 10 13120 12235 12236 4953 4954 := CountSegment.single (first := 12235) (initial := 4953) row_count_12236 (by decide)
private theorem atomic_12236_agreement : RowAgreement 12235 12236 := RowAgreement.single (first := 12235) modelrow_agrees_12236
private def atomic_12237 : CountSegment 1 10 13120 12236 12237 4954 4954 := CountSegment.single (first := 12236) (initial := 4954) row_count_12237 (by decide)
private theorem atomic_12237_agreement : RowAgreement 12236 12237 := RowAgreement.single (first := 12236) modelrow_agrees_12237
private def atomic_12238 : CountSegment 1 10 13120 12237 12238 4954 4954 := CountSegment.single (first := 12237) (initial := 4954) row_count_12238 (by decide)
private theorem atomic_12238_agreement : RowAgreement 12237 12238 := RowAgreement.single (first := 12237) modelrow_agrees_12238
private def atomic_12239 : CountSegment 1 10 13120 12238 12239 4954 4954 := CountSegment.single (first := 12238) (initial := 4954) row_count_12239 (by decide)
private theorem atomic_12239_agreement : RowAgreement 12238 12239 := RowAgreement.single (first := 12238) modelrow_agrees_12239
private def atomic_12240 : CountSegment 1 10 13120 12239 12240 4954 4955 := CountSegment.single (first := 12239) (initial := 4954) row_count_12240 (by decide)
private theorem atomic_12240_agreement : RowAgreement 12239 12240 := RowAgreement.single (first := 12239) modelrow_agrees_12240
private def atomic_12241 : CountSegment 1 10 13120 12240 12241 4955 4955 := CountSegment.single (first := 12240) (initial := 4955) row_count_12241 (by decide)
private theorem atomic_12241_agreement : RowAgreement 12240 12241 := RowAgreement.single (first := 12240) modelrow_agrees_12241
private def atomic_12242 : CountSegment 1 10 13120 12241 12242 4955 4955 := CountSegment.single (first := 12241) (initial := 4955) row_count_12242 (by decide)
private theorem atomic_12242_agreement : RowAgreement 12241 12242 := RowAgreement.single (first := 12241) modelrow_agrees_12242
private def atomic_12243 : CountSegment 1 10 13120 12242 12243 4955 4956 := CountSegment.single (first := 12242) (initial := 4955) row_count_12243 (by decide)
private theorem atomic_12243_agreement : RowAgreement 12242 12243 := RowAgreement.single (first := 12242) modelrow_agrees_12243
private def atomic_12244 : CountSegment 1 10 13120 12243 12244 4956 4956 := CountSegment.single (first := 12243) (initial := 4956) row_count_12244 (by decide)
private theorem atomic_12244_agreement : RowAgreement 12243 12244 := RowAgreement.single (first := 12243) modelrow_agrees_12244
private def atomic_12245 : CountSegment 1 10 13120 12244 12245 4956 4957 := CountSegment.single (first := 12244) (initial := 4956) row_count_12245 (by decide)
private theorem atomic_12245_agreement : RowAgreement 12244 12245 := RowAgreement.single (first := 12244) modelrow_agrees_12245
private def atomic_12246 : CountSegment 1 10 13120 12245 12246 4957 4957 := CountSegment.single (first := 12245) (initial := 4957) row_count_12246 (by decide)
private theorem atomic_12246_agreement : RowAgreement 12245 12246 := RowAgreement.single (first := 12245) modelrow_agrees_12246
private def atomic_12247 : CountSegment 1 10 13120 12246 12247 4957 4958 := CountSegment.single (first := 12246) (initial := 4957) row_count_12247 (by decide)
private theorem atomic_12247_agreement : RowAgreement 12246 12247 := RowAgreement.single (first := 12246) modelrow_agrees_12247
private def atomic_12248 : CountSegment 1 10 13120 12247 12248 4958 4959 := CountSegment.single (first := 12247) (initial := 4958) row_count_12248 (by decide)
private theorem atomic_12248_agreement : RowAgreement 12247 12248 := RowAgreement.single (first := 12247) modelrow_agrees_12248
private def atomic_12249 : CountSegment 1 10 13120 12248 12249 4959 4959 := CountSegment.single (first := 12248) (initial := 4959) row_count_12249 (by decide)
private theorem atomic_12249_agreement : RowAgreement 12248 12249 := RowAgreement.single (first := 12248) modelrow_agrees_12249
private def atomic_12250 : CountSegment 1 10 13120 12249 12250 4959 4959 := CountSegment.single (first := 12249) (initial := 4959) row_count_12250 (by decide)
private theorem atomic_12250_agreement : RowAgreement 12249 12250 := RowAgreement.single (first := 12249) modelrow_agrees_12250
private def atomic_12251 : CountSegment 1 10 13120 12250 12251 4959 4959 := CountSegment.single (first := 12250) (initial := 4959) row_count_12251 (by decide)
private theorem atomic_12251_agreement : RowAgreement 12250 12251 := RowAgreement.single (first := 12250) modelrow_agrees_12251
private def atomic_12252 : CountSegment 1 10 13120 12251 12252 4959 4959 := CountSegment.single (first := 12251) (initial := 4959) row_count_12252 (by decide)
private theorem atomic_12252_agreement : RowAgreement 12251 12252 := RowAgreement.single (first := 12251) modelrow_agrees_12252
private def atomic_12253 : CountSegment 1 10 13120 12252 12253 4959 4959 := CountSegment.single (first := 12252) (initial := 4959) row_count_12253 (by decide)
private theorem atomic_12253_agreement : RowAgreement 12252 12253 := RowAgreement.single (first := 12252) modelrow_agrees_12253
private def atomic_12254 : CountSegment 1 10 13120 12253 12254 4959 4959 := CountSegment.single (first := 12253) (initial := 4959) row_count_12254 (by decide)
private theorem atomic_12254_agreement : RowAgreement 12253 12254 := RowAgreement.single (first := 12253) modelrow_agrees_12254
private def atomic_12255 : CountSegment 1 10 13120 12254 12255 4959 4960 := CountSegment.single (first := 12254) (initial := 4959) row_count_12255 (by decide)
private theorem atomic_12255_agreement : RowAgreement 12254 12255 := RowAgreement.single (first := 12254) modelrow_agrees_12255
private def atomic_12256 : CountSegment 1 10 13120 12255 12256 4960 4960 := CountSegment.single (first := 12255) (initial := 4960) row_count_12256 (by decide)
private theorem atomic_12256_agreement : RowAgreement 12255 12256 := RowAgreement.single (first := 12255) modelrow_agrees_12256
private def atomic_12257 : CountSegment 1 10 13120 12256 12257 4960 4960 := CountSegment.single (first := 12256) (initial := 4960) row_count_12257 (by decide)
private theorem atomic_12257_agreement : RowAgreement 12256 12257 := RowAgreement.single (first := 12256) modelrow_agrees_12257
private def atomic_12258 : CountSegment 1 10 13120 12257 12258 4960 4960 := CountSegment.single (first := 12257) (initial := 4960) row_count_12258 (by decide)
private theorem atomic_12258_agreement : RowAgreement 12257 12258 := RowAgreement.single (first := 12257) modelrow_agrees_12258
private def atomic_12259 : CountSegment 1 10 13120 12258 12259 4960 4960 := CountSegment.single (first := 12258) (initial := 4960) row_count_12259 (by decide)
private theorem atomic_12259_agreement : RowAgreement 12258 12259 := RowAgreement.single (first := 12258) modelrow_agrees_12259
private def atomic_12260 : CountSegment 1 10 13120 12259 12260 4960 4960 := CountSegment.single (first := 12259) (initial := 4960) row_count_12260 (by decide)
private theorem atomic_12260_agreement : RowAgreement 12259 12260 := RowAgreement.single (first := 12259) modelrow_agrees_12260
private def atomic_12261 : CountSegment 1 10 13120 12260 12261 4960 4961 := CountSegment.single (first := 12260) (initial := 4960) row_count_12261 (by decide)
private theorem atomic_12261_agreement : RowAgreement 12260 12261 := RowAgreement.single (first := 12260) modelrow_agrees_12261
private def atomic_12262 : CountSegment 1 10 13120 12261 12262 4961 4963 := CountSegment.single (first := 12261) (initial := 4961) row_count_12262 (by decide)
private theorem atomic_12262_agreement : RowAgreement 12261 12262 := RowAgreement.single (first := 12261) modelrow_agrees_12262
private def atomic_12263 : CountSegment 1 10 13120 12262 12263 4963 4963 := CountSegment.single (first := 12262) (initial := 4963) row_count_12263 (by decide)
private theorem atomic_12263_agreement : RowAgreement 12262 12263 := RowAgreement.single (first := 12262) modelrow_agrees_12263
private def atomic_12264 : CountSegment 1 10 13120 12263 12264 4963 4964 := CountSegment.single (first := 12263) (initial := 4963) row_count_12264 (by decide)
private theorem atomic_12264_agreement : RowAgreement 12263 12264 := RowAgreement.single (first := 12263) modelrow_agrees_12264
private def atomic_12265 : CountSegment 1 10 13120 12264 12265 4964 4964 := CountSegment.single (first := 12264) (initial := 4964) row_count_12265 (by decide)
private theorem atomic_12265_agreement : RowAgreement 12264 12265 := RowAgreement.single (first := 12264) modelrow_agrees_12265
private def atomic_12266 : CountSegment 1 10 13120 12265 12266 4964 4965 := CountSegment.single (first := 12265) (initial := 4964) row_count_12266 (by decide)
private theorem atomic_12266_agreement : RowAgreement 12265 12266 := RowAgreement.single (first := 12265) modelrow_agrees_12266
private def atomic_12267 : CountSegment 1 10 13120 12266 12267 4965 4967 := CountSegment.single (first := 12266) (initial := 4965) row_count_12267 (by decide)
private theorem atomic_12267_agreement : RowAgreement 12266 12267 := RowAgreement.single (first := 12266) modelrow_agrees_12267
private def atomic_12268 : CountSegment 1 10 13120 12267 12268 4967 4967 := CountSegment.single (first := 12267) (initial := 4967) row_count_12268 (by decide)
private theorem atomic_12268_agreement : RowAgreement 12267 12268 := RowAgreement.single (first := 12267) modelrow_agrees_12268
private def atomic_12269 : CountSegment 1 10 13120 12268 12269 4967 4967 := CountSegment.single (first := 12268) (initial := 4967) row_count_12269 (by decide)
private theorem atomic_12269_agreement : RowAgreement 12268 12269 := RowAgreement.single (first := 12268) modelrow_agrees_12269
private def atomic_12270 : CountSegment 1 10 13120 12269 12270 4967 4967 := CountSegment.single (first := 12269) (initial := 4967) row_count_12270 (by decide)
private theorem atomic_12270_agreement : RowAgreement 12269 12270 := RowAgreement.single (first := 12269) modelrow_agrees_12270
private def atomic_12271 : CountSegment 1 10 13120 12270 12271 4967 4967 := CountSegment.single (first := 12270) (initial := 4967) row_count_12271 (by decide)
private theorem atomic_12271_agreement : RowAgreement 12270 12271 := RowAgreement.single (first := 12270) modelrow_agrees_12271
private def atomic_12272 : CountSegment 1 10 13120 12271 12272 4967 4968 := CountSegment.single (first := 12271) (initial := 4967) row_count_12272 (by decide)
private theorem atomic_12272_agreement : RowAgreement 12271 12272 := RowAgreement.single (first := 12271) modelrow_agrees_12272
private def atomic_12273 : CountSegment 1 10 13120 12272 12273 4968 4968 := CountSegment.single (first := 12272) (initial := 4968) row_count_12273 (by decide)
private theorem atomic_12273_agreement : RowAgreement 12272 12273 := RowAgreement.single (first := 12272) modelrow_agrees_12273
private def atomic_12274 : CountSegment 1 10 13120 12273 12274 4968 4968 := CountSegment.single (first := 12273) (initial := 4968) row_count_12274 (by decide)
private theorem atomic_12274_agreement : RowAgreement 12273 12274 := RowAgreement.single (first := 12273) modelrow_agrees_12274
private def atomic_12275 : CountSegment 1 10 13120 12274 12275 4968 4968 := CountSegment.single (first := 12274) (initial := 4968) row_count_12275 (by decide)
private theorem atomic_12275_agreement : RowAgreement 12274 12275 := RowAgreement.single (first := 12274) modelrow_agrees_12275
private def atomic_12276 : CountSegment 1 10 13120 12275 12276 4968 4968 := CountSegment.single (first := 12275) (initial := 4968) row_count_12276 (by decide)
private theorem atomic_12276_agreement : RowAgreement 12275 12276 := RowAgreement.single (first := 12275) modelrow_agrees_12276
private def atomic_12277 : CountSegment 1 10 13120 12276 12277 4968 4968 := CountSegment.single (first := 12276) (initial := 4968) row_count_12277 (by decide)
private theorem atomic_12277_agreement : RowAgreement 12276 12277 := RowAgreement.single (first := 12276) modelrow_agrees_12277
private def atomic_12278 : CountSegment 1 10 13120 12277 12278 4968 4969 := CountSegment.single (first := 12277) (initial := 4968) row_count_12278 (by decide)
private theorem atomic_12278_agreement : RowAgreement 12277 12278 := RowAgreement.single (first := 12277) modelrow_agrees_12278
private def atomic_12279 : CountSegment 1 10 13120 12278 12279 4969 4969 := CountSegment.single (first := 12278) (initial := 4969) row_count_12279 (by decide)
private theorem atomic_12279_agreement : RowAgreement 12278 12279 := RowAgreement.single (first := 12278) modelrow_agrees_12279
private def atomic_12280 : CountSegment 1 10 13120 12279 12280 4969 4969 := CountSegment.single (first := 12279) (initial := 4969) row_count_12280 (by decide)
private theorem atomic_12280_agreement : RowAgreement 12279 12280 := RowAgreement.single (first := 12279) modelrow_agrees_12280
private def atomic_12281 : CountSegment 1 10 13120 12280 12281 4969 4970 := CountSegment.single (first := 12280) (initial := 4969) row_count_12281 (by decide)
private theorem atomic_12281_agreement : RowAgreement 12280 12281 := RowAgreement.single (first := 12280) modelrow_agrees_12281
private def atomic_12282 : CountSegment 1 10 13120 12281 12282 4970 4971 := CountSegment.single (first := 12281) (initial := 4970) row_count_12282 (by decide)
private theorem atomic_12282_agreement : RowAgreement 12281 12282 := RowAgreement.single (first := 12281) modelrow_agrees_12282
private def atomic_12283 : CountSegment 1 10 13120 12282 12283 4971 4971 := CountSegment.single (first := 12282) (initial := 4971) row_count_12283 (by decide)
private theorem atomic_12283_agreement : RowAgreement 12282 12283 := RowAgreement.single (first := 12282) modelrow_agrees_12283
private def atomic_12284 : CountSegment 1 10 13120 12283 12284 4971 4971 := CountSegment.single (first := 12283) (initial := 4971) row_count_12284 (by decide)
private theorem atomic_12284_agreement : RowAgreement 12283 12284 := RowAgreement.single (first := 12283) modelrow_agrees_12284
private def atomic_12285 : CountSegment 1 10 13120 12284 12285 4971 4971 := CountSegment.single (first := 12284) (initial := 4971) row_count_12285 (by decide)
private theorem atomic_12285_agreement : RowAgreement 12284 12285 := RowAgreement.single (first := 12284) modelrow_agrees_12285
private def atomic_12286 : CountSegment 1 10 13120 12285 12286 4971 4971 := CountSegment.single (first := 12285) (initial := 4971) row_count_12286 (by decide)
private theorem atomic_12286_agreement : RowAgreement 12285 12286 := RowAgreement.single (first := 12285) modelrow_agrees_12286
private def atomic_12287 : CountSegment 1 10 13120 12286 12287 4971 4971 := CountSegment.single (first := 12286) (initial := 4971) row_count_12287 (by decide)
private theorem atomic_12287_agreement : RowAgreement 12286 12287 := RowAgreement.single (first := 12286) modelrow_agrees_12287
private def atomic_12288 : CountSegment 1 10 13120 12287 12288 4971 4972 := CountSegment.single (first := 12287) (initial := 4971) row_count_12288 (by decide)
private theorem atomic_12288_agreement : RowAgreement 12287 12288 := RowAgreement.single (first := 12287) modelrow_agrees_12288
private def within_095_000 : CountSegment 1 10 13120 12160 12162 4919 4921 := CountSegment.join atomic_12161 atomic_12162
private theorem within_095_000_agreement : RowAgreement 12160 12162 := RowAgreement.join atomic_12161_agreement atomic_12162_agreement
private def within_095_001 : CountSegment 1 10 13120 12162 12164 4921 4922 := CountSegment.join atomic_12163 atomic_12164
private theorem within_095_001_agreement : RowAgreement 12162 12164 := RowAgreement.join atomic_12163_agreement atomic_12164_agreement
private def within_095_002 : CountSegment 1 10 13120 12160 12164 4919 4922 := CountSegment.join within_095_000 within_095_001
private theorem within_095_002_agreement : RowAgreement 12160 12164 := RowAgreement.join within_095_000_agreement within_095_001_agreement
private def within_095_003 : CountSegment 1 10 13120 12164 12166 4922 4923 := CountSegment.join atomic_12165 atomic_12166
private theorem within_095_003_agreement : RowAgreement 12164 12166 := RowAgreement.join atomic_12165_agreement atomic_12166_agreement
private def within_095_004 : CountSegment 1 10 13120 12166 12168 4923 4923 := CountSegment.join atomic_12167 atomic_12168
private theorem within_095_004_agreement : RowAgreement 12166 12168 := RowAgreement.join atomic_12167_agreement atomic_12168_agreement
private def within_095_005 : CountSegment 1 10 13120 12164 12168 4922 4923 := CountSegment.join within_095_003 within_095_004
private theorem within_095_005_agreement : RowAgreement 12164 12168 := RowAgreement.join within_095_003_agreement within_095_004_agreement
private def within_095_006 : CountSegment 1 10 13120 12160 12168 4919 4923 := CountSegment.join within_095_002 within_095_005
private theorem within_095_006_agreement : RowAgreement 12160 12168 := RowAgreement.join within_095_002_agreement within_095_005_agreement
private def within_095_007 : CountSegment 1 10 13120 12168 12170 4923 4924 := CountSegment.join atomic_12169 atomic_12170
private theorem within_095_007_agreement : RowAgreement 12168 12170 := RowAgreement.join atomic_12169_agreement atomic_12170_agreement
private def within_095_008 : CountSegment 1 10 13120 12170 12172 4924 4925 := CountSegment.join atomic_12171 atomic_12172
private theorem within_095_008_agreement : RowAgreement 12170 12172 := RowAgreement.join atomic_12171_agreement atomic_12172_agreement
private def within_095_009 : CountSegment 1 10 13120 12168 12172 4923 4925 := CountSegment.join within_095_007 within_095_008
private theorem within_095_009_agreement : RowAgreement 12168 12172 := RowAgreement.join within_095_007_agreement within_095_008_agreement
private def within_095_010 : CountSegment 1 10 13120 12172 12174 4925 4925 := CountSegment.join atomic_12173 atomic_12174
private theorem within_095_010_agreement : RowAgreement 12172 12174 := RowAgreement.join atomic_12173_agreement atomic_12174_agreement
private def within_095_011 : CountSegment 1 10 13120 12174 12176 4925 4926 := CountSegment.join atomic_12175 atomic_12176
private theorem within_095_011_agreement : RowAgreement 12174 12176 := RowAgreement.join atomic_12175_agreement atomic_12176_agreement
private def within_095_012 : CountSegment 1 10 13120 12172 12176 4925 4926 := CountSegment.join within_095_010 within_095_011
private theorem within_095_012_agreement : RowAgreement 12172 12176 := RowAgreement.join within_095_010_agreement within_095_011_agreement
private def within_095_013 : CountSegment 1 10 13120 12168 12176 4923 4926 := CountSegment.join within_095_009 within_095_012
private theorem within_095_013_agreement : RowAgreement 12168 12176 := RowAgreement.join within_095_009_agreement within_095_012_agreement
private def within_095_014 : CountSegment 1 10 13120 12160 12176 4919 4926 := CountSegment.join within_095_006 within_095_013
private theorem within_095_014_agreement : RowAgreement 12160 12176 := RowAgreement.join within_095_006_agreement within_095_013_agreement
private def within_095_015 : CountSegment 1 10 13120 12176 12178 4926 4927 := CountSegment.join atomic_12177 atomic_12178
private theorem within_095_015_agreement : RowAgreement 12176 12178 := RowAgreement.join atomic_12177_agreement atomic_12178_agreement
private def within_095_016 : CountSegment 1 10 13120 12178 12180 4927 4927 := CountSegment.join atomic_12179 atomic_12180
private theorem within_095_016_agreement : RowAgreement 12178 12180 := RowAgreement.join atomic_12179_agreement atomic_12180_agreement
private def within_095_017 : CountSegment 1 10 13120 12176 12180 4926 4927 := CountSegment.join within_095_015 within_095_016
private theorem within_095_017_agreement : RowAgreement 12176 12180 := RowAgreement.join within_095_015_agreement within_095_016_agreement
private def within_095_018 : CountSegment 1 10 13120 12180 12182 4927 4929 := CountSegment.join atomic_12181 atomic_12182
private theorem within_095_018_agreement : RowAgreement 12180 12182 := RowAgreement.join atomic_12181_agreement atomic_12182_agreement
private def within_095_019 : CountSegment 1 10 13120 12182 12184 4929 4931 := CountSegment.join atomic_12183 atomic_12184
private theorem within_095_019_agreement : RowAgreement 12182 12184 := RowAgreement.join atomic_12183_agreement atomic_12184_agreement
private def within_095_020 : CountSegment 1 10 13120 12180 12184 4927 4931 := CountSegment.join within_095_018 within_095_019
private theorem within_095_020_agreement : RowAgreement 12180 12184 := RowAgreement.join within_095_018_agreement within_095_019_agreement
private def within_095_021 : CountSegment 1 10 13120 12176 12184 4926 4931 := CountSegment.join within_095_017 within_095_020
private theorem within_095_021_agreement : RowAgreement 12176 12184 := RowAgreement.join within_095_017_agreement within_095_020_agreement
private def within_095_022 : CountSegment 1 10 13120 12184 12186 4931 4931 := CountSegment.join atomic_12185 atomic_12186
private theorem within_095_022_agreement : RowAgreement 12184 12186 := RowAgreement.join atomic_12185_agreement atomic_12186_agreement
private def within_095_023 : CountSegment 1 10 13120 12186 12188 4931 4931 := CountSegment.join atomic_12187 atomic_12188
private theorem within_095_023_agreement : RowAgreement 12186 12188 := RowAgreement.join atomic_12187_agreement atomic_12188_agreement
private def within_095_024 : CountSegment 1 10 13120 12184 12188 4931 4931 := CountSegment.join within_095_022 within_095_023
private theorem within_095_024_agreement : RowAgreement 12184 12188 := RowAgreement.join within_095_022_agreement within_095_023_agreement
private def within_095_025 : CountSegment 1 10 13120 12188 12190 4931 4932 := CountSegment.join atomic_12189 atomic_12190
private theorem within_095_025_agreement : RowAgreement 12188 12190 := RowAgreement.join atomic_12189_agreement atomic_12190_agreement
private def within_095_026 : CountSegment 1 10 13120 12190 12192 4932 4934 := CountSegment.join atomic_12191 atomic_12192
private theorem within_095_026_agreement : RowAgreement 12190 12192 := RowAgreement.join atomic_12191_agreement atomic_12192_agreement
private def within_095_027 : CountSegment 1 10 13120 12188 12192 4931 4934 := CountSegment.join within_095_025 within_095_026
private theorem within_095_027_agreement : RowAgreement 12188 12192 := RowAgreement.join within_095_025_agreement within_095_026_agreement
private def within_095_028 : CountSegment 1 10 13120 12184 12192 4931 4934 := CountSegment.join within_095_024 within_095_027
private theorem within_095_028_agreement : RowAgreement 12184 12192 := RowAgreement.join within_095_024_agreement within_095_027_agreement
private def within_095_029 : CountSegment 1 10 13120 12176 12192 4926 4934 := CountSegment.join within_095_021 within_095_028
private theorem within_095_029_agreement : RowAgreement 12176 12192 := RowAgreement.join within_095_021_agreement within_095_028_agreement
private def within_095_030 : CountSegment 1 10 13120 12160 12192 4919 4934 := CountSegment.join within_095_014 within_095_029
private theorem within_095_030_agreement : RowAgreement 12160 12192 := RowAgreement.join within_095_014_agreement within_095_029_agreement
private def within_095_031 : CountSegment 1 10 13120 12192 12194 4934 4934 := CountSegment.join atomic_12193 atomic_12194
private theorem within_095_031_agreement : RowAgreement 12192 12194 := RowAgreement.join atomic_12193_agreement atomic_12194_agreement
private def within_095_032 : CountSegment 1 10 13120 12194 12196 4934 4934 := CountSegment.join atomic_12195 atomic_12196
private theorem within_095_032_agreement : RowAgreement 12194 12196 := RowAgreement.join atomic_12195_agreement atomic_12196_agreement
private def within_095_033 : CountSegment 1 10 13120 12192 12196 4934 4934 := CountSegment.join within_095_031 within_095_032
private theorem within_095_033_agreement : RowAgreement 12192 12196 := RowAgreement.join within_095_031_agreement within_095_032_agreement
private def within_095_034 : CountSegment 1 10 13120 12196 12198 4934 4935 := CountSegment.join atomic_12197 atomic_12198
private theorem within_095_034_agreement : RowAgreement 12196 12198 := RowAgreement.join atomic_12197_agreement atomic_12198_agreement
private def within_095_035 : CountSegment 1 10 13120 12198 12200 4935 4936 := CountSegment.join atomic_12199 atomic_12200
private theorem within_095_035_agreement : RowAgreement 12198 12200 := RowAgreement.join atomic_12199_agreement atomic_12200_agreement
private def within_095_036 : CountSegment 1 10 13120 12196 12200 4934 4936 := CountSegment.join within_095_034 within_095_035
private theorem within_095_036_agreement : RowAgreement 12196 12200 := RowAgreement.join within_095_034_agreement within_095_035_agreement
private def within_095_037 : CountSegment 1 10 13120 12192 12200 4934 4936 := CountSegment.join within_095_033 within_095_036
private theorem within_095_037_agreement : RowAgreement 12192 12200 := RowAgreement.join within_095_033_agreement within_095_036_agreement
private def within_095_038 : CountSegment 1 10 13120 12200 12202 4936 4937 := CountSegment.join atomic_12201 atomic_12202
private theorem within_095_038_agreement : RowAgreement 12200 12202 := RowAgreement.join atomic_12201_agreement atomic_12202_agreement
private def within_095_039 : CountSegment 1 10 13120 12202 12204 4937 4938 := CountSegment.join atomic_12203 atomic_12204
private theorem within_095_039_agreement : RowAgreement 12202 12204 := RowAgreement.join atomic_12203_agreement atomic_12204_agreement
private def within_095_040 : CountSegment 1 10 13120 12200 12204 4936 4938 := CountSegment.join within_095_038 within_095_039
private theorem within_095_040_agreement : RowAgreement 12200 12204 := RowAgreement.join within_095_038_agreement within_095_039_agreement
private def within_095_041 : CountSegment 1 10 13120 12204 12206 4938 4938 := CountSegment.join atomic_12205 atomic_12206
private theorem within_095_041_agreement : RowAgreement 12204 12206 := RowAgreement.join atomic_12205_agreement atomic_12206_agreement
private def within_095_042 : CountSegment 1 10 13120 12206 12208 4938 4939 := CountSegment.join atomic_12207 atomic_12208
private theorem within_095_042_agreement : RowAgreement 12206 12208 := RowAgreement.join atomic_12207_agreement atomic_12208_agreement
private def within_095_043 : CountSegment 1 10 13120 12204 12208 4938 4939 := CountSegment.join within_095_041 within_095_042
private theorem within_095_043_agreement : RowAgreement 12204 12208 := RowAgreement.join within_095_041_agreement within_095_042_agreement
private def within_095_044 : CountSegment 1 10 13120 12200 12208 4936 4939 := CountSegment.join within_095_040 within_095_043
private theorem within_095_044_agreement : RowAgreement 12200 12208 := RowAgreement.join within_095_040_agreement within_095_043_agreement
private def within_095_045 : CountSegment 1 10 13120 12192 12208 4934 4939 := CountSegment.join within_095_037 within_095_044
private theorem within_095_045_agreement : RowAgreement 12192 12208 := RowAgreement.join within_095_037_agreement within_095_044_agreement
private def within_095_046 : CountSegment 1 10 13120 12208 12210 4939 4939 := CountSegment.join atomic_12209 atomic_12210
private theorem within_095_046_agreement : RowAgreement 12208 12210 := RowAgreement.join atomic_12209_agreement atomic_12210_agreement
private def within_095_047 : CountSegment 1 10 13120 12210 12212 4939 4939 := CountSegment.join atomic_12211 atomic_12212
private theorem within_095_047_agreement : RowAgreement 12210 12212 := RowAgreement.join atomic_12211_agreement atomic_12212_agreement
private def within_095_048 : CountSegment 1 10 13120 12208 12212 4939 4939 := CountSegment.join within_095_046 within_095_047
private theorem within_095_048_agreement : RowAgreement 12208 12212 := RowAgreement.join within_095_046_agreement within_095_047_agreement
private def within_095_049 : CountSegment 1 10 13120 12212 12214 4939 4939 := CountSegment.join atomic_12213 atomic_12214
private theorem within_095_049_agreement : RowAgreement 12212 12214 := RowAgreement.join atomic_12213_agreement atomic_12214_agreement
private def within_095_050 : CountSegment 1 10 13120 12214 12216 4939 4940 := CountSegment.join atomic_12215 atomic_12216
private theorem within_095_050_agreement : RowAgreement 12214 12216 := RowAgreement.join atomic_12215_agreement atomic_12216_agreement
private def within_095_051 : CountSegment 1 10 13120 12212 12216 4939 4940 := CountSegment.join within_095_049 within_095_050
private theorem within_095_051_agreement : RowAgreement 12212 12216 := RowAgreement.join within_095_049_agreement within_095_050_agreement
private def within_095_052 : CountSegment 1 10 13120 12208 12216 4939 4940 := CountSegment.join within_095_048 within_095_051
private theorem within_095_052_agreement : RowAgreement 12208 12216 := RowAgreement.join within_095_048_agreement within_095_051_agreement
private def within_095_053 : CountSegment 1 10 13120 12216 12218 4940 4942 := CountSegment.join atomic_12217 atomic_12218
private theorem within_095_053_agreement : RowAgreement 12216 12218 := RowAgreement.join atomic_12217_agreement atomic_12218_agreement
private def within_095_054 : CountSegment 1 10 13120 12218 12220 4942 4944 := CountSegment.join atomic_12219 atomic_12220
private theorem within_095_054_agreement : RowAgreement 12218 12220 := RowAgreement.join atomic_12219_agreement atomic_12220_agreement
private def within_095_055 : CountSegment 1 10 13120 12216 12220 4940 4944 := CountSegment.join within_095_053 within_095_054
private theorem within_095_055_agreement : RowAgreement 12216 12220 := RowAgreement.join within_095_053_agreement within_095_054_agreement
private def within_095_056 : CountSegment 1 10 13120 12220 12222 4944 4945 := CountSegment.join atomic_12221 atomic_12222
private theorem within_095_056_agreement : RowAgreement 12220 12222 := RowAgreement.join atomic_12221_agreement atomic_12222_agreement
private def within_095_057 : CountSegment 1 10 13120 12222 12224 4945 4946 := CountSegment.join atomic_12223 atomic_12224
private theorem within_095_057_agreement : RowAgreement 12222 12224 := RowAgreement.join atomic_12223_agreement atomic_12224_agreement
private def within_095_058 : CountSegment 1 10 13120 12220 12224 4944 4946 := CountSegment.join within_095_056 within_095_057
private theorem within_095_058_agreement : RowAgreement 12220 12224 := RowAgreement.join within_095_056_agreement within_095_057_agreement
private def within_095_059 : CountSegment 1 10 13120 12216 12224 4940 4946 := CountSegment.join within_095_055 within_095_058
private theorem within_095_059_agreement : RowAgreement 12216 12224 := RowAgreement.join within_095_055_agreement within_095_058_agreement
private def within_095_060 : CountSegment 1 10 13120 12208 12224 4939 4946 := CountSegment.join within_095_052 within_095_059
private theorem within_095_060_agreement : RowAgreement 12208 12224 := RowAgreement.join within_095_052_agreement within_095_059_agreement
private def within_095_061 : CountSegment 1 10 13120 12192 12224 4934 4946 := CountSegment.join within_095_045 within_095_060
private theorem within_095_061_agreement : RowAgreement 12192 12224 := RowAgreement.join within_095_045_agreement within_095_060_agreement
private def within_095_062 : CountSegment 1 10 13120 12160 12224 4919 4946 := CountSegment.join within_095_030 within_095_061
private theorem within_095_062_agreement : RowAgreement 12160 12224 := RowAgreement.join within_095_030_agreement within_095_061_agreement
private def within_095_063 : CountSegment 1 10 13120 12224 12226 4946 4947 := CountSegment.join atomic_12225 atomic_12226
private theorem within_095_063_agreement : RowAgreement 12224 12226 := RowAgreement.join atomic_12225_agreement atomic_12226_agreement
private def within_095_064 : CountSegment 1 10 13120 12226 12228 4947 4947 := CountSegment.join atomic_12227 atomic_12228
private theorem within_095_064_agreement : RowAgreement 12226 12228 := RowAgreement.join atomic_12227_agreement atomic_12228_agreement
private def within_095_065 : CountSegment 1 10 13120 12224 12228 4946 4947 := CountSegment.join within_095_063 within_095_064
private theorem within_095_065_agreement : RowAgreement 12224 12228 := RowAgreement.join within_095_063_agreement within_095_064_agreement
private def within_095_066 : CountSegment 1 10 13120 12228 12230 4947 4950 := CountSegment.join atomic_12229 atomic_12230
private theorem within_095_066_agreement : RowAgreement 12228 12230 := RowAgreement.join atomic_12229_agreement atomic_12230_agreement
private def within_095_067 : CountSegment 1 10 13120 12230 12232 4950 4950 := CountSegment.join atomic_12231 atomic_12232
private theorem within_095_067_agreement : RowAgreement 12230 12232 := RowAgreement.join atomic_12231_agreement atomic_12232_agreement
private def within_095_068 : CountSegment 1 10 13120 12228 12232 4947 4950 := CountSegment.join within_095_066 within_095_067
private theorem within_095_068_agreement : RowAgreement 12228 12232 := RowAgreement.join within_095_066_agreement within_095_067_agreement
private def within_095_069 : CountSegment 1 10 13120 12224 12232 4946 4950 := CountSegment.join within_095_065 within_095_068
private theorem within_095_069_agreement : RowAgreement 12224 12232 := RowAgreement.join within_095_065_agreement within_095_068_agreement
private def within_095_070 : CountSegment 1 10 13120 12232 12234 4950 4951 := CountSegment.join atomic_12233 atomic_12234
private theorem within_095_070_agreement : RowAgreement 12232 12234 := RowAgreement.join atomic_12233_agreement atomic_12234_agreement
private def within_095_071 : CountSegment 1 10 13120 12234 12236 4951 4954 := CountSegment.join atomic_12235 atomic_12236
private theorem within_095_071_agreement : RowAgreement 12234 12236 := RowAgreement.join atomic_12235_agreement atomic_12236_agreement
private def within_095_072 : CountSegment 1 10 13120 12232 12236 4950 4954 := CountSegment.join within_095_070 within_095_071
private theorem within_095_072_agreement : RowAgreement 12232 12236 := RowAgreement.join within_095_070_agreement within_095_071_agreement
private def within_095_073 : CountSegment 1 10 13120 12236 12238 4954 4954 := CountSegment.join atomic_12237 atomic_12238
private theorem within_095_073_agreement : RowAgreement 12236 12238 := RowAgreement.join atomic_12237_agreement atomic_12238_agreement
private def within_095_074 : CountSegment 1 10 13120 12238 12240 4954 4955 := CountSegment.join atomic_12239 atomic_12240
private theorem within_095_074_agreement : RowAgreement 12238 12240 := RowAgreement.join atomic_12239_agreement atomic_12240_agreement
private def within_095_075 : CountSegment 1 10 13120 12236 12240 4954 4955 := CountSegment.join within_095_073 within_095_074
private theorem within_095_075_agreement : RowAgreement 12236 12240 := RowAgreement.join within_095_073_agreement within_095_074_agreement
private def within_095_076 : CountSegment 1 10 13120 12232 12240 4950 4955 := CountSegment.join within_095_072 within_095_075
private theorem within_095_076_agreement : RowAgreement 12232 12240 := RowAgreement.join within_095_072_agreement within_095_075_agreement
private def within_095_077 : CountSegment 1 10 13120 12224 12240 4946 4955 := CountSegment.join within_095_069 within_095_076
private theorem within_095_077_agreement : RowAgreement 12224 12240 := RowAgreement.join within_095_069_agreement within_095_076_agreement
private def within_095_078 : CountSegment 1 10 13120 12240 12242 4955 4955 := CountSegment.join atomic_12241 atomic_12242
private theorem within_095_078_agreement : RowAgreement 12240 12242 := RowAgreement.join atomic_12241_agreement atomic_12242_agreement
private def within_095_079 : CountSegment 1 10 13120 12242 12244 4955 4956 := CountSegment.join atomic_12243 atomic_12244
private theorem within_095_079_agreement : RowAgreement 12242 12244 := RowAgreement.join atomic_12243_agreement atomic_12244_agreement
private def within_095_080 : CountSegment 1 10 13120 12240 12244 4955 4956 := CountSegment.join within_095_078 within_095_079
private theorem within_095_080_agreement : RowAgreement 12240 12244 := RowAgreement.join within_095_078_agreement within_095_079_agreement
private def within_095_081 : CountSegment 1 10 13120 12244 12246 4956 4957 := CountSegment.join atomic_12245 atomic_12246
private theorem within_095_081_agreement : RowAgreement 12244 12246 := RowAgreement.join atomic_12245_agreement atomic_12246_agreement
private def within_095_082 : CountSegment 1 10 13120 12246 12248 4957 4959 := CountSegment.join atomic_12247 atomic_12248
private theorem within_095_082_agreement : RowAgreement 12246 12248 := RowAgreement.join atomic_12247_agreement atomic_12248_agreement
private def within_095_083 : CountSegment 1 10 13120 12244 12248 4956 4959 := CountSegment.join within_095_081 within_095_082
private theorem within_095_083_agreement : RowAgreement 12244 12248 := RowAgreement.join within_095_081_agreement within_095_082_agreement
private def within_095_084 : CountSegment 1 10 13120 12240 12248 4955 4959 := CountSegment.join within_095_080 within_095_083
private theorem within_095_084_agreement : RowAgreement 12240 12248 := RowAgreement.join within_095_080_agreement within_095_083_agreement
private def within_095_085 : CountSegment 1 10 13120 12248 12250 4959 4959 := CountSegment.join atomic_12249 atomic_12250
private theorem within_095_085_agreement : RowAgreement 12248 12250 := RowAgreement.join atomic_12249_agreement atomic_12250_agreement
private def within_095_086 : CountSegment 1 10 13120 12250 12252 4959 4959 := CountSegment.join atomic_12251 atomic_12252
private theorem within_095_086_agreement : RowAgreement 12250 12252 := RowAgreement.join atomic_12251_agreement atomic_12252_agreement
private def within_095_087 : CountSegment 1 10 13120 12248 12252 4959 4959 := CountSegment.join within_095_085 within_095_086
private theorem within_095_087_agreement : RowAgreement 12248 12252 := RowAgreement.join within_095_085_agreement within_095_086_agreement
private def within_095_088 : CountSegment 1 10 13120 12252 12254 4959 4959 := CountSegment.join atomic_12253 atomic_12254
private theorem within_095_088_agreement : RowAgreement 12252 12254 := RowAgreement.join atomic_12253_agreement atomic_12254_agreement
private def within_095_089 : CountSegment 1 10 13120 12254 12256 4959 4960 := CountSegment.join atomic_12255 atomic_12256
private theorem within_095_089_agreement : RowAgreement 12254 12256 := RowAgreement.join atomic_12255_agreement atomic_12256_agreement
private def within_095_090 : CountSegment 1 10 13120 12252 12256 4959 4960 := CountSegment.join within_095_088 within_095_089
private theorem within_095_090_agreement : RowAgreement 12252 12256 := RowAgreement.join within_095_088_agreement within_095_089_agreement
private def within_095_091 : CountSegment 1 10 13120 12248 12256 4959 4960 := CountSegment.join within_095_087 within_095_090
private theorem within_095_091_agreement : RowAgreement 12248 12256 := RowAgreement.join within_095_087_agreement within_095_090_agreement
private def within_095_092 : CountSegment 1 10 13120 12240 12256 4955 4960 := CountSegment.join within_095_084 within_095_091
private theorem within_095_092_agreement : RowAgreement 12240 12256 := RowAgreement.join within_095_084_agreement within_095_091_agreement
private def within_095_093 : CountSegment 1 10 13120 12224 12256 4946 4960 := CountSegment.join within_095_077 within_095_092
private theorem within_095_093_agreement : RowAgreement 12224 12256 := RowAgreement.join within_095_077_agreement within_095_092_agreement
private def within_095_094 : CountSegment 1 10 13120 12256 12258 4960 4960 := CountSegment.join atomic_12257 atomic_12258
private theorem within_095_094_agreement : RowAgreement 12256 12258 := RowAgreement.join atomic_12257_agreement atomic_12258_agreement
private def within_095_095 : CountSegment 1 10 13120 12258 12260 4960 4960 := CountSegment.join atomic_12259 atomic_12260
private theorem within_095_095_agreement : RowAgreement 12258 12260 := RowAgreement.join atomic_12259_agreement atomic_12260_agreement
private def within_095_096 : CountSegment 1 10 13120 12256 12260 4960 4960 := CountSegment.join within_095_094 within_095_095
private theorem within_095_096_agreement : RowAgreement 12256 12260 := RowAgreement.join within_095_094_agreement within_095_095_agreement
private def within_095_097 : CountSegment 1 10 13120 12260 12262 4960 4963 := CountSegment.join atomic_12261 atomic_12262
private theorem within_095_097_agreement : RowAgreement 12260 12262 := RowAgreement.join atomic_12261_agreement atomic_12262_agreement
private def within_095_098 : CountSegment 1 10 13120 12262 12264 4963 4964 := CountSegment.join atomic_12263 atomic_12264
private theorem within_095_098_agreement : RowAgreement 12262 12264 := RowAgreement.join atomic_12263_agreement atomic_12264_agreement
private def within_095_099 : CountSegment 1 10 13120 12260 12264 4960 4964 := CountSegment.join within_095_097 within_095_098
private theorem within_095_099_agreement : RowAgreement 12260 12264 := RowAgreement.join within_095_097_agreement within_095_098_agreement
private def within_095_100 : CountSegment 1 10 13120 12256 12264 4960 4964 := CountSegment.join within_095_096 within_095_099
private theorem within_095_100_agreement : RowAgreement 12256 12264 := RowAgreement.join within_095_096_agreement within_095_099_agreement
private def within_095_101 : CountSegment 1 10 13120 12264 12266 4964 4965 := CountSegment.join atomic_12265 atomic_12266
private theorem within_095_101_agreement : RowAgreement 12264 12266 := RowAgreement.join atomic_12265_agreement atomic_12266_agreement
private def within_095_102 : CountSegment 1 10 13120 12266 12268 4965 4967 := CountSegment.join atomic_12267 atomic_12268
private theorem within_095_102_agreement : RowAgreement 12266 12268 := RowAgreement.join atomic_12267_agreement atomic_12268_agreement
private def within_095_103 : CountSegment 1 10 13120 12264 12268 4964 4967 := CountSegment.join within_095_101 within_095_102
private theorem within_095_103_agreement : RowAgreement 12264 12268 := RowAgreement.join within_095_101_agreement within_095_102_agreement
private def within_095_104 : CountSegment 1 10 13120 12268 12270 4967 4967 := CountSegment.join atomic_12269 atomic_12270
private theorem within_095_104_agreement : RowAgreement 12268 12270 := RowAgreement.join atomic_12269_agreement atomic_12270_agreement
private def within_095_105 : CountSegment 1 10 13120 12270 12272 4967 4968 := CountSegment.join atomic_12271 atomic_12272
private theorem within_095_105_agreement : RowAgreement 12270 12272 := RowAgreement.join atomic_12271_agreement atomic_12272_agreement
private def within_095_106 : CountSegment 1 10 13120 12268 12272 4967 4968 := CountSegment.join within_095_104 within_095_105
private theorem within_095_106_agreement : RowAgreement 12268 12272 := RowAgreement.join within_095_104_agreement within_095_105_agreement
private def within_095_107 : CountSegment 1 10 13120 12264 12272 4964 4968 := CountSegment.join within_095_103 within_095_106
private theorem within_095_107_agreement : RowAgreement 12264 12272 := RowAgreement.join within_095_103_agreement within_095_106_agreement
private def within_095_108 : CountSegment 1 10 13120 12256 12272 4960 4968 := CountSegment.join within_095_100 within_095_107
private theorem within_095_108_agreement : RowAgreement 12256 12272 := RowAgreement.join within_095_100_agreement within_095_107_agreement
private def within_095_109 : CountSegment 1 10 13120 12272 12274 4968 4968 := CountSegment.join atomic_12273 atomic_12274
private theorem within_095_109_agreement : RowAgreement 12272 12274 := RowAgreement.join atomic_12273_agreement atomic_12274_agreement
private def within_095_110 : CountSegment 1 10 13120 12274 12276 4968 4968 := CountSegment.join atomic_12275 atomic_12276
private theorem within_095_110_agreement : RowAgreement 12274 12276 := RowAgreement.join atomic_12275_agreement atomic_12276_agreement
private def within_095_111 : CountSegment 1 10 13120 12272 12276 4968 4968 := CountSegment.join within_095_109 within_095_110
private theorem within_095_111_agreement : RowAgreement 12272 12276 := RowAgreement.join within_095_109_agreement within_095_110_agreement
private def within_095_112 : CountSegment 1 10 13120 12276 12278 4968 4969 := CountSegment.join atomic_12277 atomic_12278
private theorem within_095_112_agreement : RowAgreement 12276 12278 := RowAgreement.join atomic_12277_agreement atomic_12278_agreement
private def within_095_113 : CountSegment 1 10 13120 12278 12280 4969 4969 := CountSegment.join atomic_12279 atomic_12280
private theorem within_095_113_agreement : RowAgreement 12278 12280 := RowAgreement.join atomic_12279_agreement atomic_12280_agreement
private def within_095_114 : CountSegment 1 10 13120 12276 12280 4968 4969 := CountSegment.join within_095_112 within_095_113
private theorem within_095_114_agreement : RowAgreement 12276 12280 := RowAgreement.join within_095_112_agreement within_095_113_agreement
private def within_095_115 : CountSegment 1 10 13120 12272 12280 4968 4969 := CountSegment.join within_095_111 within_095_114
private theorem within_095_115_agreement : RowAgreement 12272 12280 := RowAgreement.join within_095_111_agreement within_095_114_agreement
private def within_095_116 : CountSegment 1 10 13120 12280 12282 4969 4971 := CountSegment.join atomic_12281 atomic_12282
private theorem within_095_116_agreement : RowAgreement 12280 12282 := RowAgreement.join atomic_12281_agreement atomic_12282_agreement
private def within_095_117 : CountSegment 1 10 13120 12282 12284 4971 4971 := CountSegment.join atomic_12283 atomic_12284
private theorem within_095_117_agreement : RowAgreement 12282 12284 := RowAgreement.join atomic_12283_agreement atomic_12284_agreement
private def within_095_118 : CountSegment 1 10 13120 12280 12284 4969 4971 := CountSegment.join within_095_116 within_095_117
private theorem within_095_118_agreement : RowAgreement 12280 12284 := RowAgreement.join within_095_116_agreement within_095_117_agreement
private def within_095_119 : CountSegment 1 10 13120 12284 12286 4971 4971 := CountSegment.join atomic_12285 atomic_12286
private theorem within_095_119_agreement : RowAgreement 12284 12286 := RowAgreement.join atomic_12285_agreement atomic_12286_agreement
private def within_095_120 : CountSegment 1 10 13120 12286 12288 4971 4972 := CountSegment.join atomic_12287 atomic_12288
private theorem within_095_120_agreement : RowAgreement 12286 12288 := RowAgreement.join atomic_12287_agreement atomic_12288_agreement
private def within_095_121 : CountSegment 1 10 13120 12284 12288 4971 4972 := CountSegment.join within_095_119 within_095_120
private theorem within_095_121_agreement : RowAgreement 12284 12288 := RowAgreement.join within_095_119_agreement within_095_120_agreement
private def within_095_122 : CountSegment 1 10 13120 12280 12288 4969 4972 := CountSegment.join within_095_118 within_095_121
private theorem within_095_122_agreement : RowAgreement 12280 12288 := RowAgreement.join within_095_118_agreement within_095_121_agreement
private def within_095_123 : CountSegment 1 10 13120 12272 12288 4968 4972 := CountSegment.join within_095_115 within_095_122
private theorem within_095_123_agreement : RowAgreement 12272 12288 := RowAgreement.join within_095_115_agreement within_095_122_agreement
private def within_095_124 : CountSegment 1 10 13120 12256 12288 4960 4972 := CountSegment.join within_095_108 within_095_123
private theorem within_095_124_agreement : RowAgreement 12256 12288 := RowAgreement.join within_095_108_agreement within_095_123_agreement
private def within_095_125 : CountSegment 1 10 13120 12224 12288 4946 4972 := CountSegment.join within_095_093 within_095_124
private theorem within_095_125_agreement : RowAgreement 12224 12288 := RowAgreement.join within_095_093_agreement within_095_124_agreement
private def within_095_126 : CountSegment 1 10 13120 12160 12288 4919 4972 := CountSegment.join within_095_062 within_095_125
private theorem within_095_126_agreement : RowAgreement 12160 12288 := RowAgreement.join within_095_062_agreement within_095_125_agreement
def segment_095 : CountSegment 1 10 13120 12160 12288 4919 4972 := within_095_126
theorem agreement_095 : RowAgreement 12160 12288 := within_095_126_agreement
end ProjectEulerSolutions.P58Numerical
