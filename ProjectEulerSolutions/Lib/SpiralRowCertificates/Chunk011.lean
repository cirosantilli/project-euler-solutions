import ProjectEulerSolutions.Lib.SpiralPrimeSegments
import ProjectEulerSolutions.Lib.SpiralPrimeCertificates.All
import ProjectEulerSolutions.Lib.MillerRabin
import ProjectEulerSolutions.Lib.SpiralMillerAgreement
/-! Finite spiral row and counter certificates. Python execution is checked separately. -/
namespace ProjectEulerSolutions.P58Numerical
open SpiralPrimes
set_option Elab.async false
def row_1409 : RowCertificate 1409 where
  first := false
  first_correct := by
    have composite : ¬ Nat.Prime (corner 1409 1) := by change ¬ Nat.Prime 7943943; exact PrattCertificates.composite_of_product (factor := 3) (cofactor := 2647981) rfl (by decide) (by decide)
    exact decide_eq_false composite
  second := false
  second_correct := by
    have composite : ¬ Nat.Prime (corner 1409 2) := by change ¬ Nat.Prime 7941125; exact PrattCertificates.composite_of_product (factor := 5) (cofactor := 1588225) rfl (by decide) (by decide)
    exact decide_eq_false composite
  third := false
  third_correct := by
    have composite : ¬ Nat.Prime (corner 1409 3) := by change ¬ Nat.Prime 7938307; exact PrattCertificates.composite_of_product (factor := 13) (cofactor := 610639) rfl (by decide) (by decide)
    exact decide_eq_false composite
theorem row_count_1409 : row 1409 = 0 := (row_1409.checked).trans rfl
theorem model_count_1409 : modelRow 1409 = 0 := by decide
theorem modelrow_agrees_1409 : modelRow 1409 = row 1409 := model_count_1409.trans row_count_1409.symm
def row_1410 : RowCertificate 1410 where
  first := false
  first_correct := by
    have composite : ¬ Nat.Prime (corner 1410 1) := by change ¬ Nat.Prime 7955221; exact PrattCertificates.composite_of_product (factor := 79) (cofactor := 100699) rfl (by decide) (by decide)
    exact decide_eq_false composite
  second := false
  second_correct := by
    have composite : ¬ Nat.Prime (corner 1410 2) := by change ¬ Nat.Prime 7952401; exact PrattCertificates.composite_of_product (factor := 41) (cofactor := 193961) rfl (by decide) (by decide)
    exact decide_eq_false composite
  third := false
  third_correct := by
    have composite : ¬ Nat.Prime (corner 1410 3) := by change ¬ Nat.Prime 7949581; exact PrattCertificates.composite_of_product (factor := 19) (cofactor := 418399) rfl (by decide) (by decide)
    exact decide_eq_false composite
theorem row_count_1410 : row 1410 = 0 := (row_1410.checked).trans rfl
theorem model_count_1410 : modelRow 1410 = 0 := by decide
theorem modelrow_agrees_1410 : modelRow 1410 = row 1410 := model_count_1410.trans row_count_1410.symm
def row_1411 : RowCertificate 1411 where
  first := false
  first_correct := by
    have composite : ¬ Nat.Prime (corner 1411 1) := by change ¬ Nat.Prime 7966507; exact PrattCertificates.composite_of_product (factor := 37) (cofactor := 215311) rfl (by decide) (by decide)
    exact decide_eq_false composite
  second := false
  second_correct := by
    have composite : ¬ Nat.Prime (corner 1411 2) := by change ¬ Nat.Prime 7963685; exact PrattCertificates.composite_of_product (factor := 5) (cofactor := 1592737) rfl (by decide) (by decide)
    exact decide_eq_false composite
  third := false
  third_correct := by
    have composite : ¬ Nat.Prime (corner 1411 3) := by change ¬ Nat.Prime 7960863; exact PrattCertificates.composite_of_product (factor := 3) (cofactor := 2653621) rfl (by decide) (by decide)
    exact decide_eq_false composite
theorem row_count_1411 : row 1411 = 0 := (row_1411.checked).trans rfl
theorem model_count_1411 : modelRow 1411 = 0 := by decide
theorem modelrow_agrees_1411 : modelRow 1411 = row 1411 := model_count_1411.trans row_count_1411.symm
def row_1412 : RowCertificate 1412 where
  first := false
  first_correct := by
    have composite : ¬ Nat.Prime (corner 1412 1) := by change ¬ Nat.Prime 7977801; exact PrattCertificates.composite_of_product (factor := 3) (cofactor := 2659267) rfl (by decide) (by decide)
    exact decide_eq_false composite
  second := true
  second_correct := by
    have prime : Nat.Prime (corner 1412 2) := by change Nat.Prime 7974977; exact ProjectEulerSolutions.P58PrimeNodes.p_7974977.prime
    exact decide_eq_true prime
  third := false
  third_correct := by
    have composite : ¬ Nat.Prime (corner 1412 3) := by change ¬ Nat.Prime 7972153; exact PrattCertificates.composite_of_product (factor := 7) (cofactor := 1138879) rfl (by decide) (by decide)
    exact decide_eq_false composite
theorem row_count_1412 : row 1412 = 1 := (row_1412.checked).trans rfl
theorem model_count_1412 : modelRow 1412 = 1 := by decide
theorem modelrow_agrees_1412 : modelRow 1412 = row 1412 := model_count_1412.trans row_count_1412.symm
def row_1413 : RowCertificate 1413 where
  first := false
  first_correct := by
    have composite : ¬ Nat.Prime (corner 1413 1) := by change ¬ Nat.Prime 7989103; exact PrattCertificates.composite_of_product (factor := 31) (cofactor := 257713) rfl (by decide) (by decide)
    exact decide_eq_false composite
  second := false
  second_correct := by
    have composite : ¬ Nat.Prime (corner 1413 2) := by change ¬ Nat.Prime 7986277; exact PrattCertificates.composite_of_product (factor := 13) (cofactor := 614329) rfl (by decide) (by decide)
    exact decide_eq_false composite
  third := false
  third_correct := by
    have composite : ¬ Nat.Prime (corner 1413 3) := by change ¬ Nat.Prime 7983451; exact PrattCertificates.composite_of_product (factor := 7) (cofactor := 1140493) rfl (by decide) (by decide)
    exact decide_eq_false composite
theorem row_count_1413 : row 1413 = 0 := (row_1413.checked).trans rfl
theorem model_count_1413 : modelRow 1413 = 0 := by decide
theorem modelrow_agrees_1413 : modelRow 1413 = row 1413 := model_count_1413.trans row_count_1413.symm
def row_1414 : RowCertificate 1414 where
  first := true
  first_correct := by
    have prime : Nat.Prime (corner 1414 1) := by change Nat.Prime 8000413; exact ProjectEulerSolutions.P58PrimeNodes.p_8000413.prime
    exact decide_eq_true prime
  second := false
  second_correct := by
    have composite : ¬ Nat.Prime (corner 1414 2) := by change ¬ Nat.Prime 7997585; exact PrattCertificates.composite_of_product (factor := 5) (cofactor := 1599517) rfl (by decide) (by decide)
    exact decide_eq_false composite
  third := false
  third_correct := by
    have composite : ¬ Nat.Prime (corner 1414 3) := by change ¬ Nat.Prime 7994757; exact PrattCertificates.composite_of_product (factor := 3) (cofactor := 2664919) rfl (by decide) (by decide)
    exact decide_eq_false composite
theorem row_count_1414 : row 1414 = 1 := (row_1414.checked).trans rfl
theorem model_count_1414 : modelRow 1414 = 1 := by decide
theorem modelrow_agrees_1414 : modelRow 1414 = row 1414 := model_count_1414.trans row_count_1414.symm
def row_1415 : RowCertificate 1415 where
  first := false
  first_correct := by
    have composite : ¬ Nat.Prime (corner 1415 1) := by change ¬ Nat.Prime 8011731; exact PrattCertificates.composite_of_product (factor := 3) (cofactor := 2670577) rfl (by decide) (by decide)
    exact decide_eq_false composite
  second := false
  second_correct := by
    have composite : ¬ Nat.Prime (corner 1415 2) := by change ¬ Nat.Prime 8008901; exact PrattCertificates.composite_of_product (factor := 29) (cofactor := 276169) rfl (by decide) (by decide)
    exact decide_eq_false composite
  third := false
  third_correct := by
    have composite : ¬ Nat.Prime (corner 1415 3) := by change ¬ Nat.Prime 8006071; exact PrattCertificates.composite_of_product (factor := 163) (cofactor := 49117) rfl (by decide) (by decide)
    exact decide_eq_false composite
theorem row_count_1415 : row 1415 = 0 := (row_1415.checked).trans rfl
theorem model_count_1415 : modelRow 1415 = 0 := by decide
theorem modelrow_agrees_1415 : modelRow 1415 = row 1415 := model_count_1415.trans row_count_1415.symm
def row_1416 : RowCertificate 1416 where
  first := false
  first_correct := by
    have composite : ¬ Nat.Prime (corner 1416 1) := by change ¬ Nat.Prime 8023057; exact PrattCertificates.composite_of_product (factor := 7) (cofactor := 1146151) rfl (by decide) (by decide)
    exact decide_eq_false composite
  second := false
  second_correct := by
    have composite : ¬ Nat.Prime (corner 1416 2) := by change ¬ Nat.Prime 8020225; exact PrattCertificates.composite_of_product (factor := 5) (cofactor := 1604045) rfl (by decide) (by decide)
    exact decide_eq_false composite
  third := false
  third_correct := by
    have composite : ¬ Nat.Prime (corner 1416 3) := by change ¬ Nat.Prime 8017393; exact PrattCertificates.composite_of_product (factor := 43) (cofactor := 186451) rfl (by decide) (by decide)
    exact decide_eq_false composite
theorem row_count_1416 : row 1416 = 0 := (row_1416.checked).trans rfl
theorem model_count_1416 : modelRow 1416 = 0 := by decide
theorem modelrow_agrees_1416 : modelRow 1416 = row 1416 := model_count_1416.trans row_count_1416.symm
def row_1417 : RowCertificate 1417 where
  first := false
  first_correct := by
    have composite : ¬ Nat.Prime (corner 1417 1) := by change ¬ Nat.Prime 8034391; exact PrattCertificates.composite_of_product (factor := 541) (cofactor := 14851) rfl (by decide) (by decide)
    exact decide_eq_false composite
  second := true
  second_correct := by
    have prime : Nat.Prime (corner 1417 2) := by change Nat.Prime 8031557; exact ProjectEulerSolutions.P58PrimeNodes.p_8031557.prime
    exact decide_eq_true prime
  third := false
  third_correct := by
    have composite : ¬ Nat.Prime (corner 1417 3) := by change ¬ Nat.Prime 8028723; exact PrattCertificates.composite_of_product (factor := 3) (cofactor := 2676241) rfl (by decide) (by decide)
    exact decide_eq_false composite
theorem row_count_1417 : row 1417 = 1 := (row_1417.checked).trans rfl
theorem model_count_1417 : modelRow 1417 = 1 := by decide
theorem modelrow_agrees_1417 : modelRow 1417 = row 1417 := model_count_1417.trans row_count_1417.symm
def row_1418 : RowCertificate 1418 where
  first := false
  first_correct := by
    have composite : ¬ Nat.Prime (corner 1418 1) := by change ¬ Nat.Prime 8045733; exact PrattCertificates.composite_of_product (factor := 3) (cofactor := 2681911) rfl (by decide) (by decide)
    exact decide_eq_false composite
  second := true
  second_correct := by
    have prime : Nat.Prime (corner 1418 2) := by change Nat.Prime 8042897; exact ProjectEulerSolutions.P58PrimeNodes.p_8042897.prime
    exact decide_eq_true prime
  third := false
  third_correct := by
    have composite : ¬ Nat.Prime (corner 1418 3) := by change ¬ Nat.Prime 8040061; exact PrattCertificates.composite_of_product (factor := 2371) (cofactor := 3391) rfl (by decide) (by decide)
    exact decide_eq_false composite
theorem row_count_1418 : row 1418 = 1 := (row_1418.checked).trans rfl
theorem model_count_1418 : modelRow 1418 = 1 := by decide
theorem modelrow_agrees_1418 : modelRow 1418 = row 1418 := model_count_1418.trans row_count_1418.symm
def row_1419 : RowCertificate 1419 where
  first := false
  first_correct := by
    have composite : ¬ Nat.Prime (corner 1419 1) := by change ¬ Nat.Prime 8057083; exact PrattCertificates.composite_of_product (factor := 19) (cofactor := 424057) rfl (by decide) (by decide)
    exact decide_eq_false composite
  second := false
  second_correct := by
    have composite : ¬ Nat.Prime (corner 1419 2) := by change ¬ Nat.Prime 8054245; exact PrattCertificates.composite_of_product (factor := 5) (cofactor := 1610849) rfl (by decide) (by decide)
    exact decide_eq_false composite
  third := false
  third_correct := by
    have composite : ¬ Nat.Prime (corner 1419 3) := by change ¬ Nat.Prime 8051407; exact PrattCertificates.composite_of_product (factor := 7) (cofactor := 1150201) rfl (by decide) (by decide)
    exact decide_eq_false composite
theorem row_count_1419 : row 1419 = 0 := (row_1419.checked).trans rfl
theorem model_count_1419 : modelRow 1419 = 0 := by decide
theorem modelrow_agrees_1419 : modelRow 1419 = row 1419 := model_count_1419.trans row_count_1419.symm
def row_1420 : RowCertificate 1420 where
  first := true
  first_correct := by
    have prime : Nat.Prime (corner 1420 1) := by change Nat.Prime 8068441; exact ProjectEulerSolutions.P58PrimeNodes.p_8068441.prime
    exact decide_eq_true prime
  second := false
  second_correct := by
    have composite : ¬ Nat.Prime (corner 1420 2) := by change ¬ Nat.Prime 8065601; exact PrattCertificates.composite_of_product (factor := 113) (cofactor := 71377) rfl (by decide) (by decide)
    exact decide_eq_false composite
  third := false
  third_correct := by
    have composite : ¬ Nat.Prime (corner 1420 3) := by change ¬ Nat.Prime 8062761; exact PrattCertificates.composite_of_product (factor := 3) (cofactor := 2687587) rfl (by decide) (by decide)
    exact decide_eq_false composite
theorem row_count_1420 : row 1420 = 1 := (row_1420.checked).trans rfl
theorem model_count_1420 : modelRow 1420 = 1 := by decide
theorem modelrow_agrees_1420 : modelRow 1420 = row 1420 := model_count_1420.trans row_count_1420.symm
def row_1421 : RowCertificate 1421 where
  first := false
  first_correct := by
    have composite : ¬ Nat.Prime (corner 1421 1) := by change ¬ Nat.Prime 8079807; exact PrattCertificates.composite_of_product (factor := 3) (cofactor := 2693269) rfl (by decide) (by decide)
    exact decide_eq_false composite
  second := false
  second_correct := by
    have composite : ¬ Nat.Prime (corner 1421 2) := by change ¬ Nat.Prime 8076965; exact PrattCertificates.composite_of_product (factor := 5) (cofactor := 1615393) rfl (by decide) (by decide)
    exact decide_eq_false composite
  third := true
  third_correct := by
    have prime : Nat.Prime (corner 1421 3) := by change Nat.Prime 8074123; exact ProjectEulerSolutions.P58PrimeNodes.p_8074123.prime
    exact decide_eq_true prime
theorem row_count_1421 : row 1421 = 1 := (row_1421.checked).trans rfl
theorem model_count_1421 : modelRow 1421 = 1 := by decide
theorem modelrow_agrees_1421 : modelRow 1421 = row 1421 := model_count_1421.trans row_count_1421.symm
def row_1422 : RowCertificate 1422 where
  first := false
  first_correct := by
    have composite : ¬ Nat.Prime (corner 1422 1) := by change ¬ Nat.Prime 8091181; exact PrattCertificates.composite_of_product (factor := 7) (cofactor := 1155883) rfl (by decide) (by decide)
    exact decide_eq_false composite
  second := false
  second_correct := by
    have composite : ¬ Nat.Prime (corner 1422 2) := by change ¬ Nat.Prime 8088337; exact PrattCertificates.composite_of_product (factor := 337) (cofactor := 24001) rfl (by decide) (by decide)
    exact decide_eq_false composite
  third := false
  third_correct := by
    have composite : ¬ Nat.Prime (corner 1422 3) := by change ¬ Nat.Prime 8085493; exact PrattCertificates.composite_of_product (factor := 13) (cofactor := 621961) rfl (by decide) (by decide)
    exact decide_eq_false composite
theorem row_count_1422 : row 1422 = 0 := (row_1422.checked).trans rfl
theorem model_count_1422 : modelRow 1422 = 0 := by decide
theorem modelrow_agrees_1422 : modelRow 1422 = row 1422 := model_count_1422.trans row_count_1422.symm
def row_1423 : RowCertificate 1423 where
  first := false
  first_correct := by
    have composite : ¬ Nat.Prime (corner 1423 1) := by change ¬ Nat.Prime 8102563; exact PrattCertificates.composite_of_product (factor := 7) (cofactor := 1157509) rfl (by decide) (by decide)
    exact decide_eq_false composite
  second := false
  second_correct := by
    have composite : ¬ Nat.Prime (corner 1423 2) := by change ¬ Nat.Prime 8099717; exact PrattCertificates.composite_of_product (factor := 509) (cofactor := 15913) rfl (by decide) (by decide)
    exact decide_eq_false composite
  third := false
  third_correct := by
    have composite : ¬ Nat.Prime (corner 1423 3) := by change ¬ Nat.Prime 8096871; exact PrattCertificates.composite_of_product (factor := 3) (cofactor := 2698957) rfl (by decide) (by decide)
    exact decide_eq_false composite
theorem row_count_1423 : row 1423 = 0 := (row_1423.checked).trans rfl
theorem model_count_1423 : modelRow 1423 = 0 := by decide
theorem modelrow_agrees_1423 : modelRow 1423 = row 1423 := model_count_1423.trans row_count_1423.symm
def row_1424 : RowCertificate 1424 where
  first := false
  first_correct := by
    have composite : ¬ Nat.Prime (corner 1424 1) := by change ¬ Nat.Prime 8113953; exact PrattCertificates.composite_of_product (factor := 3) (cofactor := 2704651) rfl (by decide) (by decide)
    exact decide_eq_false composite
  second := false
  second_correct := by
    have composite : ¬ Nat.Prime (corner 1424 2) := by change ¬ Nat.Prime 8111105; exact PrattCertificates.composite_of_product (factor := 5) (cofactor := 1622221) rfl (by decide) (by decide)
    exact decide_eq_false composite
  third := false
  third_correct := by
    have composite : ¬ Nat.Prime (corner 1424 3) := by change ¬ Nat.Prime 8108257; exact PrattCertificates.composite_of_product (factor := 181) (cofactor := 44797) rfl (by decide) (by decide)
    exact decide_eq_false composite
theorem row_count_1424 : row 1424 = 0 := (row_1424.checked).trans rfl
theorem model_count_1424 : modelRow 1424 = 0 := by decide
theorem modelrow_agrees_1424 : modelRow 1424 = row 1424 := model_count_1424.trans row_count_1424.symm
def row_1425 : RowCertificate 1425 where
  first := false
  first_correct := by
    have composite : ¬ Nat.Prime (corner 1425 1) := by change ¬ Nat.Prime 8125351; exact PrattCertificates.composite_of_product (factor := 13) (cofactor := 625027) rfl (by decide) (by decide)
    exact decide_eq_false composite
  second := true
  second_correct := by
    have prime : Nat.Prime (corner 1425 2) := by change Nat.Prime 8122501; exact ProjectEulerSolutions.P58PrimeNodes.p_8122501.prime
    exact decide_eq_true prime
  third := true
  third_correct := by
    have prime : Nat.Prime (corner 1425 3) := by change Nat.Prime 8119651; exact ProjectEulerSolutions.P58PrimeNodes.p_8119651.prime
    exact decide_eq_true prime
theorem row_count_1425 : row 1425 = 2 := (row_1425.checked).trans rfl
theorem model_count_1425 : modelRow 1425 = 2 := by decide
theorem modelrow_agrees_1425 : modelRow 1425 = row 1425 := model_count_1425.trans row_count_1425.symm
def row_1426 : RowCertificate 1426 where
  first := false
  first_correct := by
    have composite : ¬ Nat.Prime (corner 1426 1) := by change ¬ Nat.Prime 8136757; exact PrattCertificates.composite_of_product (factor := 367) (cofactor := 22171) rfl (by decide) (by decide)
    exact decide_eq_false composite
  second := false
  second_correct := by
    have composite : ¬ Nat.Prime (corner 1426 2) := by change ¬ Nat.Prime 8133905; exact PrattCertificates.composite_of_product (factor := 5) (cofactor := 1626781) rfl (by decide) (by decide)
    exact decide_eq_false composite
  third := false
  third_correct := by
    have composite : ¬ Nat.Prime (corner 1426 3) := by change ¬ Nat.Prime 8131053; exact PrattCertificates.composite_of_product (factor := 3) (cofactor := 2710351) rfl (by decide) (by decide)
    exact decide_eq_false composite
theorem row_count_1426 : row 1426 = 0 := (row_1426.checked).trans rfl
theorem model_count_1426 : modelRow 1426 = 0 := by decide
theorem modelrow_agrees_1426 : modelRow 1426 = row 1426 := model_count_1426.trans row_count_1426.symm
def row_1427 : RowCertificate 1427 where
  first := false
  first_correct := by
    have composite : ¬ Nat.Prime (corner 1427 1) := by change ¬ Nat.Prime 8148171; exact PrattCertificates.composite_of_product (factor := 3) (cofactor := 2716057) rfl (by decide) (by decide)
    exact decide_eq_false composite
  second := false
  second_correct := by
    have composite : ¬ Nat.Prime (corner 1427 2) := by change ¬ Nat.Prime 8145317; exact PrattCertificates.composite_of_product (factor := 29) (cofactor := 280873) rfl (by decide) (by decide)
    exact decide_eq_false composite
  third := false
  third_correct := by
    have composite : ¬ Nat.Prime (corner 1427 3) := by change ¬ Nat.Prime 8142463; exact PrattCertificates.composite_of_product (factor := 7) (cofactor := 1163209) rfl (by decide) (by decide)
    exact decide_eq_false composite
theorem row_count_1427 : row 1427 = 0 := (row_1427.checked).trans rfl
theorem model_count_1427 : modelRow 1427 = 0 := by decide
theorem modelrow_agrees_1427 : modelRow 1427 = row 1427 := model_count_1427.trans row_count_1427.symm
def row_1428 : RowCertificate 1428 where
  first := false
  first_correct := by
    have composite : ¬ Nat.Prime (corner 1428 1) := by change ¬ Nat.Prime 8159593; exact PrattCertificates.composite_of_product (factor := 13) (cofactor := 627661) rfl (by decide) (by decide)
    exact decide_eq_false composite
  second := false
  second_correct := by
    have composite : ¬ Nat.Prime (corner 1428 2) := by change ¬ Nat.Prime 8156737; exact PrattCertificates.composite_of_product (factor := 61) (cofactor := 133717) rfl (by decide) (by decide)
    exact decide_eq_false composite
  third := false
  third_correct := by
    have composite : ¬ Nat.Prime (corner 1428 3) := by change ¬ Nat.Prime 8153881; exact PrattCertificates.composite_of_product (factor := 73) (cofactor := 111697) rfl (by decide) (by decide)
    exact decide_eq_false composite
theorem row_count_1428 : row 1428 = 0 := (row_1428.checked).trans rfl
theorem model_count_1428 : modelRow 1428 = 0 := by decide
theorem modelrow_agrees_1428 : modelRow 1428 = row 1428 := model_count_1428.trans row_count_1428.symm
def row_1429 : RowCertificate 1429 where
  first := false
  first_correct := by
    have composite : ¬ Nat.Prime (corner 1429 1) := by change ¬ Nat.Prime 8171023; exact PrattCertificates.composite_of_product (factor := 7) (cofactor := 1167289) rfl (by decide) (by decide)
    exact decide_eq_false composite
  second := false
  second_correct := by
    have composite : ¬ Nat.Prime (corner 1429 2) := by change ¬ Nat.Prime 8168165; exact PrattCertificates.composite_of_product (factor := 5) (cofactor := 1633633) rfl (by decide) (by decide)
    exact decide_eq_false composite
  third := false
  third_correct := by
    have composite : ¬ Nat.Prime (corner 1429 3) := by change ¬ Nat.Prime 8165307; exact PrattCertificates.composite_of_product (factor := 3) (cofactor := 2721769) rfl (by decide) (by decide)
    exact decide_eq_false composite
theorem row_count_1429 : row 1429 = 0 := (row_1429.checked).trans rfl
theorem model_count_1429 : modelRow 1429 = 0 := by decide
theorem modelrow_agrees_1429 : modelRow 1429 = row 1429 := model_count_1429.trans row_count_1429.symm
def row_1430 : RowCertificate 1430 where
  first := false
  first_correct := by
    have composite : ¬ Nat.Prime (corner 1430 1) := by change ¬ Nat.Prime 8182461; exact PrattCertificates.composite_of_product (factor := 3) (cofactor := 2727487) rfl (by decide) (by decide)
    exact decide_eq_false composite
  second := false
  second_correct := by
    have composite : ¬ Nat.Prime (corner 1430 2) := by change ¬ Nat.Prime 8179601; exact PrattCertificates.composite_of_product (factor := 17) (cofactor := 481153) rfl (by decide) (by decide)
    exact decide_eq_false composite
  third := false
  third_correct := by
    have composite : ¬ Nat.Prime (corner 1430 3) := by change ¬ Nat.Prime 8176741; exact PrattCertificates.composite_of_product (factor := 37) (cofactor := 220993) rfl (by decide) (by decide)
    exact decide_eq_false composite
theorem row_count_1430 : row 1430 = 0 := (row_1430.checked).trans rfl
theorem model_count_1430 : modelRow 1430 = 0 := by decide
theorem modelrow_agrees_1430 : modelRow 1430 = row 1430 := model_count_1430.trans row_count_1430.symm
def row_1431 : RowCertificate 1431 where
  first := true
  first_correct := by
    have prime : Nat.Prime (corner 1431 1) := by change Nat.Prime 8193907; exact ProjectEulerSolutions.P58PrimeNodes.p_8193907.prime
    exact decide_eq_true prime
  second := false
  second_correct := by
    have composite : ¬ Nat.Prime (corner 1431 2) := by change ¬ Nat.Prime 8191045; exact PrattCertificates.composite_of_product (factor := 5) (cofactor := 1638209) rfl (by decide) (by decide)
    exact decide_eq_false composite
  third := false
  third_correct := by
    have composite : ¬ Nat.Prime (corner 1431 3) := by change ¬ Nat.Prime 8188183; exact PrattCertificates.composite_of_product (factor := 19) (cofactor := 430957) rfl (by decide) (by decide)
    exact decide_eq_false composite
theorem row_count_1431 : row 1431 = 1 := (row_1431.checked).trans rfl
theorem model_count_1431 : modelRow 1431 = 1 := by decide
theorem modelrow_agrees_1431 : modelRow 1431 = row 1431 := model_count_1431.trans row_count_1431.symm
def row_1432 : RowCertificate 1432 where
  first := true
  first_correct := by
    have prime : Nat.Prime (corner 1432 1) := by change Nat.Prime 8205361; exact ProjectEulerSolutions.P58PrimeNodes.p_8205361.prime
    exact decide_eq_true prime
  second := true
  second_correct := by
    have prime : Nat.Prime (corner 1432 2) := by change Nat.Prime 8202497; exact ProjectEulerSolutions.P58PrimeNodes.p_8202497.prime
    exact decide_eq_true prime
  third := false
  third_correct := by
    have composite : ¬ Nat.Prime (corner 1432 3) := by change ¬ Nat.Prime 8199633; exact PrattCertificates.composite_of_product (factor := 3) (cofactor := 2733211) rfl (by decide) (by decide)
    exact decide_eq_false composite
theorem row_count_1432 : row 1432 = 2 := (row_1432.checked).trans rfl
theorem model_count_1432 : modelRow 1432 = 2 := by decide
theorem modelrow_agrees_1432 : modelRow 1432 = row 1432 := model_count_1432.trans row_count_1432.symm
def row_1433 : RowCertificate 1433 where
  first := false
  first_correct := by
    have composite : ¬ Nat.Prime (corner 1433 1) := by change ¬ Nat.Prime 8216823; exact PrattCertificates.composite_of_product (factor := 3) (cofactor := 2738941) rfl (by decide) (by decide)
    exact decide_eq_false composite
  second := false
  second_correct := by
    have composite : ¬ Nat.Prime (corner 1433 2) := by change ¬ Nat.Prime 8213957; exact PrattCertificates.composite_of_product (factor := 353) (cofactor := 23269) rfl (by decide) (by decide)
    exact decide_eq_false composite
  third := false
  third_correct := by
    have composite : ¬ Nat.Prime (corner 1433 3) := by change ¬ Nat.Prime 8211091; exact PrattCertificates.composite_of_product (factor := 7) (cofactor := 1173013) rfl (by decide) (by decide)
    exact decide_eq_false composite
theorem row_count_1433 : row 1433 = 0 := (row_1433.checked).trans rfl
theorem model_count_1433 : modelRow 1433 = 0 := by decide
theorem modelrow_agrees_1433 : modelRow 1433 = row 1433 := model_count_1433.trans row_count_1433.symm
def row_1434 : RowCertificate 1434 where
  first := true
  first_correct := by
    have prime : Nat.Prime (corner 1434 1) := by change Nat.Prime 8228293; exact ProjectEulerSolutions.P58PrimeNodes.p_8228293.prime
    exact decide_eq_true prime
  second := false
  second_correct := by
    have composite : ¬ Nat.Prime (corner 1434 2) := by change ¬ Nat.Prime 8225425; exact PrattCertificates.composite_of_product (factor := 5) (cofactor := 1645085) rfl (by decide) (by decide)
    exact decide_eq_false composite
  third := false
  third_correct := by
    have composite : ¬ Nat.Prime (corner 1434 3) := by change ¬ Nat.Prime 8222557; exact PrattCertificates.composite_of_product (factor := 7) (cofactor := 1174651) rfl (by decide) (by decide)
    exact decide_eq_false composite
theorem row_count_1434 : row 1434 = 1 := (row_1434.checked).trans rfl
theorem model_count_1434 : modelRow 1434 = 1 := by decide
theorem modelrow_agrees_1434 : modelRow 1434 = row 1434 := model_count_1434.trans row_count_1434.symm
def row_1435 : RowCertificate 1435 where
  first := true
  first_correct := by
    have prime : Nat.Prime (corner 1435 1) := by change Nat.Prime 8239771; exact ProjectEulerSolutions.P58PrimeNodes.p_8239771.prime
    exact decide_eq_true prime
  second := false
  second_correct := by
    have composite : ¬ Nat.Prime (corner 1435 2) := by change ¬ Nat.Prime 8236901; exact PrattCertificates.composite_of_product (factor := 229) (cofactor := 35969) rfl (by decide) (by decide)
    exact decide_eq_false composite
  third := false
  third_correct := by
    have composite : ¬ Nat.Prime (corner 1435 3) := by change ¬ Nat.Prime 8234031; exact PrattCertificates.composite_of_product (factor := 3) (cofactor := 2744677) rfl (by decide) (by decide)
    exact decide_eq_false composite
theorem row_count_1435 : row 1435 = 1 := (row_1435.checked).trans rfl
theorem model_count_1435 : modelRow 1435 = 1 := by decide
theorem modelrow_agrees_1435 : modelRow 1435 = row 1435 := model_count_1435.trans row_count_1435.symm
def row_1436 : RowCertificate 1436 where
  first := false
  first_correct := by
    have composite : ¬ Nat.Prime (corner 1436 1) := by change ¬ Nat.Prime 8251257; exact PrattCertificates.composite_of_product (factor := 3) (cofactor := 2750419) rfl (by decide) (by decide)
    exact decide_eq_false composite
  second := false
  second_correct := by
    have composite : ¬ Nat.Prime (corner 1436 2) := by change ¬ Nat.Prime 8248385; exact PrattCertificates.composite_of_product (factor := 5) (cofactor := 1649677) rfl (by decide) (by decide)
    exact decide_eq_false composite
  third := true
  third_correct := by
    have prime : Nat.Prime (corner 1436 3) := by change Nat.Prime 8245513; exact ProjectEulerSolutions.P58PrimeNodes.p_8245513.prime
    exact decide_eq_true prime
theorem row_count_1436 : row 1436 = 1 := (row_1436.checked).trans rfl
theorem model_count_1436 : modelRow 1436 = 1 := by decide
theorem modelrow_agrees_1436 : modelRow 1436 = row 1436 := model_count_1436.trans row_count_1436.symm
def row_1437 : RowCertificate 1437 where
  first := false
  first_correct := by
    have composite : ¬ Nat.Prime (corner 1437 1) := by change ¬ Nat.Prime 8262751; exact PrattCertificates.composite_of_product (factor := 7) (cofactor := 1180393) rfl (by decide) (by decide)
    exact decide_eq_false composite
  second := false
  second_correct := by
    have composite : ¬ Nat.Prime (corner 1437 2) := by change ¬ Nat.Prime 8259877; exact PrattCertificates.composite_of_product (factor := 73) (cofactor := 113149) rfl (by decide) (by decide)
    exact decide_eq_false composite
  third := true
  third_correct := by
    have prime : Nat.Prime (corner 1437 3) := by change Nat.Prime 8257003; exact ProjectEulerSolutions.P58PrimeNodes.p_8257003.prime
    exact decide_eq_true prime
theorem row_count_1437 : row 1437 = 1 := (row_1437.checked).trans rfl
theorem model_count_1437 : modelRow 1437 = 1 := by decide
theorem modelrow_agrees_1437 : modelRow 1437 = row 1437 := model_count_1437.trans row_count_1437.symm
def row_1438 : RowCertificate 1438 where
  first := false
  first_correct := by
    have composite : ¬ Nat.Prime (corner 1438 1) := by change ¬ Nat.Prime 8274253; exact PrattCertificates.composite_of_product (factor := 13) (cofactor := 636481) rfl (by decide) (by decide)
    exact decide_eq_false composite
  second := true
  second_correct := by
    have prime : Nat.Prime (corner 1438 2) := by change Nat.Prime 8271377; exact ProjectEulerSolutions.P58PrimeNodes.p_8271377.prime
    exact decide_eq_true prime
  third := false
  third_correct := by
    have composite : ¬ Nat.Prime (corner 1438 3) := by change ¬ Nat.Prime 8268501; exact PrattCertificates.composite_of_product (factor := 3) (cofactor := 2756167) rfl (by decide) (by decide)
    exact decide_eq_false composite
theorem row_count_1438 : row 1438 = 1 := (row_1438.checked).trans rfl
theorem model_count_1438 : modelRow 1438 = 1 := by decide
theorem modelrow_agrees_1438 : modelRow 1438 = row 1438 := model_count_1438.trans row_count_1438.symm
def row_1439 : RowCertificate 1439 where
  first := false
  first_correct := by
    have composite : ¬ Nat.Prime (corner 1439 1) := by change ¬ Nat.Prime 8285763; exact PrattCertificates.composite_of_product (factor := 3) (cofactor := 2761921) rfl (by decide) (by decide)
    exact decide_eq_false composite
  second := false
  second_correct := by
    have composite : ¬ Nat.Prime (corner 1439 2) := by change ¬ Nat.Prime 8282885; exact PrattCertificates.composite_of_product (factor := 5) (cofactor := 1656577) rfl (by decide) (by decide)
    exact decide_eq_false composite
  third := false
  third_correct := by
    have composite : ¬ Nat.Prime (corner 1439 3) := by change ¬ Nat.Prime 8280007; exact PrattCertificates.composite_of_product (factor := 31) (cofactor := 267097) rfl (by decide) (by decide)
    exact decide_eq_false composite
theorem row_count_1439 : row 1439 = 0 := (row_1439.checked).trans rfl
theorem model_count_1439 : modelRow 1439 = 0 := by decide
theorem modelrow_agrees_1439 : modelRow 1439 = row 1439 := model_count_1439.trans row_count_1439.symm
def row_1440 : RowCertificate 1440 where
  first := false
  first_correct := by
    have composite : ¬ Nat.Prime (corner 1440 1) := by change ¬ Nat.Prime 8297281; exact PrattCertificates.composite_of_product (factor := 19) (cofactor := 436699) rfl (by decide) (by decide)
    exact decide_eq_false composite
  second := false
  second_correct := by
    have composite : ¬ Nat.Prime (corner 1440 2) := by change ¬ Nat.Prime 8294401; exact PrattCertificates.composite_of_product (factor := 37) (cofactor := 224173) rfl (by decide) (by decide)
    exact decide_eq_false composite
  third := false
  third_correct := by
    have composite : ¬ Nat.Prime (corner 1440 3) := by change ¬ Nat.Prime 8291521; exact PrattCertificates.composite_of_product (factor := 7) (cofactor := 1184503) rfl (by decide) (by decide)
    exact decide_eq_false composite
theorem row_count_1440 : row 1440 = 0 := (row_1440.checked).trans rfl
theorem model_count_1440 : modelRow 1440 = 0 := by decide
theorem modelrow_agrees_1440 : modelRow 1440 = row 1440 := model_count_1440.trans row_count_1440.symm
def row_1441 : RowCertificate 1441 where
  first := false
  first_correct := by
    have composite : ¬ Nat.Prime (corner 1441 1) := by change ¬ Nat.Prime 8308807; exact PrattCertificates.composite_of_product (factor := 13) (cofactor := 639139) rfl (by decide) (by decide)
    exact decide_eq_false composite
  second := false
  second_correct := by
    have composite : ¬ Nat.Prime (corner 1441 2) := by change ¬ Nat.Prime 8305925; exact PrattCertificates.composite_of_product (factor := 5) (cofactor := 1661185) rfl (by decide) (by decide)
    exact decide_eq_false composite
  third := false
  third_correct := by
    have composite : ¬ Nat.Prime (corner 1441 3) := by change ¬ Nat.Prime 8303043; exact PrattCertificates.composite_of_product (factor := 3) (cofactor := 2767681) rfl (by decide) (by decide)
    exact decide_eq_false composite
theorem row_count_1441 : row 1441 = 0 := (row_1441.checked).trans rfl
theorem model_count_1441 : modelRow 1441 = 0 := by decide
theorem modelrow_agrees_1441 : modelRow 1441 = row 1441 := model_count_1441.trans row_count_1441.symm
def row_1442 : RowCertificate 1442 where
  first := false
  first_correct := by
    have composite : ¬ Nat.Prime (corner 1442 1) := by change ¬ Nat.Prime 8320341; exact PrattCertificates.composite_of_product (factor := 3) (cofactor := 2773447) rfl (by decide) (by decide)
    exact decide_eq_false composite
  second := true
  second_correct := by
    have prime : Nat.Prime (corner 1442 2) := by change Nat.Prime 8317457; exact ProjectEulerSolutions.P58PrimeNodes.p_8317457.prime
    exact decide_eq_true prime
  third := true
  third_correct := by
    have prime : Nat.Prime (corner 1442 3) := by change Nat.Prime 8314573; exact ProjectEulerSolutions.P58PrimeNodes.p_8314573.prime
    exact decide_eq_true prime
theorem row_count_1442 : row 1442 = 2 := (row_1442.checked).trans rfl
theorem model_count_1442 : modelRow 1442 = 2 := by decide
theorem modelrow_agrees_1442 : modelRow 1442 = row 1442 := model_count_1442.trans row_count_1442.symm
def row_1443 : RowCertificate 1443 where
  first := false
  first_correct := by
    have composite : ¬ Nat.Prime (corner 1443 1) := by change ¬ Nat.Prime 8331883; exact PrattCertificates.composite_of_product (factor := 7) (cofactor := 1190269) rfl (by decide) (by decide)
    exact decide_eq_false composite
  second := false
  second_correct := by
    have composite : ¬ Nat.Prime (corner 1443 2) := by change ¬ Nat.Prime 8328997; exact PrattCertificates.composite_of_product (factor := 17) (cofactor := 489941) rfl (by decide) (by decide)
    exact decide_eq_false composite
  third := true
  third_correct := by
    have prime : Nat.Prime (corner 1443 3) := by change Nat.Prime 8326111; exact ProjectEulerSolutions.P58PrimeNodes.p_8326111.prime
    exact decide_eq_true prime
theorem row_count_1443 : row 1443 = 1 := (row_1443.checked).trans rfl
theorem model_count_1443 : modelRow 1443 = 1 := by decide
theorem modelrow_agrees_1443 : modelRow 1443 = row 1443 := model_count_1443.trans row_count_1443.symm
def row_1444 : RowCertificate 1444 where
  first := false
  first_correct := by
    have composite : ¬ Nat.Prime (corner 1444 1) := by change ¬ Nat.Prime 8343433; exact PrattCertificates.composite_of_product (factor := 7) (cofactor := 1191919) rfl (by decide) (by decide)
    exact decide_eq_false composite
  second := false
  second_correct := by
    have composite : ¬ Nat.Prime (corner 1444 2) := by change ¬ Nat.Prime 8340545; exact PrattCertificates.composite_of_product (factor := 5) (cofactor := 1668109) rfl (by decide) (by decide)
    exact decide_eq_false composite
  third := false
  third_correct := by
    have composite : ¬ Nat.Prime (corner 1444 3) := by change ¬ Nat.Prime 8337657; exact PrattCertificates.composite_of_product (factor := 3) (cofactor := 2779219) rfl (by decide) (by decide)
    exact decide_eq_false composite
theorem row_count_1444 : row 1444 = 0 := (row_1444.checked).trans rfl
theorem model_count_1444 : modelRow 1444 = 0 := by decide
theorem modelrow_agrees_1444 : modelRow 1444 = row 1444 := model_count_1444.trans row_count_1444.symm
def row_1445 : RowCertificate 1445 where
  first := false
  first_correct := by
    have composite : ¬ Nat.Prime (corner 1445 1) := by change ¬ Nat.Prime 8354991; exact PrattCertificates.composite_of_product (factor := 3) (cofactor := 2784997) rfl (by decide) (by decide)
    exact decide_eq_false composite
  second := true
  second_correct := by
    have prime : Nat.Prime (corner 1445 2) := by change Nat.Prime 8352101; exact ProjectEulerSolutions.P58PrimeNodes.p_8352101.prime
    exact decide_eq_true prime
  third := false
  third_correct := by
    have composite : ¬ Nat.Prime (corner 1445 3) := by change ¬ Nat.Prime 8349211; exact PrattCertificates.composite_of_product (factor := 13) (cofactor := 642247) rfl (by decide) (by decide)
    exact decide_eq_false composite
theorem row_count_1445 : row 1445 = 1 := (row_1445.checked).trans rfl
theorem model_count_1445 : modelRow 1445 = 1 := by decide
theorem modelrow_agrees_1445 : modelRow 1445 = row 1445 := model_count_1445.trans row_count_1445.symm
def row_1446 : RowCertificate 1446 where
  first := false
  first_correct := by
    have composite : ¬ Nat.Prime (corner 1446 1) := by change ¬ Nat.Prime 8366557; exact PrattCertificates.composite_of_product (factor := 199) (cofactor := 42043) rfl (by decide) (by decide)
    exact decide_eq_false composite
  second := false
  second_correct := by
    have composite : ¬ Nat.Prime (corner 1446 2) := by change ¬ Nat.Prime 8363665; exact PrattCertificates.composite_of_product (factor := 5) (cofactor := 1672733) rfl (by decide) (by decide)
    exact decide_eq_false composite
  third := true
  third_correct := by
    have prime : Nat.Prime (corner 1446 3) := by change Nat.Prime 8360773; exact ProjectEulerSolutions.P58PrimeNodes.p_8360773.prime
    exact decide_eq_true prime
theorem row_count_1446 : row 1446 = 1 := (row_1446.checked).trans rfl
theorem model_count_1446 : modelRow 1446 = 1 := by decide
theorem modelrow_agrees_1446 : modelRow 1446 = row 1446 := model_count_1446.trans row_count_1446.symm
def row_1447 : RowCertificate 1447 where
  first := true
  first_correct := by
    have prime : Nat.Prime (corner 1447 1) := by change Nat.Prime 8378131; exact ProjectEulerSolutions.P58PrimeNodes.p_8378131.prime
    exact decide_eq_true prime
  second := false
  second_correct := by
    have composite : ¬ Nat.Prime (corner 1447 2) := by change ¬ Nat.Prime 8375237; exact PrattCertificates.composite_of_product (factor := 13) (cofactor := 644249) rfl (by decide) (by decide)
    exact decide_eq_false composite
  third := false
  third_correct := by
    have composite : ¬ Nat.Prime (corner 1447 3) := by change ¬ Nat.Prime 8372343; exact PrattCertificates.composite_of_product (factor := 3) (cofactor := 2790781) rfl (by decide) (by decide)
    exact decide_eq_false composite
theorem row_count_1447 : row 1447 = 1 := (row_1447.checked).trans rfl
theorem model_count_1447 : modelRow 1447 = 1 := by decide
theorem modelrow_agrees_1447 : modelRow 1447 = row 1447 := model_count_1447.trans row_count_1447.symm
def row_1448 : RowCertificate 1448 where
  first := false
  first_correct := by
    have composite : ¬ Nat.Prime (corner 1448 1) := by change ¬ Nat.Prime 8389713; exact PrattCertificates.composite_of_product (factor := 3) (cofactor := 2796571) rfl (by decide) (by decide)
    exact decide_eq_false composite
  second := true
  second_correct := by
    have prime : Nat.Prime (corner 1448 2) := by change Nat.Prime 8386817; exact ProjectEulerSolutions.P58PrimeNodes.p_8386817.prime
    exact decide_eq_true prime
  third := false
  third_correct := by
    have composite : ¬ Nat.Prime (corner 1448 3) := by change ¬ Nat.Prime 8383921; exact PrattCertificates.composite_of_product (factor := 7) (cofactor := 1197703) rfl (by decide) (by decide)
    exact decide_eq_false composite
theorem row_count_1448 : row 1448 = 1 := (row_1448.checked).trans rfl
theorem model_count_1448 : modelRow 1448 = 1 := by decide
theorem modelrow_agrees_1448 : modelRow 1448 = row 1448 := model_count_1448.trans row_count_1448.symm
def row_1449 : RowCertificate 1449 where
  first := true
  first_correct := by
    have prime : Nat.Prime (corner 1449 1) := by change Nat.Prime 8401303; exact ProjectEulerSolutions.P58PrimeNodes.p_8401303.prime
    exact decide_eq_true prime
  second := false
  second_correct := by
    have composite : ¬ Nat.Prime (corner 1449 2) := by change ¬ Nat.Prime 8398405; exact PrattCertificates.composite_of_product (factor := 5) (cofactor := 1679681) rfl (by decide) (by decide)
    exact decide_eq_false composite
  third := false
  third_correct := by
    have composite : ¬ Nat.Prime (corner 1449 3) := by change ¬ Nat.Prime 8395507; exact PrattCertificates.composite_of_product (factor := 109) (cofactor := 77023) rfl (by decide) (by decide)
    exact decide_eq_false composite
theorem row_count_1449 : row 1449 = 1 := (row_1449.checked).trans rfl
theorem model_count_1449 : modelRow 1449 = 1 := by decide
theorem modelrow_agrees_1449 : modelRow 1449 = row 1449 := model_count_1449.trans row_count_1449.symm
def row_1450 : RowCertificate 1450 where
  first := false
  first_correct := by
    have composite : ¬ Nat.Prime (corner 1450 1) := by change ¬ Nat.Prime 8412901; exact PrattCertificates.composite_of_product (factor := 7) (cofactor := 1201843) rfl (by decide) (by decide)
    exact decide_eq_false composite
  second := true
  second_correct := by
    have prime : Nat.Prime (corner 1450 2) := by change Nat.Prime 8410001; exact ProjectEulerSolutions.P58PrimeNodes.p_8410001.prime
    exact decide_eq_true prime
  third := false
  third_correct := by
    have composite : ¬ Nat.Prime (corner 1450 3) := by change ¬ Nat.Prime 8407101; exact PrattCertificates.composite_of_product (factor := 3) (cofactor := 2802367) rfl (by decide) (by decide)
    exact decide_eq_false composite
theorem row_count_1450 : row 1450 = 1 := (row_1450.checked).trans rfl
theorem model_count_1450 : modelRow 1450 = 1 := by decide
theorem modelrow_agrees_1450 : modelRow 1450 = row 1450 := model_count_1450.trans row_count_1450.symm
def row_1451 : RowCertificate 1451 where
  first := false
  first_correct := by
    have composite : ¬ Nat.Prime (corner 1451 1) := by change ¬ Nat.Prime 8424507; exact PrattCertificates.composite_of_product (factor := 3) (cofactor := 2808169) rfl (by decide) (by decide)
    exact decide_eq_false composite
  second := false
  second_correct := by
    have composite : ¬ Nat.Prime (corner 1451 2) := by change ¬ Nat.Prime 8421605; exact PrattCertificates.composite_of_product (factor := 5) (cofactor := 1684321) rfl (by decide) (by decide)
    exact decide_eq_false composite
  third := false
  third_correct := by
    have composite : ¬ Nat.Prime (corner 1451 3) := by change ¬ Nat.Prime 8418703; exact PrattCertificates.composite_of_product (factor := 127) (cofactor := 66289) rfl (by decide) (by decide)
    exact decide_eq_false composite
theorem row_count_1451 : row 1451 = 0 := (row_1451.checked).trans rfl
theorem model_count_1451 : modelRow 1451 = 0 := by decide
theorem modelrow_agrees_1451 : modelRow 1451 = row 1451 := model_count_1451.trans row_count_1451.symm
def row_1452 : RowCertificate 1452 where
  first := false
  first_correct := by
    have composite : ¬ Nat.Prime (corner 1452 1) := by change ¬ Nat.Prime 8436121; exact PrattCertificates.composite_of_product (factor := 337) (cofactor := 25033) rfl (by decide) (by decide)
    exact decide_eq_false composite
  second := false
  second_correct := by
    have composite : ¬ Nat.Prime (corner 1452 2) := by change ¬ Nat.Prime 8433217; exact PrattCertificates.composite_of_product (factor := 13) (cofactor := 648709) rfl (by decide) (by decide)
    exact decide_eq_false composite
  third := false
  third_correct := by
    have composite : ¬ Nat.Prime (corner 1452 3) := by change ¬ Nat.Prime 8430313; exact PrattCertificates.composite_of_product (factor := 1033) (cofactor := 8161) rfl (by decide) (by decide)
    exact decide_eq_false composite
theorem row_count_1452 : row 1452 = 0 := (row_1452.checked).trans rfl
theorem model_count_1452 : modelRow 1452 = 0 := by decide
theorem modelrow_agrees_1452 : modelRow 1452 = row 1452 := model_count_1452.trans row_count_1452.symm
def row_1453 : RowCertificate 1453 where
  first := true
  first_correct := by
    have prime : Nat.Prime (corner 1453 1) := by change Nat.Prime 8447743; exact ProjectEulerSolutions.P58PrimeNodes.p_8447743.prime
    exact decide_eq_true prime
  second := false
  second_correct := by
    have composite : ¬ Nat.Prime (corner 1453 2) := by change ¬ Nat.Prime 8444837; exact PrattCertificates.composite_of_product (factor := 2549) (cofactor := 3313) rfl (by decide) (by decide)
    exact decide_eq_false composite
  third := false
  third_correct := by
    have composite : ¬ Nat.Prime (corner 1453 3) := by change ¬ Nat.Prime 8441931; exact PrattCertificates.composite_of_product (factor := 3) (cofactor := 2813977) rfl (by decide) (by decide)
    exact decide_eq_false composite
theorem row_count_1453 : row 1453 = 1 := (row_1453.checked).trans rfl
theorem model_count_1453 : modelRow 1453 = 1 := by decide
theorem modelrow_agrees_1453 : modelRow 1453 = row 1453 := model_count_1453.trans row_count_1453.symm
def row_1454 : RowCertificate 1454 where
  first := false
  first_correct := by
    have composite : ¬ Nat.Prime (corner 1454 1) := by change ¬ Nat.Prime 8459373; exact PrattCertificates.composite_of_product (factor := 3) (cofactor := 2819791) rfl (by decide) (by decide)
    exact decide_eq_false composite
  second := false
  second_correct := by
    have composite : ¬ Nat.Prime (corner 1454 2) := by change ¬ Nat.Prime 8456465; exact PrattCertificates.composite_of_product (factor := 5) (cofactor := 1691293) rfl (by decide) (by decide)
    exact decide_eq_false composite
  third := false
  third_correct := by
    have composite : ¬ Nat.Prime (corner 1454 3) := by change ¬ Nat.Prime 8453557; exact PrattCertificates.composite_of_product (factor := 7) (cofactor := 1207651) rfl (by decide) (by decide)
    exact decide_eq_false composite
theorem row_count_1454 : row 1454 = 0 := (row_1454.checked).trans rfl
theorem model_count_1454 : modelRow 1454 = 0 := by decide
theorem modelrow_agrees_1454 : modelRow 1454 = row 1454 := model_count_1454.trans row_count_1454.symm
def row_1455 : RowCertificate 1455 where
  first := false
  first_correct := by
    have composite : ¬ Nat.Prime (corner 1455 1) := by change ¬ Nat.Prime 8471011; exact PrattCertificates.composite_of_product (factor := 67) (cofactor := 126433) rfl (by decide) (by decide)
    exact decide_eq_false composite
  second := false
  second_correct := by
    have composite : ¬ Nat.Prime (corner 1455 2) := by change ¬ Nat.Prime 8468101; exact PrattCertificates.composite_of_product (factor := 109) (cofactor := 77689) rfl (by decide) (by decide)
    exact decide_eq_false composite
  third := false
  third_correct := by
    have composite : ¬ Nat.Prime (corner 1455 3) := by change ¬ Nat.Prime 8465191; exact PrattCertificates.composite_of_product (factor := 7) (cofactor := 1209313) rfl (by decide) (by decide)
    exact decide_eq_false composite
theorem row_count_1455 : row 1455 = 0 := (row_1455.checked).trans rfl
theorem model_count_1455 : modelRow 1455 = 0 := by decide
theorem modelrow_agrees_1455 : modelRow 1455 = row 1455 := model_count_1455.trans row_count_1455.symm
def row_1456 : RowCertificate 1456 where
  first := false
  first_correct := by
    have composite : ¬ Nat.Prime (corner 1456 1) := by change ¬ Nat.Prime 8482657; exact PrattCertificates.composite_of_product (factor := 37) (cofactor := 229261) rfl (by decide) (by decide)
    exact decide_eq_false composite
  second := false
  second_correct := by
    have composite : ¬ Nat.Prime (corner 1456 2) := by change ¬ Nat.Prime 8479745; exact PrattCertificates.composite_of_product (factor := 5) (cofactor := 1695949) rfl (by decide) (by decide)
    exact decide_eq_false composite
  third := false
  third_correct := by
    have composite : ¬ Nat.Prime (corner 1456 3) := by change ¬ Nat.Prime 8476833; exact PrattCertificates.composite_of_product (factor := 3) (cofactor := 2825611) rfl (by decide) (by decide)
    exact decide_eq_false composite
theorem row_count_1456 : row 1456 = 0 := (row_1456.checked).trans rfl
theorem model_count_1456 : modelRow 1456 = 0 := by decide
theorem modelrow_agrees_1456 : modelRow 1456 = row 1456 := model_count_1456.trans row_count_1456.symm
def row_1457 : RowCertificate 1457 where
  first := false
  first_correct := by
    have composite : ¬ Nat.Prime (corner 1457 1) := by change ¬ Nat.Prime 8494311; exact PrattCertificates.composite_of_product (factor := 3) (cofactor := 2831437) rfl (by decide) (by decide)
    exact decide_eq_false composite
  second := false
  second_correct := by
    have composite : ¬ Nat.Prime (corner 1457 2) := by change ¬ Nat.Prime 8491397; exact PrattCertificates.composite_of_product (factor := 137) (cofactor := 61981) rfl (by decide) (by decide)
    exact decide_eq_false composite
  third := false
  third_correct := by
    have composite : ¬ Nat.Prime (corner 1457 3) := by change ¬ Nat.Prime 8488483; exact PrattCertificates.composite_of_product (factor := 877) (cofactor := 9679) rfl (by decide) (by decide)
    exact decide_eq_false composite
theorem row_count_1457 : row 1457 = 0 := (row_1457.checked).trans rfl
theorem model_count_1457 : modelRow 1457 = 0 := by decide
theorem modelrow_agrees_1457 : modelRow 1457 = row 1457 := model_count_1457.trans row_count_1457.symm
def row_1458 : RowCertificate 1458 where
  first := false
  first_correct := by
    have composite : ¬ Nat.Prime (corner 1458 1) := by change ¬ Nat.Prime 8505973; exact PrattCertificates.composite_of_product (factor := 7) (cofactor := 1215139) rfl (by decide) (by decide)
    exact decide_eq_false composite
  second := true
  second_correct := by
    have prime : Nat.Prime (corner 1458 2) := by change Nat.Prime 8503057; exact ProjectEulerSolutions.P58PrimeNodes.p_8503057.prime
    exact decide_eq_true prime
  third := false
  third_correct := by
    have composite : ¬ Nat.Prime (corner 1458 3) := by change ¬ Nat.Prime 8500141; exact PrattCertificates.composite_of_product (factor := 13) (cofactor := 653857) rfl (by decide) (by decide)
    exact decide_eq_false composite
theorem row_count_1458 : row 1458 = 1 := (row_1458.checked).trans rfl
theorem model_count_1458 : modelRow 1458 = 1 := by decide
theorem modelrow_agrees_1458 : modelRow 1458 = row 1458 := model_count_1458.trans row_count_1458.symm
def row_1459 : RowCertificate 1459 where
  first := false
  first_correct := by
    have composite : ¬ Nat.Prime (corner 1459 1) := by change ¬ Nat.Prime 8517643; exact PrattCertificates.composite_of_product (factor := 19) (cofactor := 448297) rfl (by decide) (by decide)
    exact decide_eq_false composite
  second := false
  second_correct := by
    have composite : ¬ Nat.Prime (corner 1459 2) := by change ¬ Nat.Prime 8514725; exact PrattCertificates.composite_of_product (factor := 5) (cofactor := 1702945) rfl (by decide) (by decide)
    exact decide_eq_false composite
  third := false
  third_correct := by
    have composite : ¬ Nat.Prime (corner 1459 3) := by change ¬ Nat.Prime 8511807; exact PrattCertificates.composite_of_product (factor := 3) (cofactor := 2837269) rfl (by decide) (by decide)
    exact decide_eq_false composite
theorem row_count_1459 : row 1459 = 0 := (row_1459.checked).trans rfl
theorem model_count_1459 : modelRow 1459 = 0 := by decide
theorem modelrow_agrees_1459 : modelRow 1459 = row 1459 := model_count_1459.trans row_count_1459.symm
def row_1460 : RowCertificate 1460 where
  first := false
  first_correct := by
    have composite : ¬ Nat.Prime (corner 1460 1) := by change ¬ Nat.Prime 8529321; exact PrattCertificates.composite_of_product (factor := 3) (cofactor := 2843107) rfl (by decide) (by decide)
    exact decide_eq_false composite
  second := false
  second_correct := by
    have composite : ¬ Nat.Prime (corner 1460 2) := by change ¬ Nat.Prime 8526401; exact PrattCertificates.composite_of_product (factor := 13) (cofactor := 655877) rfl (by decide) (by decide)
    exact decide_eq_false composite
  third := false
  third_correct := by
    have composite : ¬ Nat.Prime (corner 1460 3) := by change ¬ Nat.Prime 8523481; exact PrattCertificates.composite_of_product (factor := 31) (cofactor := 274951) rfl (by decide) (by decide)
    exact decide_eq_false composite
theorem row_count_1460 : row 1460 = 0 := (row_1460.checked).trans rfl
theorem model_count_1460 : modelRow 1460 = 0 := by decide
theorem modelrow_agrees_1460 : modelRow 1460 = row 1460 := model_count_1460.trans row_count_1460.symm
def row_1461 : RowCertificate 1461 where
  first := true
  first_correct := by
    have prime : Nat.Prime (corner 1461 1) := by change Nat.Prime 8541007; exact ProjectEulerSolutions.P58PrimeNodes.p_8541007.prime
    exact decide_eq_true prime
  second := false
  second_correct := by
    have composite : ¬ Nat.Prime (corner 1461 2) := by change ¬ Nat.Prime 8538085; exact PrattCertificates.composite_of_product (factor := 5) (cofactor := 1707617) rfl (by decide) (by decide)
    exact decide_eq_false composite
  third := false
  third_correct := by
    have composite : ¬ Nat.Prime (corner 1461 3) := by change ¬ Nat.Prime 8535163; exact PrattCertificates.composite_of_product (factor := 7) (cofactor := 1219309) rfl (by decide) (by decide)
    exact decide_eq_false composite
theorem row_count_1461 : row 1461 = 1 := (row_1461.checked).trans rfl
theorem model_count_1461 : modelRow 1461 = 1 := by decide
theorem modelrow_agrees_1461 : modelRow 1461 = row 1461 := model_count_1461.trans row_count_1461.symm
def row_1462 : RowCertificate 1462 where
  first := false
  first_correct := by
    have composite : ¬ Nat.Prime (corner 1462 1) := by change ¬ Nat.Prime 8552701; exact PrattCertificates.composite_of_product (factor := 1777) (cofactor := 4813) rfl (by decide) (by decide)
    exact decide_eq_false composite
  second := true
  second_correct := by
    have prime : Nat.Prime (corner 1462 2) := by change Nat.Prime 8549777; exact ProjectEulerSolutions.P58PrimeNodes.p_8549777.prime
    exact decide_eq_true prime
  third := false
  third_correct := by
    have composite : ¬ Nat.Prime (corner 1462 3) := by change ¬ Nat.Prime 8546853; exact PrattCertificates.composite_of_product (factor := 3) (cofactor := 2848951) rfl (by decide) (by decide)
    exact decide_eq_false composite
theorem row_count_1462 : row 1462 = 1 := (row_1462.checked).trans rfl
theorem model_count_1462 : modelRow 1462 = 1 := by decide
theorem modelrow_agrees_1462 : modelRow 1462 = row 1462 := model_count_1462.trans row_count_1462.symm
def row_1463 : RowCertificate 1463 where
  first := false
  first_correct := by
    have composite : ¬ Nat.Prime (corner 1463 1) := by change ¬ Nat.Prime 8564403; exact PrattCertificates.composite_of_product (factor := 3) (cofactor := 2854801) rfl (by decide) (by decide)
    exact decide_eq_false composite
  second := true
  second_correct := by
    have prime : Nat.Prime (corner 1463 2) := by change Nat.Prime 8561477; exact ProjectEulerSolutions.P58PrimeNodes.p_8561477.prime
    exact decide_eq_true prime
  third := true
  third_correct := by
    have prime : Nat.Prime (corner 1463 3) := by change Nat.Prime 8558551; exact ProjectEulerSolutions.P58PrimeNodes.p_8558551.prime
    exact decide_eq_true prime
theorem row_count_1463 : row 1463 = 2 := (row_1463.checked).trans rfl
theorem model_count_1463 : modelRow 1463 = 2 := by decide
theorem modelrow_agrees_1463 : modelRow 1463 = row 1463 := model_count_1463.trans row_count_1463.symm
def row_1464 : RowCertificate 1464 where
  first := false
  first_correct := by
    have composite : ¬ Nat.Prime (corner 1464 1) := by change ¬ Nat.Prime 8576113; exact PrattCertificates.composite_of_product (factor := 7) (cofactor := 1225159) rfl (by decide) (by decide)
    exact decide_eq_false composite
  second := false
  second_correct := by
    have composite : ¬ Nat.Prime (corner 1464 2) := by change ¬ Nat.Prime 8573185; exact PrattCertificates.composite_of_product (factor := 5) (cofactor := 1714637) rfl (by decide) (by decide)
    exact decide_eq_false composite
  third := false
  third_correct := by
    have composite : ¬ Nat.Prime (corner 1464 3) := by change ¬ Nat.Prime 8570257; exact PrattCertificates.composite_of_product (factor := 1483) (cofactor := 5779) rfl (by decide) (by decide)
    exact decide_eq_false composite
theorem row_count_1464 : row 1464 = 0 := (row_1464.checked).trans rfl
theorem model_count_1464 : modelRow 1464 = 0 := by decide
theorem modelrow_agrees_1464 : modelRow 1464 = row 1464 := model_count_1464.trans row_count_1464.symm
def row_1465 : RowCertificate 1465 where
  first := false
  first_correct := by
    have composite : ¬ Nat.Prime (corner 1465 1) := by change ¬ Nat.Prime 8587831; exact PrattCertificates.composite_of_product (factor := 7) (cofactor := 1226833) rfl (by decide) (by decide)
    exact decide_eq_false composite
  second := false
  second_correct := by
    have composite : ¬ Nat.Prime (corner 1465 2) := by change ¬ Nat.Prime 8584901; exact PrattCertificates.composite_of_product (factor := 13) (cofactor := 660377) rfl (by decide) (by decide)
    exact decide_eq_false composite
  third := false
  third_correct := by
    have composite : ¬ Nat.Prime (corner 1465 3) := by change ¬ Nat.Prime 8581971; exact PrattCertificates.composite_of_product (factor := 3) (cofactor := 2860657) rfl (by decide) (by decide)
    exact decide_eq_false composite
theorem row_count_1465 : row 1465 = 0 := (row_1465.checked).trans rfl
theorem model_count_1465 : modelRow 1465 = 0 := by decide
theorem modelrow_agrees_1465 : modelRow 1465 = row 1465 := model_count_1465.trans row_count_1465.symm
def row_1466 : RowCertificate 1466 where
  first := false
  first_correct := by
    have composite : ¬ Nat.Prime (corner 1466 1) := by change ¬ Nat.Prime 8599557; exact PrattCertificates.composite_of_product (factor := 3) (cofactor := 2866519) rfl (by decide) (by decide)
    exact decide_eq_false composite
  second := false
  second_correct := by
    have composite : ¬ Nat.Prime (corner 1466 2) := by change ¬ Nat.Prime 8596625; exact PrattCertificates.composite_of_product (factor := 5) (cofactor := 1719325) rfl (by decide) (by decide)
    exact decide_eq_false composite
  third := false
  third_correct := by
    have composite : ¬ Nat.Prime (corner 1466 3) := by change ¬ Nat.Prime 8593693; exact PrattCertificates.composite_of_product (factor := 751) (cofactor := 11443) rfl (by decide) (by decide)
    exact decide_eq_false composite
theorem row_count_1466 : row 1466 = 0 := (row_1466.checked).trans rfl
theorem model_count_1466 : modelRow 1466 = 0 := by decide
theorem modelrow_agrees_1466 : modelRow 1466 = row 1466 := model_count_1466.trans row_count_1466.symm
def row_1467 : RowCertificate 1467 where
  first := false
  first_correct := by
    have composite : ¬ Nat.Prime (corner 1467 1) := by change ¬ Nat.Prime 8611291; exact PrattCertificates.composite_of_product (factor := 13) (cofactor := 662407) rfl (by decide) (by decide)
    exact decide_eq_false composite
  second := true
  second_correct := by
    have prime : Nat.Prime (corner 1467 2) := by change Nat.Prime 8608357; exact ProjectEulerSolutions.P58PrimeNodes.p_8608357.prime
    exact decide_eq_true prime
  third := false
  third_correct := by
    have composite : ¬ Nat.Prime (corner 1467 3) := by change ¬ Nat.Prime 8605423; exact PrattCertificates.composite_of_product (factor := 19) (cofactor := 452917) rfl (by decide) (by decide)
    exact decide_eq_false composite
theorem row_count_1467 : row 1467 = 1 := (row_1467.checked).trans rfl
theorem model_count_1467 : modelRow 1467 = 1 := by decide
theorem modelrow_agrees_1467 : modelRow 1467 = row 1467 := model_count_1467.trans row_count_1467.symm
def row_1468 : RowCertificate 1468 where
  first := true
  first_correct := by
    have prime : Nat.Prime (corner 1468 1) := by change Nat.Prime 8623033; exact ProjectEulerSolutions.P58PrimeNodes.p_8623033.prime
    exact decide_eq_true prime
  second := false
  second_correct := by
    have composite : ¬ Nat.Prime (corner 1468 2) := by change ¬ Nat.Prime 8620097; exact PrattCertificates.composite_of_product (factor := 149) (cofactor := 57853) rfl (by decide) (by decide)
    exact decide_eq_false composite
  third := false
  third_correct := by
    have composite : ¬ Nat.Prime (corner 1468 3) := by change ¬ Nat.Prime 8617161; exact PrattCertificates.composite_of_product (factor := 3) (cofactor := 2872387) rfl (by decide) (by decide)
    exact decide_eq_false composite
theorem row_count_1468 : row 1468 = 1 := (row_1468.checked).trans rfl
theorem model_count_1468 : modelRow 1468 = 1 := by decide
theorem modelrow_agrees_1468 : modelRow 1468 = row 1468 := model_count_1468.trans row_count_1468.symm
def row_1469 : RowCertificate 1469 where
  first := false
  first_correct := by
    have composite : ¬ Nat.Prime (corner 1469 1) := by change ¬ Nat.Prime 8634783; exact PrattCertificates.composite_of_product (factor := 3) (cofactor := 2878261) rfl (by decide) (by decide)
    exact decide_eq_false composite
  second := false
  second_correct := by
    have composite : ¬ Nat.Prime (corner 1469 2) := by change ¬ Nat.Prime 8631845; exact PrattCertificates.composite_of_product (factor := 5) (cofactor := 1726369) rfl (by decide) (by decide)
    exact decide_eq_false composite
  third := false
  third_correct := by
    have composite : ¬ Nat.Prime (corner 1469 3) := by change ¬ Nat.Prime 8628907; exact PrattCertificates.composite_of_product (factor := 7) (cofactor := 1232701) rfl (by decide) (by decide)
    exact decide_eq_false composite
theorem row_count_1469 : row 1469 = 0 := (row_1469.checked).trans rfl
theorem model_count_1469 : modelRow 1469 = 0 := by decide
theorem modelrow_agrees_1469 : modelRow 1469 = row 1469 := model_count_1469.trans row_count_1469.symm
def row_1470 : RowCertificate 1470 where
  first := false
  first_correct := by
    have composite : ¬ Nat.Prime (corner 1470 1) := by change ¬ Nat.Prime 8646541; exact PrattCertificates.composite_of_product (factor := 103) (cofactor := 83947) rfl (by decide) (by decide)
    exact decide_eq_false composite
  second := false
  second_correct := by
    have composite : ¬ Nat.Prime (corner 1470 2) := by change ¬ Nat.Prime 8643601; exact PrattCertificates.composite_of_product (factor := 233) (cofactor := 37097) rfl (by decide) (by decide)
    exact decide_eq_false composite
  third := false
  third_correct := by
    have composite : ¬ Nat.Prime (corner 1470 3) := by change ¬ Nat.Prime 8640661; exact PrattCertificates.composite_of_product (factor := 31) (cofactor := 278731) rfl (by decide) (by decide)
    exact decide_eq_false composite
theorem row_count_1470 : row 1470 = 0 := (row_1470.checked).trans rfl
theorem model_count_1470 : modelRow 1470 = 0 := by decide
theorem modelrow_agrees_1470 : modelRow 1470 = row 1470 := model_count_1470.trans row_count_1470.symm
def row_1471 : RowCertificate 1471 where
  first := false
  first_correct := by
    have composite : ¬ Nat.Prime (corner 1471 1) := by change ¬ Nat.Prime 8658307; exact PrattCertificates.composite_of_product (factor := 7) (cofactor := 1236901) rfl (by decide) (by decide)
    exact decide_eq_false composite
  second := false
  second_correct := by
    have composite : ¬ Nat.Prime (corner 1471 2) := by change ¬ Nat.Prime 8655365; exact PrattCertificates.composite_of_product (factor := 5) (cofactor := 1731073) rfl (by decide) (by decide)
    exact decide_eq_false composite
  third := false
  third_correct := by
    have composite : ¬ Nat.Prime (corner 1471 3) := by change ¬ Nat.Prime 8652423; exact PrattCertificates.composite_of_product (factor := 3) (cofactor := 2884141) rfl (by decide) (by decide)
    exact decide_eq_false composite
theorem row_count_1471 : row 1471 = 0 := (row_1471.checked).trans rfl
theorem model_count_1471 : modelRow 1471 = 0 := by decide
theorem modelrow_agrees_1471 : modelRow 1471 = row 1471 := model_count_1471.trans row_count_1471.symm
def row_1472 : RowCertificate 1472 where
  first := false
  first_correct := by
    have composite : ¬ Nat.Prime (corner 1472 1) := by change ¬ Nat.Prime 8670081; exact PrattCertificates.composite_of_product (factor := 3) (cofactor := 2890027) rfl (by decide) (by decide)
    exact decide_eq_false composite
  second := true
  second_correct := by
    have prime : Nat.Prime (corner 1472 2) := by change Nat.Prime 8667137; exact ProjectEulerSolutions.P58PrimeNodes.p_8667137.prime
    exact decide_eq_true prime
  third := true
  third_correct := by
    have prime : Nat.Prime (corner 1472 3) := by change Nat.Prime 8664193; exact ProjectEulerSolutions.P58PrimeNodes.p_8664193.prime
    exact decide_eq_true prime
theorem row_count_1472 : row 1472 = 2 := (row_1472.checked).trans rfl
theorem model_count_1472 : modelRow 1472 = 2 := by decide
theorem modelrow_agrees_1472 : modelRow 1472 = row 1472 := model_count_1472.trans row_count_1472.symm
def row_1473 : RowCertificate 1473 where
  first := false
  first_correct := by
    have composite : ¬ Nat.Prime (corner 1473 1) := by change ¬ Nat.Prime 8681863; exact PrattCertificates.composite_of_product (factor := 79) (cofactor := 109897) rfl (by decide) (by decide)
    exact decide_eq_false composite
  second := false
  second_correct := by
    have composite : ¬ Nat.Prime (corner 1473 2) := by change ¬ Nat.Prime 8678917; exact PrattCertificates.composite_of_product (factor := 13) (cofactor := 667609) rfl (by decide) (by decide)
    exact decide_eq_false composite
  third := false
  third_correct := by
    have composite : ¬ Nat.Prime (corner 1473 3) := by change ¬ Nat.Prime 8675971; exact PrattCertificates.composite_of_product (factor := 97) (cofactor := 89443) rfl (by decide) (by decide)
    exact decide_eq_false composite
theorem row_count_1473 : row 1473 = 0 := (row_1473.checked).trans rfl
theorem model_count_1473 : modelRow 1473 = 0 := by decide
theorem modelrow_agrees_1473 : modelRow 1473 = row 1473 := model_count_1473.trans row_count_1473.symm
def row_1474 : RowCertificate 1474 where
  first := true
  first_correct := by
    have prime : Nat.Prime (corner 1474 1) := by change Nat.Prime 8693653; exact ProjectEulerSolutions.P58PrimeNodes.p_8693653.prime
    exact decide_eq_true prime
  second := false
  second_correct := by
    have composite : ¬ Nat.Prime (corner 1474 2) := by change ¬ Nat.Prime 8690705; exact PrattCertificates.composite_of_product (factor := 5) (cofactor := 1738141) rfl (by decide) (by decide)
    exact decide_eq_false composite
  third := false
  third_correct := by
    have composite : ¬ Nat.Prime (corner 1474 3) := by change ¬ Nat.Prime 8687757; exact PrattCertificates.composite_of_product (factor := 3) (cofactor := 2895919) rfl (by decide) (by decide)
    exact decide_eq_false composite
theorem row_count_1474 : row 1474 = 1 := (row_1474.checked).trans rfl
theorem model_count_1474 : modelRow 1474 = 1 := by decide
theorem modelrow_agrees_1474 : modelRow 1474 = row 1474 := model_count_1474.trans row_count_1474.symm
def row_1475 : RowCertificate 1475 where
  first := false
  first_correct := by
    have composite : ¬ Nat.Prime (corner 1475 1) := by change ¬ Nat.Prime 8705451; exact PrattCertificates.composite_of_product (factor := 3) (cofactor := 2901817) rfl (by decide) (by decide)
    exact decide_eq_false composite
  second := false
  second_correct := by
    have composite : ¬ Nat.Prime (corner 1475 2) := by change ¬ Nat.Prime 8702501; exact PrattCertificates.composite_of_product (factor := 1097) (cofactor := 7933) rfl (by decide) (by decide)
    exact decide_eq_false composite
  third := false
  third_correct := by
    have composite : ¬ Nat.Prime (corner 1475 3) := by change ¬ Nat.Prime 8699551; exact PrattCertificates.composite_of_product (factor := 7) (cofactor := 1242793) rfl (by decide) (by decide)
    exact decide_eq_false composite
theorem row_count_1475 : row 1475 = 0 := (row_1475.checked).trans rfl
theorem model_count_1475 : modelRow 1475 = 0 := by decide
theorem modelrow_agrees_1475 : modelRow 1475 = row 1475 := model_count_1475.trans row_count_1475.symm
def row_1476 : RowCertificate 1476 where
  first := false
  first_correct := by
    have composite : ¬ Nat.Prime (corner 1476 1) := by change ¬ Nat.Prime 8717257; exact PrattCertificates.composite_of_product (factor := 19) (cofactor := 458803) rfl (by decide) (by decide)
    exact decide_eq_false composite
  second := false
  second_correct := by
    have composite : ¬ Nat.Prime (corner 1476 2) := by change ¬ Nat.Prime 8714305; exact PrattCertificates.composite_of_product (factor := 5) (cofactor := 1742861) rfl (by decide) (by decide)
    exact decide_eq_false composite
  third := false
  third_correct := by
    have composite : ¬ Nat.Prime (corner 1476 3) := by change ¬ Nat.Prime 8711353; exact PrattCertificates.composite_of_product (factor := 7) (cofactor := 1244479) rfl (by decide) (by decide)
    exact decide_eq_false composite
theorem row_count_1476 : row 1476 = 0 := (row_1476.checked).trans rfl
theorem model_count_1476 : modelRow 1476 = 0 := by decide
theorem modelrow_agrees_1476 : modelRow 1476 = row 1476 := model_count_1476.trans row_count_1476.symm
def row_1477 : RowCertificate 1477 where
  first := false
  first_correct := by
    have composite : ¬ Nat.Prime (corner 1477 1) := by change ¬ Nat.Prime 8729071; exact PrattCertificates.composite_of_product (factor := 13) (cofactor := 671467) rfl (by decide) (by decide)
    exact decide_eq_false composite
  second := false
  second_correct := by
    have composite : ¬ Nat.Prime (corner 1477 2) := by change ¬ Nat.Prime 8726117; exact PrattCertificates.composite_of_product (factor := 17) (cofactor := 513301) rfl (by decide) (by decide)
    exact decide_eq_false composite
  third := false
  third_correct := by
    have composite : ¬ Nat.Prime (corner 1477 3) := by change ¬ Nat.Prime 8723163; exact PrattCertificates.composite_of_product (factor := 3) (cofactor := 2907721) rfl (by decide) (by decide)
    exact decide_eq_false composite
theorem row_count_1477 : row 1477 = 0 := (row_1477.checked).trans rfl
theorem model_count_1477 : modelRow 1477 = 0 := by decide
theorem modelrow_agrees_1477 : modelRow 1477 = row 1477 := model_count_1477.trans row_count_1477.symm
def row_1478 : RowCertificate 1478 where
  first := false
  first_correct := by
    have composite : ¬ Nat.Prime (corner 1478 1) := by change ¬ Nat.Prime 8740893; exact PrattCertificates.composite_of_product (factor := 3) (cofactor := 2913631) rfl (by decide) (by decide)
    exact decide_eq_false composite
  second := false
  second_correct := by
    have composite : ¬ Nat.Prime (corner 1478 2) := by change ¬ Nat.Prime 8737937; exact PrattCertificates.composite_of_product (factor := 13) (cofactor := 672149) rfl (by decide) (by decide)
    exact decide_eq_false composite
  third := true
  third_correct := by
    have prime : Nat.Prime (corner 1478 3) := by change Nat.Prime 8734981; exact ProjectEulerSolutions.P58PrimeNodes.p_8734981.prime
    exact decide_eq_true prime
theorem row_count_1478 : row 1478 = 1 := (row_1478.checked).trans rfl
theorem model_count_1478 : modelRow 1478 = 1 := by decide
theorem modelrow_agrees_1478 : modelRow 1478 = row 1478 := model_count_1478.trans row_count_1478.symm
def row_1479 : RowCertificate 1479 where
  first := false
  first_correct := by
    have composite : ¬ Nat.Prime (corner 1479 1) := by change ¬ Nat.Prime 8752723; exact PrattCertificates.composite_of_product (factor := 7) (cofactor := 1250389) rfl (by decide) (by decide)
    exact decide_eq_false composite
  second := false
  second_correct := by
    have composite : ¬ Nat.Prime (corner 1479 2) := by change ¬ Nat.Prime 8749765; exact PrattCertificates.composite_of_product (factor := 5) (cofactor := 1749953) rfl (by decide) (by decide)
    exact decide_eq_false composite
  third := true
  third_correct := by
    have prime : Nat.Prime (corner 1479 3) := by change Nat.Prime 8746807; exact ProjectEulerSolutions.P58PrimeNodes.p_8746807.prime
    exact decide_eq_true prime
theorem row_count_1479 : row 1479 = 1 := (row_1479.checked).trans rfl
theorem model_count_1479 : modelRow 1479 = 1 := by decide
theorem modelrow_agrees_1479 : modelRow 1479 = row 1479 := model_count_1479.trans row_count_1479.symm
def row_1480 : RowCertificate 1480 where
  first := false
  first_correct := by
    have composite : ¬ Nat.Prime (corner 1480 1) := by change ¬ Nat.Prime 8764561; exact PrattCertificates.composite_of_product (factor := 13) (cofactor := 674197) rfl (by decide) (by decide)
    exact decide_eq_false composite
  second := true
  second_correct := by
    have prime : Nat.Prime (corner 1480 2) := by change Nat.Prime 8761601; exact ProjectEulerSolutions.P58PrimeNodes.p_8761601.prime
    exact decide_eq_true prime
  third := false
  third_correct := by
    have composite : ¬ Nat.Prime (corner 1480 3) := by change ¬ Nat.Prime 8758641; exact PrattCertificates.composite_of_product (factor := 3) (cofactor := 2919547) rfl (by decide) (by decide)
    exact decide_eq_false composite
theorem row_count_1480 : row 1480 = 1 := (row_1480.checked).trans rfl
theorem model_count_1480 : modelRow 1480 = 1 := by decide
theorem modelrow_agrees_1480 : modelRow 1480 = row 1480 := model_count_1480.trans row_count_1480.symm
def row_1481 : RowCertificate 1481 where
  first := false
  first_correct := by
    have composite : ¬ Nat.Prime (corner 1481 1) := by change ¬ Nat.Prime 8776407; exact PrattCertificates.composite_of_product (factor := 3) (cofactor := 2925469) rfl (by decide) (by decide)
    exact decide_eq_false composite
  second := false
  second_correct := by
    have composite : ¬ Nat.Prime (corner 1481 2) := by change ¬ Nat.Prime 8773445; exact PrattCertificates.composite_of_product (factor := 5) (cofactor := 1754689) rfl (by decide) (by decide)
    exact decide_eq_false composite
  third := false
  third_correct := by
    have composite : ¬ Nat.Prime (corner 1481 3) := by change ¬ Nat.Prime 8770483; exact PrattCertificates.composite_of_product (factor := 139) (cofactor := 63097) rfl (by decide) (by decide)
    exact decide_eq_false composite
theorem row_count_1481 : row 1481 = 0 := (row_1481.checked).trans rfl
theorem model_count_1481 : modelRow 1481 = 0 := by decide
theorem modelrow_agrees_1481 : modelRow 1481 = row 1481 := model_count_1481.trans row_count_1481.symm
def row_1482 : RowCertificate 1482 where
  first := false
  first_correct := by
    have composite : ¬ Nat.Prime (corner 1482 1) := by change ¬ Nat.Prime 8788261; exact PrattCertificates.composite_of_product (factor := 571) (cofactor := 15391) rfl (by decide) (by decide)
    exact decide_eq_false composite
  second := true
  second_correct := by
    have prime : Nat.Prime (corner 1482 2) := by change Nat.Prime 8785297; exact ProjectEulerSolutions.P58PrimeNodes.p_8785297.prime
    exact decide_eq_true prime
  third := false
  third_correct := by
    have composite : ¬ Nat.Prime (corner 1482 3) := by change ¬ Nat.Prime 8782333; exact PrattCertificates.composite_of_product (factor := 7) (cofactor := 1254619) rfl (by decide) (by decide)
    exact decide_eq_false composite
theorem row_count_1482 : row 1482 = 1 := (row_1482.checked).trans rfl
theorem model_count_1482 : modelRow 1482 = 1 := by decide
theorem modelrow_agrees_1482 : modelRow 1482 = row 1482 := model_count_1482.trans row_count_1482.symm
def row_1483 : RowCertificate 1483 where
  first := true
  first_correct := by
    have prime : Nat.Prime (corner 1483 1) := by change Nat.Prime 8800123; exact ProjectEulerSolutions.P58PrimeNodes.p_8800123.prime
    exact decide_eq_true prime
  second := false
  second_correct := by
    have composite : ¬ Nat.Prime (corner 1483 2) := by change ¬ Nat.Prime 8797157; exact PrattCertificates.composite_of_product (factor := 37) (cofactor := 237761) rfl (by decide) (by decide)
    exact decide_eq_false composite
  third := false
  third_correct := by
    have composite : ¬ Nat.Prime (corner 1483 3) := by change ¬ Nat.Prime 8794191; exact PrattCertificates.composite_of_product (factor := 3) (cofactor := 2931397) rfl (by decide) (by decide)
    exact decide_eq_false composite
theorem row_count_1483 : row 1483 = 1 := (row_1483.checked).trans rfl
theorem model_count_1483 : modelRow 1483 = 1 := by decide
theorem modelrow_agrees_1483 : modelRow 1483 = row 1483 := model_count_1483.trans row_count_1483.symm
def row_1484 : RowCertificate 1484 where
  first := false
  first_correct := by
    have composite : ¬ Nat.Prime (corner 1484 1) := by change ¬ Nat.Prime 8811993; exact PrattCertificates.composite_of_product (factor := 3) (cofactor := 2937331) rfl (by decide) (by decide)
    exact decide_eq_false composite
  second := false
  second_correct := by
    have composite : ¬ Nat.Prime (corner 1484 2) := by change ¬ Nat.Prime 8809025; exact PrattCertificates.composite_of_product (factor := 5) (cofactor := 1761805) rfl (by decide) (by decide)
    exact decide_eq_false composite
  third := false
  third_correct := by
    have composite : ¬ Nat.Prime (corner 1484 3) := by change ¬ Nat.Prime 8806057; exact PrattCertificates.composite_of_product (factor := 13) (cofactor := 677389) rfl (by decide) (by decide)
    exact decide_eq_false composite
theorem row_count_1484 : row 1484 = 0 := (row_1484.checked).trans rfl
theorem model_count_1484 : modelRow 1484 = 0 := by decide
theorem modelrow_agrees_1484 : modelRow 1484 = row 1484 := model_count_1484.trans row_count_1484.symm
def row_1485 : RowCertificate 1485 where
  first := false
  first_correct := by
    have composite : ¬ Nat.Prime (corner 1485 1) := by change ¬ Nat.Prime 8823871; exact PrattCertificates.composite_of_product (factor := 7) (cofactor := 1260553) rfl (by decide) (by decide)
    exact decide_eq_false composite
  second := false
  second_correct := by
    have composite : ¬ Nat.Prime (corner 1485 2) := by change ¬ Nat.Prime 8820901; exact PrattCertificates.composite_of_product (factor := 29) (cofactor := 304169) rfl (by decide) (by decide)
    exact decide_eq_false composite
  third := true
  third_correct := by
    have prime : Nat.Prime (corner 1485 3) := by change Nat.Prime 8817931; exact ProjectEulerSolutions.P58PrimeNodes.p_8817931.prime
    exact decide_eq_true prime
theorem row_count_1485 : row 1485 = 1 := (row_1485.checked).trans rfl
theorem model_count_1485 : modelRow 1485 = 1 := by decide
theorem modelrow_agrees_1485 : modelRow 1485 = row 1485 := model_count_1485.trans row_count_1485.symm
def row_1486 : RowCertificate 1486 where
  first := false
  first_correct := by
    have composite : ¬ Nat.Prime (corner 1486 1) := by change ¬ Nat.Prime 8835757; exact PrattCertificates.composite_of_product (factor := 7) (cofactor := 1262251) rfl (by decide) (by decide)
    exact decide_eq_false composite
  second := false
  second_correct := by
    have composite : ¬ Nat.Prime (corner 1486 2) := by change ¬ Nat.Prime 8832785; exact PrattCertificates.composite_of_product (factor := 5) (cofactor := 1766557) rfl (by decide) (by decide)
    exact decide_eq_false composite
  third := false
  third_correct := by
    have composite : ¬ Nat.Prime (corner 1486 3) := by change ¬ Nat.Prime 8829813; exact PrattCertificates.composite_of_product (factor := 3) (cofactor := 2943271) rfl (by decide) (by decide)
    exact decide_eq_false composite
theorem row_count_1486 : row 1486 = 0 := (row_1486.checked).trans rfl
theorem model_count_1486 : modelRow 1486 = 0 := by decide
theorem modelrow_agrees_1486 : modelRow 1486 = row 1486 := model_count_1486.trans row_count_1486.symm
def row_1487 : RowCertificate 1487 where
  first := false
  first_correct := by
    have composite : ¬ Nat.Prime (corner 1487 1) := by change ¬ Nat.Prime 8847651; exact PrattCertificates.composite_of_product (factor := 3) (cofactor := 2949217) rfl (by decide) (by decide)
    exact decide_eq_false composite
  second := true
  second_correct := by
    have prime : Nat.Prime (corner 1487 2) := by change Nat.Prime 8844677; exact ProjectEulerSolutions.P58PrimeNodes.p_8844677.prime
    exact decide_eq_true prime
  third := false
  third_correct := by
    have composite : ¬ Nat.Prime (corner 1487 3) := by change ¬ Nat.Prime 8841703; exact PrattCertificates.composite_of_product (factor := 13) (cofactor := 680131) rfl (by decide) (by decide)
    exact decide_eq_false composite
theorem row_count_1487 : row 1487 = 1 := (row_1487.checked).trans rfl
theorem model_count_1487 : modelRow 1487 = 1 := by decide
theorem modelrow_agrees_1487 : modelRow 1487 = row 1487 := model_count_1487.trans row_count_1487.symm
def row_1488 : RowCertificate 1488 where
  first := false
  first_correct := by
    have composite : ¬ Nat.Prime (corner 1488 1) := by change ¬ Nat.Prime 8859553; exact PrattCertificates.composite_of_product (factor := 1039) (cofactor := 8527) rfl (by decide) (by decide)
    exact decide_eq_false composite
  second := false
  second_correct := by
    have composite : ¬ Nat.Prime (corner 1488 2) := by change ¬ Nat.Prime 8856577; exact PrattCertificates.composite_of_product (factor := 109) (cofactor := 81253) rfl (by decide) (by decide)
    exact decide_eq_false composite
  third := false
  third_correct := by
    have composite : ¬ Nat.Prime (corner 1488 3) := by change ¬ Nat.Prime 8853601; exact PrattCertificates.composite_of_product (factor := 19) (cofactor := 465979) rfl (by decide) (by decide)
    exact decide_eq_false composite
theorem row_count_1488 : row 1488 = 0 := (row_1488.checked).trans rfl
theorem model_count_1488 : modelRow 1488 = 0 := by decide
theorem modelrow_agrees_1488 : modelRow 1488 = row 1488 := model_count_1488.trans row_count_1488.symm
def row_1489 : RowCertificate 1489 where
  first := false
  first_correct := by
    have composite : ¬ Nat.Prime (corner 1489 1) := by change ¬ Nat.Prime 8871463; exact PrattCertificates.composite_of_product (factor := 79) (cofactor := 112297) rfl (by decide) (by decide)
    exact decide_eq_false composite
  second := false
  second_correct := by
    have composite : ¬ Nat.Prime (corner 1489 2) := by change ¬ Nat.Prime 8868485; exact PrattCertificates.composite_of_product (factor := 5) (cofactor := 1773697) rfl (by decide) (by decide)
    exact decide_eq_false composite
  third := false
  third_correct := by
    have composite : ¬ Nat.Prime (corner 1489 3) := by change ¬ Nat.Prime 8865507; exact PrattCertificates.composite_of_product (factor := 3) (cofactor := 2955169) rfl (by decide) (by decide)
    exact decide_eq_false composite
theorem row_count_1489 : row 1489 = 0 := (row_1489.checked).trans rfl
theorem model_count_1489 : modelRow 1489 = 0 := by decide
theorem modelrow_agrees_1489 : modelRow 1489 = row 1489 := model_count_1489.trans row_count_1489.symm
def row_1490 : RowCertificate 1490 where
  first := false
  first_correct := by
    have composite : ¬ Nat.Prime (corner 1490 1) := by change ¬ Nat.Prime 8883381; exact PrattCertificates.composite_of_product (factor := 3) (cofactor := 2961127) rfl (by decide) (by decide)
    exact decide_eq_false composite
  second := false
  second_correct := by
    have composite : ¬ Nat.Prime (corner 1490 2) := by change ¬ Nat.Prime 8880401; exact PrattCertificates.composite_of_product (factor := 1697) (cofactor := 5233) rfl (by decide) (by decide)
    exact decide_eq_false composite
  third := false
  third_correct := by
    have composite : ¬ Nat.Prime (corner 1490 3) := by change ¬ Nat.Prime 8877421; exact PrattCertificates.composite_of_product (factor := 7) (cofactor := 1268203) rfl (by decide) (by decide)
    exact decide_eq_false composite
theorem row_count_1490 : row 1490 = 0 := (row_1490.checked).trans rfl
theorem model_count_1490 : modelRow 1490 = 0 := by decide
theorem modelrow_agrees_1490 : modelRow 1490 = row 1490 := model_count_1490.trans row_count_1490.symm
def row_1491 : RowCertificate 1491 where
  first := true
  first_correct := by
    have prime : Nat.Prime (corner 1491 1) := by change Nat.Prime 8895307; exact ProjectEulerSolutions.P58PrimeNodes.p_8895307.prime
    exact decide_eq_true prime
  second := false
  second_correct := by
    have composite : ¬ Nat.Prime (corner 1491 2) := by change ¬ Nat.Prime 8892325; exact PrattCertificates.composite_of_product (factor := 5) (cofactor := 1778465) rfl (by decide) (by decide)
    exact decide_eq_false composite
  third := false
  third_correct := by
    have composite : ¬ Nat.Prime (corner 1491 3) := by change ¬ Nat.Prime 8889343; exact PrattCertificates.composite_of_product (factor := 31) (cofactor := 286753) rfl (by decide) (by decide)
    exact decide_eq_false composite
theorem row_count_1491 : row 1491 = 1 := (row_1491.checked).trans rfl
theorem model_count_1491 : modelRow 1491 = 1 := by decide
theorem modelrow_agrees_1491 : modelRow 1491 = row 1491 := model_count_1491.trans row_count_1491.symm
def row_1492 : RowCertificate 1492 where
  first := false
  first_correct := by
    have composite : ¬ Nat.Prime (corner 1492 1) := by change ¬ Nat.Prime 8907241; exact PrattCertificates.composite_of_product (factor := 7) (cofactor := 1272463) rfl (by decide) (by decide)
    exact decide_eq_false composite
  second := false
  second_correct := by
    have composite : ¬ Nat.Prime (corner 1492 2) := by change ¬ Nat.Prime 8904257; exact PrattCertificates.composite_of_product (factor := 41) (cofactor := 217177) rfl (by decide) (by decide)
    exact decide_eq_false composite
  third := false
  third_correct := by
    have composite : ¬ Nat.Prime (corner 1492 3) := by change ¬ Nat.Prime 8901273; exact PrattCertificates.composite_of_product (factor := 3) (cofactor := 2967091) rfl (by decide) (by decide)
    exact decide_eq_false composite
theorem row_count_1492 : row 1492 = 0 := (row_1492.checked).trans rfl
theorem model_count_1492 : modelRow 1492 = 0 := by decide
theorem modelrow_agrees_1492 : modelRow 1492 = row 1492 := model_count_1492.trans row_count_1492.symm
def row_1493 : RowCertificate 1493 where
  first := false
  first_correct := by
    have composite : ¬ Nat.Prime (corner 1493 1) := by change ¬ Nat.Prime 8919183; exact PrattCertificates.composite_of_product (factor := 3) (cofactor := 2973061) rfl (by decide) (by decide)
    exact decide_eq_false composite
  second := true
  second_correct := by
    have prime : Nat.Prime (corner 1493 2) := by change Nat.Prime 8916197; exact ProjectEulerSolutions.P58PrimeNodes.p_8916197.prime
    exact decide_eq_true prime
  third := false
  third_correct := by
    have composite : ¬ Nat.Prime (corner 1493 3) := by change ¬ Nat.Prime 8913211; exact PrattCertificates.composite_of_product (factor := 67) (cofactor := 133033) rfl (by decide) (by decide)
    exact decide_eq_false composite
theorem row_count_1493 : row 1493 = 1 := (row_1493.checked).trans rfl
theorem model_count_1493 : modelRow 1493 = 1 := by decide
theorem modelrow_agrees_1493 : modelRow 1493 = row 1493 := model_count_1493.trans row_count_1493.symm
def row_1494 : RowCertificate 1494 where
  first := false
  first_correct := by
    have composite : ¬ Nat.Prime (corner 1494 1) := by change ¬ Nat.Prime 8931133; exact PrattCertificates.composite_of_product (factor := 109) (cofactor := 81937) rfl (by decide) (by decide)
    exact decide_eq_false composite
  second := false
  second_correct := by
    have composite : ¬ Nat.Prime (corner 1494 2) := by change ¬ Nat.Prime 8928145; exact PrattCertificates.composite_of_product (factor := 5) (cofactor := 1785629) rfl (by decide) (by decide)
    exact decide_eq_false composite
  third := false
  third_correct := by
    have composite : ¬ Nat.Prime (corner 1494 3) := by change ¬ Nat.Prime 8925157; exact PrattCertificates.composite_of_product (factor := 151) (cofactor := 59107) rfl (by decide) (by decide)
    exact decide_eq_false composite
theorem row_count_1494 : row 1494 = 0 := (row_1494.checked).trans rfl
theorem model_count_1494 : modelRow 1494 = 0 := by decide
theorem modelrow_agrees_1494 : modelRow 1494 = row 1494 := model_count_1494.trans row_count_1494.symm
def row_1495 : RowCertificate 1495 where
  first := false
  first_correct := by
    have composite : ¬ Nat.Prime (corner 1495 1) := by change ¬ Nat.Prime 8943091; exact PrattCertificates.composite_of_product (factor := 19) (cofactor := 470689) rfl (by decide) (by decide)
    exact decide_eq_false composite
  second := false
  second_correct := by
    have composite : ¬ Nat.Prime (corner 1495 2) := by change ¬ Nat.Prime 8940101; exact PrattCertificates.composite_of_product (factor := 2357) (cofactor := 3793) rfl (by decide) (by decide)
    exact decide_eq_false composite
  third := false
  third_correct := by
    have composite : ¬ Nat.Prime (corner 1495 3) := by change ¬ Nat.Prime 8937111; exact PrattCertificates.composite_of_product (factor := 3) (cofactor := 2979037) rfl (by decide) (by decide)
    exact decide_eq_false composite
theorem row_count_1495 : row 1495 = 0 := (row_1495.checked).trans rfl
theorem model_count_1495 : modelRow 1495 = 0 := by decide
theorem modelrow_agrees_1495 : modelRow 1495 = row 1495 := model_count_1495.trans row_count_1495.symm
def row_1496 : RowCertificate 1496 where
  first := false
  first_correct := by
    have composite : ¬ Nat.Prime (corner 1496 1) := by change ¬ Nat.Prime 8955057; exact PrattCertificates.composite_of_product (factor := 3) (cofactor := 2985019) rfl (by decide) (by decide)
    exact decide_eq_false composite
  second := false
  second_correct := by
    have composite : ¬ Nat.Prime (corner 1496 2) := by change ¬ Nat.Prime 8952065; exact PrattCertificates.composite_of_product (factor := 5) (cofactor := 1790413) rfl (by decide) (by decide)
    exact decide_eq_false composite
  third := false
  third_correct := by
    have composite : ¬ Nat.Prime (corner 1496 3) := by change ¬ Nat.Prime 8949073; exact PrattCertificates.composite_of_product (factor := 7) (cofactor := 1278439) rfl (by decide) (by decide)
    exact decide_eq_false composite
theorem row_count_1496 : row 1496 = 0 := (row_1496.checked).trans rfl
theorem model_count_1496 : modelRow 1496 = 0 := by decide
theorem modelrow_agrees_1496 : modelRow 1496 = row 1496 := model_count_1496.trans row_count_1496.symm
def row_1497 : RowCertificate 1497 where
  first := false
  first_correct := by
    have composite : ¬ Nat.Prime (corner 1497 1) := by change ¬ Nat.Prime 8967031; exact PrattCertificates.composite_of_product (factor := 19) (cofactor := 471949) rfl (by decide) (by decide)
    exact decide_eq_false composite
  second := false
  second_correct := by
    have composite : ¬ Nat.Prime (corner 1497 2) := by change ¬ Nat.Prime 8964037; exact PrattCertificates.composite_of_product (factor := 1013) (cofactor := 8849) rfl (by decide) (by decide)
    exact decide_eq_false composite
  third := false
  third_correct := by
    have composite : ¬ Nat.Prime (corner 1497 3) := by change ¬ Nat.Prime 8961043; exact PrattCertificates.composite_of_product (factor := 7) (cofactor := 1280149) rfl (by decide) (by decide)
    exact decide_eq_false composite
theorem row_count_1497 : row 1497 = 0 := (row_1497.checked).trans rfl
theorem model_count_1497 : modelRow 1497 = 0 := by decide
theorem modelrow_agrees_1497 : modelRow 1497 = row 1497 := model_count_1497.trans row_count_1497.symm
def row_1498 : RowCertificate 1498 where
  first := true
  first_correct := by
    have prime : Nat.Prime (corner 1498 1) := by change Nat.Prime 8979013; exact ProjectEulerSolutions.P58PrimeNodes.p_8979013.prime
    exact decide_eq_true prime
  second := false
  second_correct := by
    have composite : ¬ Nat.Prime (corner 1498 2) := by change ¬ Nat.Prime 8976017; exact PrattCertificates.composite_of_product (factor := 17) (cofactor := 528001) rfl (by decide) (by decide)
    exact decide_eq_false composite
  third := false
  third_correct := by
    have composite : ¬ Nat.Prime (corner 1498 3) := by change ¬ Nat.Prime 8973021; exact PrattCertificates.composite_of_product (factor := 3) (cofactor := 2991007) rfl (by decide) (by decide)
    exact decide_eq_false composite
theorem row_count_1498 : row 1498 = 1 := (row_1498.checked).trans rfl
theorem model_count_1498 : modelRow 1498 = 1 := by decide
theorem modelrow_agrees_1498 : modelRow 1498 = row 1498 := model_count_1498.trans row_count_1498.symm
def row_1499 : RowCertificate 1499 where
  first := false
  first_correct := by
    have composite : ¬ Nat.Prime (corner 1499 1) := by change ¬ Nat.Prime 8991003; exact PrattCertificates.composite_of_product (factor := 3) (cofactor := 2997001) rfl (by decide) (by decide)
    exact decide_eq_false composite
  second := false
  second_correct := by
    have composite : ¬ Nat.Prime (corner 1499 2) := by change ¬ Nat.Prime 8988005; exact PrattCertificates.composite_of_product (factor := 5) (cofactor := 1797601) rfl (by decide) (by decide)
    exact decide_eq_false composite
  third := true
  third_correct := by
    have prime : Nat.Prime (corner 1499 3) := by change Nat.Prime 8985007; exact ProjectEulerSolutions.P58PrimeNodes.p_8985007.prime
    exact decide_eq_true prime
theorem row_count_1499 : row 1499 = 1 := (row_1499.checked).trans rfl
theorem model_count_1499 : modelRow 1499 = 1 := by decide
theorem modelrow_agrees_1499 : modelRow 1499 = row 1499 := model_count_1499.trans row_count_1499.symm
def row_1500 : RowCertificate 1500 where
  first := false
  first_correct := by
    have composite : ¬ Nat.Prime (corner 1500 1) := by change ¬ Nat.Prime 9003001; exact PrattCertificates.composite_of_product (factor := 7) (cofactor := 1286143) rfl (by decide) (by decide)
    exact decide_eq_false composite
  second := false
  second_correct := by
    have composite : ¬ Nat.Prime (corner 1500 2) := by change ¬ Nat.Prime 9000001; exact PrattCertificates.composite_of_product (factor := 61) (cofactor := 147541) rfl (by decide) (by decide)
    exact decide_eq_false composite
  third := false
  third_correct := by
    have composite : ¬ Nat.Prime (corner 1500 3) := by change ¬ Nat.Prime 8997001; exact PrattCertificates.composite_of_product (factor := 13) (cofactor := 692077) rfl (by decide) (by decide)
    exact decide_eq_false composite
theorem row_count_1500 : row 1500 = 0 := (row_1500.checked).trans rfl
theorem model_count_1500 : modelRow 1500 = 0 := by decide
theorem modelrow_agrees_1500 : modelRow 1500 = row 1500 := model_count_1500.trans row_count_1500.symm
def row_1501 : RowCertificate 1501 where
  first := false
  first_correct := by
    have composite : ¬ Nat.Prime (corner 1501 1) := by change ¬ Nat.Prime 9015007; exact PrattCertificates.composite_of_product (factor := 61) (cofactor := 147787) rfl (by decide) (by decide)
    exact decide_eq_false composite
  second := false
  second_correct := by
    have composite : ¬ Nat.Prime (corner 1501 2) := by change ¬ Nat.Prime 9012005; exact PrattCertificates.composite_of_product (factor := 5) (cofactor := 1802401) rfl (by decide) (by decide)
    exact decide_eq_false composite
  third := false
  third_correct := by
    have composite : ¬ Nat.Prime (corner 1501 3) := by change ¬ Nat.Prime 9009003; exact PrattCertificates.composite_of_product (factor := 3) (cofactor := 3003001) rfl (by decide) (by decide)
    exact decide_eq_false composite
theorem row_count_1501 : row 1501 = 0 := (row_1501.checked).trans rfl
theorem model_count_1501 : modelRow 1501 = 0 := by decide
theorem modelrow_agrees_1501 : modelRow 1501 = row 1501 := model_count_1501.trans row_count_1501.symm
def row_1502 : RowCertificate 1502 where
  first := false
  first_correct := by
    have composite : ¬ Nat.Prime (corner 1502 1) := by change ¬ Nat.Prime 9027021; exact PrattCertificates.composite_of_product (factor := 3) (cofactor := 3009007) rfl (by decide) (by decide)
    exact decide_eq_false composite
  second := false
  second_correct := by
    have composite : ¬ Nat.Prime (corner 1502 2) := by change ¬ Nat.Prime 9024017; exact PrattCertificates.composite_of_product (factor := 29) (cofactor := 311173) rfl (by decide) (by decide)
    exact decide_eq_false composite
  third := false
  third_correct := by
    have composite : ¬ Nat.Prime (corner 1502 3) := by change ¬ Nat.Prime 9021013; exact PrattCertificates.composite_of_product (factor := 43) (cofactor := 209791) rfl (by decide) (by decide)
    exact decide_eq_false composite
theorem row_count_1502 : row 1502 = 0 := (row_1502.checked).trans rfl
theorem model_count_1502 : modelRow 1502 = 0 := by decide
theorem modelrow_agrees_1502 : modelRow 1502 = row 1502 := model_count_1502.trans row_count_1502.symm
def row_1503 : RowCertificate 1503 where
  first := false
  first_correct := by
    have composite : ¬ Nat.Prime (corner 1503 1) := by change ¬ Nat.Prime 9039043; exact PrattCertificates.composite_of_product (factor := 13) (cofactor := 695311) rfl (by decide) (by decide)
    exact decide_eq_false composite
  second := false
  second_correct := by
    have composite : ¬ Nat.Prime (corner 1503 2) := by change ¬ Nat.Prime 9036037; exact PrattCertificates.composite_of_product (factor := 409) (cofactor := 22093) rfl (by decide) (by decide)
    exact decide_eq_false composite
  third := false
  third_correct := by
    have composite : ¬ Nat.Prime (corner 1503 3) := by change ¬ Nat.Prime 9033031; exact PrattCertificates.composite_of_product (factor := 7) (cofactor := 1290433) rfl (by decide) (by decide)
    exact decide_eq_false composite
theorem row_count_1503 : row 1503 = 0 := (row_1503.checked).trans rfl
theorem model_count_1503 : modelRow 1503 = 0 := by decide
theorem modelrow_agrees_1503 : modelRow 1503 = row 1503 := model_count_1503.trans row_count_1503.symm
def row_1504 : RowCertificate 1504 where
  first := false
  first_correct := by
    have composite : ¬ Nat.Prime (corner 1504 1) := by change ¬ Nat.Prime 9051073; exact PrattCertificates.composite_of_product (factor := 661) (cofactor := 13693) rfl (by decide) (by decide)
    exact decide_eq_false composite
  second := false
  second_correct := by
    have composite : ¬ Nat.Prime (corner 1504 2) := by change ¬ Nat.Prime 9048065; exact PrattCertificates.composite_of_product (factor := 5) (cofactor := 1809613) rfl (by decide) (by decide)
    exact decide_eq_false composite
  third := false
  third_correct := by
    have composite : ¬ Nat.Prime (corner 1504 3) := by change ¬ Nat.Prime 9045057; exact PrattCertificates.composite_of_product (factor := 3) (cofactor := 3015019) rfl (by decide) (by decide)
    exact decide_eq_false composite
theorem row_count_1504 : row 1504 = 0 := (row_1504.checked).trans rfl
theorem model_count_1504 : modelRow 1504 = 0 := by decide
theorem modelrow_agrees_1504 : modelRow 1504 = row 1504 := model_count_1504.trans row_count_1504.symm
def row_1505 : RowCertificate 1505 where
  first := false
  first_correct := by
    have composite : ¬ Nat.Prime (corner 1505 1) := by change ¬ Nat.Prime 9063111; exact PrattCertificates.composite_of_product (factor := 3) (cofactor := 3021037) rfl (by decide) (by decide)
    exact decide_eq_false composite
  second := false
  second_correct := by
    have composite : ¬ Nat.Prime (corner 1505 2) := by change ¬ Nat.Prime 9060101; exact PrattCertificates.composite_of_product (factor := 1549) (cofactor := 5849) rfl (by decide) (by decide)
    exact decide_eq_false composite
  third := false
  third_correct := by
    have composite : ¬ Nat.Prime (corner 1505 3) := by change ¬ Nat.Prime 9057091; exact PrattCertificates.composite_of_product (factor := 19) (cofactor := 476689) rfl (by decide) (by decide)
    exact decide_eq_false composite
theorem row_count_1505 : row 1505 = 0 := (row_1505.checked).trans rfl
theorem model_count_1505 : modelRow 1505 = 0 := by decide
theorem modelrow_agrees_1505 : modelRow 1505 = row 1505 := model_count_1505.trans row_count_1505.symm
def row_1506 : RowCertificate 1506 where
  first := false
  first_correct := by
    have composite : ¬ Nat.Prime (corner 1506 1) := by change ¬ Nat.Prime 9075157; exact PrattCertificates.composite_of_product (factor := 7) (cofactor := 1296451) rfl (by decide) (by decide)
    exact decide_eq_false composite
  second := false
  second_correct := by
    have composite : ¬ Nat.Prime (corner 1506 2) := by change ¬ Nat.Prime 9072145; exact PrattCertificates.composite_of_product (factor := 5) (cofactor := 1814429) rfl (by decide) (by decide)
    exact decide_eq_false composite
  third := true
  third_correct := by
    have prime : Nat.Prime (corner 1506 3) := by change Nat.Prime 9069133; exact ProjectEulerSolutions.P58PrimeNodes.p_9069133.prime
    exact decide_eq_true prime
theorem row_count_1506 : row 1506 = 1 := (row_1506.checked).trans rfl
theorem model_count_1506 : modelRow 1506 = 1 := by decide
theorem modelrow_agrees_1506 : modelRow 1506 = row 1506 := model_count_1506.trans row_count_1506.symm
def row_1507 : RowCertificate 1507 where
  first := false
  first_correct := by
    have composite : ¬ Nat.Prime (corner 1507 1) := by change ¬ Nat.Prime 9087211; exact PrattCertificates.composite_of_product (factor := 7) (cofactor := 1298173) rfl (by decide) (by decide)
    exact decide_eq_false composite
  second := false
  second_correct := by
    have composite : ¬ Nat.Prime (corner 1507 2) := by change ¬ Nat.Prime 9084197; exact PrattCertificates.composite_of_product (factor := 769) (cofactor := 11813) rfl (by decide) (by decide)
    exact decide_eq_false composite
  third := false
  third_correct := by
    have composite : ¬ Nat.Prime (corner 1507 3) := by change ¬ Nat.Prime 9081183; exact PrattCertificates.composite_of_product (factor := 3) (cofactor := 3027061) rfl (by decide) (by decide)
    exact decide_eq_false composite
theorem row_count_1507 : row 1507 = 0 := (row_1507.checked).trans rfl
theorem model_count_1507 : modelRow 1507 = 0 := by decide
theorem modelrow_agrees_1507 : modelRow 1507 = row 1507 := model_count_1507.trans row_count_1507.symm
def row_1508 : RowCertificate 1508 where
  first := false
  first_correct := by
    have composite : ¬ Nat.Prime (corner 1508 1) := by change ¬ Nat.Prime 9099273; exact PrattCertificates.composite_of_product (factor := 3) (cofactor := 3033091) rfl (by decide) (by decide)
    exact decide_eq_false composite
  second := true
  second_correct := by
    have prime : Nat.Prime (corner 1508 2) := by change Nat.Prime 9096257; exact ProjectEulerSolutions.P58PrimeNodes.p_9096257.prime
    exact decide_eq_true prime
  third := false
  third_correct := by
    have composite : ¬ Nat.Prime (corner 1508 3) := by change ¬ Nat.Prime 9093241; exact PrattCertificates.composite_of_product (factor := 139) (cofactor := 65419) rfl (by decide) (by decide)
    exact decide_eq_false composite
theorem row_count_1508 : row 1508 = 1 := (row_1508.checked).trans rfl
theorem model_count_1508 : modelRow 1508 = 1 := by decide
theorem modelrow_agrees_1508 : modelRow 1508 = row 1508 := model_count_1508.trans row_count_1508.symm
def row_1509 : RowCertificate 1509 where
  first := false
  first_correct := by
    have composite : ¬ Nat.Prime (corner 1509 1) := by change ¬ Nat.Prime 9111343; exact PrattCertificates.composite_of_product (factor := 349) (cofactor := 26107) rfl (by decide) (by decide)
    exact decide_eq_false composite
  second := false
  second_correct := by
    have composite : ¬ Nat.Prime (corner 1509 2) := by change ¬ Nat.Prime 9108325; exact PrattCertificates.composite_of_product (factor := 5) (cofactor := 1821665) rfl (by decide) (by decide)
    exact decide_eq_false composite
  third := false
  third_correct := by
    have composite : ¬ Nat.Prime (corner 1509 3) := by change ¬ Nat.Prime 9105307; exact PrattCertificates.composite_of_product (factor := 691) (cofactor := 13177) rfl (by decide) (by decide)
    exact decide_eq_false composite
theorem row_count_1509 : row 1509 = 0 := (row_1509.checked).trans rfl
theorem model_count_1509 : modelRow 1509 = 0 := by decide
theorem modelrow_agrees_1509 : modelRow 1509 = row 1509 := model_count_1509.trans row_count_1509.symm
def row_1510 : RowCertificate 1510 where
  first := true
  first_correct := by
    have prime : Nat.Prime (corner 1510 1) := by change Nat.Prime 9123421; exact ProjectEulerSolutions.P58PrimeNodes.p_9123421.prime
    exact decide_eq_true prime
  second := false
  second_correct := by
    have composite : ¬ Nat.Prime (corner 1510 2) := by change ¬ Nat.Prime 9120401; exact PrattCertificates.composite_of_product (factor := 73) (cofactor := 124937) rfl (by decide) (by decide)
    exact decide_eq_false composite
  third := false
  third_correct := by
    have composite : ¬ Nat.Prime (corner 1510 3) := by change ¬ Nat.Prime 9117381; exact PrattCertificates.composite_of_product (factor := 3) (cofactor := 3039127) rfl (by decide) (by decide)
    exact decide_eq_false composite
theorem row_count_1510 : row 1510 = 1 := (row_1510.checked).trans rfl
theorem model_count_1510 : modelRow 1510 = 1 := by decide
theorem modelrow_agrees_1510 : modelRow 1510 = row 1510 := model_count_1510.trans row_count_1510.symm
def row_1511 : RowCertificate 1511 where
  first := false
  first_correct := by
    have composite : ¬ Nat.Prime (corner 1511 1) := by change ¬ Nat.Prime 9135507; exact PrattCertificates.composite_of_product (factor := 3) (cofactor := 3045169) rfl (by decide) (by decide)
    exact decide_eq_false composite
  second := false
  second_correct := by
    have composite : ¬ Nat.Prime (corner 1511 2) := by change ¬ Nat.Prime 9132485; exact PrattCertificates.composite_of_product (factor := 5) (cofactor := 1826497) rfl (by decide) (by decide)
    exact decide_eq_false composite
  third := false
  third_correct := by
    have composite : ¬ Nat.Prime (corner 1511 3) := by change ¬ Nat.Prime 9129463; exact PrattCertificates.composite_of_product (factor := 7) (cofactor := 1304209) rfl (by decide) (by decide)
    exact decide_eq_false composite
theorem row_count_1511 : row 1511 = 0 := (row_1511.checked).trans rfl
theorem model_count_1511 : modelRow 1511 = 0 := by decide
theorem modelrow_agrees_1511 : modelRow 1511 = row 1511 := model_count_1511.trans row_count_1511.symm
def row_1512 : RowCertificate 1512 where
  first := true
  first_correct := by
    have prime : Nat.Prime (corner 1512 1) := by change Nat.Prime 9147601; exact ProjectEulerSolutions.P58PrimeNodes.p_9147601.prime
    exact decide_eq_true prime
  second := false
  second_correct := by
    have composite : ¬ Nat.Prime (corner 1512 2) := by change ¬ Nat.Prime 9144577; exact PrattCertificates.composite_of_product (factor := 13) (cofactor := 703429) rfl (by decide) (by decide)
    exact decide_eq_false composite
  third := false
  third_correct := by
    have composite : ¬ Nat.Prime (corner 1512 3) := by change ¬ Nat.Prime 9141553; exact PrattCertificates.composite_of_product (factor := 37) (cofactor := 247069) rfl (by decide) (by decide)
    exact decide_eq_false composite
theorem row_count_1512 : row 1512 = 1 := (row_1512.checked).trans rfl
theorem model_count_1512 : modelRow 1512 = 1 := by decide
theorem modelrow_agrees_1512 : modelRow 1512 = row 1512 := model_count_1512.trans row_count_1512.symm
def row_1513 : RowCertificate 1513 where
  first := false
  first_correct := by
    have composite : ¬ Nat.Prime (corner 1513 1) := by change ¬ Nat.Prime 9159703; exact PrattCertificates.composite_of_product (factor := 7) (cofactor := 1308529) rfl (by decide) (by decide)
    exact decide_eq_false composite
  second := true
  second_correct := by
    have prime : Nat.Prime (corner 1513 2) := by change Nat.Prime 9156677; exact ProjectEulerSolutions.P58PrimeNodes.p_9156677.prime
    exact decide_eq_true prime
  third := false
  third_correct := by
    have composite : ¬ Nat.Prime (corner 1513 3) := by change ¬ Nat.Prime 9153651; exact PrattCertificates.composite_of_product (factor := 3) (cofactor := 3051217) rfl (by decide) (by decide)
    exact decide_eq_false composite
theorem row_count_1513 : row 1513 = 1 := (row_1513.checked).trans rfl
theorem model_count_1513 : modelRow 1513 = 1 := by decide
theorem modelrow_agrees_1513 : modelRow 1513 = row 1513 := model_count_1513.trans row_count_1513.symm
def row_1514 : RowCertificate 1514 where
  first := false
  first_correct := by
    have composite : ¬ Nat.Prime (corner 1514 1) := by change ¬ Nat.Prime 9171813; exact PrattCertificates.composite_of_product (factor := 3) (cofactor := 3057271) rfl (by decide) (by decide)
    exact decide_eq_false composite
  second := false
  second_correct := by
    have composite : ¬ Nat.Prime (corner 1514 2) := by change ¬ Nat.Prime 9168785; exact PrattCertificates.composite_of_product (factor := 5) (cofactor := 1833757) rfl (by decide) (by decide)
    exact decide_eq_false composite
  third := true
  third_correct := by
    have prime : Nat.Prime (corner 1514 3) := by change Nat.Prime 9165757; exact ProjectEulerSolutions.P58PrimeNodes.p_9165757.prime
    exact decide_eq_true prime
theorem row_count_1514 : row 1514 = 1 := (row_1514.checked).trans rfl
theorem model_count_1514 : modelRow 1514 = 1 := by decide
theorem modelrow_agrees_1514 : modelRow 1514 = row 1514 := model_count_1514.trans row_count_1514.symm
def row_1515 : RowCertificate 1515 where
  first := true
  first_correct := by
    have prime : Nat.Prime (corner 1515 1) := by change Nat.Prime 9183931; exact ProjectEulerSolutions.P58PrimeNodes.p_9183931.prime
    exact decide_eq_true prime
  second := false
  second_correct := by
    have composite : ¬ Nat.Prime (corner 1515 2) := by change ¬ Nat.Prime 9180901; exact PrattCertificates.composite_of_product (factor := 17) (cofactor := 540053) rfl (by decide) (by decide)
    exact decide_eq_false composite
  third := false
  third_correct := by
    have composite : ¬ Nat.Prime (corner 1515 3) := by change ¬ Nat.Prime 9177871; exact PrattCertificates.composite_of_product (factor := 601) (cofactor := 15271) rfl (by decide) (by decide)
    exact decide_eq_false composite
theorem row_count_1515 : row 1515 = 1 := (row_1515.checked).trans rfl
theorem model_count_1515 : modelRow 1515 = 1 := by decide
theorem modelrow_agrees_1515 : modelRow 1515 = row 1515 := model_count_1515.trans row_count_1515.symm
def row_1516 : RowCertificate 1516 where
  first := false
  first_correct := by
    have composite : ¬ Nat.Prime (corner 1516 1) := by change ¬ Nat.Prime 9196057; exact PrattCertificates.composite_of_product (factor := 13) (cofactor := 707389) rfl (by decide) (by decide)
    exact decide_eq_false composite
  second := false
  second_correct := by
    have composite : ¬ Nat.Prime (corner 1516 2) := by change ¬ Nat.Prime 9193025; exact PrattCertificates.composite_of_product (factor := 5) (cofactor := 1838605) rfl (by decide) (by decide)
    exact decide_eq_false composite
  third := false
  third_correct := by
    have composite : ¬ Nat.Prime (corner 1516 3) := by change ¬ Nat.Prime 9189993; exact PrattCertificates.composite_of_product (factor := 3) (cofactor := 3063331) rfl (by decide) (by decide)
    exact decide_eq_false composite
theorem row_count_1516 : row 1516 = 0 := (row_1516.checked).trans rfl
theorem model_count_1516 : modelRow 1516 = 0 := by decide
theorem modelrow_agrees_1516 : modelRow 1516 = row 1516 := model_count_1516.trans row_count_1516.symm
def row_1517 : RowCertificate 1517 where
  first := false
  first_correct := by
    have composite : ¬ Nat.Prime (corner 1517 1) := by change ¬ Nat.Prime 9208191; exact PrattCertificates.composite_of_product (factor := 3) (cofactor := 3069397) rfl (by decide) (by decide)
    exact decide_eq_false composite
  second := false
  second_correct := by
    have composite : ¬ Nat.Prime (corner 1517 2) := by change ¬ Nat.Prime 9205157; exact PrattCertificates.composite_of_product (factor := 13) (cofactor := 708089) rfl (by decide) (by decide)
    exact decide_eq_false composite
  third := false
  third_correct := by
    have composite : ¬ Nat.Prime (corner 1517 3) := by change ¬ Nat.Prime 9202123; exact PrattCertificates.composite_of_product (factor := 7) (cofactor := 1314589) rfl (by decide) (by decide)
    exact decide_eq_false composite
theorem row_count_1517 : row 1517 = 0 := (row_1517.checked).trans rfl
theorem model_count_1517 : modelRow 1517 = 0 := by decide
theorem modelrow_agrees_1517 : modelRow 1517 = row 1517 := model_count_1517.trans row_count_1517.symm
def row_1518 : RowCertificate 1518 where
  first := false
  first_correct := by
    have composite : ¬ Nat.Prime (corner 1518 1) := by change ¬ Nat.Prime 9220333; exact PrattCertificates.composite_of_product (factor := 61) (cofactor := 151153) rfl (by decide) (by decide)
    exact decide_eq_false composite
  second := false
  second_correct := by
    have composite : ¬ Nat.Prime (corner 1518 2) := by change ¬ Nat.Prime 9217297; exact PrattCertificates.composite_of_product (factor := 113) (cofactor := 81569) rfl (by decide) (by decide)
    exact decide_eq_false composite
  third := false
  third_correct := by
    have composite : ¬ Nat.Prime (corner 1518 3) := by change ¬ Nat.Prime 9214261; exact PrattCertificates.composite_of_product (factor := 7) (cofactor := 1316323) rfl (by decide) (by decide)
    exact decide_eq_false composite
theorem row_count_1518 : row 1518 = 0 := (row_1518.checked).trans rfl
theorem model_count_1518 : modelRow 1518 = 0 := by decide
theorem modelrow_agrees_1518 : modelRow 1518 = row 1518 := model_count_1518.trans row_count_1518.symm
def row_1519 : RowCertificate 1519 where
  first := false
  first_correct := by
    have composite : ¬ Nat.Prime (corner 1519 1) := by change ¬ Nat.Prime 9232483; exact PrattCertificates.composite_of_product (factor := 13) (cofactor := 710191) rfl (by decide) (by decide)
    exact decide_eq_false composite
  second := false
  second_correct := by
    have composite : ¬ Nat.Prime (corner 1519 2) := by change ¬ Nat.Prime 9229445; exact PrattCertificates.composite_of_product (factor := 5) (cofactor := 1845889) rfl (by decide) (by decide)
    exact decide_eq_false composite
  third := false
  third_correct := by
    have composite : ¬ Nat.Prime (corner 1519 3) := by change ¬ Nat.Prime 9226407; exact PrattCertificates.composite_of_product (factor := 3) (cofactor := 3075469) rfl (by decide) (by decide)
    exact decide_eq_false composite
theorem row_count_1519 : row 1519 = 0 := (row_1519.checked).trans rfl
theorem model_count_1519 : modelRow 1519 = 0 := by decide
theorem modelrow_agrees_1519 : modelRow 1519 = row 1519 := model_count_1519.trans row_count_1519.symm
def row_1520 : RowCertificate 1520 where
  first := false
  first_correct := by
    have composite : ¬ Nat.Prime (corner 1520 1) := by change ¬ Nat.Prime 9244641; exact PrattCertificates.composite_of_product (factor := 3) (cofactor := 3081547) rfl (by decide) (by decide)
    exact decide_eq_false composite
  second := false
  second_correct := by
    have composite : ¬ Nat.Prime (corner 1520 2) := by change ¬ Nat.Prime 9241601; exact PrattCertificates.composite_of_product (factor := 37) (cofactor := 249773) rfl (by decide) (by decide)
    exact decide_eq_false composite
  third := true
  third_correct := by
    have prime : Nat.Prime (corner 1520 3) := by change Nat.Prime 9238561; exact ProjectEulerSolutions.P58PrimeNodes.p_9238561.prime
    exact decide_eq_true prime
theorem row_count_1520 : row 1520 = 1 := (row_1520.checked).trans rfl
theorem model_count_1520 : modelRow 1520 = 1 := by decide
theorem modelrow_agrees_1520 : modelRow 1520 = row 1520 := model_count_1520.trans row_count_1520.symm
def row_1521 : RowCertificate 1521 where
  first := false
  first_correct := by
    have composite : ¬ Nat.Prime (corner 1521 1) := by change ¬ Nat.Prime 9256807; exact PrattCertificates.composite_of_product (factor := 7) (cofactor := 1322401) rfl (by decide) (by decide)
    exact decide_eq_false composite
  second := false
  second_correct := by
    have composite : ¬ Nat.Prime (corner 1521 2) := by change ¬ Nat.Prime 9253765; exact PrattCertificates.composite_of_product (factor := 5) (cofactor := 1850753) rfl (by decide) (by decide)
    exact decide_eq_false composite
  third := true
  third_correct := by
    have prime : Nat.Prime (corner 1521 3) := by change Nat.Prime 9250723; exact ProjectEulerSolutions.P58PrimeNodes.p_9250723.prime
    exact decide_eq_true prime
theorem row_count_1521 : row 1521 = 1 := (row_1521.checked).trans rfl
theorem model_count_1521 : modelRow 1521 = 1 := by decide
theorem modelrow_agrees_1521 : modelRow 1521 = row 1521 := model_count_1521.trans row_count_1521.symm
def row_1522 : RowCertificate 1522 where
  first := false
  first_correct := by
    have composite : ¬ Nat.Prime (corner 1522 1) := by change ¬ Nat.Prime 9268981; exact PrattCertificates.composite_of_product (factor := 37) (cofactor := 250513) rfl (by decide) (by decide)
    exact decide_eq_false composite
  second := false
  second_correct := by
    have composite : ¬ Nat.Prime (corner 1522 2) := by change ¬ Nat.Prime 9265937; exact PrattCertificates.composite_of_product (factor := 53) (cofactor := 174829) rfl (by decide) (by decide)
    exact decide_eq_false composite
  third := false
  third_correct := by
    have composite : ¬ Nat.Prime (corner 1522 3) := by change ¬ Nat.Prime 9262893; exact PrattCertificates.composite_of_product (factor := 3) (cofactor := 3087631) rfl (by decide) (by decide)
    exact decide_eq_false composite
theorem row_count_1522 : row 1522 = 0 := (row_1522.checked).trans rfl
theorem model_count_1522 : modelRow 1522 = 0 := by decide
theorem modelrow_agrees_1522 : modelRow 1522 = row 1522 := model_count_1522.trans row_count_1522.symm
def row_1523 : RowCertificate 1523 where
  first := false
  first_correct := by
    have composite : ¬ Nat.Prime (corner 1523 1) := by change ¬ Nat.Prime 9281163; exact PrattCertificates.composite_of_product (factor := 3) (cofactor := 3093721) rfl (by decide) (by decide)
    exact decide_eq_false composite
  second := true
  second_correct := by
    have prime : Nat.Prime (corner 1523 2) := by change Nat.Prime 9278117; exact ProjectEulerSolutions.P58PrimeNodes.p_9278117.prime
    exact decide_eq_true prime
  third := false
  third_correct := by
    have composite : ¬ Nat.Prime (corner 1523 3) := by change ¬ Nat.Prime 9275071; exact PrattCertificates.composite_of_product (factor := 13) (cofactor := 713467) rfl (by decide) (by decide)
    exact decide_eq_false composite
theorem row_count_1523 : row 1523 = 1 := (row_1523.checked).trans rfl
theorem model_count_1523 : modelRow 1523 = 1 := by decide
theorem modelrow_agrees_1523 : modelRow 1523 = row 1523 := model_count_1523.trans row_count_1523.symm
def row_1524 : RowCertificate 1524 where
  first := true
  first_correct := by
    have prime : Nat.Prime (corner 1524 1) := by change Nat.Prime 9293353; exact ProjectEulerSolutions.P58PrimeNodes.p_9293353.prime
    exact decide_eq_true prime
  second := false
  second_correct := by
    have composite : ¬ Nat.Prime (corner 1524 2) := by change ¬ Nat.Prime 9290305; exact PrattCertificates.composite_of_product (factor := 5) (cofactor := 1858061) rfl (by decide) (by decide)
    exact decide_eq_false composite
  third := false
  third_correct := by
    have composite : ¬ Nat.Prime (corner 1524 3) := by change ¬ Nat.Prime 9287257; exact PrattCertificates.composite_of_product (factor := 7) (cofactor := 1326751) rfl (by decide) (by decide)
    exact decide_eq_false composite
theorem row_count_1524 : row 1524 = 1 := (row_1524.checked).trans rfl
theorem model_count_1524 : modelRow 1524 = 1 := by decide
theorem modelrow_agrees_1524 : modelRow 1524 = row 1524 := model_count_1524.trans row_count_1524.symm
def row_1525 : RowCertificate 1525 where
  first := true
  first_correct := by
    have prime : Nat.Prime (corner 1525 1) := by change Nat.Prime 9305551; exact ProjectEulerSolutions.P58PrimeNodes.p_9305551.prime
    exact decide_eq_true prime
  second := false
  second_correct := by
    have composite : ¬ Nat.Prime (corner 1525 2) := by change ¬ Nat.Prime 9302501; exact PrattCertificates.composite_of_product (factor := 13) (cofactor := 715577) rfl (by decide) (by decide)
    exact decide_eq_false composite
  third := false
  third_correct := by
    have composite : ¬ Nat.Prime (corner 1525 3) := by change ¬ Nat.Prime 9299451; exact PrattCertificates.composite_of_product (factor := 3) (cofactor := 3099817) rfl (by decide) (by decide)
    exact decide_eq_false composite
theorem row_count_1525 : row 1525 = 1 := (row_1525.checked).trans rfl
theorem model_count_1525 : modelRow 1525 = 1 := by decide
theorem modelrow_agrees_1525 : modelRow 1525 = row 1525 := model_count_1525.trans row_count_1525.symm
def row_1526 : RowCertificate 1526 where
  first := false
  first_correct := by
    have composite : ¬ Nat.Prime (corner 1526 1) := by change ¬ Nat.Prime 9317757; exact PrattCertificates.composite_of_product (factor := 3) (cofactor := 3105919) rfl (by decide) (by decide)
    exact decide_eq_false composite
  second := false
  second_correct := by
    have composite : ¬ Nat.Prime (corner 1526 2) := by change ¬ Nat.Prime 9314705; exact PrattCertificates.composite_of_product (factor := 5) (cofactor := 1862941) rfl (by decide) (by decide)
    exact decide_eq_false composite
  third := false
  third_correct := by
    have composite : ¬ Nat.Prime (corner 1526 3) := by change ¬ Nat.Prime 9311653; exact PrattCertificates.composite_of_product (factor := 13) (cofactor := 716281) rfl (by decide) (by decide)
    exact decide_eq_false composite
theorem row_count_1526 : row 1526 = 0 := (row_1526.checked).trans rfl
theorem model_count_1526 : modelRow 1526 = 0 := by decide
theorem modelrow_agrees_1526 : modelRow 1526 = row 1526 := model_count_1526.trans row_count_1526.symm
def row_1527 : RowCertificate 1527 where
  first := false
  first_correct := by
    have composite : ¬ Nat.Prime (corner 1527 1) := by change ¬ Nat.Prime 9329971; exact PrattCertificates.composite_of_product (factor := 7) (cofactor := 1332853) rfl (by decide) (by decide)
    exact decide_eq_false composite
  second := true
  second_correct := by
    have prime : Nat.Prime (corner 1527 2) := by change Nat.Prime 9326917; exact ProjectEulerSolutions.P58PrimeNodes.p_9326917.prime
    exact decide_eq_true prime
  third := true
  third_correct := by
    have prime : Nat.Prime (corner 1527 3) := by change Nat.Prime 9323863; exact ProjectEulerSolutions.P58PrimeNodes.p_9323863.prime
    exact decide_eq_true prime
theorem row_count_1527 : row 1527 = 2 := (row_1527.checked).trans rfl
theorem model_count_1527 : modelRow 1527 = 2 := by decide
theorem modelrow_agrees_1527 : modelRow 1527 = row 1527 := model_count_1527.trans row_count_1527.symm
def row_1528 : RowCertificate 1528 where
  first := false
  first_correct := by
    have composite : ¬ Nat.Prime (corner 1528 1) := by change ¬ Nat.Prime 9342193; exact PrattCertificates.composite_of_product (factor := 7) (cofactor := 1334599) rfl (by decide) (by decide)
    exact decide_eq_false composite
  second := false
  second_correct := by
    have composite : ¬ Nat.Prime (corner 1528 2) := by change ¬ Nat.Prime 9339137; exact PrattCertificates.composite_of_product (factor := 17) (cofactor := 549361) rfl (by decide) (by decide)
    exact decide_eq_false composite
  third := false
  third_correct := by
    have composite : ¬ Nat.Prime (corner 1528 3) := by change ¬ Nat.Prime 9336081; exact PrattCertificates.composite_of_product (factor := 3) (cofactor := 3112027) rfl (by decide) (by decide)
    exact decide_eq_false composite
theorem row_count_1528 : row 1528 = 0 := (row_1528.checked).trans rfl
theorem model_count_1528 : modelRow 1528 = 0 := by decide
theorem modelrow_agrees_1528 : modelRow 1528 = row 1528 := model_count_1528.trans row_count_1528.symm
def row_1529 : RowCertificate 1529 where
  first := false
  first_correct := by
    have composite : ¬ Nat.Prime (corner 1529 1) := by change ¬ Nat.Prime 9354423; exact PrattCertificates.composite_of_product (factor := 3) (cofactor := 3118141) rfl (by decide) (by decide)
    exact decide_eq_false composite
  second := false
  second_correct := by
    have composite : ¬ Nat.Prime (corner 1529 2) := by change ¬ Nat.Prime 9351365; exact PrattCertificates.composite_of_product (factor := 5) (cofactor := 1870273) rfl (by decide) (by decide)
    exact decide_eq_false composite
  third := false
  third_correct := by
    have composite : ¬ Nat.Prime (corner 1529 3) := by change ¬ Nat.Prime 9348307; exact PrattCertificates.composite_of_product (factor := 73) (cofactor := 128059) rfl (by decide) (by decide)
    exact decide_eq_false composite
theorem row_count_1529 : row 1529 = 0 := (row_1529.checked).trans rfl
theorem model_count_1529 : modelRow 1529 = 0 := by decide
theorem modelrow_agrees_1529 : modelRow 1529 = row 1529 := model_count_1529.trans row_count_1529.symm
def row_1530 : RowCertificate 1530 where
  first := false
  first_correct := by
    have composite : ¬ Nat.Prime (corner 1530 1) := by change ¬ Nat.Prime 9366661; exact PrattCertificates.composite_of_product (factor := 37) (cofactor := 253153) rfl (by decide) (by decide)
    exact decide_eq_false composite
  second := false
  second_correct := by
    have composite : ¬ Nat.Prime (corner 1530 2) := by change ¬ Nat.Prime 9363601; exact PrattCertificates.composite_of_product (factor := 13) (cofactor := 720277) rfl (by decide) (by decide)
    exact decide_eq_false composite
  third := false
  third_correct := by
    have composite : ¬ Nat.Prime (corner 1530 3) := by change ¬ Nat.Prime 9360541; exact PrattCertificates.composite_of_product (factor := 43) (cofactor := 217687) rfl (by decide) (by decide)
    exact decide_eq_false composite
theorem row_count_1530 : row 1530 = 0 := (row_1530.checked).trans rfl
theorem model_count_1530 : modelRow 1530 = 0 := by decide
theorem modelrow_agrees_1530 : modelRow 1530 = row 1530 := model_count_1530.trans row_count_1530.symm
def row_1531 : RowCertificate 1531 where
  first := false
  first_correct := by
    have composite : ¬ Nat.Prime (corner 1531 1) := by change ¬ Nat.Prime 9378907; exact PrattCertificates.composite_of_product (factor := 1279) (cofactor := 7333) rfl (by decide) (by decide)
    exact decide_eq_false composite
  second := false
  second_correct := by
    have composite : ¬ Nat.Prime (corner 1531 2) := by change ¬ Nat.Prime 9375845; exact PrattCertificates.composite_of_product (factor := 5) (cofactor := 1875169) rfl (by decide) (by decide)
    exact decide_eq_false composite
  third := false
  third_correct := by
    have composite : ¬ Nat.Prime (corner 1531 3) := by change ¬ Nat.Prime 9372783; exact PrattCertificates.composite_of_product (factor := 3) (cofactor := 3124261) rfl (by decide) (by decide)
    exact decide_eq_false composite
theorem row_count_1531 : row 1531 = 0 := (row_1531.checked).trans rfl
theorem model_count_1531 : modelRow 1531 = 0 := by decide
theorem modelrow_agrees_1531 : modelRow 1531 = row 1531 := model_count_1531.trans row_count_1531.symm
def row_1532 : RowCertificate 1532 where
  first := false
  first_correct := by
    have composite : ¬ Nat.Prime (corner 1532 1) := by change ¬ Nat.Prime 9391161; exact PrattCertificates.composite_of_product (factor := 3) (cofactor := 3130387) rfl (by decide) (by decide)
    exact decide_eq_false composite
  second := false
  second_correct := by
    have composite : ¬ Nat.Prime (corner 1532 2) := by change ¬ Nat.Prime 9388097; exact PrattCertificates.composite_of_product (factor := 17) (cofactor := 552241) rfl (by decide) (by decide)
    exact decide_eq_false composite
  third := false
  third_correct := by
    have composite : ¬ Nat.Prime (corner 1532 3) := by change ¬ Nat.Prime 9385033; exact PrattCertificates.composite_of_product (factor := 7) (cofactor := 1340719) rfl (by decide) (by decide)
    exact decide_eq_false composite
theorem row_count_1532 : row 1532 = 0 := (row_1532.checked).trans rfl
theorem model_count_1532 : modelRow 1532 = 0 := by decide
theorem modelrow_agrees_1532 : modelRow 1532 = row 1532 := model_count_1532.trans row_count_1532.symm
def row_1533 : RowCertificate 1533 where
  first := false
  first_correct := by
    have composite : ¬ Nat.Prime (corner 1533 1) := by change ¬ Nat.Prime 9403423; exact PrattCertificates.composite_of_product (factor := 19) (cofactor := 494917) rfl (by decide) (by decide)
    exact decide_eq_false composite
  second := false
  second_correct := by
    have composite : ¬ Nat.Prime (corner 1533 2) := by change ¬ Nat.Prime 9400357; exact PrattCertificates.composite_of_product (factor := 41) (cofactor := 229277) rfl (by decide) (by decide)
    exact decide_eq_false composite
  third := false
  third_correct := by
    have composite : ¬ Nat.Prime (corner 1533 3) := by change ¬ Nat.Prime 9397291; exact PrattCertificates.composite_of_product (factor := 457) (cofactor := 20563) rfl (by decide) (by decide)
    exact decide_eq_false composite
theorem row_count_1533 : row 1533 = 0 := (row_1533.checked).trans rfl
theorem model_count_1533 : modelRow 1533 = 0 := by decide
theorem modelrow_agrees_1533 : modelRow 1533 = row 1533 := model_count_1533.trans row_count_1533.symm
def row_1534 : RowCertificate 1534 where
  first := false
  first_correct := by
    have composite : ¬ Nat.Prime (corner 1534 1) := by change ¬ Nat.Prime 9415693; exact PrattCertificates.composite_of_product (factor := 7) (cofactor := 1345099) rfl (by decide) (by decide)
    exact decide_eq_false composite
  second := false
  second_correct := by
    have composite : ¬ Nat.Prime (corner 1534 2) := by change ¬ Nat.Prime 9412625; exact PrattCertificates.composite_of_product (factor := 5) (cofactor := 1882525) rfl (by decide) (by decide)
    exact decide_eq_false composite
  third := false
  third_correct := by
    have composite : ¬ Nat.Prime (corner 1534 3) := by change ¬ Nat.Prime 9409557; exact PrattCertificates.composite_of_product (factor := 3) (cofactor := 3136519) rfl (by decide) (by decide)
    exact decide_eq_false composite
theorem row_count_1534 : row 1534 = 0 := (row_1534.checked).trans rfl
theorem model_count_1534 : modelRow 1534 = 0 := by decide
theorem modelrow_agrees_1534 : modelRow 1534 = row 1534 := model_count_1534.trans row_count_1534.symm
def row_1535 : RowCertificate 1535 where
  first := false
  first_correct := by
    have composite : ¬ Nat.Prime (corner 1535 1) := by change ¬ Nat.Prime 9427971; exact PrattCertificates.composite_of_product (factor := 3) (cofactor := 3142657) rfl (by decide) (by decide)
    exact decide_eq_false composite
  second := false
  second_correct := by
    have composite : ¬ Nat.Prime (corner 1535 2) := by change ¬ Nat.Prime 9424901; exact PrattCertificates.composite_of_product (factor := 829) (cofactor := 11369) rfl (by decide) (by decide)
    exact decide_eq_false composite
  third := true
  third_correct := by
    have prime : Nat.Prime (corner 1535 3) := by change Nat.Prime 9421831; exact ProjectEulerSolutions.P58PrimeNodes.p_9421831.prime
    exact decide_eq_true prime
theorem row_count_1535 : row 1535 = 1 := (row_1535.checked).trans rfl
theorem model_count_1535 : modelRow 1535 = 1 := by decide
theorem modelrow_agrees_1535 : modelRow 1535 = row 1535 := model_count_1535.trans row_count_1535.symm
def row_1536 : RowCertificate 1536 where
  first := false
  first_correct := by
    have composite : ¬ Nat.Prime (corner 1536 1) := by change ¬ Nat.Prime 9440257; exact PrattCertificates.composite_of_product (factor := 373) (cofactor := 25309) rfl (by decide) (by decide)
    exact decide_eq_false composite
  second := false
  second_correct := by
    have composite : ¬ Nat.Prime (corner 1536 2) := by change ¬ Nat.Prime 9437185; exact PrattCertificates.composite_of_product (factor := 5) (cofactor := 1887437) rfl (by decide) (by decide)
    exact decide_eq_false composite
  third := false
  third_correct := by
    have composite : ¬ Nat.Prime (corner 1536 3) := by change ¬ Nat.Prime 9434113; exact PrattCertificates.composite_of_product (factor := 13) (cofactor := 725701) rfl (by decide) (by decide)
    exact decide_eq_false composite
theorem row_count_1536 : row 1536 = 0 := (row_1536.checked).trans rfl
theorem model_count_1536 : modelRow 1536 = 0 := by decide
theorem modelrow_agrees_1536 : modelRow 1536 = row 1536 := model_count_1536.trans row_count_1536.symm
private def atomic_1409 : CountSegment 1 10 13120 1408 1409 753 753 := CountSegment.single (first := 1408) (initial := 753) row_count_1409 (by decide)
private theorem atomic_1409_agreement : RowAgreement 1408 1409 := RowAgreement.single (first := 1408) modelrow_agrees_1409
private def atomic_1410 : CountSegment 1 10 13120 1409 1410 753 753 := CountSegment.single (first := 1409) (initial := 753) row_count_1410 (by decide)
private theorem atomic_1410_agreement : RowAgreement 1409 1410 := RowAgreement.single (first := 1409) modelrow_agrees_1410
private def atomic_1411 : CountSegment 1 10 13120 1410 1411 753 753 := CountSegment.single (first := 1410) (initial := 753) row_count_1411 (by decide)
private theorem atomic_1411_agreement : RowAgreement 1410 1411 := RowAgreement.single (first := 1410) modelrow_agrees_1411
private def atomic_1412 : CountSegment 1 10 13120 1411 1412 753 754 := CountSegment.single (first := 1411) (initial := 753) row_count_1412 (by decide)
private theorem atomic_1412_agreement : RowAgreement 1411 1412 := RowAgreement.single (first := 1411) modelrow_agrees_1412
private def atomic_1413 : CountSegment 1 10 13120 1412 1413 754 754 := CountSegment.single (first := 1412) (initial := 754) row_count_1413 (by decide)
private theorem atomic_1413_agreement : RowAgreement 1412 1413 := RowAgreement.single (first := 1412) modelrow_agrees_1413
private def atomic_1414 : CountSegment 1 10 13120 1413 1414 754 755 := CountSegment.single (first := 1413) (initial := 754) row_count_1414 (by decide)
private theorem atomic_1414_agreement : RowAgreement 1413 1414 := RowAgreement.single (first := 1413) modelrow_agrees_1414
private def atomic_1415 : CountSegment 1 10 13120 1414 1415 755 755 := CountSegment.single (first := 1414) (initial := 755) row_count_1415 (by decide)
private theorem atomic_1415_agreement : RowAgreement 1414 1415 := RowAgreement.single (first := 1414) modelrow_agrees_1415
private def atomic_1416 : CountSegment 1 10 13120 1415 1416 755 755 := CountSegment.single (first := 1415) (initial := 755) row_count_1416 (by decide)
private theorem atomic_1416_agreement : RowAgreement 1415 1416 := RowAgreement.single (first := 1415) modelrow_agrees_1416
private def atomic_1417 : CountSegment 1 10 13120 1416 1417 755 756 := CountSegment.single (first := 1416) (initial := 755) row_count_1417 (by decide)
private theorem atomic_1417_agreement : RowAgreement 1416 1417 := RowAgreement.single (first := 1416) modelrow_agrees_1417
private def atomic_1418 : CountSegment 1 10 13120 1417 1418 756 757 := CountSegment.single (first := 1417) (initial := 756) row_count_1418 (by decide)
private theorem atomic_1418_agreement : RowAgreement 1417 1418 := RowAgreement.single (first := 1417) modelrow_agrees_1418
private def atomic_1419 : CountSegment 1 10 13120 1418 1419 757 757 := CountSegment.single (first := 1418) (initial := 757) row_count_1419 (by decide)
private theorem atomic_1419_agreement : RowAgreement 1418 1419 := RowAgreement.single (first := 1418) modelrow_agrees_1419
private def atomic_1420 : CountSegment 1 10 13120 1419 1420 757 758 := CountSegment.single (first := 1419) (initial := 757) row_count_1420 (by decide)
private theorem atomic_1420_agreement : RowAgreement 1419 1420 := RowAgreement.single (first := 1419) modelrow_agrees_1420
private def atomic_1421 : CountSegment 1 10 13120 1420 1421 758 759 := CountSegment.single (first := 1420) (initial := 758) row_count_1421 (by decide)
private theorem atomic_1421_agreement : RowAgreement 1420 1421 := RowAgreement.single (first := 1420) modelrow_agrees_1421
private def atomic_1422 : CountSegment 1 10 13120 1421 1422 759 759 := CountSegment.single (first := 1421) (initial := 759) row_count_1422 (by decide)
private theorem atomic_1422_agreement : RowAgreement 1421 1422 := RowAgreement.single (first := 1421) modelrow_agrees_1422
private def atomic_1423 : CountSegment 1 10 13120 1422 1423 759 759 := CountSegment.single (first := 1422) (initial := 759) row_count_1423 (by decide)
private theorem atomic_1423_agreement : RowAgreement 1422 1423 := RowAgreement.single (first := 1422) modelrow_agrees_1423
private def atomic_1424 : CountSegment 1 10 13120 1423 1424 759 759 := CountSegment.single (first := 1423) (initial := 759) row_count_1424 (by decide)
private theorem atomic_1424_agreement : RowAgreement 1423 1424 := RowAgreement.single (first := 1423) modelrow_agrees_1424
private def atomic_1425 : CountSegment 1 10 13120 1424 1425 759 761 := CountSegment.single (first := 1424) (initial := 759) row_count_1425 (by decide)
private theorem atomic_1425_agreement : RowAgreement 1424 1425 := RowAgreement.single (first := 1424) modelrow_agrees_1425
private def atomic_1426 : CountSegment 1 10 13120 1425 1426 761 761 := CountSegment.single (first := 1425) (initial := 761) row_count_1426 (by decide)
private theorem atomic_1426_agreement : RowAgreement 1425 1426 := RowAgreement.single (first := 1425) modelrow_agrees_1426
private def atomic_1427 : CountSegment 1 10 13120 1426 1427 761 761 := CountSegment.single (first := 1426) (initial := 761) row_count_1427 (by decide)
private theorem atomic_1427_agreement : RowAgreement 1426 1427 := RowAgreement.single (first := 1426) modelrow_agrees_1427
private def atomic_1428 : CountSegment 1 10 13120 1427 1428 761 761 := CountSegment.single (first := 1427) (initial := 761) row_count_1428 (by decide)
private theorem atomic_1428_agreement : RowAgreement 1427 1428 := RowAgreement.single (first := 1427) modelrow_agrees_1428
private def atomic_1429 : CountSegment 1 10 13120 1428 1429 761 761 := CountSegment.single (first := 1428) (initial := 761) row_count_1429 (by decide)
private theorem atomic_1429_agreement : RowAgreement 1428 1429 := RowAgreement.single (first := 1428) modelrow_agrees_1429
private def atomic_1430 : CountSegment 1 10 13120 1429 1430 761 761 := CountSegment.single (first := 1429) (initial := 761) row_count_1430 (by decide)
private theorem atomic_1430_agreement : RowAgreement 1429 1430 := RowAgreement.single (first := 1429) modelrow_agrees_1430
private def atomic_1431 : CountSegment 1 10 13120 1430 1431 761 762 := CountSegment.single (first := 1430) (initial := 761) row_count_1431 (by decide)
private theorem atomic_1431_agreement : RowAgreement 1430 1431 := RowAgreement.single (first := 1430) modelrow_agrees_1431
private def atomic_1432 : CountSegment 1 10 13120 1431 1432 762 764 := CountSegment.single (first := 1431) (initial := 762) row_count_1432 (by decide)
private theorem atomic_1432_agreement : RowAgreement 1431 1432 := RowAgreement.single (first := 1431) modelrow_agrees_1432
private def atomic_1433 : CountSegment 1 10 13120 1432 1433 764 764 := CountSegment.single (first := 1432) (initial := 764) row_count_1433 (by decide)
private theorem atomic_1433_agreement : RowAgreement 1432 1433 := RowAgreement.single (first := 1432) modelrow_agrees_1433
private def atomic_1434 : CountSegment 1 10 13120 1433 1434 764 765 := CountSegment.single (first := 1433) (initial := 764) row_count_1434 (by decide)
private theorem atomic_1434_agreement : RowAgreement 1433 1434 := RowAgreement.single (first := 1433) modelrow_agrees_1434
private def atomic_1435 : CountSegment 1 10 13120 1434 1435 765 766 := CountSegment.single (first := 1434) (initial := 765) row_count_1435 (by decide)
private theorem atomic_1435_agreement : RowAgreement 1434 1435 := RowAgreement.single (first := 1434) modelrow_agrees_1435
private def atomic_1436 : CountSegment 1 10 13120 1435 1436 766 767 := CountSegment.single (first := 1435) (initial := 766) row_count_1436 (by decide)
private theorem atomic_1436_agreement : RowAgreement 1435 1436 := RowAgreement.single (first := 1435) modelrow_agrees_1436
private def atomic_1437 : CountSegment 1 10 13120 1436 1437 767 768 := CountSegment.single (first := 1436) (initial := 767) row_count_1437 (by decide)
private theorem atomic_1437_agreement : RowAgreement 1436 1437 := RowAgreement.single (first := 1436) modelrow_agrees_1437
private def atomic_1438 : CountSegment 1 10 13120 1437 1438 768 769 := CountSegment.single (first := 1437) (initial := 768) row_count_1438 (by decide)
private theorem atomic_1438_agreement : RowAgreement 1437 1438 := RowAgreement.single (first := 1437) modelrow_agrees_1438
private def atomic_1439 : CountSegment 1 10 13120 1438 1439 769 769 := CountSegment.single (first := 1438) (initial := 769) row_count_1439 (by decide)
private theorem atomic_1439_agreement : RowAgreement 1438 1439 := RowAgreement.single (first := 1438) modelrow_agrees_1439
private def atomic_1440 : CountSegment 1 10 13120 1439 1440 769 769 := CountSegment.single (first := 1439) (initial := 769) row_count_1440 (by decide)
private theorem atomic_1440_agreement : RowAgreement 1439 1440 := RowAgreement.single (first := 1439) modelrow_agrees_1440
private def atomic_1441 : CountSegment 1 10 13120 1440 1441 769 769 := CountSegment.single (first := 1440) (initial := 769) row_count_1441 (by decide)
private theorem atomic_1441_agreement : RowAgreement 1440 1441 := RowAgreement.single (first := 1440) modelrow_agrees_1441
private def atomic_1442 : CountSegment 1 10 13120 1441 1442 769 771 := CountSegment.single (first := 1441) (initial := 769) row_count_1442 (by decide)
private theorem atomic_1442_agreement : RowAgreement 1441 1442 := RowAgreement.single (first := 1441) modelrow_agrees_1442
private def atomic_1443 : CountSegment 1 10 13120 1442 1443 771 772 := CountSegment.single (first := 1442) (initial := 771) row_count_1443 (by decide)
private theorem atomic_1443_agreement : RowAgreement 1442 1443 := RowAgreement.single (first := 1442) modelrow_agrees_1443
private def atomic_1444 : CountSegment 1 10 13120 1443 1444 772 772 := CountSegment.single (first := 1443) (initial := 772) row_count_1444 (by decide)
private theorem atomic_1444_agreement : RowAgreement 1443 1444 := RowAgreement.single (first := 1443) modelrow_agrees_1444
private def atomic_1445 : CountSegment 1 10 13120 1444 1445 772 773 := CountSegment.single (first := 1444) (initial := 772) row_count_1445 (by decide)
private theorem atomic_1445_agreement : RowAgreement 1444 1445 := RowAgreement.single (first := 1444) modelrow_agrees_1445
private def atomic_1446 : CountSegment 1 10 13120 1445 1446 773 774 := CountSegment.single (first := 1445) (initial := 773) row_count_1446 (by decide)
private theorem atomic_1446_agreement : RowAgreement 1445 1446 := RowAgreement.single (first := 1445) modelrow_agrees_1446
private def atomic_1447 : CountSegment 1 10 13120 1446 1447 774 775 := CountSegment.single (first := 1446) (initial := 774) row_count_1447 (by decide)
private theorem atomic_1447_agreement : RowAgreement 1446 1447 := RowAgreement.single (first := 1446) modelrow_agrees_1447
private def atomic_1448 : CountSegment 1 10 13120 1447 1448 775 776 := CountSegment.single (first := 1447) (initial := 775) row_count_1448 (by decide)
private theorem atomic_1448_agreement : RowAgreement 1447 1448 := RowAgreement.single (first := 1447) modelrow_agrees_1448
private def atomic_1449 : CountSegment 1 10 13120 1448 1449 776 777 := CountSegment.single (first := 1448) (initial := 776) row_count_1449 (by decide)
private theorem atomic_1449_agreement : RowAgreement 1448 1449 := RowAgreement.single (first := 1448) modelrow_agrees_1449
private def atomic_1450 : CountSegment 1 10 13120 1449 1450 777 778 := CountSegment.single (first := 1449) (initial := 777) row_count_1450 (by decide)
private theorem atomic_1450_agreement : RowAgreement 1449 1450 := RowAgreement.single (first := 1449) modelrow_agrees_1450
private def atomic_1451 : CountSegment 1 10 13120 1450 1451 778 778 := CountSegment.single (first := 1450) (initial := 778) row_count_1451 (by decide)
private theorem atomic_1451_agreement : RowAgreement 1450 1451 := RowAgreement.single (first := 1450) modelrow_agrees_1451
private def atomic_1452 : CountSegment 1 10 13120 1451 1452 778 778 := CountSegment.single (first := 1451) (initial := 778) row_count_1452 (by decide)
private theorem atomic_1452_agreement : RowAgreement 1451 1452 := RowAgreement.single (first := 1451) modelrow_agrees_1452
private def atomic_1453 : CountSegment 1 10 13120 1452 1453 778 779 := CountSegment.single (first := 1452) (initial := 778) row_count_1453 (by decide)
private theorem atomic_1453_agreement : RowAgreement 1452 1453 := RowAgreement.single (first := 1452) modelrow_agrees_1453
private def atomic_1454 : CountSegment 1 10 13120 1453 1454 779 779 := CountSegment.single (first := 1453) (initial := 779) row_count_1454 (by decide)
private theorem atomic_1454_agreement : RowAgreement 1453 1454 := RowAgreement.single (first := 1453) modelrow_agrees_1454
private def atomic_1455 : CountSegment 1 10 13120 1454 1455 779 779 := CountSegment.single (first := 1454) (initial := 779) row_count_1455 (by decide)
private theorem atomic_1455_agreement : RowAgreement 1454 1455 := RowAgreement.single (first := 1454) modelrow_agrees_1455
private def atomic_1456 : CountSegment 1 10 13120 1455 1456 779 779 := CountSegment.single (first := 1455) (initial := 779) row_count_1456 (by decide)
private theorem atomic_1456_agreement : RowAgreement 1455 1456 := RowAgreement.single (first := 1455) modelrow_agrees_1456
private def atomic_1457 : CountSegment 1 10 13120 1456 1457 779 779 := CountSegment.single (first := 1456) (initial := 779) row_count_1457 (by decide)
private theorem atomic_1457_agreement : RowAgreement 1456 1457 := RowAgreement.single (first := 1456) modelrow_agrees_1457
private def atomic_1458 : CountSegment 1 10 13120 1457 1458 779 780 := CountSegment.single (first := 1457) (initial := 779) row_count_1458 (by decide)
private theorem atomic_1458_agreement : RowAgreement 1457 1458 := RowAgreement.single (first := 1457) modelrow_agrees_1458
private def atomic_1459 : CountSegment 1 10 13120 1458 1459 780 780 := CountSegment.single (first := 1458) (initial := 780) row_count_1459 (by decide)
private theorem atomic_1459_agreement : RowAgreement 1458 1459 := RowAgreement.single (first := 1458) modelrow_agrees_1459
private def atomic_1460 : CountSegment 1 10 13120 1459 1460 780 780 := CountSegment.single (first := 1459) (initial := 780) row_count_1460 (by decide)
private theorem atomic_1460_agreement : RowAgreement 1459 1460 := RowAgreement.single (first := 1459) modelrow_agrees_1460
private def atomic_1461 : CountSegment 1 10 13120 1460 1461 780 781 := CountSegment.single (first := 1460) (initial := 780) row_count_1461 (by decide)
private theorem atomic_1461_agreement : RowAgreement 1460 1461 := RowAgreement.single (first := 1460) modelrow_agrees_1461
private def atomic_1462 : CountSegment 1 10 13120 1461 1462 781 782 := CountSegment.single (first := 1461) (initial := 781) row_count_1462 (by decide)
private theorem atomic_1462_agreement : RowAgreement 1461 1462 := RowAgreement.single (first := 1461) modelrow_agrees_1462
private def atomic_1463 : CountSegment 1 10 13120 1462 1463 782 784 := CountSegment.single (first := 1462) (initial := 782) row_count_1463 (by decide)
private theorem atomic_1463_agreement : RowAgreement 1462 1463 := RowAgreement.single (first := 1462) modelrow_agrees_1463
private def atomic_1464 : CountSegment 1 10 13120 1463 1464 784 784 := CountSegment.single (first := 1463) (initial := 784) row_count_1464 (by decide)
private theorem atomic_1464_agreement : RowAgreement 1463 1464 := RowAgreement.single (first := 1463) modelrow_agrees_1464
private def atomic_1465 : CountSegment 1 10 13120 1464 1465 784 784 := CountSegment.single (first := 1464) (initial := 784) row_count_1465 (by decide)
private theorem atomic_1465_agreement : RowAgreement 1464 1465 := RowAgreement.single (first := 1464) modelrow_agrees_1465
private def atomic_1466 : CountSegment 1 10 13120 1465 1466 784 784 := CountSegment.single (first := 1465) (initial := 784) row_count_1466 (by decide)
private theorem atomic_1466_agreement : RowAgreement 1465 1466 := RowAgreement.single (first := 1465) modelrow_agrees_1466
private def atomic_1467 : CountSegment 1 10 13120 1466 1467 784 785 := CountSegment.single (first := 1466) (initial := 784) row_count_1467 (by decide)
private theorem atomic_1467_agreement : RowAgreement 1466 1467 := RowAgreement.single (first := 1466) modelrow_agrees_1467
private def atomic_1468 : CountSegment 1 10 13120 1467 1468 785 786 := CountSegment.single (first := 1467) (initial := 785) row_count_1468 (by decide)
private theorem atomic_1468_agreement : RowAgreement 1467 1468 := RowAgreement.single (first := 1467) modelrow_agrees_1468
private def atomic_1469 : CountSegment 1 10 13120 1468 1469 786 786 := CountSegment.single (first := 1468) (initial := 786) row_count_1469 (by decide)
private theorem atomic_1469_agreement : RowAgreement 1468 1469 := RowAgreement.single (first := 1468) modelrow_agrees_1469
private def atomic_1470 : CountSegment 1 10 13120 1469 1470 786 786 := CountSegment.single (first := 1469) (initial := 786) row_count_1470 (by decide)
private theorem atomic_1470_agreement : RowAgreement 1469 1470 := RowAgreement.single (first := 1469) modelrow_agrees_1470
private def atomic_1471 : CountSegment 1 10 13120 1470 1471 786 786 := CountSegment.single (first := 1470) (initial := 786) row_count_1471 (by decide)
private theorem atomic_1471_agreement : RowAgreement 1470 1471 := RowAgreement.single (first := 1470) modelrow_agrees_1471
private def atomic_1472 : CountSegment 1 10 13120 1471 1472 786 788 := CountSegment.single (first := 1471) (initial := 786) row_count_1472 (by decide)
private theorem atomic_1472_agreement : RowAgreement 1471 1472 := RowAgreement.single (first := 1471) modelrow_agrees_1472
private def atomic_1473 : CountSegment 1 10 13120 1472 1473 788 788 := CountSegment.single (first := 1472) (initial := 788) row_count_1473 (by decide)
private theorem atomic_1473_agreement : RowAgreement 1472 1473 := RowAgreement.single (first := 1472) modelrow_agrees_1473
private def atomic_1474 : CountSegment 1 10 13120 1473 1474 788 789 := CountSegment.single (first := 1473) (initial := 788) row_count_1474 (by decide)
private theorem atomic_1474_agreement : RowAgreement 1473 1474 := RowAgreement.single (first := 1473) modelrow_agrees_1474
private def atomic_1475 : CountSegment 1 10 13120 1474 1475 789 789 := CountSegment.single (first := 1474) (initial := 789) row_count_1475 (by decide)
private theorem atomic_1475_agreement : RowAgreement 1474 1475 := RowAgreement.single (first := 1474) modelrow_agrees_1475
private def atomic_1476 : CountSegment 1 10 13120 1475 1476 789 789 := CountSegment.single (first := 1475) (initial := 789) row_count_1476 (by decide)
private theorem atomic_1476_agreement : RowAgreement 1475 1476 := RowAgreement.single (first := 1475) modelrow_agrees_1476
private def atomic_1477 : CountSegment 1 10 13120 1476 1477 789 789 := CountSegment.single (first := 1476) (initial := 789) row_count_1477 (by decide)
private theorem atomic_1477_agreement : RowAgreement 1476 1477 := RowAgreement.single (first := 1476) modelrow_agrees_1477
private def atomic_1478 : CountSegment 1 10 13120 1477 1478 789 790 := CountSegment.single (first := 1477) (initial := 789) row_count_1478 (by decide)
private theorem atomic_1478_agreement : RowAgreement 1477 1478 := RowAgreement.single (first := 1477) modelrow_agrees_1478
private def atomic_1479 : CountSegment 1 10 13120 1478 1479 790 791 := CountSegment.single (first := 1478) (initial := 790) row_count_1479 (by decide)
private theorem atomic_1479_agreement : RowAgreement 1478 1479 := RowAgreement.single (first := 1478) modelrow_agrees_1479
private def atomic_1480 : CountSegment 1 10 13120 1479 1480 791 792 := CountSegment.single (first := 1479) (initial := 791) row_count_1480 (by decide)
private theorem atomic_1480_agreement : RowAgreement 1479 1480 := RowAgreement.single (first := 1479) modelrow_agrees_1480
private def atomic_1481 : CountSegment 1 10 13120 1480 1481 792 792 := CountSegment.single (first := 1480) (initial := 792) row_count_1481 (by decide)
private theorem atomic_1481_agreement : RowAgreement 1480 1481 := RowAgreement.single (first := 1480) modelrow_agrees_1481
private def atomic_1482 : CountSegment 1 10 13120 1481 1482 792 793 := CountSegment.single (first := 1481) (initial := 792) row_count_1482 (by decide)
private theorem atomic_1482_agreement : RowAgreement 1481 1482 := RowAgreement.single (first := 1481) modelrow_agrees_1482
private def atomic_1483 : CountSegment 1 10 13120 1482 1483 793 794 := CountSegment.single (first := 1482) (initial := 793) row_count_1483 (by decide)
private theorem atomic_1483_agreement : RowAgreement 1482 1483 := RowAgreement.single (first := 1482) modelrow_agrees_1483
private def atomic_1484 : CountSegment 1 10 13120 1483 1484 794 794 := CountSegment.single (first := 1483) (initial := 794) row_count_1484 (by decide)
private theorem atomic_1484_agreement : RowAgreement 1483 1484 := RowAgreement.single (first := 1483) modelrow_agrees_1484
private def atomic_1485 : CountSegment 1 10 13120 1484 1485 794 795 := CountSegment.single (first := 1484) (initial := 794) row_count_1485 (by decide)
private theorem atomic_1485_agreement : RowAgreement 1484 1485 := RowAgreement.single (first := 1484) modelrow_agrees_1485
private def atomic_1486 : CountSegment 1 10 13120 1485 1486 795 795 := CountSegment.single (first := 1485) (initial := 795) row_count_1486 (by decide)
private theorem atomic_1486_agreement : RowAgreement 1485 1486 := RowAgreement.single (first := 1485) modelrow_agrees_1486
private def atomic_1487 : CountSegment 1 10 13120 1486 1487 795 796 := CountSegment.single (first := 1486) (initial := 795) row_count_1487 (by decide)
private theorem atomic_1487_agreement : RowAgreement 1486 1487 := RowAgreement.single (first := 1486) modelrow_agrees_1487
private def atomic_1488 : CountSegment 1 10 13120 1487 1488 796 796 := CountSegment.single (first := 1487) (initial := 796) row_count_1488 (by decide)
private theorem atomic_1488_agreement : RowAgreement 1487 1488 := RowAgreement.single (first := 1487) modelrow_agrees_1488
private def atomic_1489 : CountSegment 1 10 13120 1488 1489 796 796 := CountSegment.single (first := 1488) (initial := 796) row_count_1489 (by decide)
private theorem atomic_1489_agreement : RowAgreement 1488 1489 := RowAgreement.single (first := 1488) modelrow_agrees_1489
private def atomic_1490 : CountSegment 1 10 13120 1489 1490 796 796 := CountSegment.single (first := 1489) (initial := 796) row_count_1490 (by decide)
private theorem atomic_1490_agreement : RowAgreement 1489 1490 := RowAgreement.single (first := 1489) modelrow_agrees_1490
private def atomic_1491 : CountSegment 1 10 13120 1490 1491 796 797 := CountSegment.single (first := 1490) (initial := 796) row_count_1491 (by decide)
private theorem atomic_1491_agreement : RowAgreement 1490 1491 := RowAgreement.single (first := 1490) modelrow_agrees_1491
private def atomic_1492 : CountSegment 1 10 13120 1491 1492 797 797 := CountSegment.single (first := 1491) (initial := 797) row_count_1492 (by decide)
private theorem atomic_1492_agreement : RowAgreement 1491 1492 := RowAgreement.single (first := 1491) modelrow_agrees_1492
private def atomic_1493 : CountSegment 1 10 13120 1492 1493 797 798 := CountSegment.single (first := 1492) (initial := 797) row_count_1493 (by decide)
private theorem atomic_1493_agreement : RowAgreement 1492 1493 := RowAgreement.single (first := 1492) modelrow_agrees_1493
private def atomic_1494 : CountSegment 1 10 13120 1493 1494 798 798 := CountSegment.single (first := 1493) (initial := 798) row_count_1494 (by decide)
private theorem atomic_1494_agreement : RowAgreement 1493 1494 := RowAgreement.single (first := 1493) modelrow_agrees_1494
private def atomic_1495 : CountSegment 1 10 13120 1494 1495 798 798 := CountSegment.single (first := 1494) (initial := 798) row_count_1495 (by decide)
private theorem atomic_1495_agreement : RowAgreement 1494 1495 := RowAgreement.single (first := 1494) modelrow_agrees_1495
private def atomic_1496 : CountSegment 1 10 13120 1495 1496 798 798 := CountSegment.single (first := 1495) (initial := 798) row_count_1496 (by decide)
private theorem atomic_1496_agreement : RowAgreement 1495 1496 := RowAgreement.single (first := 1495) modelrow_agrees_1496
private def atomic_1497 : CountSegment 1 10 13120 1496 1497 798 798 := CountSegment.single (first := 1496) (initial := 798) row_count_1497 (by decide)
private theorem atomic_1497_agreement : RowAgreement 1496 1497 := RowAgreement.single (first := 1496) modelrow_agrees_1497
private def atomic_1498 : CountSegment 1 10 13120 1497 1498 798 799 := CountSegment.single (first := 1497) (initial := 798) row_count_1498 (by decide)
private theorem atomic_1498_agreement : RowAgreement 1497 1498 := RowAgreement.single (first := 1497) modelrow_agrees_1498
private def atomic_1499 : CountSegment 1 10 13120 1498 1499 799 800 := CountSegment.single (first := 1498) (initial := 799) row_count_1499 (by decide)
private theorem atomic_1499_agreement : RowAgreement 1498 1499 := RowAgreement.single (first := 1498) modelrow_agrees_1499
private def atomic_1500 : CountSegment 1 10 13120 1499 1500 800 800 := CountSegment.single (first := 1499) (initial := 800) row_count_1500 (by decide)
private theorem atomic_1500_agreement : RowAgreement 1499 1500 := RowAgreement.single (first := 1499) modelrow_agrees_1500
private def atomic_1501 : CountSegment 1 10 13120 1500 1501 800 800 := CountSegment.single (first := 1500) (initial := 800) row_count_1501 (by decide)
private theorem atomic_1501_agreement : RowAgreement 1500 1501 := RowAgreement.single (first := 1500) modelrow_agrees_1501
private def atomic_1502 : CountSegment 1 10 13120 1501 1502 800 800 := CountSegment.single (first := 1501) (initial := 800) row_count_1502 (by decide)
private theorem atomic_1502_agreement : RowAgreement 1501 1502 := RowAgreement.single (first := 1501) modelrow_agrees_1502
private def atomic_1503 : CountSegment 1 10 13120 1502 1503 800 800 := CountSegment.single (first := 1502) (initial := 800) row_count_1503 (by decide)
private theorem atomic_1503_agreement : RowAgreement 1502 1503 := RowAgreement.single (first := 1502) modelrow_agrees_1503
private def atomic_1504 : CountSegment 1 10 13120 1503 1504 800 800 := CountSegment.single (first := 1503) (initial := 800) row_count_1504 (by decide)
private theorem atomic_1504_agreement : RowAgreement 1503 1504 := RowAgreement.single (first := 1503) modelrow_agrees_1504
private def atomic_1505 : CountSegment 1 10 13120 1504 1505 800 800 := CountSegment.single (first := 1504) (initial := 800) row_count_1505 (by decide)
private theorem atomic_1505_agreement : RowAgreement 1504 1505 := RowAgreement.single (first := 1504) modelrow_agrees_1505
private def atomic_1506 : CountSegment 1 10 13120 1505 1506 800 801 := CountSegment.single (first := 1505) (initial := 800) row_count_1506 (by decide)
private theorem atomic_1506_agreement : RowAgreement 1505 1506 := RowAgreement.single (first := 1505) modelrow_agrees_1506
private def atomic_1507 : CountSegment 1 10 13120 1506 1507 801 801 := CountSegment.single (first := 1506) (initial := 801) row_count_1507 (by decide)
private theorem atomic_1507_agreement : RowAgreement 1506 1507 := RowAgreement.single (first := 1506) modelrow_agrees_1507
private def atomic_1508 : CountSegment 1 10 13120 1507 1508 801 802 := CountSegment.single (first := 1507) (initial := 801) row_count_1508 (by decide)
private theorem atomic_1508_agreement : RowAgreement 1507 1508 := RowAgreement.single (first := 1507) modelrow_agrees_1508
private def atomic_1509 : CountSegment 1 10 13120 1508 1509 802 802 := CountSegment.single (first := 1508) (initial := 802) row_count_1509 (by decide)
private theorem atomic_1509_agreement : RowAgreement 1508 1509 := RowAgreement.single (first := 1508) modelrow_agrees_1509
private def atomic_1510 : CountSegment 1 10 13120 1509 1510 802 803 := CountSegment.single (first := 1509) (initial := 802) row_count_1510 (by decide)
private theorem atomic_1510_agreement : RowAgreement 1509 1510 := RowAgreement.single (first := 1509) modelrow_agrees_1510
private def atomic_1511 : CountSegment 1 10 13120 1510 1511 803 803 := CountSegment.single (first := 1510) (initial := 803) row_count_1511 (by decide)
private theorem atomic_1511_agreement : RowAgreement 1510 1511 := RowAgreement.single (first := 1510) modelrow_agrees_1511
private def atomic_1512 : CountSegment 1 10 13120 1511 1512 803 804 := CountSegment.single (first := 1511) (initial := 803) row_count_1512 (by decide)
private theorem atomic_1512_agreement : RowAgreement 1511 1512 := RowAgreement.single (first := 1511) modelrow_agrees_1512
private def atomic_1513 : CountSegment 1 10 13120 1512 1513 804 805 := CountSegment.single (first := 1512) (initial := 804) row_count_1513 (by decide)
private theorem atomic_1513_agreement : RowAgreement 1512 1513 := RowAgreement.single (first := 1512) modelrow_agrees_1513
private def atomic_1514 : CountSegment 1 10 13120 1513 1514 805 806 := CountSegment.single (first := 1513) (initial := 805) row_count_1514 (by decide)
private theorem atomic_1514_agreement : RowAgreement 1513 1514 := RowAgreement.single (first := 1513) modelrow_agrees_1514
private def atomic_1515 : CountSegment 1 10 13120 1514 1515 806 807 := CountSegment.single (first := 1514) (initial := 806) row_count_1515 (by decide)
private theorem atomic_1515_agreement : RowAgreement 1514 1515 := RowAgreement.single (first := 1514) modelrow_agrees_1515
private def atomic_1516 : CountSegment 1 10 13120 1515 1516 807 807 := CountSegment.single (first := 1515) (initial := 807) row_count_1516 (by decide)
private theorem atomic_1516_agreement : RowAgreement 1515 1516 := RowAgreement.single (first := 1515) modelrow_agrees_1516
private def atomic_1517 : CountSegment 1 10 13120 1516 1517 807 807 := CountSegment.single (first := 1516) (initial := 807) row_count_1517 (by decide)
private theorem atomic_1517_agreement : RowAgreement 1516 1517 := RowAgreement.single (first := 1516) modelrow_agrees_1517
private def atomic_1518 : CountSegment 1 10 13120 1517 1518 807 807 := CountSegment.single (first := 1517) (initial := 807) row_count_1518 (by decide)
private theorem atomic_1518_agreement : RowAgreement 1517 1518 := RowAgreement.single (first := 1517) modelrow_agrees_1518
private def atomic_1519 : CountSegment 1 10 13120 1518 1519 807 807 := CountSegment.single (first := 1518) (initial := 807) row_count_1519 (by decide)
private theorem atomic_1519_agreement : RowAgreement 1518 1519 := RowAgreement.single (first := 1518) modelrow_agrees_1519
private def atomic_1520 : CountSegment 1 10 13120 1519 1520 807 808 := CountSegment.single (first := 1519) (initial := 807) row_count_1520 (by decide)
private theorem atomic_1520_agreement : RowAgreement 1519 1520 := RowAgreement.single (first := 1519) modelrow_agrees_1520
private def atomic_1521 : CountSegment 1 10 13120 1520 1521 808 809 := CountSegment.single (first := 1520) (initial := 808) row_count_1521 (by decide)
private theorem atomic_1521_agreement : RowAgreement 1520 1521 := RowAgreement.single (first := 1520) modelrow_agrees_1521
private def atomic_1522 : CountSegment 1 10 13120 1521 1522 809 809 := CountSegment.single (first := 1521) (initial := 809) row_count_1522 (by decide)
private theorem atomic_1522_agreement : RowAgreement 1521 1522 := RowAgreement.single (first := 1521) modelrow_agrees_1522
private def atomic_1523 : CountSegment 1 10 13120 1522 1523 809 810 := CountSegment.single (first := 1522) (initial := 809) row_count_1523 (by decide)
private theorem atomic_1523_agreement : RowAgreement 1522 1523 := RowAgreement.single (first := 1522) modelrow_agrees_1523
private def atomic_1524 : CountSegment 1 10 13120 1523 1524 810 811 := CountSegment.single (first := 1523) (initial := 810) row_count_1524 (by decide)
private theorem atomic_1524_agreement : RowAgreement 1523 1524 := RowAgreement.single (first := 1523) modelrow_agrees_1524
private def atomic_1525 : CountSegment 1 10 13120 1524 1525 811 812 := CountSegment.single (first := 1524) (initial := 811) row_count_1525 (by decide)
private theorem atomic_1525_agreement : RowAgreement 1524 1525 := RowAgreement.single (first := 1524) modelrow_agrees_1525
private def atomic_1526 : CountSegment 1 10 13120 1525 1526 812 812 := CountSegment.single (first := 1525) (initial := 812) row_count_1526 (by decide)
private theorem atomic_1526_agreement : RowAgreement 1525 1526 := RowAgreement.single (first := 1525) modelrow_agrees_1526
private def atomic_1527 : CountSegment 1 10 13120 1526 1527 812 814 := CountSegment.single (first := 1526) (initial := 812) row_count_1527 (by decide)
private theorem atomic_1527_agreement : RowAgreement 1526 1527 := RowAgreement.single (first := 1526) modelrow_agrees_1527
private def atomic_1528 : CountSegment 1 10 13120 1527 1528 814 814 := CountSegment.single (first := 1527) (initial := 814) row_count_1528 (by decide)
private theorem atomic_1528_agreement : RowAgreement 1527 1528 := RowAgreement.single (first := 1527) modelrow_agrees_1528
private def atomic_1529 : CountSegment 1 10 13120 1528 1529 814 814 := CountSegment.single (first := 1528) (initial := 814) row_count_1529 (by decide)
private theorem atomic_1529_agreement : RowAgreement 1528 1529 := RowAgreement.single (first := 1528) modelrow_agrees_1529
private def atomic_1530 : CountSegment 1 10 13120 1529 1530 814 814 := CountSegment.single (first := 1529) (initial := 814) row_count_1530 (by decide)
private theorem atomic_1530_agreement : RowAgreement 1529 1530 := RowAgreement.single (first := 1529) modelrow_agrees_1530
private def atomic_1531 : CountSegment 1 10 13120 1530 1531 814 814 := CountSegment.single (first := 1530) (initial := 814) row_count_1531 (by decide)
private theorem atomic_1531_agreement : RowAgreement 1530 1531 := RowAgreement.single (first := 1530) modelrow_agrees_1531
private def atomic_1532 : CountSegment 1 10 13120 1531 1532 814 814 := CountSegment.single (first := 1531) (initial := 814) row_count_1532 (by decide)
private theorem atomic_1532_agreement : RowAgreement 1531 1532 := RowAgreement.single (first := 1531) modelrow_agrees_1532
private def atomic_1533 : CountSegment 1 10 13120 1532 1533 814 814 := CountSegment.single (first := 1532) (initial := 814) row_count_1533 (by decide)
private theorem atomic_1533_agreement : RowAgreement 1532 1533 := RowAgreement.single (first := 1532) modelrow_agrees_1533
private def atomic_1534 : CountSegment 1 10 13120 1533 1534 814 814 := CountSegment.single (first := 1533) (initial := 814) row_count_1534 (by decide)
private theorem atomic_1534_agreement : RowAgreement 1533 1534 := RowAgreement.single (first := 1533) modelrow_agrees_1534
private def atomic_1535 : CountSegment 1 10 13120 1534 1535 814 815 := CountSegment.single (first := 1534) (initial := 814) row_count_1535 (by decide)
private theorem atomic_1535_agreement : RowAgreement 1534 1535 := RowAgreement.single (first := 1534) modelrow_agrees_1535
private def atomic_1536 : CountSegment 1 10 13120 1535 1536 815 815 := CountSegment.single (first := 1535) (initial := 815) row_count_1536 (by decide)
private theorem atomic_1536_agreement : RowAgreement 1535 1536 := RowAgreement.single (first := 1535) modelrow_agrees_1536
private def within_011_000 : CountSegment 1 10 13120 1408 1410 753 753 := CountSegment.join atomic_1409 atomic_1410
private theorem within_011_000_agreement : RowAgreement 1408 1410 := RowAgreement.join atomic_1409_agreement atomic_1410_agreement
private def within_011_001 : CountSegment 1 10 13120 1410 1412 753 754 := CountSegment.join atomic_1411 atomic_1412
private theorem within_011_001_agreement : RowAgreement 1410 1412 := RowAgreement.join atomic_1411_agreement atomic_1412_agreement
private def within_011_002 : CountSegment 1 10 13120 1408 1412 753 754 := CountSegment.join within_011_000 within_011_001
private theorem within_011_002_agreement : RowAgreement 1408 1412 := RowAgreement.join within_011_000_agreement within_011_001_agreement
private def within_011_003 : CountSegment 1 10 13120 1412 1414 754 755 := CountSegment.join atomic_1413 atomic_1414
private theorem within_011_003_agreement : RowAgreement 1412 1414 := RowAgreement.join atomic_1413_agreement atomic_1414_agreement
private def within_011_004 : CountSegment 1 10 13120 1414 1416 755 755 := CountSegment.join atomic_1415 atomic_1416
private theorem within_011_004_agreement : RowAgreement 1414 1416 := RowAgreement.join atomic_1415_agreement atomic_1416_agreement
private def within_011_005 : CountSegment 1 10 13120 1412 1416 754 755 := CountSegment.join within_011_003 within_011_004
private theorem within_011_005_agreement : RowAgreement 1412 1416 := RowAgreement.join within_011_003_agreement within_011_004_agreement
private def within_011_006 : CountSegment 1 10 13120 1408 1416 753 755 := CountSegment.join within_011_002 within_011_005
private theorem within_011_006_agreement : RowAgreement 1408 1416 := RowAgreement.join within_011_002_agreement within_011_005_agreement
private def within_011_007 : CountSegment 1 10 13120 1416 1418 755 757 := CountSegment.join atomic_1417 atomic_1418
private theorem within_011_007_agreement : RowAgreement 1416 1418 := RowAgreement.join atomic_1417_agreement atomic_1418_agreement
private def within_011_008 : CountSegment 1 10 13120 1418 1420 757 758 := CountSegment.join atomic_1419 atomic_1420
private theorem within_011_008_agreement : RowAgreement 1418 1420 := RowAgreement.join atomic_1419_agreement atomic_1420_agreement
private def within_011_009 : CountSegment 1 10 13120 1416 1420 755 758 := CountSegment.join within_011_007 within_011_008
private theorem within_011_009_agreement : RowAgreement 1416 1420 := RowAgreement.join within_011_007_agreement within_011_008_agreement
private def within_011_010 : CountSegment 1 10 13120 1420 1422 758 759 := CountSegment.join atomic_1421 atomic_1422
private theorem within_011_010_agreement : RowAgreement 1420 1422 := RowAgreement.join atomic_1421_agreement atomic_1422_agreement
private def within_011_011 : CountSegment 1 10 13120 1422 1424 759 759 := CountSegment.join atomic_1423 atomic_1424
private theorem within_011_011_agreement : RowAgreement 1422 1424 := RowAgreement.join atomic_1423_agreement atomic_1424_agreement
private def within_011_012 : CountSegment 1 10 13120 1420 1424 758 759 := CountSegment.join within_011_010 within_011_011
private theorem within_011_012_agreement : RowAgreement 1420 1424 := RowAgreement.join within_011_010_agreement within_011_011_agreement
private def within_011_013 : CountSegment 1 10 13120 1416 1424 755 759 := CountSegment.join within_011_009 within_011_012
private theorem within_011_013_agreement : RowAgreement 1416 1424 := RowAgreement.join within_011_009_agreement within_011_012_agreement
private def within_011_014 : CountSegment 1 10 13120 1408 1424 753 759 := CountSegment.join within_011_006 within_011_013
private theorem within_011_014_agreement : RowAgreement 1408 1424 := RowAgreement.join within_011_006_agreement within_011_013_agreement
private def within_011_015 : CountSegment 1 10 13120 1424 1426 759 761 := CountSegment.join atomic_1425 atomic_1426
private theorem within_011_015_agreement : RowAgreement 1424 1426 := RowAgreement.join atomic_1425_agreement atomic_1426_agreement
private def within_011_016 : CountSegment 1 10 13120 1426 1428 761 761 := CountSegment.join atomic_1427 atomic_1428
private theorem within_011_016_agreement : RowAgreement 1426 1428 := RowAgreement.join atomic_1427_agreement atomic_1428_agreement
private def within_011_017 : CountSegment 1 10 13120 1424 1428 759 761 := CountSegment.join within_011_015 within_011_016
private theorem within_011_017_agreement : RowAgreement 1424 1428 := RowAgreement.join within_011_015_agreement within_011_016_agreement
private def within_011_018 : CountSegment 1 10 13120 1428 1430 761 761 := CountSegment.join atomic_1429 atomic_1430
private theorem within_011_018_agreement : RowAgreement 1428 1430 := RowAgreement.join atomic_1429_agreement atomic_1430_agreement
private def within_011_019 : CountSegment 1 10 13120 1430 1432 761 764 := CountSegment.join atomic_1431 atomic_1432
private theorem within_011_019_agreement : RowAgreement 1430 1432 := RowAgreement.join atomic_1431_agreement atomic_1432_agreement
private def within_011_020 : CountSegment 1 10 13120 1428 1432 761 764 := CountSegment.join within_011_018 within_011_019
private theorem within_011_020_agreement : RowAgreement 1428 1432 := RowAgreement.join within_011_018_agreement within_011_019_agreement
private def within_011_021 : CountSegment 1 10 13120 1424 1432 759 764 := CountSegment.join within_011_017 within_011_020
private theorem within_011_021_agreement : RowAgreement 1424 1432 := RowAgreement.join within_011_017_agreement within_011_020_agreement
private def within_011_022 : CountSegment 1 10 13120 1432 1434 764 765 := CountSegment.join atomic_1433 atomic_1434
private theorem within_011_022_agreement : RowAgreement 1432 1434 := RowAgreement.join atomic_1433_agreement atomic_1434_agreement
private def within_011_023 : CountSegment 1 10 13120 1434 1436 765 767 := CountSegment.join atomic_1435 atomic_1436
private theorem within_011_023_agreement : RowAgreement 1434 1436 := RowAgreement.join atomic_1435_agreement atomic_1436_agreement
private def within_011_024 : CountSegment 1 10 13120 1432 1436 764 767 := CountSegment.join within_011_022 within_011_023
private theorem within_011_024_agreement : RowAgreement 1432 1436 := RowAgreement.join within_011_022_agreement within_011_023_agreement
private def within_011_025 : CountSegment 1 10 13120 1436 1438 767 769 := CountSegment.join atomic_1437 atomic_1438
private theorem within_011_025_agreement : RowAgreement 1436 1438 := RowAgreement.join atomic_1437_agreement atomic_1438_agreement
private def within_011_026 : CountSegment 1 10 13120 1438 1440 769 769 := CountSegment.join atomic_1439 atomic_1440
private theorem within_011_026_agreement : RowAgreement 1438 1440 := RowAgreement.join atomic_1439_agreement atomic_1440_agreement
private def within_011_027 : CountSegment 1 10 13120 1436 1440 767 769 := CountSegment.join within_011_025 within_011_026
private theorem within_011_027_agreement : RowAgreement 1436 1440 := RowAgreement.join within_011_025_agreement within_011_026_agreement
private def within_011_028 : CountSegment 1 10 13120 1432 1440 764 769 := CountSegment.join within_011_024 within_011_027
private theorem within_011_028_agreement : RowAgreement 1432 1440 := RowAgreement.join within_011_024_agreement within_011_027_agreement
private def within_011_029 : CountSegment 1 10 13120 1424 1440 759 769 := CountSegment.join within_011_021 within_011_028
private theorem within_011_029_agreement : RowAgreement 1424 1440 := RowAgreement.join within_011_021_agreement within_011_028_agreement
private def within_011_030 : CountSegment 1 10 13120 1408 1440 753 769 := CountSegment.join within_011_014 within_011_029
private theorem within_011_030_agreement : RowAgreement 1408 1440 := RowAgreement.join within_011_014_agreement within_011_029_agreement
private def within_011_031 : CountSegment 1 10 13120 1440 1442 769 771 := CountSegment.join atomic_1441 atomic_1442
private theorem within_011_031_agreement : RowAgreement 1440 1442 := RowAgreement.join atomic_1441_agreement atomic_1442_agreement
private def within_011_032 : CountSegment 1 10 13120 1442 1444 771 772 := CountSegment.join atomic_1443 atomic_1444
private theorem within_011_032_agreement : RowAgreement 1442 1444 := RowAgreement.join atomic_1443_agreement atomic_1444_agreement
private def within_011_033 : CountSegment 1 10 13120 1440 1444 769 772 := CountSegment.join within_011_031 within_011_032
private theorem within_011_033_agreement : RowAgreement 1440 1444 := RowAgreement.join within_011_031_agreement within_011_032_agreement
private def within_011_034 : CountSegment 1 10 13120 1444 1446 772 774 := CountSegment.join atomic_1445 atomic_1446
private theorem within_011_034_agreement : RowAgreement 1444 1446 := RowAgreement.join atomic_1445_agreement atomic_1446_agreement
private def within_011_035 : CountSegment 1 10 13120 1446 1448 774 776 := CountSegment.join atomic_1447 atomic_1448
private theorem within_011_035_agreement : RowAgreement 1446 1448 := RowAgreement.join atomic_1447_agreement atomic_1448_agreement
private def within_011_036 : CountSegment 1 10 13120 1444 1448 772 776 := CountSegment.join within_011_034 within_011_035
private theorem within_011_036_agreement : RowAgreement 1444 1448 := RowAgreement.join within_011_034_agreement within_011_035_agreement
private def within_011_037 : CountSegment 1 10 13120 1440 1448 769 776 := CountSegment.join within_011_033 within_011_036
private theorem within_011_037_agreement : RowAgreement 1440 1448 := RowAgreement.join within_011_033_agreement within_011_036_agreement
private def within_011_038 : CountSegment 1 10 13120 1448 1450 776 778 := CountSegment.join atomic_1449 atomic_1450
private theorem within_011_038_agreement : RowAgreement 1448 1450 := RowAgreement.join atomic_1449_agreement atomic_1450_agreement
private def within_011_039 : CountSegment 1 10 13120 1450 1452 778 778 := CountSegment.join atomic_1451 atomic_1452
private theorem within_011_039_agreement : RowAgreement 1450 1452 := RowAgreement.join atomic_1451_agreement atomic_1452_agreement
private def within_011_040 : CountSegment 1 10 13120 1448 1452 776 778 := CountSegment.join within_011_038 within_011_039
private theorem within_011_040_agreement : RowAgreement 1448 1452 := RowAgreement.join within_011_038_agreement within_011_039_agreement
private def within_011_041 : CountSegment 1 10 13120 1452 1454 778 779 := CountSegment.join atomic_1453 atomic_1454
private theorem within_011_041_agreement : RowAgreement 1452 1454 := RowAgreement.join atomic_1453_agreement atomic_1454_agreement
private def within_011_042 : CountSegment 1 10 13120 1454 1456 779 779 := CountSegment.join atomic_1455 atomic_1456
private theorem within_011_042_agreement : RowAgreement 1454 1456 := RowAgreement.join atomic_1455_agreement atomic_1456_agreement
private def within_011_043 : CountSegment 1 10 13120 1452 1456 778 779 := CountSegment.join within_011_041 within_011_042
private theorem within_011_043_agreement : RowAgreement 1452 1456 := RowAgreement.join within_011_041_agreement within_011_042_agreement
private def within_011_044 : CountSegment 1 10 13120 1448 1456 776 779 := CountSegment.join within_011_040 within_011_043
private theorem within_011_044_agreement : RowAgreement 1448 1456 := RowAgreement.join within_011_040_agreement within_011_043_agreement
private def within_011_045 : CountSegment 1 10 13120 1440 1456 769 779 := CountSegment.join within_011_037 within_011_044
private theorem within_011_045_agreement : RowAgreement 1440 1456 := RowAgreement.join within_011_037_agreement within_011_044_agreement
private def within_011_046 : CountSegment 1 10 13120 1456 1458 779 780 := CountSegment.join atomic_1457 atomic_1458
private theorem within_011_046_agreement : RowAgreement 1456 1458 := RowAgreement.join atomic_1457_agreement atomic_1458_agreement
private def within_011_047 : CountSegment 1 10 13120 1458 1460 780 780 := CountSegment.join atomic_1459 atomic_1460
private theorem within_011_047_agreement : RowAgreement 1458 1460 := RowAgreement.join atomic_1459_agreement atomic_1460_agreement
private def within_011_048 : CountSegment 1 10 13120 1456 1460 779 780 := CountSegment.join within_011_046 within_011_047
private theorem within_011_048_agreement : RowAgreement 1456 1460 := RowAgreement.join within_011_046_agreement within_011_047_agreement
private def within_011_049 : CountSegment 1 10 13120 1460 1462 780 782 := CountSegment.join atomic_1461 atomic_1462
private theorem within_011_049_agreement : RowAgreement 1460 1462 := RowAgreement.join atomic_1461_agreement atomic_1462_agreement
private def within_011_050 : CountSegment 1 10 13120 1462 1464 782 784 := CountSegment.join atomic_1463 atomic_1464
private theorem within_011_050_agreement : RowAgreement 1462 1464 := RowAgreement.join atomic_1463_agreement atomic_1464_agreement
private def within_011_051 : CountSegment 1 10 13120 1460 1464 780 784 := CountSegment.join within_011_049 within_011_050
private theorem within_011_051_agreement : RowAgreement 1460 1464 := RowAgreement.join within_011_049_agreement within_011_050_agreement
private def within_011_052 : CountSegment 1 10 13120 1456 1464 779 784 := CountSegment.join within_011_048 within_011_051
private theorem within_011_052_agreement : RowAgreement 1456 1464 := RowAgreement.join within_011_048_agreement within_011_051_agreement
private def within_011_053 : CountSegment 1 10 13120 1464 1466 784 784 := CountSegment.join atomic_1465 atomic_1466
private theorem within_011_053_agreement : RowAgreement 1464 1466 := RowAgreement.join atomic_1465_agreement atomic_1466_agreement
private def within_011_054 : CountSegment 1 10 13120 1466 1468 784 786 := CountSegment.join atomic_1467 atomic_1468
private theorem within_011_054_agreement : RowAgreement 1466 1468 := RowAgreement.join atomic_1467_agreement atomic_1468_agreement
private def within_011_055 : CountSegment 1 10 13120 1464 1468 784 786 := CountSegment.join within_011_053 within_011_054
private theorem within_011_055_agreement : RowAgreement 1464 1468 := RowAgreement.join within_011_053_agreement within_011_054_agreement
private def within_011_056 : CountSegment 1 10 13120 1468 1470 786 786 := CountSegment.join atomic_1469 atomic_1470
private theorem within_011_056_agreement : RowAgreement 1468 1470 := RowAgreement.join atomic_1469_agreement atomic_1470_agreement
private def within_011_057 : CountSegment 1 10 13120 1470 1472 786 788 := CountSegment.join atomic_1471 atomic_1472
private theorem within_011_057_agreement : RowAgreement 1470 1472 := RowAgreement.join atomic_1471_agreement atomic_1472_agreement
private def within_011_058 : CountSegment 1 10 13120 1468 1472 786 788 := CountSegment.join within_011_056 within_011_057
private theorem within_011_058_agreement : RowAgreement 1468 1472 := RowAgreement.join within_011_056_agreement within_011_057_agreement
private def within_011_059 : CountSegment 1 10 13120 1464 1472 784 788 := CountSegment.join within_011_055 within_011_058
private theorem within_011_059_agreement : RowAgreement 1464 1472 := RowAgreement.join within_011_055_agreement within_011_058_agreement
private def within_011_060 : CountSegment 1 10 13120 1456 1472 779 788 := CountSegment.join within_011_052 within_011_059
private theorem within_011_060_agreement : RowAgreement 1456 1472 := RowAgreement.join within_011_052_agreement within_011_059_agreement
private def within_011_061 : CountSegment 1 10 13120 1440 1472 769 788 := CountSegment.join within_011_045 within_011_060
private theorem within_011_061_agreement : RowAgreement 1440 1472 := RowAgreement.join within_011_045_agreement within_011_060_agreement
private def within_011_062 : CountSegment 1 10 13120 1408 1472 753 788 := CountSegment.join within_011_030 within_011_061
private theorem within_011_062_agreement : RowAgreement 1408 1472 := RowAgreement.join within_011_030_agreement within_011_061_agreement
private def within_011_063 : CountSegment 1 10 13120 1472 1474 788 789 := CountSegment.join atomic_1473 atomic_1474
private theorem within_011_063_agreement : RowAgreement 1472 1474 := RowAgreement.join atomic_1473_agreement atomic_1474_agreement
private def within_011_064 : CountSegment 1 10 13120 1474 1476 789 789 := CountSegment.join atomic_1475 atomic_1476
private theorem within_011_064_agreement : RowAgreement 1474 1476 := RowAgreement.join atomic_1475_agreement atomic_1476_agreement
private def within_011_065 : CountSegment 1 10 13120 1472 1476 788 789 := CountSegment.join within_011_063 within_011_064
private theorem within_011_065_agreement : RowAgreement 1472 1476 := RowAgreement.join within_011_063_agreement within_011_064_agreement
private def within_011_066 : CountSegment 1 10 13120 1476 1478 789 790 := CountSegment.join atomic_1477 atomic_1478
private theorem within_011_066_agreement : RowAgreement 1476 1478 := RowAgreement.join atomic_1477_agreement atomic_1478_agreement
private def within_011_067 : CountSegment 1 10 13120 1478 1480 790 792 := CountSegment.join atomic_1479 atomic_1480
private theorem within_011_067_agreement : RowAgreement 1478 1480 := RowAgreement.join atomic_1479_agreement atomic_1480_agreement
private def within_011_068 : CountSegment 1 10 13120 1476 1480 789 792 := CountSegment.join within_011_066 within_011_067
private theorem within_011_068_agreement : RowAgreement 1476 1480 := RowAgreement.join within_011_066_agreement within_011_067_agreement
private def within_011_069 : CountSegment 1 10 13120 1472 1480 788 792 := CountSegment.join within_011_065 within_011_068
private theorem within_011_069_agreement : RowAgreement 1472 1480 := RowAgreement.join within_011_065_agreement within_011_068_agreement
private def within_011_070 : CountSegment 1 10 13120 1480 1482 792 793 := CountSegment.join atomic_1481 atomic_1482
private theorem within_011_070_agreement : RowAgreement 1480 1482 := RowAgreement.join atomic_1481_agreement atomic_1482_agreement
private def within_011_071 : CountSegment 1 10 13120 1482 1484 793 794 := CountSegment.join atomic_1483 atomic_1484
private theorem within_011_071_agreement : RowAgreement 1482 1484 := RowAgreement.join atomic_1483_agreement atomic_1484_agreement
private def within_011_072 : CountSegment 1 10 13120 1480 1484 792 794 := CountSegment.join within_011_070 within_011_071
private theorem within_011_072_agreement : RowAgreement 1480 1484 := RowAgreement.join within_011_070_agreement within_011_071_agreement
private def within_011_073 : CountSegment 1 10 13120 1484 1486 794 795 := CountSegment.join atomic_1485 atomic_1486
private theorem within_011_073_agreement : RowAgreement 1484 1486 := RowAgreement.join atomic_1485_agreement atomic_1486_agreement
private def within_011_074 : CountSegment 1 10 13120 1486 1488 795 796 := CountSegment.join atomic_1487 atomic_1488
private theorem within_011_074_agreement : RowAgreement 1486 1488 := RowAgreement.join atomic_1487_agreement atomic_1488_agreement
private def within_011_075 : CountSegment 1 10 13120 1484 1488 794 796 := CountSegment.join within_011_073 within_011_074
private theorem within_011_075_agreement : RowAgreement 1484 1488 := RowAgreement.join within_011_073_agreement within_011_074_agreement
private def within_011_076 : CountSegment 1 10 13120 1480 1488 792 796 := CountSegment.join within_011_072 within_011_075
private theorem within_011_076_agreement : RowAgreement 1480 1488 := RowAgreement.join within_011_072_agreement within_011_075_agreement
private def within_011_077 : CountSegment 1 10 13120 1472 1488 788 796 := CountSegment.join within_011_069 within_011_076
private theorem within_011_077_agreement : RowAgreement 1472 1488 := RowAgreement.join within_011_069_agreement within_011_076_agreement
private def within_011_078 : CountSegment 1 10 13120 1488 1490 796 796 := CountSegment.join atomic_1489 atomic_1490
private theorem within_011_078_agreement : RowAgreement 1488 1490 := RowAgreement.join atomic_1489_agreement atomic_1490_agreement
private def within_011_079 : CountSegment 1 10 13120 1490 1492 796 797 := CountSegment.join atomic_1491 atomic_1492
private theorem within_011_079_agreement : RowAgreement 1490 1492 := RowAgreement.join atomic_1491_agreement atomic_1492_agreement
private def within_011_080 : CountSegment 1 10 13120 1488 1492 796 797 := CountSegment.join within_011_078 within_011_079
private theorem within_011_080_agreement : RowAgreement 1488 1492 := RowAgreement.join within_011_078_agreement within_011_079_agreement
private def within_011_081 : CountSegment 1 10 13120 1492 1494 797 798 := CountSegment.join atomic_1493 atomic_1494
private theorem within_011_081_agreement : RowAgreement 1492 1494 := RowAgreement.join atomic_1493_agreement atomic_1494_agreement
private def within_011_082 : CountSegment 1 10 13120 1494 1496 798 798 := CountSegment.join atomic_1495 atomic_1496
private theorem within_011_082_agreement : RowAgreement 1494 1496 := RowAgreement.join atomic_1495_agreement atomic_1496_agreement
private def within_011_083 : CountSegment 1 10 13120 1492 1496 797 798 := CountSegment.join within_011_081 within_011_082
private theorem within_011_083_agreement : RowAgreement 1492 1496 := RowAgreement.join within_011_081_agreement within_011_082_agreement
private def within_011_084 : CountSegment 1 10 13120 1488 1496 796 798 := CountSegment.join within_011_080 within_011_083
private theorem within_011_084_agreement : RowAgreement 1488 1496 := RowAgreement.join within_011_080_agreement within_011_083_agreement
private def within_011_085 : CountSegment 1 10 13120 1496 1498 798 799 := CountSegment.join atomic_1497 atomic_1498
private theorem within_011_085_agreement : RowAgreement 1496 1498 := RowAgreement.join atomic_1497_agreement atomic_1498_agreement
private def within_011_086 : CountSegment 1 10 13120 1498 1500 799 800 := CountSegment.join atomic_1499 atomic_1500
private theorem within_011_086_agreement : RowAgreement 1498 1500 := RowAgreement.join atomic_1499_agreement atomic_1500_agreement
private def within_011_087 : CountSegment 1 10 13120 1496 1500 798 800 := CountSegment.join within_011_085 within_011_086
private theorem within_011_087_agreement : RowAgreement 1496 1500 := RowAgreement.join within_011_085_agreement within_011_086_agreement
private def within_011_088 : CountSegment 1 10 13120 1500 1502 800 800 := CountSegment.join atomic_1501 atomic_1502
private theorem within_011_088_agreement : RowAgreement 1500 1502 := RowAgreement.join atomic_1501_agreement atomic_1502_agreement
private def within_011_089 : CountSegment 1 10 13120 1502 1504 800 800 := CountSegment.join atomic_1503 atomic_1504
private theorem within_011_089_agreement : RowAgreement 1502 1504 := RowAgreement.join atomic_1503_agreement atomic_1504_agreement
private def within_011_090 : CountSegment 1 10 13120 1500 1504 800 800 := CountSegment.join within_011_088 within_011_089
private theorem within_011_090_agreement : RowAgreement 1500 1504 := RowAgreement.join within_011_088_agreement within_011_089_agreement
private def within_011_091 : CountSegment 1 10 13120 1496 1504 798 800 := CountSegment.join within_011_087 within_011_090
private theorem within_011_091_agreement : RowAgreement 1496 1504 := RowAgreement.join within_011_087_agreement within_011_090_agreement
private def within_011_092 : CountSegment 1 10 13120 1488 1504 796 800 := CountSegment.join within_011_084 within_011_091
private theorem within_011_092_agreement : RowAgreement 1488 1504 := RowAgreement.join within_011_084_agreement within_011_091_agreement
private def within_011_093 : CountSegment 1 10 13120 1472 1504 788 800 := CountSegment.join within_011_077 within_011_092
private theorem within_011_093_agreement : RowAgreement 1472 1504 := RowAgreement.join within_011_077_agreement within_011_092_agreement
private def within_011_094 : CountSegment 1 10 13120 1504 1506 800 801 := CountSegment.join atomic_1505 atomic_1506
private theorem within_011_094_agreement : RowAgreement 1504 1506 := RowAgreement.join atomic_1505_agreement atomic_1506_agreement
private def within_011_095 : CountSegment 1 10 13120 1506 1508 801 802 := CountSegment.join atomic_1507 atomic_1508
private theorem within_011_095_agreement : RowAgreement 1506 1508 := RowAgreement.join atomic_1507_agreement atomic_1508_agreement
private def within_011_096 : CountSegment 1 10 13120 1504 1508 800 802 := CountSegment.join within_011_094 within_011_095
private theorem within_011_096_agreement : RowAgreement 1504 1508 := RowAgreement.join within_011_094_agreement within_011_095_agreement
private def within_011_097 : CountSegment 1 10 13120 1508 1510 802 803 := CountSegment.join atomic_1509 atomic_1510
private theorem within_011_097_agreement : RowAgreement 1508 1510 := RowAgreement.join atomic_1509_agreement atomic_1510_agreement
private def within_011_098 : CountSegment 1 10 13120 1510 1512 803 804 := CountSegment.join atomic_1511 atomic_1512
private theorem within_011_098_agreement : RowAgreement 1510 1512 := RowAgreement.join atomic_1511_agreement atomic_1512_agreement
private def within_011_099 : CountSegment 1 10 13120 1508 1512 802 804 := CountSegment.join within_011_097 within_011_098
private theorem within_011_099_agreement : RowAgreement 1508 1512 := RowAgreement.join within_011_097_agreement within_011_098_agreement
private def within_011_100 : CountSegment 1 10 13120 1504 1512 800 804 := CountSegment.join within_011_096 within_011_099
private theorem within_011_100_agreement : RowAgreement 1504 1512 := RowAgreement.join within_011_096_agreement within_011_099_agreement
private def within_011_101 : CountSegment 1 10 13120 1512 1514 804 806 := CountSegment.join atomic_1513 atomic_1514
private theorem within_011_101_agreement : RowAgreement 1512 1514 := RowAgreement.join atomic_1513_agreement atomic_1514_agreement
private def within_011_102 : CountSegment 1 10 13120 1514 1516 806 807 := CountSegment.join atomic_1515 atomic_1516
private theorem within_011_102_agreement : RowAgreement 1514 1516 := RowAgreement.join atomic_1515_agreement atomic_1516_agreement
private def within_011_103 : CountSegment 1 10 13120 1512 1516 804 807 := CountSegment.join within_011_101 within_011_102
private theorem within_011_103_agreement : RowAgreement 1512 1516 := RowAgreement.join within_011_101_agreement within_011_102_agreement
private def within_011_104 : CountSegment 1 10 13120 1516 1518 807 807 := CountSegment.join atomic_1517 atomic_1518
private theorem within_011_104_agreement : RowAgreement 1516 1518 := RowAgreement.join atomic_1517_agreement atomic_1518_agreement
private def within_011_105 : CountSegment 1 10 13120 1518 1520 807 808 := CountSegment.join atomic_1519 atomic_1520
private theorem within_011_105_agreement : RowAgreement 1518 1520 := RowAgreement.join atomic_1519_agreement atomic_1520_agreement
private def within_011_106 : CountSegment 1 10 13120 1516 1520 807 808 := CountSegment.join within_011_104 within_011_105
private theorem within_011_106_agreement : RowAgreement 1516 1520 := RowAgreement.join within_011_104_agreement within_011_105_agreement
private def within_011_107 : CountSegment 1 10 13120 1512 1520 804 808 := CountSegment.join within_011_103 within_011_106
private theorem within_011_107_agreement : RowAgreement 1512 1520 := RowAgreement.join within_011_103_agreement within_011_106_agreement
private def within_011_108 : CountSegment 1 10 13120 1504 1520 800 808 := CountSegment.join within_011_100 within_011_107
private theorem within_011_108_agreement : RowAgreement 1504 1520 := RowAgreement.join within_011_100_agreement within_011_107_agreement
private def within_011_109 : CountSegment 1 10 13120 1520 1522 808 809 := CountSegment.join atomic_1521 atomic_1522
private theorem within_011_109_agreement : RowAgreement 1520 1522 := RowAgreement.join atomic_1521_agreement atomic_1522_agreement
private def within_011_110 : CountSegment 1 10 13120 1522 1524 809 811 := CountSegment.join atomic_1523 atomic_1524
private theorem within_011_110_agreement : RowAgreement 1522 1524 := RowAgreement.join atomic_1523_agreement atomic_1524_agreement
private def within_011_111 : CountSegment 1 10 13120 1520 1524 808 811 := CountSegment.join within_011_109 within_011_110
private theorem within_011_111_agreement : RowAgreement 1520 1524 := RowAgreement.join within_011_109_agreement within_011_110_agreement
private def within_011_112 : CountSegment 1 10 13120 1524 1526 811 812 := CountSegment.join atomic_1525 atomic_1526
private theorem within_011_112_agreement : RowAgreement 1524 1526 := RowAgreement.join atomic_1525_agreement atomic_1526_agreement
private def within_011_113 : CountSegment 1 10 13120 1526 1528 812 814 := CountSegment.join atomic_1527 atomic_1528
private theorem within_011_113_agreement : RowAgreement 1526 1528 := RowAgreement.join atomic_1527_agreement atomic_1528_agreement
private def within_011_114 : CountSegment 1 10 13120 1524 1528 811 814 := CountSegment.join within_011_112 within_011_113
private theorem within_011_114_agreement : RowAgreement 1524 1528 := RowAgreement.join within_011_112_agreement within_011_113_agreement
private def within_011_115 : CountSegment 1 10 13120 1520 1528 808 814 := CountSegment.join within_011_111 within_011_114
private theorem within_011_115_agreement : RowAgreement 1520 1528 := RowAgreement.join within_011_111_agreement within_011_114_agreement
private def within_011_116 : CountSegment 1 10 13120 1528 1530 814 814 := CountSegment.join atomic_1529 atomic_1530
private theorem within_011_116_agreement : RowAgreement 1528 1530 := RowAgreement.join atomic_1529_agreement atomic_1530_agreement
private def within_011_117 : CountSegment 1 10 13120 1530 1532 814 814 := CountSegment.join atomic_1531 atomic_1532
private theorem within_011_117_agreement : RowAgreement 1530 1532 := RowAgreement.join atomic_1531_agreement atomic_1532_agreement
private def within_011_118 : CountSegment 1 10 13120 1528 1532 814 814 := CountSegment.join within_011_116 within_011_117
private theorem within_011_118_agreement : RowAgreement 1528 1532 := RowAgreement.join within_011_116_agreement within_011_117_agreement
private def within_011_119 : CountSegment 1 10 13120 1532 1534 814 814 := CountSegment.join atomic_1533 atomic_1534
private theorem within_011_119_agreement : RowAgreement 1532 1534 := RowAgreement.join atomic_1533_agreement atomic_1534_agreement
private def within_011_120 : CountSegment 1 10 13120 1534 1536 814 815 := CountSegment.join atomic_1535 atomic_1536
private theorem within_011_120_agreement : RowAgreement 1534 1536 := RowAgreement.join atomic_1535_agreement atomic_1536_agreement
private def within_011_121 : CountSegment 1 10 13120 1532 1536 814 815 := CountSegment.join within_011_119 within_011_120
private theorem within_011_121_agreement : RowAgreement 1532 1536 := RowAgreement.join within_011_119_agreement within_011_120_agreement
private def within_011_122 : CountSegment 1 10 13120 1528 1536 814 815 := CountSegment.join within_011_118 within_011_121
private theorem within_011_122_agreement : RowAgreement 1528 1536 := RowAgreement.join within_011_118_agreement within_011_121_agreement
private def within_011_123 : CountSegment 1 10 13120 1520 1536 808 815 := CountSegment.join within_011_115 within_011_122
private theorem within_011_123_agreement : RowAgreement 1520 1536 := RowAgreement.join within_011_115_agreement within_011_122_agreement
private def within_011_124 : CountSegment 1 10 13120 1504 1536 800 815 := CountSegment.join within_011_108 within_011_123
private theorem within_011_124_agreement : RowAgreement 1504 1536 := RowAgreement.join within_011_108_agreement within_011_123_agreement
private def within_011_125 : CountSegment 1 10 13120 1472 1536 788 815 := CountSegment.join within_011_093 within_011_124
private theorem within_011_125_agreement : RowAgreement 1472 1536 := RowAgreement.join within_011_093_agreement within_011_124_agreement
private def within_011_126 : CountSegment 1 10 13120 1408 1536 753 815 := CountSegment.join within_011_062 within_011_125
private theorem within_011_126_agreement : RowAgreement 1408 1536 := RowAgreement.join within_011_062_agreement within_011_125_agreement
def segment_011 : CountSegment 1 10 13120 1408 1536 753 815 := within_011_126
theorem agreement_011 : RowAgreement 1408 1536 := within_011_126_agreement
end ProjectEulerSolutions.P58Numerical
