import ProjectEulerSolutions.Lib.PrimePairCertificate

/-! Generated bounded vertex intervals; every list is kernel checked. -/
namespace ProjectEulerSolutions.PrimePairVertices
open PrimePairCertificate

def interval_2 : List Nat := [3, 7, 11, 13, 17, 19, 23, 29, 31, 37, 41, 43, 47, 53, 59, 61, 67, 71, 73, 79, 83, 89, 97, 101, 103, 107, 109, 113, 127]
theorem interval_2_checked : primeSegment 2 128 = interval_2 := by
  rw [← computedSegment_eq_primeSegment 2 128 (by decide)]
  decide

def interval_130 : List Nat := [131, 137, 139, 149, 151, 157, 163, 167, 173, 179, 181, 191, 193, 197, 199, 211, 223, 227, 229, 233, 239, 241, 251, 257]
theorem interval_130_checked : primeSegment 130 128 = interval_130 := by
  rw [← computedSegment_eq_primeSegment 130 128 (by decide)]
  decide

def interval_258 : List Nat := [263, 269, 271, 277, 281, 283, 293, 307, 311, 313, 317, 331, 337, 347, 349, 353, 359, 367, 373, 379, 383]
theorem interval_258_checked : primeSegment 258 128 = interval_258 := by
  rw [← computedSegment_eq_primeSegment 258 128 (by decide)]
  decide

def interval_386 : List Nat := [389, 397, 401, 409, 419, 421, 431, 433, 439, 443, 449, 457, 461, 463, 467, 479, 487, 491, 499, 503, 509]
theorem interval_386_checked : primeSegment 386 128 = interval_386 := by
  rw [← computedSegment_eq_primeSegment 386 128 (by decide)]
  decide

def interval_514 : List Nat := [521, 523, 541, 547, 557, 563, 569, 571, 577, 587, 593, 599, 601, 607, 613, 617, 619, 631, 641]
theorem interval_514_checked : primeSegment 514 128 = interval_514 := by
  rw [← computedSegment_eq_primeSegment 514 128 (by decide)]
  decide

def interval_642 : List Nat := [643, 647, 653, 659, 661, 673, 677, 683, 691, 701, 709, 719, 727, 733, 739, 743, 751, 757, 761, 769]
theorem interval_642_checked : primeSegment 642 128 = interval_642 := by
  rw [← computedSegment_eq_primeSegment 642 128 (by decide)]
  decide

def interval_770 : List Nat := [773, 787, 797, 809, 811, 821, 823, 827, 829, 839, 853, 857, 859, 863, 877, 881, 883, 887]
theorem interval_770_checked : primeSegment 770 128 = interval_770 := by
  rw [← computedSegment_eq_primeSegment 770 128 (by decide)]
  decide

def interval_898 : List Nat := [907, 911, 919, 929, 937, 941, 947, 953, 967, 971, 977, 983, 991, 997, 1009, 1013, 1019, 1021]
theorem interval_898_checked : primeSegment 898 128 = interval_898 := by
  rw [← computedSegment_eq_primeSegment 898 128 (by decide)]
  decide

def interval_1026 : List Nat := [1031, 1033, 1039, 1049, 1051, 1061, 1063, 1069, 1087, 1091, 1093, 1097, 1103, 1109, 1117, 1123, 1129, 1151, 1153]
theorem interval_1026_checked : primeSegment 1026 128 = interval_1026 := by
  rw [← computedSegment_eq_primeSegment 1026 128 (by decide)]
  decide

def interval_1154 : List Nat := [1163, 1171, 1181, 1187, 1193, 1201, 1213, 1217, 1223, 1229, 1231, 1237, 1249, 1259, 1277, 1279]
theorem interval_1154_checked : primeSegment 1154 128 = interval_1154 := by
  rw [← computedSegment_eq_primeSegment 1154 128 (by decide)]
  decide

def interval_1282 : List Nat := [1283, 1289, 1291, 1297, 1301, 1303, 1307, 1319, 1321, 1327, 1361, 1367, 1373, 1381, 1399, 1409]
theorem interval_1282_checked : primeSegment 1282 128 = interval_1282 := by
  rw [← computedSegment_eq_primeSegment 1282 128 (by decide)]
  decide

def interval_1410 : List Nat := [1423, 1427, 1429, 1433, 1439, 1447, 1451, 1453, 1459, 1471, 1481, 1483, 1487, 1489, 1493, 1499, 1511, 1523, 1531]
theorem interval_1410_checked : primeSegment 1410 128 = interval_1410 := by
  rw [← computedSegment_eq_primeSegment 1410 128 (by decide)]
  decide

def interval_1538 : List Nat := [1543, 1549, 1553, 1559, 1567, 1571, 1579, 1583, 1597, 1601, 1607, 1609, 1613, 1619, 1621, 1627, 1637, 1657, 1663]
theorem interval_1538_checked : primeSegment 1538 128 = interval_1538 := by
  rw [← computedSegment_eq_primeSegment 1538 128 (by decide)]
  decide

def interval_1666 : List Nat := [1667, 1669, 1693, 1697, 1699, 1709, 1721, 1723, 1733, 1741, 1747, 1753, 1759, 1777, 1783, 1787, 1789]
theorem interval_1666_checked : primeSegment 1666 128 = interval_1666 := by
  rw [← computedSegment_eq_primeSegment 1666 128 (by decide)]
  decide

def interval_1794 : List Nat := [1801, 1811, 1823, 1831, 1847, 1861, 1867, 1871, 1873, 1877, 1879, 1889, 1901, 1907, 1913]
theorem interval_1794_checked : primeSegment 1794 128 = interval_1794 := by
  rw [← computedSegment_eq_primeSegment 1794 128 (by decide)]
  decide

def interval_1922 : List Nat := [1931, 1933, 1949, 1951, 1973, 1979, 1987, 1993, 1997, 1999, 2003, 2011, 2017, 2027, 2029, 2039]
theorem interval_1922_checked : primeSegment 1922 128 = interval_1922 := by
  rw [← computedSegment_eq_primeSegment 1922 128 (by decide)]
  decide

def block_000_left_left_left : List Nat := interval_2 ++ interval_130
theorem block_000_left_left_left_checked : primeSegment 2 256 = block_000_left_left_left := by
  change primeSegment 2 (128 + 128) = _
  rw [primeSegment_append, interval_2_checked, interval_130_checked]
  rfl

def block_000_left_left_right : List Nat := interval_258 ++ interval_386
theorem block_000_left_left_right_checked : primeSegment 258 256 = block_000_left_left_right := by
  change primeSegment 258 (128 + 128) = _
  rw [primeSegment_append, interval_258_checked, interval_386_checked]
  rfl

def block_000_left_left : List Nat := block_000_left_left_left ++ block_000_left_left_right
theorem block_000_left_left_checked : primeSegment 2 512 = block_000_left_left := by
  change primeSegment 2 (256 + 256) = _
  rw [primeSegment_append, block_000_left_left_left_checked, block_000_left_left_right_checked]
  rfl

def block_000_left_right_left : List Nat := interval_514 ++ interval_642
theorem block_000_left_right_left_checked : primeSegment 514 256 = block_000_left_right_left := by
  change primeSegment 514 (128 + 128) = _
  rw [primeSegment_append, interval_514_checked, interval_642_checked]
  rfl

def block_000_left_right_right : List Nat := interval_770 ++ interval_898
theorem block_000_left_right_right_checked : primeSegment 770 256 = block_000_left_right_right := by
  change primeSegment 770 (128 + 128) = _
  rw [primeSegment_append, interval_770_checked, interval_898_checked]
  rfl

def block_000_left_right : List Nat := block_000_left_right_left ++ block_000_left_right_right
theorem block_000_left_right_checked : primeSegment 514 512 = block_000_left_right := by
  change primeSegment 514 (256 + 256) = _
  rw [primeSegment_append, block_000_left_right_left_checked, block_000_left_right_right_checked]
  rfl

def block_000_left : List Nat := block_000_left_left ++ block_000_left_right
theorem block_000_left_checked : primeSegment 2 1024 = block_000_left := by
  change primeSegment 2 (512 + 512) = _
  rw [primeSegment_append, block_000_left_left_checked, block_000_left_right_checked]
  rfl

def block_000_right_left_left : List Nat := interval_1026 ++ interval_1154
theorem block_000_right_left_left_checked : primeSegment 1026 256 = block_000_right_left_left := by
  change primeSegment 1026 (128 + 128) = _
  rw [primeSegment_append, interval_1026_checked, interval_1154_checked]
  rfl

def block_000_right_left_right : List Nat := interval_1282 ++ interval_1410
theorem block_000_right_left_right_checked : primeSegment 1282 256 = block_000_right_left_right := by
  change primeSegment 1282 (128 + 128) = _
  rw [primeSegment_append, interval_1282_checked, interval_1410_checked]
  rfl

def block_000_right_left : List Nat := block_000_right_left_left ++ block_000_right_left_right
theorem block_000_right_left_checked : primeSegment 1026 512 = block_000_right_left := by
  change primeSegment 1026 (256 + 256) = _
  rw [primeSegment_append, block_000_right_left_left_checked, block_000_right_left_right_checked]
  rfl

def block_000_right_right_left : List Nat := interval_1538 ++ interval_1666
theorem block_000_right_right_left_checked : primeSegment 1538 256 = block_000_right_right_left := by
  change primeSegment 1538 (128 + 128) = _
  rw [primeSegment_append, interval_1538_checked, interval_1666_checked]
  rfl

def block_000_right_right_right : List Nat := interval_1794 ++ interval_1922
theorem block_000_right_right_right_checked : primeSegment 1794 256 = block_000_right_right_right := by
  change primeSegment 1794 (128 + 128) = _
  rw [primeSegment_append, interval_1794_checked, interval_1922_checked]
  rfl

def block_000_right_right : List Nat := block_000_right_right_left ++ block_000_right_right_right
theorem block_000_right_right_checked : primeSegment 1538 512 = block_000_right_right := by
  change primeSegment 1538 (256 + 256) = _
  rw [primeSegment_append, block_000_right_right_left_checked, block_000_right_right_right_checked]
  rfl

def block_000_right : List Nat := block_000_right_left ++ block_000_right_right
theorem block_000_right_checked : primeSegment 1026 1024 = block_000_right := by
  change primeSegment 1026 (512 + 512) = _
  rw [primeSegment_append, block_000_right_left_checked, block_000_right_right_checked]
  rfl

def block_000 : List Nat := block_000_left ++ block_000_right
theorem block_000_checked : primeSegment 2 2048 = block_000 := by
  change primeSegment 2 (1024 + 1024) = _
  rw [primeSegment_append, block_000_left_checked, block_000_right_checked]
  rfl

end ProjectEulerSolutions.PrimePairVertices
