import ProjectEulerSolutions.Lib.PrimePairLarge.Bounds

namespace ProjectEulerSolutions.PrimePairLarge
open PrimePairCertificate

def row11831_small : List Nat := []
theorem row11831_small_covers : Budget.RowUpperBound PrimePairVertices.tree_small 26034 11831 row11831_small :=
  Budget.empty_row_before _ _ _ 8389 (fun q member => (Bounds.small q member).2) (by decide)

def row11831_block_005 : List Nat := []
theorem row11831_block_005_covers : Budget.RowUpperBound PrimePairVertices.tree_block_005 26034 11831 row11831_block_005 :=
  Budget.empty_row_before _ _ _ 10370 (fun q member => (Bounds.block_005 q member).2) (by decide)

def row11831_block_006 : List Nat := []
def plan11831_block_006 : List Budget.Choice := [.outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .first, .first, .first, (.firstFactor 7), .first, (.firstFactor 11), (.secondFactor 17), (.firstFactor 7), .first, (.firstFactor 23), .first, (.secondFactor 7), .first, .first, .first, (.firstFactor 13), .first, (.secondFactor 7), .first, .first, .second, .first, (.secondFactor 13), .first, .first, .second, .first, .first, (.firstFactor 29), (.secondFactor 7), .first, .first, (.firstFactor 7), (.secondFactor 19), (.secondFactor 7), .first, (.firstFactor 7), .first, (.firstFactor 19), (.firstFactor 7), .first, (.firstFactor 11), .second, .first, .second, .first, (.firstFactor 17), (.secondFactor 31), .first, (.firstFactor 13), .first, .first, .first, (.firstFactor 7), .first, .first, .first, .second, .first, .first, (.secondFactor 7), .first, (.firstFactor 7)]
set_option maxRecDepth 2048 in
theorem row11831_block_006_covers : Budget.RowUpperBound PrimePairVertices.tree_block_006 26034 11831 row11831_block_006 := by
  apply Budget.row_upper_of_plan _ _ _ plan11831_block_006
  · decide
  · rfl

def row11831_block_007 : List Nat := [13649]
def plan11831_block_007 : List Budget.Choice := [.first, .first, (.firstFactor 13), .first, .first, .first, (.firstFactor 23), .first, (.firstFactor 11), (.firstFactor 7), .first, .first, .first, (.secondFactor 7), (.firstFactor 7), .first, .first, .first, (.firstFactor 17), .first, .first, .first, .first, (.secondFactor 7), .first, .first, .first, (.secondFactor 31), .second, (.firstFactor 19), .first, (.firstFactor 13), (.firstFactor 11), .first, .first, (.secondFactor 19), .first, .first, (.secondFactor 13), .first, .first, .first, (.firstFactor 7), .first, .first, (.firstFactor 11), .first, .first, .first, .first, .first, (.firstFactor 23), (.secondFactor 13), .first, (.secondFactor 23), (.firstFactor 7), .first, (.firstFactor 31), (.secondFactor 19), (.firstFactor 11), (.firstFactor 7), .first, .first, .first, (.firstFactor 13), (.firstFactor 7), .first, .first, .first, .first, .first, (.firstFactor 7), (.secondFactor 23), .first, .first, .first, .first, (.firstFactor 19), .first, (.firstFactor 7), .first, (.firstFactor 11), .first, .second, .first, .first, .first, (.secondFactor 23), (.firstFactor 11), .first, (.secondFactor 17), (.secondFactor 7), .first, .first, .first, .first, .first, .first, .first, .first, (.secondFactor 17), (.firstFactor 7), .first, (.secondFactor 7), .first, (.secondFactor 19), .first, .first, .first, (.firstFactor 7), .first, (.firstFactor 13), .first, (.firstFactor 7), .first, .first, (.firstFactor 17), .second, .first, (.secondFactor 13), .first, .first, .first, (.secondFactor 7), .first, .first, (.firstFactor 11), (.secondFactor 7), .first, .first, .possible, .first, (.firstFactor 11), .first, .first, (.firstFactor 17), .first, .first, .second, .first, (.firstFactor 23), .first, .first, .first, (.firstFactor 7), .first, (.firstFactor 13), .first, .first, (.firstFactor 7), .first, (.secondFactor 7), .first, (.firstFactor 7), (.firstFactor 23), .first, (.firstFactor 11), .first, (.firstFactor 7), .first, .first, (.firstFactor 19), (.secondFactor 7), .first, (.secondFactor 29), .first, .first, (.firstFactor 7), (.firstFactor 13), .first, (.firstFactor 7), .first, .first, .first, (.firstFactor 7), .first, .first, (.secondFactor 7), .first, .first, .first, .first, .first, (.firstFactor 13), .first, .first, (.firstFactor 7), .first, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside]
set_option maxRecDepth 2048 in
theorem row11831_block_007_covers : Budget.RowUpperBound PrimePairVertices.tree_block_007 26034 11831 row11831_block_007 := by
  apply Budget.row_upper_of_plan _ _ _ plan11831_block_007
  · decide
  · rfl

def row11831_block_008 : List Nat := []
theorem row11831_block_008_covers : Budget.RowUpperBound PrimePairVertices.tree_block_008 26034 11831 row11831_block_008 :=
  Budget.empty_row_after _ _ _ 14466 (fun q member => (Bounds.block_008 q member).1) (by decide)

def row11831_block_009 : List Nat := []
theorem row11831_block_009_covers : Budget.RowUpperBound PrimePairVertices.tree_block_009 26034 11831 row11831_block_009 :=
  Budget.empty_row_after _ _ _ 16514 (fun q member => (Bounds.block_009 q member).1) (by decide)

def row11831_block_010 : List Nat := []
theorem row11831_block_010_covers : Budget.RowUpperBound PrimePairVertices.tree_block_010 26034 11831 row11831_block_010 :=
  Budget.empty_row_after _ _ _ 18562 (fun q member => (Bounds.block_010 q member).1) (by decide)

def row11831_block_011 : List Nat := []
theorem row11831_block_011_covers : Budget.RowUpperBound PrimePairVertices.tree_block_011 26034 11831 row11831_block_011 :=
  Budget.empty_row_after _ _ _ 20610 (fun q member => (Bounds.block_011 q member).1) (by decide)

def row11831_block_012 : List Nat := []
theorem row11831_block_012_covers : Budget.RowUpperBound PrimePairVertices.tree_block_012 26034 11831 row11831_block_012 :=
  Budget.empty_row_after _ _ _ 22658 (fun q member => (Bounds.block_012 q member).1) (by decide)

def row11831_block_013 : List Nat := []
theorem row11831_block_013_covers : Budget.RowUpperBound PrimePairVertices.tree_block_013 26034 11831 row11831_block_013 :=
  Budget.empty_row_after _ _ _ 24706 (fun q member => (Bounds.block_013 q member).1) (by decide)

def row11831_large_left_left : List Nat := row11831_small ++ row11831_block_005
theorem row11831_large_left_left_covers : Budget.RowUpperBound PrimePairVertices.tree_large_left_left 26034 11831 row11831_large_left_left :=
  Budget.RowUpperBound.branch row11831_small_covers row11831_block_005_covers

def row11831_large_left_right_right : List Nat := row11831_block_007 ++ row11831_block_008
theorem row11831_large_left_right_right_covers : Budget.RowUpperBound PrimePairVertices.tree_large_left_right_right 26034 11831 row11831_large_left_right_right :=
  Budget.RowUpperBound.branch row11831_block_007_covers row11831_block_008_covers

def row11831_large_left_right : List Nat := row11831_block_006 ++ row11831_large_left_right_right
theorem row11831_large_left_right_covers : Budget.RowUpperBound PrimePairVertices.tree_large_left_right 26034 11831 row11831_large_left_right :=
  Budget.RowUpperBound.branch row11831_block_006_covers row11831_large_left_right_right_covers

def row11831_large_left : List Nat := row11831_large_left_left ++ row11831_large_left_right
theorem row11831_large_left_covers : Budget.RowUpperBound PrimePairVertices.tree_large_left 26034 11831 row11831_large_left :=
  Budget.RowUpperBound.branch row11831_large_left_left_covers row11831_large_left_right_covers

def row11831_large_right_left : List Nat := row11831_block_009 ++ row11831_block_010
theorem row11831_large_right_left_covers : Budget.RowUpperBound PrimePairVertices.tree_large_right_left 26034 11831 row11831_large_right_left :=
  Budget.RowUpperBound.branch row11831_block_009_covers row11831_block_010_covers

def row11831_large_right_right_right : List Nat := row11831_block_012 ++ row11831_block_013
theorem row11831_large_right_right_right_covers : Budget.RowUpperBound PrimePairVertices.tree_large_right_right_right 26034 11831 row11831_large_right_right_right :=
  Budget.RowUpperBound.branch row11831_block_012_covers row11831_block_013_covers

def row11831_large_right_right : List Nat := row11831_block_011 ++ row11831_large_right_right_right
theorem row11831_large_right_right_covers : Budget.RowUpperBound PrimePairVertices.tree_large_right_right 26034 11831 row11831_large_right_right :=
  Budget.RowUpperBound.branch row11831_block_011_covers row11831_large_right_right_right_covers

def row11831_large_right : List Nat := row11831_large_right_left ++ row11831_large_right_right
theorem row11831_large_right_covers : Budget.RowUpperBound PrimePairVertices.tree_large_right 26034 11831 row11831_large_right :=
  Budget.RowUpperBound.branch row11831_large_right_left_covers row11831_large_right_right_covers

def row11831_large : List Nat := row11831_large_left ++ row11831_large_right
theorem row11831_large_covers : Budget.RowUpperBound PrimePairVertices.tree_large 26034 11831 row11831_large :=
  Budget.RowUpperBound.branch row11831_large_left_covers row11831_large_right_covers

def row11831 : List Nat := row11831_large
theorem row11831_covers : Budget.RowUpperBound PrimePairVertices.largeTree 26034 11831 row11831 := row11831_large_covers

def row11833_small : List Nat := []
theorem row11833_small_covers : Budget.RowUpperBound PrimePairVertices.tree_small 26034 11833 row11833_small :=
  Budget.empty_row_before _ _ _ 8389 (fun q member => (Bounds.small q member).2) (by decide)

def row11833_block_005 : List Nat := []
theorem row11833_block_005_covers : Budget.RowUpperBound PrimePairVertices.tree_block_005 26034 11833 row11833_block_005 :=
  Budget.empty_row_before _ _ _ 10370 (fun q member => (Bounds.block_005 q member).2) (by decide)

def row11833_block_006 : List Nat := [12301, 12379]
def plan11833_block_006 : List Budget.Choice := [.outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, (.secondFactor 13), (.secondFactor 7), .first, (.firstFactor 19), .first, .first, .first, .first, .first, .first, .first, (.firstFactor 7), .first, .first, .first, .first, .first, .first, .first, .first, .first, .first, .first, (.secondFactor 23), .first, (.firstFactor 23), (.firstFactor 11), .first, .first, (.firstFactor 7), .first, .first, .first, .first, (.secondFactor 7), .first, (.firstFactor 11), .first, .first, (.firstFactor 23), .first, .first, (.secondFactor 7), .first, (.firstFactor 31), .first, .first, (.firstFactor 7), .first, .second, .possible, .first, .first, (.firstFactor 19), .first, (.secondFactor 17), .first, .possible, (.secondFactor 37), .first, (.secondFactor 7), .first]
set_option maxRecDepth 2048 in
theorem row11833_block_006_covers : Budget.RowUpperBound PrimePairVertices.tree_block_006 26034 11833 row11833_block_006 := by
  apply Budget.row_upper_of_plan _ _ _ plan11833_block_006
  · decide
  · rfl

def row11833_block_007 : List Nat := [13267, 14029]
def plan11833_block_007 : List Budget.Choice := [(.firstFactor 17), (.firstFactor 29), .first, (.secondFactor 7), (.firstFactor 19), .first, .first, (.firstFactor 7), .first, .first, .first, .first, (.secondFactor 29), .first, .first, (.secondFactor 13), .second, .first, .first, (.firstFactor 37), .first, .first, (.secondFactor 23), .first, (.firstFactor 7), (.secondFactor 7), (.firstFactor 13), .first, .first, .first, .first, .first, .first, (.firstFactor 7), (.secondFactor 7), .first, .first, (.firstFactor 7), .first, (.firstFactor 11), (.firstFactor 23), (.firstFactor 7), .first, (.firstFactor 19), .first, .first, (.firstFactor 7), (.firstFactor 17), .first, (.secondFactor 13), (.firstFactor 11), .first, .first, (.firstFactor 7), .first, .first, .first, .first, .first, .first, .first, .second, .first, .first, .first, .first, (.secondFactor 31), .first, (.secondFactor 13), (.firstFactor 7), .first, .first, .first, .first, .first, .first, .first, .first, .first, .first, .second, .first, (.firstFactor 7), .first, .second, (.firstFactor 23), (.firstFactor 13), .first, .first, (.firstFactor 11), .first, .first, (.secondFactor 7), .first, .possible, (.secondFactor 7), .first, .first, .first, (.firstFactor 7), .first, .first, (.firstFactor 13), .first, .first, .first, (.secondFactor 13), (.firstFactor 7), (.firstFactor 11), .first, (.firstFactor 17), .first, .first, .first, .first, (.firstFactor 29), .first, .first, .first, .first, (.firstFactor 7), .first, (.secondFactor 23), .first, (.firstFactor 23), (.firstFactor 19), .first, .first, (.secondFactor 7), (.secondFactor 13), .first, (.secondFactor 7), .first, (.firstFactor 11), (.firstFactor 37), .first, .second, .first, .first, (.firstFactor 19), .first, (.secondFactor 37), (.firstFactor 13), .first, .first, .second, .first, .first, (.firstFactor 7), .first, (.firstFactor 13), .first, (.firstFactor 7), .first, .first, (.firstFactor 7), .first, (.firstFactor 11), .first, .first, (.secondFactor 17), .first, .first, (.secondFactor 7), .first, (.secondFactor 31), (.firstFactor 13), .first, .first, (.firstFactor 7), .first, (.firstFactor 11), .possible, .first, .first, .first, .first, .first, (.firstFactor 7), .first, (.secondFactor 17), (.firstFactor 11), .first, .first, .first, (.firstFactor 29), .first, (.firstFactor 13), .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside]
set_option maxRecDepth 2048 in
theorem row11833_block_007_covers : Budget.RowUpperBound PrimePairVertices.tree_block_007 26034 11833 row11833_block_007 := by
  apply Budget.row_upper_of_plan _ _ _ plan11833_block_007
  · decide
  · rfl

def row11833_block_008 : List Nat := []
theorem row11833_block_008_covers : Budget.RowUpperBound PrimePairVertices.tree_block_008 26034 11833 row11833_block_008 :=
  Budget.empty_row_after _ _ _ 14466 (fun q member => (Bounds.block_008 q member).1) (by decide)

def row11833_block_009 : List Nat := []
theorem row11833_block_009_covers : Budget.RowUpperBound PrimePairVertices.tree_block_009 26034 11833 row11833_block_009 :=
  Budget.empty_row_after _ _ _ 16514 (fun q member => (Bounds.block_009 q member).1) (by decide)

def row11833_block_010 : List Nat := []
theorem row11833_block_010_covers : Budget.RowUpperBound PrimePairVertices.tree_block_010 26034 11833 row11833_block_010 :=
  Budget.empty_row_after _ _ _ 18562 (fun q member => (Bounds.block_010 q member).1) (by decide)

def row11833_block_011 : List Nat := []
theorem row11833_block_011_covers : Budget.RowUpperBound PrimePairVertices.tree_block_011 26034 11833 row11833_block_011 :=
  Budget.empty_row_after _ _ _ 20610 (fun q member => (Bounds.block_011 q member).1) (by decide)

def row11833_block_012 : List Nat := []
theorem row11833_block_012_covers : Budget.RowUpperBound PrimePairVertices.tree_block_012 26034 11833 row11833_block_012 :=
  Budget.empty_row_after _ _ _ 22658 (fun q member => (Bounds.block_012 q member).1) (by decide)

def row11833_block_013 : List Nat := []
theorem row11833_block_013_covers : Budget.RowUpperBound PrimePairVertices.tree_block_013 26034 11833 row11833_block_013 :=
  Budget.empty_row_after _ _ _ 24706 (fun q member => (Bounds.block_013 q member).1) (by decide)

def row11833_large_left_left : List Nat := row11833_small ++ row11833_block_005
theorem row11833_large_left_left_covers : Budget.RowUpperBound PrimePairVertices.tree_large_left_left 26034 11833 row11833_large_left_left :=
  Budget.RowUpperBound.branch row11833_small_covers row11833_block_005_covers

def row11833_large_left_right_right : List Nat := row11833_block_007 ++ row11833_block_008
theorem row11833_large_left_right_right_covers : Budget.RowUpperBound PrimePairVertices.tree_large_left_right_right 26034 11833 row11833_large_left_right_right :=
  Budget.RowUpperBound.branch row11833_block_007_covers row11833_block_008_covers

def row11833_large_left_right : List Nat := row11833_block_006 ++ row11833_large_left_right_right
theorem row11833_large_left_right_covers : Budget.RowUpperBound PrimePairVertices.tree_large_left_right 26034 11833 row11833_large_left_right :=
  Budget.RowUpperBound.branch row11833_block_006_covers row11833_large_left_right_right_covers

def row11833_large_left : List Nat := row11833_large_left_left ++ row11833_large_left_right
theorem row11833_large_left_covers : Budget.RowUpperBound PrimePairVertices.tree_large_left 26034 11833 row11833_large_left :=
  Budget.RowUpperBound.branch row11833_large_left_left_covers row11833_large_left_right_covers

def row11833_large_right_left : List Nat := row11833_block_009 ++ row11833_block_010
theorem row11833_large_right_left_covers : Budget.RowUpperBound PrimePairVertices.tree_large_right_left 26034 11833 row11833_large_right_left :=
  Budget.RowUpperBound.branch row11833_block_009_covers row11833_block_010_covers

def row11833_large_right_right_right : List Nat := row11833_block_012 ++ row11833_block_013
theorem row11833_large_right_right_right_covers : Budget.RowUpperBound PrimePairVertices.tree_large_right_right_right 26034 11833 row11833_large_right_right_right :=
  Budget.RowUpperBound.branch row11833_block_012_covers row11833_block_013_covers

def row11833_large_right_right : List Nat := row11833_block_011 ++ row11833_large_right_right_right
theorem row11833_large_right_right_covers : Budget.RowUpperBound PrimePairVertices.tree_large_right_right 26034 11833 row11833_large_right_right :=
  Budget.RowUpperBound.branch row11833_block_011_covers row11833_large_right_right_right_covers

def row11833_large_right : List Nat := row11833_large_right_left ++ row11833_large_right_right
theorem row11833_large_right_covers : Budget.RowUpperBound PrimePairVertices.tree_large_right 26034 11833 row11833_large_right :=
  Budget.RowUpperBound.branch row11833_large_right_left_covers row11833_large_right_right_covers

def row11833_large : List Nat := row11833_large_left ++ row11833_large_right
theorem row11833_large_covers : Budget.RowUpperBound PrimePairVertices.tree_large 26034 11833 row11833_large :=
  Budget.RowUpperBound.branch row11833_large_left_covers row11833_large_right_covers

def row11833 : List Nat := row11833_large
theorem row11833_covers : Budget.RowUpperBound PrimePairVertices.largeTree 26034 11833 row11833 := row11833_large_covers

def row11839_small : List Nat := []
theorem row11839_small_covers : Budget.RowUpperBound PrimePairVertices.tree_small 26034 11839 row11839_small :=
  Budget.empty_row_before _ _ _ 8389 (fun q member => (Bounds.small q member).2) (by decide)

def row11839_block_005 : List Nat := []
theorem row11839_block_005_covers : Budget.RowUpperBound PrimePairVertices.tree_block_005 26034 11839 row11839_block_005 :=
  Budget.empty_row_before _ _ _ 10370 (fun q member => (Bounds.block_005 q member).2) (by decide)

def row11839_block_006 : List Nat := []
def plan11839_block_006 : List Budget.Choice := [.outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, (.firstFactor 29), .first, (.secondFactor 7), .first, .first, .first, .second, .first, .first, .first, .first, (.firstFactor 7), (.firstFactor 23), .first, (.firstFactor 11), .first, .first, .first, .first, (.firstFactor 7), .first, (.secondFactor 31), (.secondFactor 17), .first, .first, (.firstFactor 19), .first, .first, (.secondFactor 37), .first, .first, .first, .first, .second, .first, (.firstFactor 7), .first, .first, (.firstFactor 19), .first, .first, .first, .first, (.secondFactor 17), .first, .first, (.secondFactor 29), .first, (.firstFactor 7), (.firstFactor 11), .first, .first, (.secondFactor 19), .first, (.firstFactor 7), .first, (.firstFactor 13), (.secondFactor 7), .first, .first, .first]
set_option maxRecDepth 2048 in
theorem row11839_block_006_covers : Budget.RowUpperBound PrimePairVertices.tree_block_006 26034 11839 row11839_block_006 := by
  apply Budget.row_upper_of_plan _ _ _ plan11839_block_006
  · decide
  · rfl

def row11839_block_007 : List Nat := [12721]
def plan11839_block_007 : List Budget.Choice := [(.firstFactor 37), (.firstFactor 11), .first, (.secondFactor 29), (.firstFactor 7), .first, .first, (.firstFactor 17), .first, .first, .first, (.firstFactor 23), (.secondFactor 7), .first, .first, (.firstFactor 7), .first, (.firstFactor 19), .first, .first, (.firstFactor 7), (.firstFactor 17), (.secondFactor 7), .first, (.firstFactor 13), .second, (.secondFactor 13), .first, .first, .first, .first, .first, .first, (.firstFactor 11), .first, .first, .possible, .first, .first, .second, (.firstFactor 11), (.firstFactor 19), .first, (.secondFactor 19), .first, .first, .first, (.firstFactor 11), .first, (.secondFactor 7), .first, .first, .first, (.firstFactor 29), .first, .first, (.firstFactor 7), .first, .first, .first, .first, (.secondFactor 17), (.secondFactor 29), (.secondFactor 7), .first, .first, (.firstFactor 7), .first, (.firstFactor 19), (.firstFactor 31), .first, .first, .first, (.firstFactor 23), (.firstFactor 11), (.firstFactor 17), .first, .first, .first, .first, (.secondFactor 7), .first, (.firstFactor 11), .first, (.firstFactor 7), .first, (.secondFactor 13), .first, .first, (.firstFactor 31), .first, .first, .first, .first, .first, (.firstFactor 11), (.firstFactor 7), (.firstFactor 37), .first, (.secondFactor 23), .first, .first, (.firstFactor 7), .first, (.firstFactor 7), .first, (.secondFactor 7), .first, (.secondFactor 13), .first, (.secondFactor 7), .first, .first, .first, .first, (.firstFactor 23), .first, .first, .first, .first, .first, .first, (.secondFactor 7), .first, (.firstFactor 7), (.secondFactor 19), .first, .first, (.firstFactor 13), (.firstFactor 7), .first, (.secondFactor 29), .first, (.secondFactor 17), (.firstFactor 11), .first, (.firstFactor 19), .first, .first, (.firstFactor 17), .first, .second, (.secondFactor 13), .first, .first, (.firstFactor 7), .first, .first, .first, .first, (.firstFactor 19), .first, .first, .first, .first, .second, .first, (.secondFactor 23), .first, .first, (.secondFactor 7), .first, .first, (.firstFactor 19), .first, .first, (.firstFactor 31), .first, .first, .first, .first, (.firstFactor 7), (.firstFactor 23), .first, .first, .first, (.secondFactor 7), .first, (.firstFactor 11), .first, (.secondFactor 37), .first, (.firstFactor 11), .first, .first, (.firstFactor 13), .first, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside]
set_option maxRecDepth 2048 in
theorem row11839_block_007_covers : Budget.RowUpperBound PrimePairVertices.tree_block_007 26034 11839 row11839_block_007 := by
  apply Budget.row_upper_of_plan _ _ _ plan11839_block_007
  · decide
  · rfl

def row11839_block_008 : List Nat := []
theorem row11839_block_008_covers : Budget.RowUpperBound PrimePairVertices.tree_block_008 26034 11839 row11839_block_008 :=
  Budget.empty_row_after _ _ _ 14466 (fun q member => (Bounds.block_008 q member).1) (by decide)

def row11839_block_009 : List Nat := []
theorem row11839_block_009_covers : Budget.RowUpperBound PrimePairVertices.tree_block_009 26034 11839 row11839_block_009 :=
  Budget.empty_row_after _ _ _ 16514 (fun q member => (Bounds.block_009 q member).1) (by decide)

def row11839_block_010 : List Nat := []
theorem row11839_block_010_covers : Budget.RowUpperBound PrimePairVertices.tree_block_010 26034 11839 row11839_block_010 :=
  Budget.empty_row_after _ _ _ 18562 (fun q member => (Bounds.block_010 q member).1) (by decide)

def row11839_block_011 : List Nat := []
theorem row11839_block_011_covers : Budget.RowUpperBound PrimePairVertices.tree_block_011 26034 11839 row11839_block_011 :=
  Budget.empty_row_after _ _ _ 20610 (fun q member => (Bounds.block_011 q member).1) (by decide)

def row11839_block_012 : List Nat := []
theorem row11839_block_012_covers : Budget.RowUpperBound PrimePairVertices.tree_block_012 26034 11839 row11839_block_012 :=
  Budget.empty_row_after _ _ _ 22658 (fun q member => (Bounds.block_012 q member).1) (by decide)

def row11839_block_013 : List Nat := []
theorem row11839_block_013_covers : Budget.RowUpperBound PrimePairVertices.tree_block_013 26034 11839 row11839_block_013 :=
  Budget.empty_row_after _ _ _ 24706 (fun q member => (Bounds.block_013 q member).1) (by decide)

def row11839_large_left_left : List Nat := row11839_small ++ row11839_block_005
theorem row11839_large_left_left_covers : Budget.RowUpperBound PrimePairVertices.tree_large_left_left 26034 11839 row11839_large_left_left :=
  Budget.RowUpperBound.branch row11839_small_covers row11839_block_005_covers

def row11839_large_left_right_right : List Nat := row11839_block_007 ++ row11839_block_008
theorem row11839_large_left_right_right_covers : Budget.RowUpperBound PrimePairVertices.tree_large_left_right_right 26034 11839 row11839_large_left_right_right :=
  Budget.RowUpperBound.branch row11839_block_007_covers row11839_block_008_covers

def row11839_large_left_right : List Nat := row11839_block_006 ++ row11839_large_left_right_right
theorem row11839_large_left_right_covers : Budget.RowUpperBound PrimePairVertices.tree_large_left_right 26034 11839 row11839_large_left_right :=
  Budget.RowUpperBound.branch row11839_block_006_covers row11839_large_left_right_right_covers

def row11839_large_left : List Nat := row11839_large_left_left ++ row11839_large_left_right
theorem row11839_large_left_covers : Budget.RowUpperBound PrimePairVertices.tree_large_left 26034 11839 row11839_large_left :=
  Budget.RowUpperBound.branch row11839_large_left_left_covers row11839_large_left_right_covers

def row11839_large_right_left : List Nat := row11839_block_009 ++ row11839_block_010
theorem row11839_large_right_left_covers : Budget.RowUpperBound PrimePairVertices.tree_large_right_left 26034 11839 row11839_large_right_left :=
  Budget.RowUpperBound.branch row11839_block_009_covers row11839_block_010_covers

def row11839_large_right_right_right : List Nat := row11839_block_012 ++ row11839_block_013
theorem row11839_large_right_right_right_covers : Budget.RowUpperBound PrimePairVertices.tree_large_right_right_right 26034 11839 row11839_large_right_right_right :=
  Budget.RowUpperBound.branch row11839_block_012_covers row11839_block_013_covers

def row11839_large_right_right : List Nat := row11839_block_011 ++ row11839_large_right_right_right
theorem row11839_large_right_right_covers : Budget.RowUpperBound PrimePairVertices.tree_large_right_right 26034 11839 row11839_large_right_right :=
  Budget.RowUpperBound.branch row11839_block_011_covers row11839_large_right_right_right_covers

def row11839_large_right : List Nat := row11839_large_right_left ++ row11839_large_right_right
theorem row11839_large_right_covers : Budget.RowUpperBound PrimePairVertices.tree_large_right 26034 11839 row11839_large_right :=
  Budget.RowUpperBound.branch row11839_large_right_left_covers row11839_large_right_right_covers

def row11839_large : List Nat := row11839_large_left ++ row11839_large_right
theorem row11839_large_covers : Budget.RowUpperBound PrimePairVertices.tree_large 26034 11839 row11839_large :=
  Budget.RowUpperBound.branch row11839_large_left_covers row11839_large_right_covers

def row11839 : List Nat := row11839_large
theorem row11839_covers : Budget.RowUpperBound PrimePairVertices.largeTree 26034 11839 row11839 := row11839_large_covers

def row11863_small : List Nat := []
theorem row11863_small_covers : Budget.RowUpperBound PrimePairVertices.tree_small 26034 11863 row11863_small :=
  Budget.empty_row_before _ _ _ 8389 (fun q member => (Bounds.small q member).2) (by decide)

def row11863_block_005 : List Nat := []
theorem row11863_block_005_covers : Budget.RowUpperBound PrimePairVertices.tree_block_005 26034 11863 row11863_block_005 :=
  Budget.empty_row_before _ _ _ 10370 (fun q member => (Bounds.block_005 q member).2) (by decide)

def row11863_block_006 : List Nat := [12157]
def plan11863_block_006 : List Budget.Choice := [.outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .first, .first, .first, .first, .first, (.secondFactor 31), .first, .first, .first, (.secondFactor 7), (.secondFactor 23), (.firstFactor 7), .first, (.firstFactor 13), .first, .first, (.secondFactor 13), .first, (.firstFactor 17), .first, (.firstFactor 7), (.firstFactor 13), .first, .first, .first, .first, .first, (.secondFactor 7), .first, .first, .first, .first, .possible, .first, (.secondFactor 13), .first, .first, (.firstFactor 7), .first, .first, (.firstFactor 17), .first, (.firstFactor 7), .first, .first, (.secondFactor 7), .first, .first, .first, .first, .first, (.firstFactor 17), .first, (.secondFactor 19), .first, (.firstFactor 7), (.firstFactor 11), .first, .first, .first]
set_option maxRecDepth 2048 in
theorem row11863_block_006_covers : Budget.RowUpperBound PrimePairVertices.tree_block_006 26034 11863 row11863_block_006 := by
  apply Budget.row_upper_of_plan _ _ _ plan11863_block_006
  · decide
  · rfl

def row11863_block_007 : List Nat := [13903]
def plan11863_block_007 : List Budget.Choice := [(.firstFactor 7), (.firstFactor 31), .first, .first, (.firstFactor 11), .first, .first, (.secondFactor 7), .first, .first, .first, .first, (.firstFactor 13), .first, .first, (.firstFactor 29), (.firstFactor 7), (.secondFactor 13), .first, .first, .first, (.firstFactor 7), (.secondFactor 19), .first, (.secondFactor 7), (.firstFactor 31), .first, .first, .first, .first, .first, .first, .first, (.secondFactor 7), .second, .first, (.firstFactor 11), (.secondFactor 7), .first, (.firstFactor 7), .second, (.secondFactor 7), .first, (.firstFactor 7), .first, .first, (.firstFactor 19), (.firstFactor 13), (.firstFactor 7), (.firstFactor 11), (.firstFactor 29), .first, .first, (.firstFactor 13), .first, .first, (.firstFactor 11), .first, .first, .first, .first, (.firstFactor 7), .first, (.secondFactor 37), .first, .first, .first, .first, (.firstFactor 7), (.secondFactor 7), .first, .first, .first, (.firstFactor 13), (.firstFactor 7), (.secondFactor 13), .first, .first, .first, .first, .first, .first, (.firstFactor 17), .first, (.secondFactor 19), (.firstFactor 7), (.firstFactor 11), .first, .first, (.firstFactor 7), .first, .first, (.firstFactor 11), .first, .first, .first, (.firstFactor 13), .first, .first, (.secondFactor 7), .first, .first, .first, .first, (.firstFactor 11), .first, (.secondFactor 19), (.firstFactor 29), .first, .first, .second, .first, .first, .first, .first, .first, .first, .first, (.firstFactor 7), .first, (.secondFactor 7), .first, (.firstFactor 17), .first, .first, (.firstFactor 7), .first, .first, (.secondFactor 19), .second, .first, (.firstFactor 17), .first, (.firstFactor 7), (.firstFactor 13), .first, .first, .first, .first, (.firstFactor 11), .first, (.firstFactor 7), .first, .first, .first, (.firstFactor 29), .first, .first, (.secondFactor 7), .first, (.firstFactor 7), .first, (.secondFactor 7), .first, .first, (.firstFactor 17), .first, (.secondFactor 13), .first, .first, .possible, .first, .first, (.firstFactor 13), .first, (.firstFactor 7), (.firstFactor 19), .first, .first, (.firstFactor 13), .first, .first, .first, .first, .first, .first, .first, .first, (.secondFactor 7), .first, (.firstFactor 11), (.firstFactor 7), .first, .first, .first, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside]
set_option maxRecDepth 2048 in
theorem row11863_block_007_covers : Budget.RowUpperBound PrimePairVertices.tree_block_007 26034 11863 row11863_block_007 := by
  apply Budget.row_upper_of_plan _ _ _ plan11863_block_007
  · decide
  · rfl

def row11863_block_008 : List Nat := []
theorem row11863_block_008_covers : Budget.RowUpperBound PrimePairVertices.tree_block_008 26034 11863 row11863_block_008 :=
  Budget.empty_row_after _ _ _ 14466 (fun q member => (Bounds.block_008 q member).1) (by decide)

def row11863_block_009 : List Nat := []
theorem row11863_block_009_covers : Budget.RowUpperBound PrimePairVertices.tree_block_009 26034 11863 row11863_block_009 :=
  Budget.empty_row_after _ _ _ 16514 (fun q member => (Bounds.block_009 q member).1) (by decide)

def row11863_block_010 : List Nat := []
theorem row11863_block_010_covers : Budget.RowUpperBound PrimePairVertices.tree_block_010 26034 11863 row11863_block_010 :=
  Budget.empty_row_after _ _ _ 18562 (fun q member => (Bounds.block_010 q member).1) (by decide)

def row11863_block_011 : List Nat := []
theorem row11863_block_011_covers : Budget.RowUpperBound PrimePairVertices.tree_block_011 26034 11863 row11863_block_011 :=
  Budget.empty_row_after _ _ _ 20610 (fun q member => (Bounds.block_011 q member).1) (by decide)

def row11863_block_012 : List Nat := []
theorem row11863_block_012_covers : Budget.RowUpperBound PrimePairVertices.tree_block_012 26034 11863 row11863_block_012 :=
  Budget.empty_row_after _ _ _ 22658 (fun q member => (Bounds.block_012 q member).1) (by decide)

def row11863_block_013 : List Nat := []
theorem row11863_block_013_covers : Budget.RowUpperBound PrimePairVertices.tree_block_013 26034 11863 row11863_block_013 :=
  Budget.empty_row_after _ _ _ 24706 (fun q member => (Bounds.block_013 q member).1) (by decide)

def row11863_large_left_left : List Nat := row11863_small ++ row11863_block_005
theorem row11863_large_left_left_covers : Budget.RowUpperBound PrimePairVertices.tree_large_left_left 26034 11863 row11863_large_left_left :=
  Budget.RowUpperBound.branch row11863_small_covers row11863_block_005_covers

def row11863_large_left_right_right : List Nat := row11863_block_007 ++ row11863_block_008
theorem row11863_large_left_right_right_covers : Budget.RowUpperBound PrimePairVertices.tree_large_left_right_right 26034 11863 row11863_large_left_right_right :=
  Budget.RowUpperBound.branch row11863_block_007_covers row11863_block_008_covers

def row11863_large_left_right : List Nat := row11863_block_006 ++ row11863_large_left_right_right
theorem row11863_large_left_right_covers : Budget.RowUpperBound PrimePairVertices.tree_large_left_right 26034 11863 row11863_large_left_right :=
  Budget.RowUpperBound.branch row11863_block_006_covers row11863_large_left_right_right_covers

def row11863_large_left : List Nat := row11863_large_left_left ++ row11863_large_left_right
theorem row11863_large_left_covers : Budget.RowUpperBound PrimePairVertices.tree_large_left 26034 11863 row11863_large_left :=
  Budget.RowUpperBound.branch row11863_large_left_left_covers row11863_large_left_right_covers

def row11863_large_right_left : List Nat := row11863_block_009 ++ row11863_block_010
theorem row11863_large_right_left_covers : Budget.RowUpperBound PrimePairVertices.tree_large_right_left 26034 11863 row11863_large_right_left :=
  Budget.RowUpperBound.branch row11863_block_009_covers row11863_block_010_covers

def row11863_large_right_right_right : List Nat := row11863_block_012 ++ row11863_block_013
theorem row11863_large_right_right_right_covers : Budget.RowUpperBound PrimePairVertices.tree_large_right_right_right 26034 11863 row11863_large_right_right_right :=
  Budget.RowUpperBound.branch row11863_block_012_covers row11863_block_013_covers

def row11863_large_right_right : List Nat := row11863_block_011 ++ row11863_large_right_right_right
theorem row11863_large_right_right_covers : Budget.RowUpperBound PrimePairVertices.tree_large_right_right 26034 11863 row11863_large_right_right :=
  Budget.RowUpperBound.branch row11863_block_011_covers row11863_large_right_right_right_covers

def row11863_large_right : List Nat := row11863_large_right_left ++ row11863_large_right_right
theorem row11863_large_right_covers : Budget.RowUpperBound PrimePairVertices.tree_large_right 26034 11863 row11863_large_right :=
  Budget.RowUpperBound.branch row11863_large_right_left_covers row11863_large_right_right_covers

def row11863_large : List Nat := row11863_large_left ++ row11863_large_right
theorem row11863_large_covers : Budget.RowUpperBound PrimePairVertices.tree_large 26034 11863 row11863_large :=
  Budget.RowUpperBound.branch row11863_large_left_covers row11863_large_right_covers

def row11863 : List Nat := row11863_large
theorem row11863_covers : Budget.RowUpperBound PrimePairVertices.largeTree 26034 11863 row11863 := row11863_large_covers

end ProjectEulerSolutions.PrimePairLarge
