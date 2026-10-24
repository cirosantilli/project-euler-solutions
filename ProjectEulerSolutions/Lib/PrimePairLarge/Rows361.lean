import ProjectEulerSolutions.Lib.PrimePairLarge.Bounds

namespace ProjectEulerSolutions.PrimePairLarge
open PrimePairCertificate

def row12097_small : List Nat := []
theorem row12097_small_covers : Budget.RowUpperBound PrimePairVertices.tree_small 26034 12097 row12097_small :=
  Budget.empty_row_before _ _ _ 8389 (fun q member => (Bounds.small q member).2) (by decide)

def row12097_block_005 : List Nat := []
theorem row12097_block_005_covers : Budget.RowUpperBound PrimePairVertices.tree_block_005 26034 12097 row12097_block_005 :=
  Budget.empty_row_before _ _ _ 10370 (fun q member => (Bounds.block_005 q member).2) (by decide)

def row12097_block_006 : List Nat := [12109]
def plan12097_block_006 : List Budget.Choice := [.outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .first, .first, .possible, .first, .first, .first, .first, .first, .first, (.firstFactor 11), .first, .first, .first, .first, .first, (.secondFactor 13), .first, (.secondFactor 23), .first, .first, .second, .first, (.secondFactor 7), (.firstFactor 7), .first, .first, (.firstFactor 7), .first, (.firstFactor 37), .first, .first, (.secondFactor 23), .first, (.firstFactor 29), .first]
set_option maxRecDepth 2048 in
theorem row12097_block_006_covers : Budget.RowUpperBound PrimePairVertices.tree_block_006 26034 12097 row12097_block_006 := by
  apply Budget.row_upper_of_plan _ _ _ plan12097_block_006
  · decide
  · rfl

def row12097_block_007 : List Nat := [13513, 13537]
def plan12097_block_007 : List Budget.Choice := [.first, .first, .first, .first, (.secondFactor 7), .first, .first, .first, .first, .first, .first, (.firstFactor 7), (.firstFactor 13), .first, .first, (.secondFactor 7), .second, (.firstFactor 7), .first, (.firstFactor 19), (.firstFactor 29), .first, (.secondFactor 17), .first, (.secondFactor 31), .second, (.firstFactor 7), .first, .first, .first, .first, .first, .first, .first, (.secondFactor 17), .first, (.firstFactor 7), .second, .first, (.firstFactor 11), (.firstFactor 7), .second, .first, (.secondFactor 31), .first, .first, (.firstFactor 11), (.firstFactor 13), .first, (.firstFactor 23), (.firstFactor 7), .first, .first, (.firstFactor 13), .first, .first, (.firstFactor 19), .first, .first, .first, .first, .first, (.firstFactor 7), (.firstFactor 17), .first, .first, (.secondFactor 7), .first, (.secondFactor 17), (.firstFactor 19), .first, .first, .first, (.firstFactor 13), (.firstFactor 31), (.firstFactor 7), .first, .first, .first, .first, (.firstFactor 19), .first, .first, .first, (.secondFactor 7), (.secondFactor 13), (.firstFactor 7), .first, .first, (.firstFactor 11), .first, .first, .second, .first, (.firstFactor 7), .first, (.firstFactor 13), (.firstFactor 7), .first, .first, .first, .first, (.secondFactor 7), .first, (.secondFactor 7), .first, .first, (.secondFactor 13), (.firstFactor 11), .first, (.secondFactor 29), .first, .first, .first, .first, (.firstFactor 7), .first, .first, .possible, .first, .possible, .first, (.secondFactor 13), .first, (.firstFactor 17), .first, .first, .first, (.firstFactor 29), (.secondFactor 7), .first, .first, .first, (.firstFactor 11), (.firstFactor 7), .first, (.firstFactor 17), .first, .first, .first, .first, (.secondFactor 13), (.firstFactor 7), .first, .first, (.secondFactor 7), .first, .first, (.secondFactor 29), .first, .first, .first, (.firstFactor 19), .first, .first, .first, .first, (.firstFactor 11), .first, .first, .first, .first, .first, (.firstFactor 13), .first, .first, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside]
set_option maxRecDepth 2048 in
theorem row12097_block_007_covers : Budget.RowUpperBound PrimePairVertices.tree_block_007 26034 12097 row12097_block_007 := by
  apply Budget.row_upper_of_plan _ _ _ plan12097_block_007
  · decide
  · rfl

def row12097_block_008 : List Nat := []
theorem row12097_block_008_covers : Budget.RowUpperBound PrimePairVertices.tree_block_008 26034 12097 row12097_block_008 :=
  Budget.empty_row_after _ _ _ 14466 (fun q member => (Bounds.block_008 q member).1) (by decide)

def row12097_block_009 : List Nat := []
theorem row12097_block_009_covers : Budget.RowUpperBound PrimePairVertices.tree_block_009 26034 12097 row12097_block_009 :=
  Budget.empty_row_after _ _ _ 16514 (fun q member => (Bounds.block_009 q member).1) (by decide)

def row12097_block_010 : List Nat := []
theorem row12097_block_010_covers : Budget.RowUpperBound PrimePairVertices.tree_block_010 26034 12097 row12097_block_010 :=
  Budget.empty_row_after _ _ _ 18562 (fun q member => (Bounds.block_010 q member).1) (by decide)

def row12097_block_011 : List Nat := []
theorem row12097_block_011_covers : Budget.RowUpperBound PrimePairVertices.tree_block_011 26034 12097 row12097_block_011 :=
  Budget.empty_row_after _ _ _ 20610 (fun q member => (Bounds.block_011 q member).1) (by decide)

def row12097_block_012 : List Nat := []
theorem row12097_block_012_covers : Budget.RowUpperBound PrimePairVertices.tree_block_012 26034 12097 row12097_block_012 :=
  Budget.empty_row_after _ _ _ 22658 (fun q member => (Bounds.block_012 q member).1) (by decide)

def row12097_block_013 : List Nat := []
theorem row12097_block_013_covers : Budget.RowUpperBound PrimePairVertices.tree_block_013 26034 12097 row12097_block_013 :=
  Budget.empty_row_after _ _ _ 24706 (fun q member => (Bounds.block_013 q member).1) (by decide)

def row12097_large_left_left : List Nat := row12097_small ++ row12097_block_005
theorem row12097_large_left_left_covers : Budget.RowUpperBound PrimePairVertices.tree_large_left_left 26034 12097 row12097_large_left_left :=
  Budget.RowUpperBound.branch row12097_small_covers row12097_block_005_covers

def row12097_large_left_right_right : List Nat := row12097_block_007 ++ row12097_block_008
theorem row12097_large_left_right_right_covers : Budget.RowUpperBound PrimePairVertices.tree_large_left_right_right 26034 12097 row12097_large_left_right_right :=
  Budget.RowUpperBound.branch row12097_block_007_covers row12097_block_008_covers

def row12097_large_left_right : List Nat := row12097_block_006 ++ row12097_large_left_right_right
theorem row12097_large_left_right_covers : Budget.RowUpperBound PrimePairVertices.tree_large_left_right 26034 12097 row12097_large_left_right :=
  Budget.RowUpperBound.branch row12097_block_006_covers row12097_large_left_right_right_covers

def row12097_large_left : List Nat := row12097_large_left_left ++ row12097_large_left_right
theorem row12097_large_left_covers : Budget.RowUpperBound PrimePairVertices.tree_large_left 26034 12097 row12097_large_left :=
  Budget.RowUpperBound.branch row12097_large_left_left_covers row12097_large_left_right_covers

def row12097_large_right_left : List Nat := row12097_block_009 ++ row12097_block_010
theorem row12097_large_right_left_covers : Budget.RowUpperBound PrimePairVertices.tree_large_right_left 26034 12097 row12097_large_right_left :=
  Budget.RowUpperBound.branch row12097_block_009_covers row12097_block_010_covers

def row12097_large_right_right_right : List Nat := row12097_block_012 ++ row12097_block_013
theorem row12097_large_right_right_right_covers : Budget.RowUpperBound PrimePairVertices.tree_large_right_right_right 26034 12097 row12097_large_right_right_right :=
  Budget.RowUpperBound.branch row12097_block_012_covers row12097_block_013_covers

def row12097_large_right_right : List Nat := row12097_block_011 ++ row12097_large_right_right_right
theorem row12097_large_right_right_covers : Budget.RowUpperBound PrimePairVertices.tree_large_right_right 26034 12097 row12097_large_right_right :=
  Budget.RowUpperBound.branch row12097_block_011_covers row12097_large_right_right_right_covers

def row12097_large_right : List Nat := row12097_large_right_left ++ row12097_large_right_right
theorem row12097_large_right_covers : Budget.RowUpperBound PrimePairVertices.tree_large_right 26034 12097 row12097_large_right :=
  Budget.RowUpperBound.branch row12097_large_right_left_covers row12097_large_right_right_covers

def row12097_large : List Nat := row12097_large_left ++ row12097_large_right
theorem row12097_large_covers : Budget.RowUpperBound PrimePairVertices.tree_large 26034 12097 row12097_large :=
  Budget.RowUpperBound.branch row12097_large_left_covers row12097_large_right_covers

def row12097 : List Nat := row12097_large
theorem row12097_covers : Budget.RowUpperBound PrimePairVertices.largeTree 26034 12097 row12097 := row12097_large_covers

def row12101_small : List Nat := []
theorem row12101_small_covers : Budget.RowUpperBound PrimePairVertices.tree_small 26034 12101 row12101_small :=
  Budget.empty_row_before _ _ _ 8389 (fun q member => (Bounds.small q member).2) (by decide)

def row12101_block_005 : List Nat := []
theorem row12101_block_005_covers : Budget.RowUpperBound PrimePairVertices.tree_block_005 26034 12101 row12101_block_005 :=
  Budget.empty_row_before _ _ _ 10370 (fun q member => (Bounds.block_005 q member).2) (by decide)

def row12101_block_006 : List Nat := [12251]
def plan12101_block_006 : List Budget.Choice := [.outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .first, .first, (.firstFactor 7), .first, .first, (.secondFactor 13), .first, .first, .first, (.firstFactor 7), (.firstFactor 29), .first, (.secondFactor 13), (.firstFactor 7), .first, .possible, .first, (.secondFactor 7), (.firstFactor 37), .first, (.firstFactor 7), .first, .first, (.firstFactor 7), .first, .first, (.secondFactor 7), .first, (.firstFactor 17), .first, .first, .first, .first, (.firstFactor 19)]
set_option maxRecDepth 2048 in
theorem row12101_block_006_covers : Budget.RowUpperBound PrimePairVertices.tree_block_006 26034 12101 row12101_block_006 := by
  apply Budget.row_upper_of_plan _ _ _ plan12101_block_006
  · decide
  · rfl

def row12101_block_007 : List Nat := [12437, 12689]
def plan12101_block_007 : List Budget.Choice := [.first, .first, .possible, .first, .first, (.secondFactor 7), (.firstFactor 17), .first, (.firstFactor 7), (.firstFactor 11), .first, .first, .first, (.firstFactor 13), (.secondFactor 13), .first, .first, .first, .first, .first, .first, .first, .first, .first, .first, .first, .first, (.firstFactor 19), .first, .second, (.firstFactor 7), (.secondFactor 37), .possible, .first, .first, (.firstFactor 37), .first, .first, (.firstFactor 7), .first, .first, .first, .second, .first, (.secondFactor 7), .second, .first, .first, .first, .first, .first, (.firstFactor 11), (.firstFactor 29), .first, (.firstFactor 7), (.firstFactor 13), .first, .first, .second, (.firstFactor 7), (.firstFactor 11), .first, .first, .first, (.firstFactor 19), .second, .first, (.firstFactor 23), .first, .first, (.firstFactor 7), .first, (.secondFactor 29), .first, .first, .first, (.secondFactor 7), .second, (.firstFactor 7), .second, .first, (.firstFactor 13), .first, (.firstFactor 7), .first, .first, .first, (.secondFactor 7), .first, .first, (.firstFactor 13), (.firstFactor 31), .first, .first, .first, .first, .first, .first, (.secondFactor 7), .first, (.firstFactor 7), (.secondFactor 17), .first, .first, .first, (.firstFactor 17), .first, .first, .first, (.firstFactor 11), .first, .second, (.firstFactor 7), (.firstFactor 13), .first, .first, (.firstFactor 11), (.firstFactor 7), .first, (.secondFactor 7), .first, (.firstFactor 11), .first, .first, .first, .first, (.firstFactor 31), (.firstFactor 11), .first, .first, (.secondFactor 7), .first, .first, .first, .first, (.secondFactor 7), .first, (.firstFactor 13), (.firstFactor 7), .first, .first, .first, .first, (.firstFactor 7), .first, .first, (.secondFactor 19), (.firstFactor 19), .first, (.firstFactor 31), .first, .first, .first, (.secondFactor 23), (.secondFactor 7), .first, (.firstFactor 7), .first, (.firstFactor 11), (.secondFactor 7), .first, (.firstFactor 17), .first, .first, (.firstFactor 13), .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside]
set_option maxRecDepth 2048 in
theorem row12101_block_007_covers : Budget.RowUpperBound PrimePairVertices.tree_block_007 26034 12101 row12101_block_007 := by
  apply Budget.row_upper_of_plan _ _ _ plan12101_block_007
  · decide
  · rfl

def row12101_block_008 : List Nat := []
theorem row12101_block_008_covers : Budget.RowUpperBound PrimePairVertices.tree_block_008 26034 12101 row12101_block_008 :=
  Budget.empty_row_after _ _ _ 14466 (fun q member => (Bounds.block_008 q member).1) (by decide)

def row12101_block_009 : List Nat := []
theorem row12101_block_009_covers : Budget.RowUpperBound PrimePairVertices.tree_block_009 26034 12101 row12101_block_009 :=
  Budget.empty_row_after _ _ _ 16514 (fun q member => (Bounds.block_009 q member).1) (by decide)

def row12101_block_010 : List Nat := []
theorem row12101_block_010_covers : Budget.RowUpperBound PrimePairVertices.tree_block_010 26034 12101 row12101_block_010 :=
  Budget.empty_row_after _ _ _ 18562 (fun q member => (Bounds.block_010 q member).1) (by decide)

def row12101_block_011 : List Nat := []
theorem row12101_block_011_covers : Budget.RowUpperBound PrimePairVertices.tree_block_011 26034 12101 row12101_block_011 :=
  Budget.empty_row_after _ _ _ 20610 (fun q member => (Bounds.block_011 q member).1) (by decide)

def row12101_block_012 : List Nat := []
theorem row12101_block_012_covers : Budget.RowUpperBound PrimePairVertices.tree_block_012 26034 12101 row12101_block_012 :=
  Budget.empty_row_after _ _ _ 22658 (fun q member => (Bounds.block_012 q member).1) (by decide)

def row12101_block_013 : List Nat := []
theorem row12101_block_013_covers : Budget.RowUpperBound PrimePairVertices.tree_block_013 26034 12101 row12101_block_013 :=
  Budget.empty_row_after _ _ _ 24706 (fun q member => (Bounds.block_013 q member).1) (by decide)

def row12101_large_left_left : List Nat := row12101_small ++ row12101_block_005
theorem row12101_large_left_left_covers : Budget.RowUpperBound PrimePairVertices.tree_large_left_left 26034 12101 row12101_large_left_left :=
  Budget.RowUpperBound.branch row12101_small_covers row12101_block_005_covers

def row12101_large_left_right_right : List Nat := row12101_block_007 ++ row12101_block_008
theorem row12101_large_left_right_right_covers : Budget.RowUpperBound PrimePairVertices.tree_large_left_right_right 26034 12101 row12101_large_left_right_right :=
  Budget.RowUpperBound.branch row12101_block_007_covers row12101_block_008_covers

def row12101_large_left_right : List Nat := row12101_block_006 ++ row12101_large_left_right_right
theorem row12101_large_left_right_covers : Budget.RowUpperBound PrimePairVertices.tree_large_left_right 26034 12101 row12101_large_left_right :=
  Budget.RowUpperBound.branch row12101_block_006_covers row12101_large_left_right_right_covers

def row12101_large_left : List Nat := row12101_large_left_left ++ row12101_large_left_right
theorem row12101_large_left_covers : Budget.RowUpperBound PrimePairVertices.tree_large_left 26034 12101 row12101_large_left :=
  Budget.RowUpperBound.branch row12101_large_left_left_covers row12101_large_left_right_covers

def row12101_large_right_left : List Nat := row12101_block_009 ++ row12101_block_010
theorem row12101_large_right_left_covers : Budget.RowUpperBound PrimePairVertices.tree_large_right_left 26034 12101 row12101_large_right_left :=
  Budget.RowUpperBound.branch row12101_block_009_covers row12101_block_010_covers

def row12101_large_right_right_right : List Nat := row12101_block_012 ++ row12101_block_013
theorem row12101_large_right_right_right_covers : Budget.RowUpperBound PrimePairVertices.tree_large_right_right_right 26034 12101 row12101_large_right_right_right :=
  Budget.RowUpperBound.branch row12101_block_012_covers row12101_block_013_covers

def row12101_large_right_right : List Nat := row12101_block_011 ++ row12101_large_right_right_right
theorem row12101_large_right_right_covers : Budget.RowUpperBound PrimePairVertices.tree_large_right_right 26034 12101 row12101_large_right_right :=
  Budget.RowUpperBound.branch row12101_block_011_covers row12101_large_right_right_right_covers

def row12101_large_right : List Nat := row12101_large_right_left ++ row12101_large_right_right
theorem row12101_large_right_covers : Budget.RowUpperBound PrimePairVertices.tree_large_right 26034 12101 row12101_large_right :=
  Budget.RowUpperBound.branch row12101_large_right_left_covers row12101_large_right_right_covers

def row12101_large : List Nat := row12101_large_left ++ row12101_large_right
theorem row12101_large_covers : Budget.RowUpperBound PrimePairVertices.tree_large 26034 12101 row12101_large :=
  Budget.RowUpperBound.branch row12101_large_left_covers row12101_large_right_covers

def row12101 : List Nat := row12101_large
theorem row12101_covers : Budget.RowUpperBound PrimePairVertices.largeTree 26034 12101 row12101 := row12101_large_covers

def row12107_small : List Nat := []
theorem row12107_small_covers : Budget.RowUpperBound PrimePairVertices.tree_small 26034 12107 row12107_small :=
  Budget.empty_row_before _ _ _ 8389 (fun q member => (Bounds.small q member).2) (by decide)

def row12107_block_005 : List Nat := []
theorem row12107_block_005_covers : Budget.RowUpperBound PrimePairVertices.tree_block_005 26034 12107 row12107_block_005 :=
  Budget.empty_row_before _ _ _ 10370 (fun q member => (Bounds.block_005 q member).2) (by decide)

def row12107_block_006 : List Nat := []
def plan12107_block_006 : List Budget.Choice := [.outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .first, (.firstFactor 13), (.secondFactor 7), .first, .first, .first, (.firstFactor 31), .first, (.secondFactor 17), (.firstFactor 23), .first, .first, (.firstFactor 11), .first, (.firstFactor 7), .first, .second, (.firstFactor 13), .first, (.firstFactor 19), .first, .first, (.secondFactor 13), (.firstFactor 29), .first, (.firstFactor 13), .first, (.firstFactor 7), .first, .first, (.secondFactor 13), .first, (.secondFactor 7)]
set_option maxRecDepth 2048 in
theorem row12107_block_006_covers : Budget.RowUpperBound PrimePairVertices.tree_block_006 26034 12107 row12107_block_006 := by
  apply Budget.row_upper_of_plan _ _ _ plan12107_block_006
  · decide
  · rfl

def row12107_block_007 : List Nat := [12491, 12653, 13487]
def plan12107_block_007 : List Budget.Choice := [.first, .first, (.firstFactor 11), .first, .first, .first, (.firstFactor 23), .first, .possible, (.secondFactor 7), (.firstFactor 7), .first, .first, .first, (.secondFactor 7), .first, .first, .first, (.firstFactor 11), .first, .first, .first, .first, (.secondFactor 37), .first, .first, .first, .second, (.firstFactor 17), .possible, (.firstFactor 13), (.firstFactor 7), .first, .first, .first, (.firstFactor 7), .first, .first, .second, .first, .first, .first, (.secondFactor 7), .first, (.secondFactor 17), .first, .first, .first, .first, .first, .first, (.firstFactor 13), (.firstFactor 11), .first, (.secondFactor 17), (.secondFactor 7), .first, (.firstFactor 7), .second, (.firstFactor 17), (.secondFactor 7), .first, .first, .first, (.secondFactor 29), (.firstFactor 37), .first, (.firstFactor 7), .first, .first, .second, (.secondFactor 7), (.firstFactor 7), .first, .first, .first, (.secondFactor 13), .first, (.secondFactor 31), (.firstFactor 13), .first, .first, .first, (.firstFactor 11), .first, .first, .first, (.secondFactor 23), (.firstFactor 7), .first, (.firstFactor 11), .first, .first, (.firstFactor 7), .first, .first, .first, .first, .first, .first, (.secondFactor 29), (.secondFactor 7), .first, .first, .first, .first, .first, .first, .first, (.firstFactor 19), .first, .second, (.secondFactor 19), (.firstFactor 17), (.firstFactor 7), .first, .possible, (.secondFactor 37), .first, (.secondFactor 17), .first, (.firstFactor 7), .first, .first, .first, .first, .first, .first, .first, .first, (.firstFactor 19), .first, (.firstFactor 7), .first, .first, (.firstFactor 11), .first, .first, .first, .first, (.firstFactor 7), .first, .first, (.firstFactor 13), (.firstFactor 11), .first, (.firstFactor 7), .first, .first, (.secondFactor 7), .first, (.firstFactor 13), .first, (.secondFactor 7), (.firstFactor 23), .first, (.firstFactor 19), .first, (.secondFactor 7), .first, .first, (.firstFactor 13), (.secondFactor 19), .first, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside]
set_option maxRecDepth 2048 in
theorem row12107_block_007_covers : Budget.RowUpperBound PrimePairVertices.tree_block_007 26034 12107 row12107_block_007 := by
  apply Budget.row_upper_of_plan _ _ _ plan12107_block_007
  · decide
  · rfl

def row12107_block_008 : List Nat := []
theorem row12107_block_008_covers : Budget.RowUpperBound PrimePairVertices.tree_block_008 26034 12107 row12107_block_008 :=
  Budget.empty_row_after _ _ _ 14466 (fun q member => (Bounds.block_008 q member).1) (by decide)

def row12107_block_009 : List Nat := []
theorem row12107_block_009_covers : Budget.RowUpperBound PrimePairVertices.tree_block_009 26034 12107 row12107_block_009 :=
  Budget.empty_row_after _ _ _ 16514 (fun q member => (Bounds.block_009 q member).1) (by decide)

def row12107_block_010 : List Nat := []
theorem row12107_block_010_covers : Budget.RowUpperBound PrimePairVertices.tree_block_010 26034 12107 row12107_block_010 :=
  Budget.empty_row_after _ _ _ 18562 (fun q member => (Bounds.block_010 q member).1) (by decide)

def row12107_block_011 : List Nat := []
theorem row12107_block_011_covers : Budget.RowUpperBound PrimePairVertices.tree_block_011 26034 12107 row12107_block_011 :=
  Budget.empty_row_after _ _ _ 20610 (fun q member => (Bounds.block_011 q member).1) (by decide)

def row12107_block_012 : List Nat := []
theorem row12107_block_012_covers : Budget.RowUpperBound PrimePairVertices.tree_block_012 26034 12107 row12107_block_012 :=
  Budget.empty_row_after _ _ _ 22658 (fun q member => (Bounds.block_012 q member).1) (by decide)

def row12107_block_013 : List Nat := []
theorem row12107_block_013_covers : Budget.RowUpperBound PrimePairVertices.tree_block_013 26034 12107 row12107_block_013 :=
  Budget.empty_row_after _ _ _ 24706 (fun q member => (Bounds.block_013 q member).1) (by decide)

def row12107_large_left_left : List Nat := row12107_small ++ row12107_block_005
theorem row12107_large_left_left_covers : Budget.RowUpperBound PrimePairVertices.tree_large_left_left 26034 12107 row12107_large_left_left :=
  Budget.RowUpperBound.branch row12107_small_covers row12107_block_005_covers

def row12107_large_left_right_right : List Nat := row12107_block_007 ++ row12107_block_008
theorem row12107_large_left_right_right_covers : Budget.RowUpperBound PrimePairVertices.tree_large_left_right_right 26034 12107 row12107_large_left_right_right :=
  Budget.RowUpperBound.branch row12107_block_007_covers row12107_block_008_covers

def row12107_large_left_right : List Nat := row12107_block_006 ++ row12107_large_left_right_right
theorem row12107_large_left_right_covers : Budget.RowUpperBound PrimePairVertices.tree_large_left_right 26034 12107 row12107_large_left_right :=
  Budget.RowUpperBound.branch row12107_block_006_covers row12107_large_left_right_right_covers

def row12107_large_left : List Nat := row12107_large_left_left ++ row12107_large_left_right
theorem row12107_large_left_covers : Budget.RowUpperBound PrimePairVertices.tree_large_left 26034 12107 row12107_large_left :=
  Budget.RowUpperBound.branch row12107_large_left_left_covers row12107_large_left_right_covers

def row12107_large_right_left : List Nat := row12107_block_009 ++ row12107_block_010
theorem row12107_large_right_left_covers : Budget.RowUpperBound PrimePairVertices.tree_large_right_left 26034 12107 row12107_large_right_left :=
  Budget.RowUpperBound.branch row12107_block_009_covers row12107_block_010_covers

def row12107_large_right_right_right : List Nat := row12107_block_012 ++ row12107_block_013
theorem row12107_large_right_right_right_covers : Budget.RowUpperBound PrimePairVertices.tree_large_right_right_right 26034 12107 row12107_large_right_right_right :=
  Budget.RowUpperBound.branch row12107_block_012_covers row12107_block_013_covers

def row12107_large_right_right : List Nat := row12107_block_011 ++ row12107_large_right_right_right
theorem row12107_large_right_right_covers : Budget.RowUpperBound PrimePairVertices.tree_large_right_right 26034 12107 row12107_large_right_right :=
  Budget.RowUpperBound.branch row12107_block_011_covers row12107_large_right_right_right_covers

def row12107_large_right : List Nat := row12107_large_right_left ++ row12107_large_right_right
theorem row12107_large_right_covers : Budget.RowUpperBound PrimePairVertices.tree_large_right 26034 12107 row12107_large_right :=
  Budget.RowUpperBound.branch row12107_large_right_left_covers row12107_large_right_right_covers

def row12107_large : List Nat := row12107_large_left ++ row12107_large_right
theorem row12107_large_covers : Budget.RowUpperBound PrimePairVertices.tree_large 26034 12107 row12107_large :=
  Budget.RowUpperBound.branch row12107_large_left_covers row12107_large_right_covers

def row12107 : List Nat := row12107_large
theorem row12107_covers : Budget.RowUpperBound PrimePairVertices.largeTree 26034 12107 row12107 := row12107_large_covers

def row12109_small : List Nat := []
theorem row12109_small_covers : Budget.RowUpperBound PrimePairVertices.tree_small 26034 12109 row12109_small :=
  Budget.empty_row_before _ _ _ 8389 (fun q member => (Bounds.small q member).2) (by decide)

def row12109_block_005 : List Nat := []
theorem row12109_block_005_covers : Budget.RowUpperBound PrimePairVertices.tree_block_005 26034 12109 row12109_block_005 :=
  Budget.empty_row_before _ _ _ 10370 (fun q member => (Bounds.block_005 q member).2) (by decide)

def row12109_block_006 : List Nat := []
def plan12109_block_006 : List Budget.Choice := [.outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .first, .first, .first, .first, (.firstFactor 7), .first, .first, .first, .first, (.firstFactor 23), .first, .first, (.firstFactor 7), .first, (.secondFactor 7), .first, .first, (.secondFactor 13), .first, .first, .first, .first, .first, .second, .first, (.firstFactor 11), .first, (.secondFactor 7), (.firstFactor 13), .first, (.firstFactor 7), .first]
set_option maxRecDepth 2048 in
theorem row12109_block_006_covers : Budget.RowUpperBound PrimePairVertices.tree_block_006 26034 12109 row12109_block_006 := by
  apply Budget.row_upper_of_plan _ _ _ plan12109_block_006
  · decide
  · rfl

def row12109_block_007 : List Nat := [13063]
def plan12109_block_007 : List Budget.Choice := [(.firstFactor 31), (.secondFactor 13), .first, (.firstFactor 7), .first, .first, .first, (.firstFactor 23), .first, .first, .first, (.secondFactor 13), (.secondFactor 37), .first, .first, (.firstFactor 19), (.firstFactor 13), (.secondFactor 31), .first, (.firstFactor 7), .first, (.secondFactor 7), (.firstFactor 17), .first, (.secondFactor 29), (.firstFactor 7), (.firstFactor 11), .first, .first, .first, .first, .first, .first, .first, (.firstFactor 7), .first, .first, (.secondFactor 23), .first, (.secondFactor 7), (.firstFactor 23), (.firstFactor 13), .first, (.secondFactor 7), .first, .first, (.secondFactor 13), (.firstFactor 7), (.secondFactor 7), .first, .first, .first, .first, (.firstFactor 17), .first, .first, .first, .first, .first, .first, .first, (.firstFactor 11), (.secondFactor 17), (.firstFactor 31), .first, .first, .first, .first, (.firstFactor 17), (.firstFactor 11), .first, .first, .first, .possible, (.firstFactor 13), (.firstFactor 11), .first, .first, .first, .first, .first, .first, .first, .first, (.firstFactor 13), (.firstFactor 23), (.firstFactor 29), .first, .first, (.secondFactor 7), .first, .first, (.firstFactor 7), .first, .first, (.firstFactor 7), (.firstFactor 11), (.secondFactor 29), .first, (.firstFactor 13), .first, .first, (.firstFactor 19), .first, (.secondFactor 17), .first, .first, .first, (.firstFactor 7), .first, .first, .first, .first, .first, .first, .first, .first, .first, (.secondFactor 7), .first, (.firstFactor 31), .first, (.firstFactor 19), .first, (.firstFactor 23), (.secondFactor 7), .first, .first, (.firstFactor 7), .second, .first, (.firstFactor 7), .first, (.firstFactor 19), (.secondFactor 17), .first, (.firstFactor 11), .first, .first, (.firstFactor 7), .first, (.firstFactor 17), (.firstFactor 23), .first, .first, (.firstFactor 11), .first, .first, (.secondFactor 17), .first, (.secondFactor 7), .first, (.secondFactor 29), .first, .first, (.firstFactor 13), .first, (.firstFactor 7), .first, .first, .first, .first, .first, (.firstFactor 7), .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside]
set_option maxRecDepth 2048 in
theorem row12109_block_007_covers : Budget.RowUpperBound PrimePairVertices.tree_block_007 26034 12109 row12109_block_007 := by
  apply Budget.row_upper_of_plan _ _ _ plan12109_block_007
  · decide
  · rfl

def row12109_block_008 : List Nat := []
theorem row12109_block_008_covers : Budget.RowUpperBound PrimePairVertices.tree_block_008 26034 12109 row12109_block_008 :=
  Budget.empty_row_after _ _ _ 14466 (fun q member => (Bounds.block_008 q member).1) (by decide)

def row12109_block_009 : List Nat := []
theorem row12109_block_009_covers : Budget.RowUpperBound PrimePairVertices.tree_block_009 26034 12109 row12109_block_009 :=
  Budget.empty_row_after _ _ _ 16514 (fun q member => (Bounds.block_009 q member).1) (by decide)

def row12109_block_010 : List Nat := []
theorem row12109_block_010_covers : Budget.RowUpperBound PrimePairVertices.tree_block_010 26034 12109 row12109_block_010 :=
  Budget.empty_row_after _ _ _ 18562 (fun q member => (Bounds.block_010 q member).1) (by decide)

def row12109_block_011 : List Nat := []
theorem row12109_block_011_covers : Budget.RowUpperBound PrimePairVertices.tree_block_011 26034 12109 row12109_block_011 :=
  Budget.empty_row_after _ _ _ 20610 (fun q member => (Bounds.block_011 q member).1) (by decide)

def row12109_block_012 : List Nat := []
theorem row12109_block_012_covers : Budget.RowUpperBound PrimePairVertices.tree_block_012 26034 12109 row12109_block_012 :=
  Budget.empty_row_after _ _ _ 22658 (fun q member => (Bounds.block_012 q member).1) (by decide)

def row12109_block_013 : List Nat := []
theorem row12109_block_013_covers : Budget.RowUpperBound PrimePairVertices.tree_block_013 26034 12109 row12109_block_013 :=
  Budget.empty_row_after _ _ _ 24706 (fun q member => (Bounds.block_013 q member).1) (by decide)

def row12109_large_left_left : List Nat := row12109_small ++ row12109_block_005
theorem row12109_large_left_left_covers : Budget.RowUpperBound PrimePairVertices.tree_large_left_left 26034 12109 row12109_large_left_left :=
  Budget.RowUpperBound.branch row12109_small_covers row12109_block_005_covers

def row12109_large_left_right_right : List Nat := row12109_block_007 ++ row12109_block_008
theorem row12109_large_left_right_right_covers : Budget.RowUpperBound PrimePairVertices.tree_large_left_right_right 26034 12109 row12109_large_left_right_right :=
  Budget.RowUpperBound.branch row12109_block_007_covers row12109_block_008_covers

def row12109_large_left_right : List Nat := row12109_block_006 ++ row12109_large_left_right_right
theorem row12109_large_left_right_covers : Budget.RowUpperBound PrimePairVertices.tree_large_left_right 26034 12109 row12109_large_left_right :=
  Budget.RowUpperBound.branch row12109_block_006_covers row12109_large_left_right_right_covers

def row12109_large_left : List Nat := row12109_large_left_left ++ row12109_large_left_right
theorem row12109_large_left_covers : Budget.RowUpperBound PrimePairVertices.tree_large_left 26034 12109 row12109_large_left :=
  Budget.RowUpperBound.branch row12109_large_left_left_covers row12109_large_left_right_covers

def row12109_large_right_left : List Nat := row12109_block_009 ++ row12109_block_010
theorem row12109_large_right_left_covers : Budget.RowUpperBound PrimePairVertices.tree_large_right_left 26034 12109 row12109_large_right_left :=
  Budget.RowUpperBound.branch row12109_block_009_covers row12109_block_010_covers

def row12109_large_right_right_right : List Nat := row12109_block_012 ++ row12109_block_013
theorem row12109_large_right_right_right_covers : Budget.RowUpperBound PrimePairVertices.tree_large_right_right_right 26034 12109 row12109_large_right_right_right :=
  Budget.RowUpperBound.branch row12109_block_012_covers row12109_block_013_covers

def row12109_large_right_right : List Nat := row12109_block_011 ++ row12109_large_right_right_right
theorem row12109_large_right_right_covers : Budget.RowUpperBound PrimePairVertices.tree_large_right_right 26034 12109 row12109_large_right_right :=
  Budget.RowUpperBound.branch row12109_block_011_covers row12109_large_right_right_right_covers

def row12109_large_right : List Nat := row12109_large_right_left ++ row12109_large_right_right
theorem row12109_large_right_covers : Budget.RowUpperBound PrimePairVertices.tree_large_right 26034 12109 row12109_large_right :=
  Budget.RowUpperBound.branch row12109_large_right_left_covers row12109_large_right_right_covers

def row12109_large : List Nat := row12109_large_left ++ row12109_large_right
theorem row12109_large_covers : Budget.RowUpperBound PrimePairVertices.tree_large 26034 12109 row12109_large :=
  Budget.RowUpperBound.branch row12109_large_left_covers row12109_large_right_covers

def row12109 : List Nat := row12109_large
theorem row12109_covers : Budget.RowUpperBound PrimePairVertices.largeTree 26034 12109 row12109 := row12109_large_covers

end ProjectEulerSolutions.PrimePairLarge
