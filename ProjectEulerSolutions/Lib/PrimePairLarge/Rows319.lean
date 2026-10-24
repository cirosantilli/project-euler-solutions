import ProjectEulerSolutions.Lib.PrimePairLarge.Bounds

namespace ProjectEulerSolutions.PrimePairLarge
open PrimePairCertificate

def row10457_small : List Nat := []
theorem row10457_small_covers : Budget.RowUpperBound PrimePairVertices.tree_small 26034 10457 row10457_small :=
  Budget.empty_row_before _ _ _ 8389 (fun q member => (Bounds.small q member).2) (by decide)

def row10457_block_005 : List Nat := []
theorem row10457_block_005_covers : Budget.RowUpperBound PrimePairVertices.tree_block_005 26034 10457 row10457_block_005 :=
  Budget.empty_row_before _ _ _ 10370 (fun q member => (Bounds.block_005 q member).2) (by decide)

def row10457_block_006 : List Nat := [11621, 11717, 12377]
def plan10457_block_006 : List Budget.Choice := [.outside, .outside, .outside, .outside, .outside, .outside, .outside, .first, (.firstFactor 7), .first, .first, .first, .first, .first, .first, .first, (.secondFactor 7), .first, (.firstFactor 7), .first, (.firstFactor 13), .first, (.firstFactor 17), .first, (.firstFactor 7), .first, .first, .first, .first, .first, .first, .first, (.firstFactor 23), .first, .first, .first, (.firstFactor 19), (.firstFactor 29), .first, .first, .first, .first, (.firstFactor 7), .first, .first, (.firstFactor 19), (.firstFactor 11), .second, .first, .first, (.firstFactor 7), .first, .first, .first, .first, (.secondFactor 7), .first, (.secondFactor 29), .first, .first, (.secondFactor 7), .first, .first, (.firstFactor 31), (.secondFactor 17), .first, (.firstFactor 37), .first, (.firstFactor 13), .first, .first, (.firstFactor 29), (.firstFactor 7), .first, (.firstFactor 11), .first, .first, .first, .first, .first, .second, .first, (.firstFactor 7), .first, .second, .first, .first, .first, .first, (.firstFactor 7), (.secondFactor 7), (.firstFactor 37), .first, .first, .first, .first, .first, .first, (.secondFactor 13), .first, .first, .first, .first, (.firstFactor 23), .first, (.secondFactor 19), .first, .first, (.firstFactor 11), .first, (.firstFactor 7), (.secondFactor 7), .first, .first, .first, .first, .first, .first, .second, .first, (.firstFactor 11), .first, .first, (.firstFactor 7), .first, .possible, (.firstFactor 17), .first, .first, (.firstFactor 7), .first, (.secondFactor 31), .first, .possible, .first, .first, .first, (.firstFactor 11), .first, (.firstFactor 29), .first, .second, (.firstFactor 7), (.firstFactor 23), .first, .first, .first, .first, .first, .first, .first, .first, (.secondFactor 13), (.secondFactor 7), (.firstFactor 11), .first, (.firstFactor 13), (.firstFactor 7), (.firstFactor 17), .first, .first, .first, .first, .first, (.secondFactor 23), (.firstFactor 19), .first, .first, .first, (.firstFactor 11), .first, .first, (.secondFactor 7), .first, .first, (.firstFactor 7), (.firstFactor 11), .first, (.secondFactor 7), (.firstFactor 31), (.firstFactor 7), (.secondFactor 17), .first, (.firstFactor 13), .first, (.secondFactor 7), .first, .first, (.firstFactor 7), (.firstFactor 11), .first, (.secondFactor 17), .first, .first, (.firstFactor 7), .first, (.secondFactor 7), .first, .first, (.secondFactor 7), (.firstFactor 19), .first, (.firstFactor 17), .first, .possible, .first, .first, .first, .first, .first]
set_option maxRecDepth 2048 in
theorem row10457_block_006_covers : Budget.RowUpperBound PrimePairVertices.tree_block_006 26034 10457 row10457_block_006 := by
  apply Budget.row_upper_of_plan _ _ _ plan10457_block_006
  · decide
  · rfl

def row10457_block_007 : List Nat := [12821, 13313, 14207, 14447]
def plan10457_block_007 : List Budget.Choice := [.first, .first, (.firstFactor 7), .first, .first, (.firstFactor 13), (.firstFactor 7), .first, (.firstFactor 31), .second, (.firstFactor 11), .first, .first, .second, .first, .first, .first, .first, (.firstFactor 11), .first, .first, .first, .first, (.firstFactor 37), .first, .first, .first, (.firstFactor 23), (.firstFactor 7), (.firstFactor 17), (.secondFactor 7), (.firstFactor 19), (.firstFactor 7), .first, .first, .first, .first, .first, (.secondFactor 7), .first, .first, .first, (.secondFactor 19), .first, (.secondFactor 23), .possible, .first, .first, .first, .first, .first, .first, (.firstFactor 7), .first, (.secondFactor 7), (.firstFactor 23), .first, .first, (.firstFactor 7), (.secondFactor 7), (.firstFactor 17), .first, .first, .first, (.firstFactor 7), (.firstFactor 29), .first, .second, .first, .first, (.secondFactor 7), .first, (.firstFactor 31), .first, .first, .first, .first, (.firstFactor 7), (.secondFactor 7), (.firstFactor 19), .first, (.firstFactor 7), .first, (.firstFactor 11), .first, .first, .first, (.secondFactor 31), .second, .first, (.firstFactor 11), (.firstFactor 19), .first, .first, .first, .first, .first, .first, .possible, .first, (.firstFactor 13), .second, .first, (.firstFactor 17), .first, .first, .first, .first, .first, (.firstFactor 31), .first, .first, (.secondFactor 7), .first, (.firstFactor 17), .first, (.firstFactor 7), (.firstFactor 37), .first, (.firstFactor 29), .first, .first, .first, (.secondFactor 17), .first, .first, (.firstFactor 7), .first, .first, .first, .first, .first, (.secondFactor 17), .first, .first, (.firstFactor 11), .first, (.firstFactor 7), (.secondFactor 7), .first, (.firstFactor 13), .first, .first, (.secondFactor 7), (.firstFactor 11), .first, .first, (.firstFactor 7), .first, (.firstFactor 13), .first, .second, .first, .second, .first, .first, (.firstFactor 13), .first, (.firstFactor 23), .first, .first, (.firstFactor 7), (.secondFactor 23), .first, (.secondFactor 19), .first, .first, .first, .second, .first, .first, .first, .first, (.firstFactor 7), (.secondFactor 23), .first, .first, (.firstFactor 17), .first, (.firstFactor 11), .first, .first, .first, (.firstFactor 11), (.firstFactor 7), .first, .first, .first, .possible, .first, (.firstFactor 7), .first, .first, .first, .first, (.secondFactor 31), .first, .first, (.firstFactor 7), .first, .first, (.firstFactor 7), (.firstFactor 17), .first, .first, .first, (.firstFactor 7), .first, (.firstFactor 13), .first, .first, .possible, .first, .first]
set_option maxRecDepth 2048 in
theorem row10457_block_007_covers : Budget.RowUpperBound PrimePairVertices.tree_block_007 26034 10457 row10457_block_007 := by
  apply Budget.row_upper_of_plan _ _ _ plan10457_block_007
  · decide
  · rfl

def row10457_block_008 : List Nat := [15287]
def plan10457_block_008 : List Budget.Choice := [.first, (.firstFactor 17), .first, .first, .first, (.firstFactor 7), .first, (.firstFactor 11), .first, .first, .first, .first, (.firstFactor 17), .first, (.firstFactor 7), (.secondFactor 13), .first, (.secondFactor 7), .first, .first, (.firstFactor 13), .first, .first, (.secondFactor 17), .first, (.secondFactor 7), (.firstFactor 19), .first, .first, (.firstFactor 29), (.firstFactor 7), .first, (.secondFactor 7), .first, .first, .first, (.secondFactor 13), .first, (.firstFactor 11), .first, .first, (.firstFactor 7), (.secondFactor 7), .first, .first, .first, (.firstFactor 11), .first, (.firstFactor 13), (.firstFactor 17), .first, .first, (.secondFactor 13), .first, (.firstFactor 19), (.firstFactor 7), (.firstFactor 13), .first, .first, (.secondFactor 13), .first, (.firstFactor 37), .first, .first, (.firstFactor 11), (.firstFactor 7), .first, (.firstFactor 17), (.secondFactor 17), .first, .first, (.secondFactor 7), .first, .second, .first, (.secondFactor 13), .first, .first, .first, .first, .first, (.secondFactor 31), .first, .first, (.firstFactor 23), .first, .first, .first, .possible, .first, (.secondFactor 19), .first, .first, .first, (.secondFactor 13), .first, .first, (.firstFactor 13), .first, .first, (.firstFactor 7), .first, .first, (.firstFactor 23), (.secondFactor 17), .first, .first, .first, .first, (.firstFactor 7), (.firstFactor 31), (.firstFactor 11), .first, (.firstFactor 37), .first, (.secondFactor 19), .first, .first, .first, (.secondFactor 23), .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside]
set_option maxRecDepth 2048 in
theorem row10457_block_008_covers : Budget.RowUpperBound PrimePairVertices.tree_block_008 26034 10457 row10457_block_008 := by
  apply Budget.row_upper_of_plan _ _ _ plan10457_block_008
  · decide
  · rfl

def row10457_block_009 : List Nat := []
theorem row10457_block_009_covers : Budget.RowUpperBound PrimePairVertices.tree_block_009 26034 10457 row10457_block_009 :=
  Budget.empty_row_after _ _ _ 16514 (fun q member => (Bounds.block_009 q member).1) (by decide)

def row10457_block_010 : List Nat := []
theorem row10457_block_010_covers : Budget.RowUpperBound PrimePairVertices.tree_block_010 26034 10457 row10457_block_010 :=
  Budget.empty_row_after _ _ _ 18562 (fun q member => (Bounds.block_010 q member).1) (by decide)

def row10457_block_011 : List Nat := []
theorem row10457_block_011_covers : Budget.RowUpperBound PrimePairVertices.tree_block_011 26034 10457 row10457_block_011 :=
  Budget.empty_row_after _ _ _ 20610 (fun q member => (Bounds.block_011 q member).1) (by decide)

def row10457_block_012 : List Nat := []
theorem row10457_block_012_covers : Budget.RowUpperBound PrimePairVertices.tree_block_012 26034 10457 row10457_block_012 :=
  Budget.empty_row_after _ _ _ 22658 (fun q member => (Bounds.block_012 q member).1) (by decide)

def row10457_block_013 : List Nat := []
theorem row10457_block_013_covers : Budget.RowUpperBound PrimePairVertices.tree_block_013 26034 10457 row10457_block_013 :=
  Budget.empty_row_after _ _ _ 24706 (fun q member => (Bounds.block_013 q member).1) (by decide)

def row10457_large_left_left : List Nat := row10457_small ++ row10457_block_005
theorem row10457_large_left_left_covers : Budget.RowUpperBound PrimePairVertices.tree_large_left_left 26034 10457 row10457_large_left_left :=
  Budget.RowUpperBound.branch row10457_small_covers row10457_block_005_covers

def row10457_large_left_right_right : List Nat := row10457_block_007 ++ row10457_block_008
theorem row10457_large_left_right_right_covers : Budget.RowUpperBound PrimePairVertices.tree_large_left_right_right 26034 10457 row10457_large_left_right_right :=
  Budget.RowUpperBound.branch row10457_block_007_covers row10457_block_008_covers

def row10457_large_left_right : List Nat := row10457_block_006 ++ row10457_large_left_right_right
theorem row10457_large_left_right_covers : Budget.RowUpperBound PrimePairVertices.tree_large_left_right 26034 10457 row10457_large_left_right :=
  Budget.RowUpperBound.branch row10457_block_006_covers row10457_large_left_right_right_covers

def row10457_large_left : List Nat := row10457_large_left_left ++ row10457_large_left_right
theorem row10457_large_left_covers : Budget.RowUpperBound PrimePairVertices.tree_large_left 26034 10457 row10457_large_left :=
  Budget.RowUpperBound.branch row10457_large_left_left_covers row10457_large_left_right_covers

def row10457_large_right_left : List Nat := row10457_block_009 ++ row10457_block_010
theorem row10457_large_right_left_covers : Budget.RowUpperBound PrimePairVertices.tree_large_right_left 26034 10457 row10457_large_right_left :=
  Budget.RowUpperBound.branch row10457_block_009_covers row10457_block_010_covers

def row10457_large_right_right_right : List Nat := row10457_block_012 ++ row10457_block_013
theorem row10457_large_right_right_right_covers : Budget.RowUpperBound PrimePairVertices.tree_large_right_right_right 26034 10457 row10457_large_right_right_right :=
  Budget.RowUpperBound.branch row10457_block_012_covers row10457_block_013_covers

def row10457_large_right_right : List Nat := row10457_block_011 ++ row10457_large_right_right_right
theorem row10457_large_right_right_covers : Budget.RowUpperBound PrimePairVertices.tree_large_right_right 26034 10457 row10457_large_right_right :=
  Budget.RowUpperBound.branch row10457_block_011_covers row10457_large_right_right_right_covers

def row10457_large_right : List Nat := row10457_large_right_left ++ row10457_large_right_right
theorem row10457_large_right_covers : Budget.RowUpperBound PrimePairVertices.tree_large_right 26034 10457 row10457_large_right :=
  Budget.RowUpperBound.branch row10457_large_right_left_covers row10457_large_right_right_covers

def row10457_large : List Nat := row10457_large_left ++ row10457_large_right
theorem row10457_large_covers : Budget.RowUpperBound PrimePairVertices.tree_large 26034 10457 row10457_large :=
  Budget.RowUpperBound.branch row10457_large_left_covers row10457_large_right_covers

def row10457 : List Nat := row10457_large
theorem row10457_covers : Budget.RowUpperBound PrimePairVertices.largeTree 26034 10457 row10457 := row10457_large_covers

def row10459_small : List Nat := []
theorem row10459_small_covers : Budget.RowUpperBound PrimePairVertices.tree_small 26034 10459 row10459_small :=
  Budget.empty_row_before _ _ _ 8389 (fun q member => (Bounds.small q member).2) (by decide)

def row10459_block_005 : List Nat := []
theorem row10459_block_005_covers : Budget.RowUpperBound PrimePairVertices.tree_block_005 26034 10459 row10459_block_005 :=
  Budget.empty_row_before _ _ _ 10370 (fun q member => (Bounds.block_005 q member).2) (by decide)

def row10459_block_006 : List Nat := [10501, 10687, 11173]
def plan10459_block_006 : List Budget.Choice := [.outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .first, (.secondFactor 17), .first, .first, .possible, .first, .first, .first, .first, (.firstFactor 13), .first, .second, .first, .first, .first, .first, .first, .first, (.firstFactor 19), (.firstFactor 11), (.firstFactor 7), .first, .possible, .first, .first, .first, (.firstFactor 11), .first, .first, .first, (.secondFactor 29), (.firstFactor 17), .first, (.firstFactor 7), .first, (.firstFactor 7), (.secondFactor 13), .first, .first, .first, (.secondFactor 7), .first, .first, .first, (.firstFactor 31), (.secondFactor 7), .first, .first, .first, .first, (.firstFactor 7), .first, .first, (.firstFactor 11), (.firstFactor 19), .first, .first, .second, .first, .first, .first, (.firstFactor 29), (.firstFactor 7), .first, .first, (.firstFactor 13), .first, (.firstFactor 11), (.firstFactor 23), (.secondFactor 13), .first, .first, .first, .possible, .first, (.secondFactor 7), .first, (.secondFactor 7), .first, (.firstFactor 7), (.secondFactor 23), .first, .first, .first, .first, .second, .first, (.firstFactor 11), .first, .first, .first, .first, .first, (.firstFactor 11), .first, .first, .first, .first, .first, .first, .first, .first, .first, .first, .first, (.secondFactor 7), (.secondFactor 17), (.firstFactor 7), .first, .first, .first, .first, .first, (.firstFactor 7), (.firstFactor 29), .first, (.secondFactor 7), .first, .first, .first, (.firstFactor 19), .first, (.firstFactor 17), .first, (.secondFactor 7), .first, .second, .first, (.secondFactor 7), .first, (.firstFactor 11), .first, .first, .first, .first, .first, (.firstFactor 23), (.secondFactor 7), .first, .first, (.firstFactor 7), .first, .first, .first, .first, .first, .first, (.firstFactor 7), .first, .first, .first, (.firstFactor 29), (.secondFactor 7), (.firstFactor 23), .first, (.firstFactor 13), .first, .first, (.firstFactor 7), .first, (.secondFactor 7), .first, (.firstFactor 11), (.firstFactor 7), .first, .second, (.firstFactor 17), .first, .first, (.firstFactor 11), .first, .first, .first, .first, (.secondFactor 19), .first, (.secondFactor 7), .first, .first, (.secondFactor 17), .first, .first, (.firstFactor 11), .first, .first, .first, .first, .first, .first, (.firstFactor 29), (.firstFactor 7), .first, .first, (.firstFactor 7), .first, (.firstFactor 11), .first, (.firstFactor 31), .first, .first, .first, .first]
set_option maxRecDepth 2048 in
theorem row10459_block_006_covers : Budget.RowUpperBound PrimePairVertices.tree_block_006 26034 10459 row10459_block_006 := by
  apply Budget.row_upper_of_plan _ _ _ plan10459_block_006
  · decide
  · rfl

def row10459_block_007 : List Nat := [13291, 13597, 13789, 14083, 14197, 14449]
def plan10459_block_007 : List Budget.Choice := [.second, (.secondFactor 31), .first, .second, (.secondFactor 7), .first, .first, .first, .first, .first, .first, (.firstFactor 7), (.firstFactor 13), .first, .first, (.secondFactor 7), .first, (.firstFactor 7), .first, .second, (.secondFactor 7), (.firstFactor 19), .first, .first, (.secondFactor 19), (.secondFactor 17), (.firstFactor 7), .first, .first, .first, .first, .first, .first, .second, (.firstFactor 11), .first, (.firstFactor 7), .first, .first, .first, (.firstFactor 7), (.firstFactor 37), .first, .first, .first, .first, (.secondFactor 17), (.firstFactor 13), (.secondFactor 19), .first, (.firstFactor 7), .first, .first, (.firstFactor 13), .first, .first, (.secondFactor 7), .first, .first, .first, .first, (.firstFactor 11), (.firstFactor 7), .first, .first, .first, (.firstFactor 37), .first, .second, (.firstFactor 11), .first, .first, .first, (.firstFactor 13), .first, (.firstFactor 7), .first, .first, .first, .first, .first, .first, (.firstFactor 19), .first, (.secondFactor 7), (.secondFactor 13), (.firstFactor 7), .first, .first, (.firstFactor 13), .first, .first, .first, .first, (.firstFactor 7), .possible, (.firstFactor 11), (.firstFactor 7), .first, (.secondFactor 23), .first, .first, (.firstFactor 23), .first, (.secondFactor 7), .first, .first, (.secondFactor 13), .second, .first, .first, .first, .first, .first, .first, (.firstFactor 7), .first, .first, .first, .first, (.secondFactor 17), .first, (.secondFactor 13), .first, (.secondFactor 7), .possible, .first, .first, (.firstFactor 11), (.secondFactor 7), .first, (.firstFactor 37), .first, (.firstFactor 29), (.firstFactor 7), .first, (.firstFactor 11), .first, .first, (.secondFactor 29), .first, (.secondFactor 13), (.firstFactor 7), .first, .first, (.firstFactor 11), .first, .first, .possible, .first, .first, .first, (.firstFactor 17), .first, .first, .second, .first, (.secondFactor 13), .first, .first, .second, .first, .first, (.firstFactor 13), .first, (.firstFactor 17), .first, .first, .first, (.firstFactor 13), .first, (.secondFactor 7), (.firstFactor 23), .first, .first, .first, (.firstFactor 19), .first, .possible, .first, (.firstFactor 7), (.secondFactor 37), (.firstFactor 7), .first, .first, .first, .first, .possible, .first, (.firstFactor 11), .first, .first, (.secondFactor 17), .first, (.secondFactor 23), .first, .first, (.secondFactor 19), .first, (.firstFactor 17), (.secondFactor 7), .first, .first, (.firstFactor 13), (.firstFactor 7), (.secondFactor 29), .first, (.firstFactor 11), .first, (.secondFactor 7), (.secondFactor 19), .first, .possible, .first]
set_option maxRecDepth 2048 in
theorem row10459_block_007_covers : Budget.RowUpperBound PrimePairVertices.tree_block_007 26034 10459 row10459_block_007 := by
  apply Budget.row_upper_of_plan _ _ _ plan10459_block_007
  · decide
  · rfl

def row10459_block_008 : List Nat := [14713, 14827, 15511]
def plan10459_block_008 : List Budget.Choice := [(.secondFactor 31), .first, (.secondFactor 13), .first, .first, .first, .first, .first, (.firstFactor 11), (.firstFactor 37), .first, .second, .first, .first, .first, .first, .first, .first, .first, (.firstFactor 7), .first, .first, (.firstFactor 11), .first, .possible, .first, .first, .first, (.firstFactor 7), .first, .first, .first, .first, (.secondFactor 7), .first, (.firstFactor 7), .first, (.firstFactor 31), .first, (.firstFactor 7), .possible, .first, .first, (.firstFactor 17), .first, (.firstFactor 19), .first, .first, .first, .first, .first, (.secondFactor 29), .first, (.firstFactor 7), .first, .first, .first, (.firstFactor 19), (.firstFactor 11), .first, (.firstFactor 7), .first, (.secondFactor 7), (.firstFactor 7), .first, .first, (.firstFactor 13), .first, .first, (.secondFactor 19), .first, .first, .first, .first, .first, .first, (.secondFactor 7), .first, (.firstFactor 7), .first, .first, .first, (.firstFactor 7), (.firstFactor 17), .first, .first, (.firstFactor 23), (.firstFactor 11), .first, .first, .first, .second, (.secondFactor 7), .second, .first, .first, (.secondFactor 19), .first, (.firstFactor 17), (.secondFactor 17), .first, .first, .second, .first, .first, .first, (.firstFactor 19), .first, (.firstFactor 7), .first, .first, .first, (.firstFactor 7), .first, .possible, .first, (.firstFactor 11), .first, (.firstFactor 13), .first, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside]
set_option maxRecDepth 2048 in
theorem row10459_block_008_covers : Budget.RowUpperBound PrimePairVertices.tree_block_008 26034 10459 row10459_block_008 := by
  apply Budget.row_upper_of_plan _ _ _ plan10459_block_008
  · decide
  · rfl

def row10459_block_009 : List Nat := []
theorem row10459_block_009_covers : Budget.RowUpperBound PrimePairVertices.tree_block_009 26034 10459 row10459_block_009 :=
  Budget.empty_row_after _ _ _ 16514 (fun q member => (Bounds.block_009 q member).1) (by decide)

def row10459_block_010 : List Nat := []
theorem row10459_block_010_covers : Budget.RowUpperBound PrimePairVertices.tree_block_010 26034 10459 row10459_block_010 :=
  Budget.empty_row_after _ _ _ 18562 (fun q member => (Bounds.block_010 q member).1) (by decide)

def row10459_block_011 : List Nat := []
theorem row10459_block_011_covers : Budget.RowUpperBound PrimePairVertices.tree_block_011 26034 10459 row10459_block_011 :=
  Budget.empty_row_after _ _ _ 20610 (fun q member => (Bounds.block_011 q member).1) (by decide)

def row10459_block_012 : List Nat := []
theorem row10459_block_012_covers : Budget.RowUpperBound PrimePairVertices.tree_block_012 26034 10459 row10459_block_012 :=
  Budget.empty_row_after _ _ _ 22658 (fun q member => (Bounds.block_012 q member).1) (by decide)

def row10459_block_013 : List Nat := []
theorem row10459_block_013_covers : Budget.RowUpperBound PrimePairVertices.tree_block_013 26034 10459 row10459_block_013 :=
  Budget.empty_row_after _ _ _ 24706 (fun q member => (Bounds.block_013 q member).1) (by decide)

def row10459_large_left_left : List Nat := row10459_small ++ row10459_block_005
theorem row10459_large_left_left_covers : Budget.RowUpperBound PrimePairVertices.tree_large_left_left 26034 10459 row10459_large_left_left :=
  Budget.RowUpperBound.branch row10459_small_covers row10459_block_005_covers

def row10459_large_left_right_right : List Nat := row10459_block_007 ++ row10459_block_008
theorem row10459_large_left_right_right_covers : Budget.RowUpperBound PrimePairVertices.tree_large_left_right_right 26034 10459 row10459_large_left_right_right :=
  Budget.RowUpperBound.branch row10459_block_007_covers row10459_block_008_covers

def row10459_large_left_right : List Nat := row10459_block_006 ++ row10459_large_left_right_right
theorem row10459_large_left_right_covers : Budget.RowUpperBound PrimePairVertices.tree_large_left_right 26034 10459 row10459_large_left_right :=
  Budget.RowUpperBound.branch row10459_block_006_covers row10459_large_left_right_right_covers

def row10459_large_left : List Nat := row10459_large_left_left ++ row10459_large_left_right
theorem row10459_large_left_covers : Budget.RowUpperBound PrimePairVertices.tree_large_left 26034 10459 row10459_large_left :=
  Budget.RowUpperBound.branch row10459_large_left_left_covers row10459_large_left_right_covers

def row10459_large_right_left : List Nat := row10459_block_009 ++ row10459_block_010
theorem row10459_large_right_left_covers : Budget.RowUpperBound PrimePairVertices.tree_large_right_left 26034 10459 row10459_large_right_left :=
  Budget.RowUpperBound.branch row10459_block_009_covers row10459_block_010_covers

def row10459_large_right_right_right : List Nat := row10459_block_012 ++ row10459_block_013
theorem row10459_large_right_right_right_covers : Budget.RowUpperBound PrimePairVertices.tree_large_right_right_right 26034 10459 row10459_large_right_right_right :=
  Budget.RowUpperBound.branch row10459_block_012_covers row10459_block_013_covers

def row10459_large_right_right : List Nat := row10459_block_011 ++ row10459_large_right_right_right
theorem row10459_large_right_right_covers : Budget.RowUpperBound PrimePairVertices.tree_large_right_right 26034 10459 row10459_large_right_right :=
  Budget.RowUpperBound.branch row10459_block_011_covers row10459_large_right_right_right_covers

def row10459_large_right : List Nat := row10459_large_right_left ++ row10459_large_right_right
theorem row10459_large_right_covers : Budget.RowUpperBound PrimePairVertices.tree_large_right 26034 10459 row10459_large_right :=
  Budget.RowUpperBound.branch row10459_large_right_left_covers row10459_large_right_right_covers

def row10459_large : List Nat := row10459_large_left ++ row10459_large_right
theorem row10459_large_covers : Budget.RowUpperBound PrimePairVertices.tree_large 26034 10459 row10459_large :=
  Budget.RowUpperBound.branch row10459_large_left_covers row10459_large_right_covers

def row10459 : List Nat := row10459_large
theorem row10459_covers : Budget.RowUpperBound PrimePairVertices.largeTree 26034 10459 row10459 := row10459_large_covers

def row10463_small : List Nat := []
theorem row10463_small_covers : Budget.RowUpperBound PrimePairVertices.tree_small 26034 10463 row10463_small :=
  Budget.empty_row_before _ _ _ 8389 (fun q member => (Bounds.small q member).2) (by decide)

def row10463_block_005 : List Nat := []
theorem row10463_block_005_covers : Budget.RowUpperBound PrimePairVertices.tree_block_005 26034 10463 row10463_block_005 :=
  Budget.empty_row_before _ _ _ 10370 (fun q member => (Bounds.block_005 q member).2) (by decide)

def row10463_block_006 : List Nat := [11177, 11657, 11807, 11831, 12269, 12329]
def plan10463_block_006 : List Budget.Choice := [.outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .first, (.firstFactor 19), (.firstFactor 13), .first, .first, (.firstFactor 11), .first, (.firstFactor 7), .first, (.secondFactor 13), .first, (.firstFactor 7), .first, .first, .first, .first, .first, .first, .first, .first, (.firstFactor 31), .first, (.firstFactor 29), (.secondFactor 7), .first, .first, .first, (.firstFactor 13), (.secondFactor 31), .first, .first, (.firstFactor 17), .first, .first, .first, .first, (.secondFactor 37), (.firstFactor 7), (.firstFactor 11), .first, .first, (.firstFactor 17), (.firstFactor 13), .first, .first, .first, (.firstFactor 7), .first, .first, .first, .first, (.firstFactor 7), .first, .first, (.secondFactor 7), .first, .first, (.firstFactor 11), .first, (.secondFactor 37), .first, .first, (.firstFactor 17), .first, .first, (.secondFactor 19), .first, .first, .first, .first, .first, (.firstFactor 19), .first, .possible, .first, (.firstFactor 29), .first, .first, .first, .first, .second, (.firstFactor 7), (.firstFactor 13), .first, .first, .first, .first, (.firstFactor 11), .first, .first, .first, (.secondFactor 13), .first, (.firstFactor 17), (.firstFactor 7), (.firstFactor 31), (.firstFactor 23), .first, .first, (.secondFactor 13), .first, .first, (.firstFactor 7), .first, .first, .first, .first, (.firstFactor 11), .first, (.secondFactor 7), .first, (.secondFactor 29), .first, .first, (.firstFactor 17), .first, (.secondFactor 17), (.secondFactor 7), .possible, .first, (.secondFactor 13), .first, (.firstFactor 17), .first, (.firstFactor 11), .first, .first, .first, (.firstFactor 7), .first, (.firstFactor 11), (.firstFactor 37), (.firstFactor 17), .possible, .second, .first, .first, .possible, .first, .first, .first, .second, .first, .first, (.firstFactor 7), (.firstFactor 29), .first, (.secondFactor 7), .second, .first, .first, .first, .first, (.firstFactor 19), .first, (.firstFactor 11), (.firstFactor 7), .first, (.firstFactor 37), .first, (.secondFactor 31), .first, .first, (.firstFactor 7), .first, .first, (.secondFactor 29), (.firstFactor 17), .first, (.firstFactor 7), .second, (.secondFactor 19), (.secondFactor 13), .first, .first, .first, (.firstFactor 7), .second, .first, (.secondFactor 13), (.firstFactor 7), .first, (.firstFactor 23), .first, (.secondFactor 7), .possible, .first, (.firstFactor 7), .first, .first, (.firstFactor 7), .possible, .first, (.secondFactor 7), .first, (.firstFactor 11), .first, .first, (.secondFactor 37), .first, (.firstFactor 17)]
set_option maxRecDepth 2048 in
theorem row10463_block_006_covers : Budget.RowUpperBound PrimePairVertices.tree_block_006 26034 10463 row10463_block_006 := by
  apply Budget.row_upper_of_plan _ _ _ plan10463_block_006
  · decide
  · rfl

def row10463_block_007 : List Nat := [12671, 12959]
def plan10463_block_007 : List Budget.Choice := [.first, .first, (.secondFactor 17), .first, .first, (.secondFactor 7), .first, .first, (.firstFactor 7), .first, .first, .first, .first, (.firstFactor 13), (.firstFactor 19), .first, .first, .first, (.secondFactor 23), .first, .first, .first, .first, .second, .first, .first, .first, (.firstFactor 11), .first, (.firstFactor 19), (.firstFactor 7), .possible, .first, .first, .first, (.firstFactor 31), .first, .first, (.firstFactor 7), .first, .first, .first, .second, .first, (.secondFactor 7), (.firstFactor 17), .first, .first, .first, .first, .first, (.secondFactor 7), (.firstFactor 31), .first, (.firstFactor 7), (.firstFactor 13), .first, (.firstFactor 17), (.firstFactor 23), (.firstFactor 7), .possible, .first, .first, .first, (.secondFactor 23), .first, .first, (.secondFactor 13), .first, .first, (.firstFactor 7), .first, (.secondFactor 17), .first, .first, .first, (.firstFactor 11), (.firstFactor 19), (.firstFactor 7), (.firstFactor 17), .first, (.firstFactor 13), .first, (.firstFactor 7), .first, .first, .first, (.secondFactor 7), (.firstFactor 23), .first, (.firstFactor 13), (.secondFactor 13), .first, (.secondFactor 23), .first, .first, .first, .first, (.secondFactor 7), .first, (.firstFactor 7), (.firstFactor 19), .first, (.firstFactor 11), .first, (.secondFactor 7), .first, .first, .first, .second, .first, (.firstFactor 19), (.firstFactor 7), (.firstFactor 13), .first, .first, .first, (.firstFactor 7), .first, (.secondFactor 7), .first, (.secondFactor 13), .first, .first, .first, .first, .second, (.firstFactor 13), .first, .first, (.firstFactor 29), .first, (.firstFactor 19), .first, .first, (.secondFactor 7), .first, (.firstFactor 11), (.firstFactor 7), .first, .first, .first, .first, (.firstFactor 7), .first, .first, (.firstFactor 11), .first, .first, .first, .first, (.firstFactor 11), .first, (.firstFactor 17), (.secondFactor 7), .first, (.firstFactor 7), .first, .first, (.secondFactor 7), .first, (.firstFactor 19), .first, .first, (.firstFactor 13), .first, .first, (.secondFactor 17), (.firstFactor 29), .first, (.firstFactor 13), .first, .first, .second, .first, .second, .first, (.secondFactor 19), .first, (.firstFactor 7), .first, .first, .first, (.secondFactor 7), (.firstFactor 11), .first, (.secondFactor 13), .first, .first, .first, (.firstFactor 13), (.firstFactor 17), .first, .first, .first, .second, (.firstFactor 13), .first, .second, .first, .first, .first, (.firstFactor 31), .first, .first, .first, (.secondFactor 13), .first, (.firstFactor 7), .first, .first, (.secondFactor 7), .first, .first]
set_option maxRecDepth 2048 in
theorem row10463_block_007_covers : Budget.RowUpperBound PrimePairVertices.tree_block_007 26034 10463 row10463_block_007 := by
  apply Budget.row_upper_of_plan _ _ _ plan10463_block_007
  · decide
  · rfl

def row10463_block_008 : List Nat := [14627]
def plan10463_block_008 : List Budget.Choice := [.first, (.firstFactor 11), .first, (.firstFactor 29), .first, (.secondFactor 19), .first, (.firstFactor 7), .first, .first, .second, .first, (.firstFactor 7), .first, (.firstFactor 11), .possible, .first, (.firstFactor 7), (.secondFactor 23), .first, (.firstFactor 17), .first, .first, (.secondFactor 7), .first, (.firstFactor 7), (.secondFactor 13), .first, .first, (.secondFactor 7), .first, (.firstFactor 11), (.firstFactor 7), .first, .first, .first, (.secondFactor 7), .first, .first, .first, .first, (.secondFactor 31), (.firstFactor 7), .first, (.firstFactor 13), .first, (.secondFactor 13), .first, .first, (.firstFactor 37), .first, .first, .first, .first, (.firstFactor 11), (.secondFactor 13), (.firstFactor 7), .first, .first, (.firstFactor 11), .first, (.firstFactor 7), .first, .first, (.secondFactor 7), (.firstFactor 11), .first, (.firstFactor 13), (.secondFactor 19), .first, (.firstFactor 31), (.firstFactor 7), .first, (.firstFactor 11), (.firstFactor 19), .first, .first, .first, .first, .first, .second, (.secondFactor 29), .first, .first, (.firstFactor 7), (.firstFactor 17), .first, .first, (.firstFactor 23), .first, .first, .first, .first, .first, (.secondFactor 7), .first, .first, .first, .first, .first, .second, .second, .first, .first, (.firstFactor 11), .first, .first, .first, .first, .second, (.secondFactor 23), (.firstFactor 7), .first, (.secondFactor 7), .first, .first, .first, .first, .first, (.firstFactor 13), .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside]
set_option maxRecDepth 2048 in
theorem row10463_block_008_covers : Budget.RowUpperBound PrimePairVertices.tree_block_008 26034 10463 row10463_block_008 := by
  apply Budget.row_upper_of_plan _ _ _ plan10463_block_008
  · decide
  · rfl

def row10463_block_009 : List Nat := []
theorem row10463_block_009_covers : Budget.RowUpperBound PrimePairVertices.tree_block_009 26034 10463 row10463_block_009 :=
  Budget.empty_row_after _ _ _ 16514 (fun q member => (Bounds.block_009 q member).1) (by decide)

def row10463_block_010 : List Nat := []
theorem row10463_block_010_covers : Budget.RowUpperBound PrimePairVertices.tree_block_010 26034 10463 row10463_block_010 :=
  Budget.empty_row_after _ _ _ 18562 (fun q member => (Bounds.block_010 q member).1) (by decide)

def row10463_block_011 : List Nat := []
theorem row10463_block_011_covers : Budget.RowUpperBound PrimePairVertices.tree_block_011 26034 10463 row10463_block_011 :=
  Budget.empty_row_after _ _ _ 20610 (fun q member => (Bounds.block_011 q member).1) (by decide)

def row10463_block_012 : List Nat := []
theorem row10463_block_012_covers : Budget.RowUpperBound PrimePairVertices.tree_block_012 26034 10463 row10463_block_012 :=
  Budget.empty_row_after _ _ _ 22658 (fun q member => (Bounds.block_012 q member).1) (by decide)

def row10463_block_013 : List Nat := []
theorem row10463_block_013_covers : Budget.RowUpperBound PrimePairVertices.tree_block_013 26034 10463 row10463_block_013 :=
  Budget.empty_row_after _ _ _ 24706 (fun q member => (Bounds.block_013 q member).1) (by decide)

def row10463_large_left_left : List Nat := row10463_small ++ row10463_block_005
theorem row10463_large_left_left_covers : Budget.RowUpperBound PrimePairVertices.tree_large_left_left 26034 10463 row10463_large_left_left :=
  Budget.RowUpperBound.branch row10463_small_covers row10463_block_005_covers

def row10463_large_left_right_right : List Nat := row10463_block_007 ++ row10463_block_008
theorem row10463_large_left_right_right_covers : Budget.RowUpperBound PrimePairVertices.tree_large_left_right_right 26034 10463 row10463_large_left_right_right :=
  Budget.RowUpperBound.branch row10463_block_007_covers row10463_block_008_covers

def row10463_large_left_right : List Nat := row10463_block_006 ++ row10463_large_left_right_right
theorem row10463_large_left_right_covers : Budget.RowUpperBound PrimePairVertices.tree_large_left_right 26034 10463 row10463_large_left_right :=
  Budget.RowUpperBound.branch row10463_block_006_covers row10463_large_left_right_right_covers

def row10463_large_left : List Nat := row10463_large_left_left ++ row10463_large_left_right
theorem row10463_large_left_covers : Budget.RowUpperBound PrimePairVertices.tree_large_left 26034 10463 row10463_large_left :=
  Budget.RowUpperBound.branch row10463_large_left_left_covers row10463_large_left_right_covers

def row10463_large_right_left : List Nat := row10463_block_009 ++ row10463_block_010
theorem row10463_large_right_left_covers : Budget.RowUpperBound PrimePairVertices.tree_large_right_left 26034 10463 row10463_large_right_left :=
  Budget.RowUpperBound.branch row10463_block_009_covers row10463_block_010_covers

def row10463_large_right_right_right : List Nat := row10463_block_012 ++ row10463_block_013
theorem row10463_large_right_right_right_covers : Budget.RowUpperBound PrimePairVertices.tree_large_right_right_right 26034 10463 row10463_large_right_right_right :=
  Budget.RowUpperBound.branch row10463_block_012_covers row10463_block_013_covers

def row10463_large_right_right : List Nat := row10463_block_011 ++ row10463_large_right_right_right
theorem row10463_large_right_right_covers : Budget.RowUpperBound PrimePairVertices.tree_large_right_right 26034 10463 row10463_large_right_right :=
  Budget.RowUpperBound.branch row10463_block_011_covers row10463_large_right_right_right_covers

def row10463_large_right : List Nat := row10463_large_right_left ++ row10463_large_right_right
theorem row10463_large_right_covers : Budget.RowUpperBound PrimePairVertices.tree_large_right 26034 10463 row10463_large_right :=
  Budget.RowUpperBound.branch row10463_large_right_left_covers row10463_large_right_right_covers

def row10463_large : List Nat := row10463_large_left ++ row10463_large_right
theorem row10463_large_covers : Budget.RowUpperBound PrimePairVertices.tree_large 26034 10463 row10463_large :=
  Budget.RowUpperBound.branch row10463_large_left_covers row10463_large_right_covers

def row10463 : List Nat := row10463_large
theorem row10463_covers : Budget.RowUpperBound PrimePairVertices.largeTree 26034 10463 row10463 := row10463_large_covers

def row10477_small : List Nat := []
theorem row10477_small_covers : Budget.RowUpperBound PrimePairVertices.tree_small 26034 10477 row10477_small :=
  Budget.empty_row_before _ _ _ 8389 (fun q member => (Bounds.small q member).2) (by decide)

def row10477_block_005 : List Nat := []
theorem row10477_block_005_covers : Budget.RowUpperBound PrimePairVertices.tree_block_005 26034 10477 row10477_block_005 :=
  Budget.empty_row_before _ _ _ 10370 (fun q member => (Bounds.block_005 q member).2) (by decide)

def row10477_block_006 : List Nat := [10567, 11131, 11551, 12097]
def plan10477_block_006 : List Budget.Choice := [.outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .first, .first, (.secondFactor 13), (.firstFactor 23), .first, (.firstFactor 7), .first, .possible, .first, (.firstFactor 19), .first, .first, .first, (.firstFactor 31), .first, (.secondFactor 7), (.firstFactor 13), (.firstFactor 7), (.firstFactor 17), .first, .second, .first, .first, (.firstFactor 19), (.secondFactor 7), (.firstFactor 13), .first, .first, (.firstFactor 29), .second, .first, (.firstFactor 23), .first, (.secondFactor 17), .first, .first, .first, .first, .first, (.firstFactor 7), .first, .first, (.secondFactor 7), (.secondFactor 29), (.firstFactor 7), .first, (.firstFactor 11), .first, .second, .first, .first, .first, (.firstFactor 7), .first, .first, (.secondFactor 13), .first, (.secondFactor 7), .first, (.firstFactor 11), .first, .first, .first, .first, .first, (.firstFactor 7), .possible, .second, .first, (.firstFactor 7), .first, (.firstFactor 17), .first, (.firstFactor 13), .first, (.secondFactor 17), .first, (.secondFactor 29), .first, .first, .first, .first, (.firstFactor 7), (.secondFactor 37), (.secondFactor 7), .second, .first, (.firstFactor 7), .first, (.firstFactor 13), .first, .first, .first, .first, .first, .first, (.secondFactor 7), (.secondFactor 17), .first, (.firstFactor 11), .first, .first, .first, .second, (.firstFactor 7), (.secondFactor 23), .first, .first, .first, .possible, .first, (.firstFactor 13), (.secondFactor 13), .first, (.firstFactor 23), .first, .first, .first, .first, .first, (.secondFactor 7), .first, .first, .first, .second, (.firstFactor 11), (.firstFactor 13), .first, (.secondFactor 23), .first, .first, .first, .first, .first, (.firstFactor 13), (.secondFactor 13), .first, (.firstFactor 7), .first, (.firstFactor 11), .first, (.firstFactor 17), .first, .first, .first, (.secondFactor 19), .first, .first, .first, (.secondFactor 7), (.secondFactor 17), (.firstFactor 7), .first, (.firstFactor 29), .first, .first, .first, .first, (.secondFactor 19), .first, (.firstFactor 7), .first, .first, .first, .possible, .first, .first, (.secondFactor 7), .first, .first, .first, .first, (.secondFactor 17), .first, .first, .first, .first, (.firstFactor 7), .first, .first, .first, .first, (.firstFactor 7), .first, .first, (.secondFactor 7), .first, (.firstFactor 13), (.firstFactor 31), .first, .first, (.secondFactor 31), .first, (.secondFactor 13), .first, (.firstFactor 7), (.firstFactor 11), .first, (.secondFactor 37), .first]
set_option maxRecDepth 2048 in
theorem row10477_block_006_covers : Budget.RowUpperBound PrimePairVertices.tree_block_006 26034 10477 row10477_block_006 := by
  apply Budget.row_upper_of_plan _ _ _ plan10477_block_006
  · decide
  · rfl

def row10477_block_007 : List Nat := [13171, 13267, 13441, 13729, 13921]
def plan10477_block_007 : List Budget.Choice := [(.firstFactor 7), .first, .first, (.secondFactor 13), (.firstFactor 11), .first, .first, (.firstFactor 31), .first, .first, .first, .first, .first, .first, .first, .first, (.firstFactor 7), .second, .first, .first, (.firstFactor 23), (.firstFactor 7), (.firstFactor 13), .first, (.secondFactor 7), .first, .first, .first, .first, .first, .first, .first, .first, (.secondFactor 7), (.firstFactor 17), .first, (.firstFactor 11), (.secondFactor 7), .first, (.firstFactor 7), (.firstFactor 19), (.secondFactor 7), .first, (.firstFactor 7), .first, .first, (.secondFactor 7), .first, (.firstFactor 7), (.firstFactor 11), .first, .first, .first, (.firstFactor 17), .first, .first, (.firstFactor 11), .first, .first, .first, .first, (.firstFactor 7), (.secondFactor 17), (.firstFactor 37), .first, .first, .first, .first, (.firstFactor 7), (.secondFactor 7), .first, .first, .first, (.secondFactor 19), (.firstFactor 7), .first, .first, .first, .first, .first, (.firstFactor 13), .first, (.secondFactor 7), .first, .possible, (.firstFactor 7), (.firstFactor 11), .first, .first, (.firstFactor 7), .first, .first, (.firstFactor 11), .first, .possible, (.secondFactor 19), (.secondFactor 23), (.secondFactor 13), .first, (.secondFactor 7), .first, .first, (.secondFactor 29), .first, (.firstFactor 11), .first, .first, (.firstFactor 23), (.firstFactor 17), .first, .possible, .first, .first, .first, .first, .first, .first, .first, (.firstFactor 7), .first, (.firstFactor 13), .first, .first, .first, .first, (.firstFactor 7), .first, .first, .first, (.secondFactor 19), .first, .second, .first, (.firstFactor 7), (.firstFactor 23), .first, (.firstFactor 13), .first, .first, (.firstFactor 11), .first, (.firstFactor 7), .possible, .first, .first, .second, .first, .first, (.firstFactor 19), .first, (.firstFactor 7), .first, (.secondFactor 7), .first, .first, (.secondFactor 7), .first, .first, .first, .first, (.firstFactor 19), .first, .first, .possible, .first, (.firstFactor 7), (.firstFactor 23), .first, .first, (.secondFactor 7), .first, (.secondFactor 13), (.firstFactor 17), .first, .first, .first, .first, .first, (.firstFactor 13), .first, (.firstFactor 11), (.firstFactor 7), .second, .first, .first, (.firstFactor 11), .first, (.secondFactor 17), .first, .first, .first, .first, (.secondFactor 7), .first, (.secondFactor 7), .first, .first, (.secondFactor 13), .first, .second, (.firstFactor 31), .first, .first, (.secondFactor 31), (.secondFactor 13), (.firstFactor 29), .first, (.secondFactor 7), .first, (.secondFactor 19), (.firstFactor 7), .first, .first, (.secondFactor 7)]
set_option maxRecDepth 2048 in
theorem row10477_block_007_covers : Budget.RowUpperBound PrimePairVertices.tree_block_007 26034 10477 row10477_block_007 := by
  apply Budget.row_upper_of_plan _ _ _ plan10477_block_007
  · decide
  · rfl

def row10477_block_008 : List Nat := [15349, 15439]
def plan10477_block_008 : List Budget.Choice := [(.firstFactor 7), .first, (.firstFactor 11), .first, (.firstFactor 31), .first, .first, .first, (.firstFactor 13), (.secondFactor 13), .first, (.firstFactor 7), .first, .first, .first, .first, (.firstFactor 13), .first, .first, (.firstFactor 23), .first, .first, .first, .first, (.secondFactor 7), .first, .first, (.firstFactor 7), .first, .first, .first, .first, .first, (.firstFactor 11), .first, .first, .first, (.secondFactor 7), .first, .first, .first, .first, .first, (.secondFactor 37), .first, (.secondFactor 13), .first, (.secondFactor 19), .first, .first, (.secondFactor 7), (.firstFactor 19), .first, (.firstFactor 17), .first, .first, .first, (.firstFactor 7), (.secondFactor 17), .first, (.firstFactor 11), .first, .first, (.secondFactor 37), .first, .first, (.firstFactor 31), .first, .first, .first, .first, .first, .first, .first, .first, .first, .second, (.firstFactor 7), (.firstFactor 37), (.secondFactor 7), .first, .first, .first, (.secondFactor 7), .first, .first, (.firstFactor 19), (.firstFactor 7), .first, .first, .first, .second, .first, (.firstFactor 7), .first, (.firstFactor 13), .possible, .first, (.firstFactor 7), .second, .first, .first, .first, .first, .first, (.firstFactor 11), .possible, .first, (.firstFactor 29), .first, .first, .first, (.firstFactor 11), .first, (.secondFactor 7), .first, .first, .first, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside]
set_option maxRecDepth 2048 in
theorem row10477_block_008_covers : Budget.RowUpperBound PrimePairVertices.tree_block_008 26034 10477 row10477_block_008 := by
  apply Budget.row_upper_of_plan _ _ _ plan10477_block_008
  · decide
  · rfl

def row10477_block_009 : List Nat := []
theorem row10477_block_009_covers : Budget.RowUpperBound PrimePairVertices.tree_block_009 26034 10477 row10477_block_009 :=
  Budget.empty_row_after _ _ _ 16514 (fun q member => (Bounds.block_009 q member).1) (by decide)

def row10477_block_010 : List Nat := []
theorem row10477_block_010_covers : Budget.RowUpperBound PrimePairVertices.tree_block_010 26034 10477 row10477_block_010 :=
  Budget.empty_row_after _ _ _ 18562 (fun q member => (Bounds.block_010 q member).1) (by decide)

def row10477_block_011 : List Nat := []
theorem row10477_block_011_covers : Budget.RowUpperBound PrimePairVertices.tree_block_011 26034 10477 row10477_block_011 :=
  Budget.empty_row_after _ _ _ 20610 (fun q member => (Bounds.block_011 q member).1) (by decide)

def row10477_block_012 : List Nat := []
theorem row10477_block_012_covers : Budget.RowUpperBound PrimePairVertices.tree_block_012 26034 10477 row10477_block_012 :=
  Budget.empty_row_after _ _ _ 22658 (fun q member => (Bounds.block_012 q member).1) (by decide)

def row10477_block_013 : List Nat := []
theorem row10477_block_013_covers : Budget.RowUpperBound PrimePairVertices.tree_block_013 26034 10477 row10477_block_013 :=
  Budget.empty_row_after _ _ _ 24706 (fun q member => (Bounds.block_013 q member).1) (by decide)

def row10477_large_left_left : List Nat := row10477_small ++ row10477_block_005
theorem row10477_large_left_left_covers : Budget.RowUpperBound PrimePairVertices.tree_large_left_left 26034 10477 row10477_large_left_left :=
  Budget.RowUpperBound.branch row10477_small_covers row10477_block_005_covers

def row10477_large_left_right_right : List Nat := row10477_block_007 ++ row10477_block_008
theorem row10477_large_left_right_right_covers : Budget.RowUpperBound PrimePairVertices.tree_large_left_right_right 26034 10477 row10477_large_left_right_right :=
  Budget.RowUpperBound.branch row10477_block_007_covers row10477_block_008_covers

def row10477_large_left_right : List Nat := row10477_block_006 ++ row10477_large_left_right_right
theorem row10477_large_left_right_covers : Budget.RowUpperBound PrimePairVertices.tree_large_left_right 26034 10477 row10477_large_left_right :=
  Budget.RowUpperBound.branch row10477_block_006_covers row10477_large_left_right_right_covers

def row10477_large_left : List Nat := row10477_large_left_left ++ row10477_large_left_right
theorem row10477_large_left_covers : Budget.RowUpperBound PrimePairVertices.tree_large_left 26034 10477 row10477_large_left :=
  Budget.RowUpperBound.branch row10477_large_left_left_covers row10477_large_left_right_covers

def row10477_large_right_left : List Nat := row10477_block_009 ++ row10477_block_010
theorem row10477_large_right_left_covers : Budget.RowUpperBound PrimePairVertices.tree_large_right_left 26034 10477 row10477_large_right_left :=
  Budget.RowUpperBound.branch row10477_block_009_covers row10477_block_010_covers

def row10477_large_right_right_right : List Nat := row10477_block_012 ++ row10477_block_013
theorem row10477_large_right_right_right_covers : Budget.RowUpperBound PrimePairVertices.tree_large_right_right_right 26034 10477 row10477_large_right_right_right :=
  Budget.RowUpperBound.branch row10477_block_012_covers row10477_block_013_covers

def row10477_large_right_right : List Nat := row10477_block_011 ++ row10477_large_right_right_right
theorem row10477_large_right_right_covers : Budget.RowUpperBound PrimePairVertices.tree_large_right_right 26034 10477 row10477_large_right_right :=
  Budget.RowUpperBound.branch row10477_block_011_covers row10477_large_right_right_right_covers

def row10477_large_right : List Nat := row10477_large_right_left ++ row10477_large_right_right
theorem row10477_large_right_covers : Budget.RowUpperBound PrimePairVertices.tree_large_right 26034 10477 row10477_large_right :=
  Budget.RowUpperBound.branch row10477_large_right_left_covers row10477_large_right_right_covers

def row10477_large : List Nat := row10477_large_left ++ row10477_large_right
theorem row10477_large_covers : Budget.RowUpperBound PrimePairVertices.tree_large 26034 10477 row10477_large :=
  Budget.RowUpperBound.branch row10477_large_left_covers row10477_large_right_covers

def row10477 : List Nat := row10477_large
theorem row10477_covers : Budget.RowUpperBound PrimePairVertices.largeTree 26034 10477 row10477 := row10477_large_covers

end ProjectEulerSolutions.PrimePairLarge
