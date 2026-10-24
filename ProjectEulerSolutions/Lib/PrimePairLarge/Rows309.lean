import ProjectEulerSolutions.Lib.PrimePairLarge.Bounds

namespace ProjectEulerSolutions.PrimePairLarge
open PrimePairCertificate

def row10093_small : List Nat := []
theorem row10093_small_covers : Budget.RowUpperBound PrimePairVertices.tree_small 26034 10093 row10093_small :=
  Budget.empty_row_before _ _ _ 8389 (fun q member => (Bounds.small q member).2) (by decide)

def row10093_block_005 : List Nat := []
def plan10093_block_005 : List Budget.Choice := [.outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, (.firstFactor 7), .first, (.firstFactor 17), .first, .first, (.firstFactor 7), .first, (.firstFactor 11), .first, .first, .first, .first, .first, .first, .first, .first, .first, .first, .first, (.firstFactor 7), .first, .first, .first, .first, (.secondFactor 19), .first, (.secondFactor 7), .first, .second, .first, .first, (.firstFactor 11), (.firstFactor 31)]
set_option maxRecDepth 2048 in
theorem row10093_block_005_covers : Budget.RowUpperBound PrimePairVertices.tree_block_005 26034 10093 row10093_block_005 := by
  apply Budget.row_upper_of_plan _ _ _ plan10093_block_005
  · decide
  · rfl

def row10093_block_006 : List Nat := [10789]
def plan10093_block_006 : List Budget.Choice := [.first, .first, .first, (.secondFactor 29), .first, (.secondFactor 17), .first, .first, .first, (.firstFactor 7), .first, .first, .first, .second, .first, (.firstFactor 37), .first, .first, .first, (.secondFactor 13), .first, .first, .first, (.firstFactor 13), .first, .second, .first, (.secondFactor 7), .first, .first, (.firstFactor 7), .first, .first, .first, (.firstFactor 17), (.firstFactor 7), .first, .first, (.firstFactor 11), (.firstFactor 7), .first, .possible, .first, (.secondFactor 13), (.firstFactor 19), .first, .first, .first, (.firstFactor 13), (.firstFactor 29), .first, .first, .first, .first, (.firstFactor 23), .first, (.firstFactor 7), .first, .second, .first, .first, (.secondFactor 13), (.secondFactor 7), .first, .first, (.firstFactor 23), .first, .first, .first, .second, (.firstFactor 11), .first, .first, (.firstFactor 31), .first, (.secondFactor 7), (.firstFactor 17), (.firstFactor 7), .first, (.secondFactor 7), .first, (.firstFactor 13), .first, (.firstFactor 37), .first, .first, .first, (.firstFactor 13), .second, .first, .first, .first, (.secondFactor 7), (.firstFactor 31), .second, (.firstFactor 7), .first, (.firstFactor 13), .first, .first, .first, .first, .first, .first, .first, .first, (.firstFactor 17), (.firstFactor 7), .first, .first, .first, .first, .first, .first, (.secondFactor 7), .first, .first, (.firstFactor 7), .first, .first, .first, (.secondFactor 37), .first, .first, .second, .first, .first, .first, (.firstFactor 11), .first, (.secondFactor 13), .first, .first, .first, (.firstFactor 13), .first, (.firstFactor 11), .first, (.firstFactor 7), .first, .first, .first, .first, .first, (.firstFactor 7), .first, .first, (.secondFactor 7), .first, (.firstFactor 7), .first, .first, .first, .first, .first, (.secondFactor 13), .first, .first, .first, (.firstFactor 11), (.firstFactor 13), (.secondFactor 7), .first, .second, .first, .first, (.firstFactor 11), .first, .second, .first, (.firstFactor 31), (.firstFactor 17), .first, (.firstFactor 7), (.secondFactor 23), .first, .first, (.firstFactor 13), .first, .first, .first, .first, (.firstFactor 7), .first, .first, .first, .first, (.secondFactor 7), .first, .first, (.firstFactor 7), .first, (.firstFactor 17), .first, .first, .first, .first, (.firstFactor 23), .second, .first, .first, (.firstFactor 13), .first, (.secondFactor 23), .first, (.secondFactor 7), (.secondFactor 13), .first, (.firstFactor 7), .first]
set_option maxRecDepth 2048 in
theorem row10093_block_006_covers : Budget.RowUpperBound PrimePairVertices.tree_block_006 26034 10093 row10093_block_006 := by
  apply Budget.row_upper_of_plan _ _ _ plan10093_block_006
  · decide
  · rfl

def row10093_block_007 : List Nat := [12721, 14281, 14323]
def plan10093_block_007 : List Budget.Choice := [(.firstFactor 13), (.firstFactor 19), .first, (.firstFactor 7), (.firstFactor 17), .first, .first, .second, .first, .first, .first, (.secondFactor 23), (.secondFactor 29), .first, .first, .first, (.firstFactor 19), .first, .first, (.firstFactor 7), (.secondFactor 19), (.secondFactor 7), (.firstFactor 11), .first, .first, (.firstFactor 7), .first, .first, .first, .first, .first, .first, .first, (.secondFactor 17), (.firstFactor 7), .first, .possible, .first, .first, (.secondFactor 7), (.firstFactor 17), (.firstFactor 29), .first, (.firstFactor 11), .first, .first, .first, (.firstFactor 7), (.firstFactor 23), .first, (.firstFactor 13), .first, .first, .first, .first, .first, (.secondFactor 37), .first, .first, .first, .first, (.firstFactor 13), (.firstFactor 31), (.firstFactor 23), .first, .first, (.firstFactor 19), .first, (.secondFactor 7), .first, .first, .first, .first, (.firstFactor 11), (.secondFactor 7), .first, .first, .first, .first, .first, (.secondFactor 31), .first, (.firstFactor 31), .first, (.firstFactor 17), (.secondFactor 7), .first, .first, .first, (.secondFactor 7), .first, .first, (.firstFactor 7), .first, (.secondFactor 19), (.firstFactor 7), .first, (.secondFactor 17), .first, (.firstFactor 11), .first, .first, (.secondFactor 23), .first, (.secondFactor 19), .first, .first, (.secondFactor 17), (.firstFactor 7), .first, .first, .first, .first, .first, .first, (.firstFactor 17), .first, .first, (.firstFactor 13), .first, .second, .first, .second, .first, (.firstFactor 11), (.secondFactor 7), .first, .first, (.firstFactor 7), .first, .first, (.firstFactor 7), .first, (.firstFactor 17), (.firstFactor 19), .first, .first, .first, .first, (.firstFactor 7), .first, (.firstFactor 11), .first, .first, .first, .first, .first, .first, (.firstFactor 11), .first, (.firstFactor 23), .first, .first, .first, .first, (.secondFactor 13), .first, (.firstFactor 7), .first, .first, (.firstFactor 13), .first, .first, (.firstFactor 7), .first, (.secondFactor 7), (.firstFactor 7), .first, .first, (.firstFactor 29), .first, .second, (.firstFactor 19), .first, .first, .first, .first, .first, (.firstFactor 23), .first, (.secondFactor 13), (.firstFactor 19), .first, .first, .first, (.firstFactor 7), .first, .first, .first, (.firstFactor 23), .first, .first, (.firstFactor 11), .possible, (.firstFactor 13), .first, .first, .possible, .first, (.firstFactor 7), (.firstFactor 29), .first, .first, .first, .first, (.secondFactor 19), .first, (.secondFactor 13), .first, (.secondFactor 17), (.secondFactor 7), .first, (.firstFactor 11), (.firstFactor 31)]
set_option maxRecDepth 2048 in
theorem row10093_block_007_covers : Budget.RowUpperBound PrimePairVertices.tree_block_007 26034 10093 row10093_block_007 := by
  apply Budget.row_upper_of_plan _ _ _ plan10093_block_007
  · decide
  · rfl

def row10093_block_008 : List Nat := [14851, 15259]
def plan10093_block_008 : List Budget.Choice := [(.secondFactor 7), .first, .second, .first, (.secondFactor 17), .first, .first, .first, (.firstFactor 7), .first, .first, (.secondFactor 7), .first, (.firstFactor 7), .first, .first, .first, .first, .first, (.secondFactor 13), .first, .first, (.firstFactor 13), .first, (.firstFactor 11), .first, .first, (.secondFactor 7), (.secondFactor 17), .first, .first, .first, .first, .first, .first, (.firstFactor 11), .first, .first, .first, (.secondFactor 31), (.firstFactor 19), .first, .first, .possible, .first, (.firstFactor 29), .first, (.firstFactor 7), .first, .first, .first, (.firstFactor 7), .first, .first, .first, .first, .first, (.secondFactor 7), (.firstFactor 7), .first, .first, .first, .first, (.firstFactor 13), .first, .first, (.secondFactor 19), .first, .first, (.secondFactor 13), .first, .first, (.firstFactor 7), .first, .first, .first, (.firstFactor 23), (.firstFactor 37), (.secondFactor 13), (.firstFactor 29), .first, .first, (.firstFactor 11), .possible, .first, .first, (.secondFactor 23), (.secondFactor 7), .first, .second, .first, (.firstFactor 7), (.firstFactor 17), (.secondFactor 7), .first, .first, (.firstFactor 7), .first, (.secondFactor 7), (.firstFactor 11), .first, .first, (.firstFactor 7), .first, .first, .first, (.firstFactor 11), .first, (.secondFactor 17), .first, .first, .first, .second, .first, (.firstFactor 19), .first, (.firstFactor 13), .first, (.firstFactor 7), .first, .first, (.secondFactor 37), (.firstFactor 7), .first, (.firstFactor 13), .first, .first, (.firstFactor 7), .first, (.secondFactor 29), (.secondFactor 19), (.secondFactor 13), .first, .first, .first, (.firstFactor 7), .first, .second, .first, (.firstFactor 19), .first, .first, .first, .first, .second, .first, .first, .first, .first, .first, (.firstFactor 17), (.firstFactor 37), (.firstFactor 23), .first, .first, (.secondFactor 19), (.firstFactor 11), (.secondFactor 7), (.firstFactor 29), .first, .first, (.firstFactor 7), .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside]
set_option maxRecDepth 2048 in
theorem row10093_block_008_covers : Budget.RowUpperBound PrimePairVertices.tree_block_008 26034 10093 row10093_block_008 := by
  apply Budget.row_upper_of_plan _ _ _ plan10093_block_008
  · decide
  · rfl

def row10093_block_009 : List Nat := []
theorem row10093_block_009_covers : Budget.RowUpperBound PrimePairVertices.tree_block_009 26034 10093 row10093_block_009 :=
  Budget.empty_row_after _ _ _ 16514 (fun q member => (Bounds.block_009 q member).1) (by decide)

def row10093_block_010 : List Nat := []
theorem row10093_block_010_covers : Budget.RowUpperBound PrimePairVertices.tree_block_010 26034 10093 row10093_block_010 :=
  Budget.empty_row_after _ _ _ 18562 (fun q member => (Bounds.block_010 q member).1) (by decide)

def row10093_block_011 : List Nat := []
theorem row10093_block_011_covers : Budget.RowUpperBound PrimePairVertices.tree_block_011 26034 10093 row10093_block_011 :=
  Budget.empty_row_after _ _ _ 20610 (fun q member => (Bounds.block_011 q member).1) (by decide)

def row10093_block_012 : List Nat := []
theorem row10093_block_012_covers : Budget.RowUpperBound PrimePairVertices.tree_block_012 26034 10093 row10093_block_012 :=
  Budget.empty_row_after _ _ _ 22658 (fun q member => (Bounds.block_012 q member).1) (by decide)

def row10093_block_013 : List Nat := []
theorem row10093_block_013_covers : Budget.RowUpperBound PrimePairVertices.tree_block_013 26034 10093 row10093_block_013 :=
  Budget.empty_row_after _ _ _ 24706 (fun q member => (Bounds.block_013 q member).1) (by decide)

def row10093_large_left_left : List Nat := row10093_small ++ row10093_block_005
theorem row10093_large_left_left_covers : Budget.RowUpperBound PrimePairVertices.tree_large_left_left 26034 10093 row10093_large_left_left :=
  Budget.RowUpperBound.branch row10093_small_covers row10093_block_005_covers

def row10093_large_left_right_right : List Nat := row10093_block_007 ++ row10093_block_008
theorem row10093_large_left_right_right_covers : Budget.RowUpperBound PrimePairVertices.tree_large_left_right_right 26034 10093 row10093_large_left_right_right :=
  Budget.RowUpperBound.branch row10093_block_007_covers row10093_block_008_covers

def row10093_large_left_right : List Nat := row10093_block_006 ++ row10093_large_left_right_right
theorem row10093_large_left_right_covers : Budget.RowUpperBound PrimePairVertices.tree_large_left_right 26034 10093 row10093_large_left_right :=
  Budget.RowUpperBound.branch row10093_block_006_covers row10093_large_left_right_right_covers

def row10093_large_left : List Nat := row10093_large_left_left ++ row10093_large_left_right
theorem row10093_large_left_covers : Budget.RowUpperBound PrimePairVertices.tree_large_left 26034 10093 row10093_large_left :=
  Budget.RowUpperBound.branch row10093_large_left_left_covers row10093_large_left_right_covers

def row10093_large_right_left : List Nat := row10093_block_009 ++ row10093_block_010
theorem row10093_large_right_left_covers : Budget.RowUpperBound PrimePairVertices.tree_large_right_left 26034 10093 row10093_large_right_left :=
  Budget.RowUpperBound.branch row10093_block_009_covers row10093_block_010_covers

def row10093_large_right_right_right : List Nat := row10093_block_012 ++ row10093_block_013
theorem row10093_large_right_right_right_covers : Budget.RowUpperBound PrimePairVertices.tree_large_right_right_right 26034 10093 row10093_large_right_right_right :=
  Budget.RowUpperBound.branch row10093_block_012_covers row10093_block_013_covers

def row10093_large_right_right : List Nat := row10093_block_011 ++ row10093_large_right_right_right
theorem row10093_large_right_right_covers : Budget.RowUpperBound PrimePairVertices.tree_large_right_right 26034 10093 row10093_large_right_right :=
  Budget.RowUpperBound.branch row10093_block_011_covers row10093_large_right_right_right_covers

def row10093_large_right : List Nat := row10093_large_right_left ++ row10093_large_right_right
theorem row10093_large_right_covers : Budget.RowUpperBound PrimePairVertices.tree_large_right 26034 10093 row10093_large_right :=
  Budget.RowUpperBound.branch row10093_large_right_left_covers row10093_large_right_right_covers

def row10093_large : List Nat := row10093_large_left ++ row10093_large_right
theorem row10093_large_covers : Budget.RowUpperBound PrimePairVertices.tree_large 26034 10093 row10093_large :=
  Budget.RowUpperBound.branch row10093_large_left_covers row10093_large_right_covers

def row10093 : List Nat := row10093_large
theorem row10093_covers : Budget.RowUpperBound PrimePairVertices.largeTree 26034 10093 row10093 := row10093_large_covers

def row10099_small : List Nat := []
theorem row10099_small_covers : Budget.RowUpperBound PrimePairVertices.tree_small 26034 10099 row10099_small :=
  Budget.empty_row_before _ _ _ 8389 (fun q member => (Bounds.small q member).2) (by decide)

def row10099_block_005 : List Nat := [10273]
def plan10099_block_005 : List Budget.Choice := [.outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .first, (.firstFactor 7), .first, .first, (.secondFactor 31), .first, .first, .first, .first, (.firstFactor 17), .first, .first, .first, .first, (.firstFactor 23), .first, .first, .first, .first, .first, .possible, .first, .first, (.secondFactor 7), .first, (.firstFactor 7), .first, (.secondFactor 17), .first, .first, .first, (.firstFactor 13)]
set_option maxRecDepth 2048 in
theorem row10099_block_005_covers : Budget.RowUpperBound PrimePairVertices.tree_block_005 26034 10099 row10099_block_005 := by
  apply Budget.row_upper_of_plan _ _ _ plan10099_block_005
  · decide
  · rfl

def row10099_block_006 : List Nat := [10831]
def plan10099_block_006 : List Budget.Choice := [.first, .first, .first, (.firstFactor 11), .first, (.secondFactor 19), .first, (.secondFactor 13), .first, (.firstFactor 19), .first, .first, .first, (.secondFactor 7), .first, (.firstFactor 7), .first, (.secondFactor 19), .first, (.firstFactor 37), .first, .first, .first, (.firstFactor 11), .first, (.secondFactor 7), .first, (.firstFactor 7), .second, .first, (.firstFactor 17), .first, .first, (.secondFactor 29), (.secondFactor 7), .first, .first, .first, (.secondFactor 23), (.secondFactor 13), .first, (.firstFactor 17), .first, .possible, (.firstFactor 13), .first, .first, .first, (.secondFactor 37), (.firstFactor 7), .first, .first, (.firstFactor 11), .second, (.firstFactor 7), .first, .first, .first, (.firstFactor 11), .first, .first, .first, (.firstFactor 7), .first, .first, (.firstFactor 19), .first, (.secondFactor 7), .first, (.firstFactor 13), (.secondFactor 13), .first, .first, .first, .first, (.firstFactor 7), .first, (.firstFactor 13), .first, (.firstFactor 7), .first, .first, .first, (.firstFactor 17), .first, (.secondFactor 13), .first, (.secondFactor 17), (.secondFactor 31), .first, .first, .first, (.firstFactor 7), (.firstFactor 17), (.secondFactor 7), (.secondFactor 13), .first, (.firstFactor 7), .first, (.firstFactor 11), .first, (.firstFactor 13), .first, .first, .first, .first, (.secondFactor 7), (.secondFactor 23), .first, .first, .first, .first, .first, .first, (.firstFactor 7), (.firstFactor 17), .first, (.secondFactor 37), .first, (.firstFactor 11), .first, .second, (.secondFactor 19), .first, (.firstFactor 11), .first, .first, .first, .first, .first, (.firstFactor 29), .first, .first, .first, (.secondFactor 23), (.firstFactor 19), .second, .first, .second, .first, .first, .first, .first, .first, (.secondFactor 19), .first, .first, (.firstFactor 7), .first, (.firstFactor 29), .first, .second, .first, .first, .first, .first, .first, .first, .first, (.secondFactor 7), .first, (.firstFactor 7), .first, (.secondFactor 37), .first, .first, (.firstFactor 13), .first, (.firstFactor 23), .first, (.firstFactor 7), (.secondFactor 19), .first, (.firstFactor 19), (.secondFactor 13), .first, .first, (.secondFactor 7), .first, .first, .first, .first, .first, .first, (.firstFactor 13), .first, .first, (.firstFactor 7), .first, .first, (.firstFactor 13), .first, (.firstFactor 7), .first, .first, (.firstFactor 11), .first, .second, (.firstFactor 19), .first, .first, (.firstFactor 11), .first, (.firstFactor 37), .first, (.firstFactor 7), (.secondFactor 19), .first, (.firstFactor 11), .first]
set_option maxRecDepth 2048 in
theorem row10099_block_006_covers : Budget.RowUpperBound PrimePairVertices.tree_block_006 26034 10099 row10099_block_006 := by
  apply Budget.row_upper_of_plan _ _ _ plan10099_block_006
  · decide
  · rfl

def row10099_block_007 : List Nat := [12919, 13063, 13297, 13921, 14071]
def plan10099_block_007 : List Budget.Choice := [(.firstFactor 7), .first, .first, (.firstFactor 23), .first, .first, .first, (.secondFactor 7), .first, .first, .first, .second, .first, .first, .first, (.firstFactor 11), (.firstFactor 7), (.firstFactor 13), .first, (.secondFactor 17), .first, (.firstFactor 7), .second, .first, (.secondFactor 7), (.secondFactor 19), (.firstFactor 31), .first, .first, .first, .first, .first, .first, (.secondFactor 7), .first, .first, (.secondFactor 13), (.firstFactor 11), .first, (.firstFactor 7), .first, (.secondFactor 7), .first, (.firstFactor 7), .first, .first, (.firstFactor 31), (.firstFactor 17), (.firstFactor 7), .first, .first, .first, .first, (.firstFactor 29), .first, .first, .possible, .first, .first, .first, .first, (.firstFactor 7), (.secondFactor 29), .first, .first, .first, (.firstFactor 11), .first, (.firstFactor 7), (.firstFactor 17), .first, .first, .first, .possible, (.firstFactor 7), (.firstFactor 13), .first, .first, .first, .first, (.secondFactor 29), .first, (.secondFactor 7), .first, .first, (.firstFactor 7), .first, .first, .first, (.firstFactor 7), .first, .first, .first, .first, (.firstFactor 11), (.secondFactor 17), .possible, .first, .first, (.firstFactor 19), .first, .first, (.firstFactor 17), .first, (.firstFactor 31), .first, (.firstFactor 11), (.firstFactor 13), (.firstFactor 23), .first, (.firstFactor 17), .first, .first, .first, .first, .first, .first, .first, (.firstFactor 7), .first, (.secondFactor 7), .first, (.firstFactor 13), .first, .first, (.firstFactor 7), .first, .first, .first, .first, .first, (.firstFactor 19), .first, (.firstFactor 7), .second, .first, (.firstFactor 23), .first, .first, .second, .first, (.firstFactor 7), (.firstFactor 11), .first, .first, (.secondFactor 19), .first, .first, (.secondFactor 7), .first, (.firstFactor 7), .first, (.firstFactor 23), .first, .first, (.secondFactor 7), .first, (.firstFactor 13), .first, .first, (.secondFactor 17), .first, .first, .possible, .first, (.firstFactor 7), .first, .first, .first, (.secondFactor 7), .first, (.firstFactor 19), .first, .first, .first, .first, .possible, .first, (.secondFactor 7), .first, (.firstFactor 23), (.firstFactor 7), (.firstFactor 37), .first, .first, .first, .first, .second, .first, .second, .first, .first, (.secondFactor 7), (.secondFactor 13), (.secondFactor 7), .first, .first, (.firstFactor 11), .first, (.secondFactor 23), (.firstFactor 13), .first, .first, (.firstFactor 11), .first, .first, .first, (.secondFactor 7), .first, .first, (.firstFactor 7), .first, .second, (.firstFactor 17)]
set_option maxRecDepth 2048 in
theorem row10099_block_007_covers : Budget.RowUpperBound PrimePairVertices.tree_block_007 26034 10099 row10099_block_007 := by
  apply Budget.row_upper_of_plan _ _ _ plan10099_block_007
  · decide
  · rfl

def row10099_block_008 : List Nat := [15667]
def plan10099_block_008 : List Budget.Choice := [(.firstFactor 7), .first, (.firstFactor 13), .first, .second, .first, .first, .first, .first, (.secondFactor 19), .first, (.firstFactor 7), .first, (.firstFactor 37), .first, .first, (.secondFactor 7), .first, .first, (.firstFactor 11), .first, .first, (.firstFactor 31), .first, (.secondFactor 7), .first, .first, (.firstFactor 7), (.firstFactor 13), .first, .first, .first, .first, (.firstFactor 17), .first, .first, .first, (.firstFactor 23), .first, (.firstFactor 29), (.secondFactor 13), .first, .first, (.firstFactor 11), .first, (.firstFactor 17), .first, (.secondFactor 29), .first, .first, (.firstFactor 19), .first, .first, .first, .first, .first, .first, (.firstFactor 7), (.secondFactor 19), .first, (.secondFactor 23), .first, (.secondFactor 13), (.firstFactor 17), .first, .first, (.secondFactor 7), .first, .first, .first, .first, .first, (.secondFactor 13), .first, .first, .first, .first, (.firstFactor 7), .first, (.secondFactor 7), .first, .first, (.firstFactor 31), (.firstFactor 37), .first, .first, .second, (.firstFactor 7), .first, .first, .first, (.secondFactor 23), (.firstFactor 11), (.firstFactor 7), .first, (.secondFactor 17), (.firstFactor 23), .first, (.firstFactor 7), (.secondFactor 13), .first, .first, .second, .first, .first, (.firstFactor 31), (.firstFactor 13), .first, (.secondFactor 13), .first, .first, .first, (.firstFactor 19), .first, (.firstFactor 11), .first, .first, .first, .second, .first, .first, (.firstFactor 17), .first, (.firstFactor 19), .first, .first, .first, (.firstFactor 11), .first, .first, .first, .possible, .first, (.secondFactor 7), .first, .second, .first, .first, .first, (.firstFactor 7), .first, .first, .first, .first, (.firstFactor 17), .first, .first, .first, .first, .first, (.firstFactor 7), (.secondFactor 23), .first, .first, .first, (.firstFactor 17), (.firstFactor 23), (.firstFactor 7), .first, (.secondFactor 13), .first, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside]
set_option maxRecDepth 2048 in
theorem row10099_block_008_covers : Budget.RowUpperBound PrimePairVertices.tree_block_008 26034 10099 row10099_block_008 := by
  apply Budget.row_upper_of_plan _ _ _ plan10099_block_008
  · decide
  · rfl

def row10099_block_009 : List Nat := []
theorem row10099_block_009_covers : Budget.RowUpperBound PrimePairVertices.tree_block_009 26034 10099 row10099_block_009 :=
  Budget.empty_row_after _ _ _ 16514 (fun q member => (Bounds.block_009 q member).1) (by decide)

def row10099_block_010 : List Nat := []
theorem row10099_block_010_covers : Budget.RowUpperBound PrimePairVertices.tree_block_010 26034 10099 row10099_block_010 :=
  Budget.empty_row_after _ _ _ 18562 (fun q member => (Bounds.block_010 q member).1) (by decide)

def row10099_block_011 : List Nat := []
theorem row10099_block_011_covers : Budget.RowUpperBound PrimePairVertices.tree_block_011 26034 10099 row10099_block_011 :=
  Budget.empty_row_after _ _ _ 20610 (fun q member => (Bounds.block_011 q member).1) (by decide)

def row10099_block_012 : List Nat := []
theorem row10099_block_012_covers : Budget.RowUpperBound PrimePairVertices.tree_block_012 26034 10099 row10099_block_012 :=
  Budget.empty_row_after _ _ _ 22658 (fun q member => (Bounds.block_012 q member).1) (by decide)

def row10099_block_013 : List Nat := []
theorem row10099_block_013_covers : Budget.RowUpperBound PrimePairVertices.tree_block_013 26034 10099 row10099_block_013 :=
  Budget.empty_row_after _ _ _ 24706 (fun q member => (Bounds.block_013 q member).1) (by decide)

def row10099_large_left_left : List Nat := row10099_small ++ row10099_block_005
theorem row10099_large_left_left_covers : Budget.RowUpperBound PrimePairVertices.tree_large_left_left 26034 10099 row10099_large_left_left :=
  Budget.RowUpperBound.branch row10099_small_covers row10099_block_005_covers

def row10099_large_left_right_right : List Nat := row10099_block_007 ++ row10099_block_008
theorem row10099_large_left_right_right_covers : Budget.RowUpperBound PrimePairVertices.tree_large_left_right_right 26034 10099 row10099_large_left_right_right :=
  Budget.RowUpperBound.branch row10099_block_007_covers row10099_block_008_covers

def row10099_large_left_right : List Nat := row10099_block_006 ++ row10099_large_left_right_right
theorem row10099_large_left_right_covers : Budget.RowUpperBound PrimePairVertices.tree_large_left_right 26034 10099 row10099_large_left_right :=
  Budget.RowUpperBound.branch row10099_block_006_covers row10099_large_left_right_right_covers

def row10099_large_left : List Nat := row10099_large_left_left ++ row10099_large_left_right
theorem row10099_large_left_covers : Budget.RowUpperBound PrimePairVertices.tree_large_left 26034 10099 row10099_large_left :=
  Budget.RowUpperBound.branch row10099_large_left_left_covers row10099_large_left_right_covers

def row10099_large_right_left : List Nat := row10099_block_009 ++ row10099_block_010
theorem row10099_large_right_left_covers : Budget.RowUpperBound PrimePairVertices.tree_large_right_left 26034 10099 row10099_large_right_left :=
  Budget.RowUpperBound.branch row10099_block_009_covers row10099_block_010_covers

def row10099_large_right_right_right : List Nat := row10099_block_012 ++ row10099_block_013
theorem row10099_large_right_right_right_covers : Budget.RowUpperBound PrimePairVertices.tree_large_right_right_right 26034 10099 row10099_large_right_right_right :=
  Budget.RowUpperBound.branch row10099_block_012_covers row10099_block_013_covers

def row10099_large_right_right : List Nat := row10099_block_011 ++ row10099_large_right_right_right
theorem row10099_large_right_right_covers : Budget.RowUpperBound PrimePairVertices.tree_large_right_right 26034 10099 row10099_large_right_right :=
  Budget.RowUpperBound.branch row10099_block_011_covers row10099_large_right_right_right_covers

def row10099_large_right : List Nat := row10099_large_right_left ++ row10099_large_right_right
theorem row10099_large_right_covers : Budget.RowUpperBound PrimePairVertices.tree_large_right 26034 10099 row10099_large_right :=
  Budget.RowUpperBound.branch row10099_large_right_left_covers row10099_large_right_right_covers

def row10099_large : List Nat := row10099_large_left ++ row10099_large_right
theorem row10099_large_covers : Budget.RowUpperBound PrimePairVertices.tree_large 26034 10099 row10099_large :=
  Budget.RowUpperBound.branch row10099_large_left_covers row10099_large_right_covers

def row10099 : List Nat := row10099_large
theorem row10099_covers : Budget.RowUpperBound PrimePairVertices.largeTree 26034 10099 row10099 := row10099_large_covers

def row10103_small : List Nat := []
theorem row10103_small_covers : Budget.RowUpperBound PrimePairVertices.tree_small 26034 10103 row10103_small :=
  Budget.empty_row_before _ _ _ 8389 (fun q member => (Bounds.small q member).2) (by decide)

def row10103_block_005 : List Nat := []
def plan10103_block_005 : List Budget.Choice := [.outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .first, (.firstFactor 7), .first, .first, (.secondFactor 7), .first, .second, (.firstFactor 11), .first, (.firstFactor 31), (.secondFactor 7), (.secondFactor 13), (.firstFactor 13), .first, .first, (.secondFactor 17), (.firstFactor 7), .first, (.firstFactor 37), .first, (.firstFactor 17), (.firstFactor 7), .first, (.firstFactor 19), .first, .second, .first, .first, (.firstFactor 7), .first, .first]
set_option maxRecDepth 2048 in
theorem row10103_block_005_covers : Budget.RowUpperBound PrimePairVertices.tree_block_005 26034 10103 row10103_block_005 := by
  apply Budget.row_upper_of_plan _ _ _ plan10103_block_005
  · decide
  · rfl

def row10103_block_006 : List Nat := [10733, 11597]
def plan10103_block_006 : List Budget.Choice := [(.firstFactor 17), .first, (.firstFactor 7), .first, (.firstFactor 11), .first, (.firstFactor 13), .first, .first, .first, (.secondFactor 7), (.firstFactor 11), .first, .first, (.secondFactor 7), .first, (.secondFactor 17), .first, .first, .first, (.secondFactor 13), .second, (.firstFactor 13), .first, (.firstFactor 11), .first, .first, .first, .first, (.secondFactor 19), .first, (.firstFactor 13), .second, .first, .first, .first, .possible, (.firstFactor 31), .first, .first, (.secondFactor 7), .first, (.firstFactor 17), .first, .first, (.firstFactor 7), .first, .second, .first, .first, (.firstFactor 19), (.firstFactor 7), .first, .first, .first, (.firstFactor 37), .first, (.firstFactor 23), .first, (.firstFactor 7), .first, .first, .first, (.firstFactor 13), (.firstFactor 11), .first, (.firstFactor 7), .first, (.secondFactor 13), .first, .first, (.firstFactor 23), (.firstFactor 11), .first, (.secondFactor 7), .first, .first, .first, (.firstFactor 11), .first, (.secondFactor 17), .first, .first, .first, .first, .first, (.secondFactor 7), .first, .first, .first, (.secondFactor 17), .first, .first, .first, .first, .first, .first, .first, (.firstFactor 7), .first, (.secondFactor 7), .first, (.firstFactor 7), .first, (.firstFactor 17), (.firstFactor 11), .first, .first, .first, .first, (.firstFactor 13), (.firstFactor 29), (.firstFactor 11), .first, .first, .first, (.firstFactor 7), .first, (.firstFactor 13), .first, (.secondFactor 7), .first, .first, .possible, .first, (.firstFactor 11), .first, (.firstFactor 29), .first, (.firstFactor 19), .first, .second, .first, (.firstFactor 17), .first, .first, .first, (.firstFactor 23), .first, (.firstFactor 13), (.secondFactor 7), .first, (.secondFactor 19), (.firstFactor 7), .first, .first, (.firstFactor 29), .first, .first, .first, .first, .first, (.firstFactor 7), .first, (.firstFactor 19), .first, (.secondFactor 13), .first, (.firstFactor 7), .first, .first, .first, .first, .first, (.firstFactor 7), (.secondFactor 17), .first, .first, .first, (.firstFactor 31), .first, .first, .first, .first, .first, .second, (.firstFactor 7), .first, .first, .first, .first, (.firstFactor 7), .first, (.secondFactor 13), .first, .first, (.secondFactor 23), .first, (.firstFactor 17), (.secondFactor 13), .first, (.firstFactor 13), .first, (.secondFactor 19), (.firstFactor 37), .first, (.firstFactor 11), .first, .first, .first, (.firstFactor 13), .first, (.firstFactor 11), .first, (.secondFactor 7), .first, .first, (.firstFactor 7), .first, (.firstFactor 11)]
set_option maxRecDepth 2048 in
theorem row10103_block_006_covers : Budget.RowUpperBound PrimePairVertices.tree_block_006 26034 10103 row10103_block_006 := by
  apply Budget.row_upper_of_plan _ _ _ plan10103_block_006
  · decide
  · rfl

def row10103_block_007 : List Nat := [12983, 13001, 14159]
def plan10103_block_007 : List Budget.Choice := [.first, .first, .first, .first, .first, (.secondFactor 13), (.firstFactor 11), .first, (.firstFactor 37), (.secondFactor 17), (.secondFactor 7), .first, .first, (.firstFactor 7), (.secondFactor 31), .first, .first, .first, (.firstFactor 7), .first, .first, .first, .first, (.firstFactor 7), .first, .first, .first, (.firstFactor 13), .second, (.firstFactor 7), .first, (.secondFactor 7), .first, .first, .first, (.firstFactor 37), .first, .first, (.firstFactor 11), .first, .first, .first, .first, .first, (.firstFactor 11), (.firstFactor 7), .first, .first, .first, .first, .first, (.secondFactor 23), .first, .first, (.secondFactor 31), .second, .first, (.secondFactor 7), (.firstFactor 11), (.firstFactor 13), .first, .first, .first, .first, .possible, .possible, .first, (.firstFactor 11), .first, .first, .first, (.firstFactor 17), (.firstFactor 19), .first, .first, .first, (.secondFactor 29), (.firstFactor 13), .first, .first, .first, .second, .first, (.firstFactor 19), .first, .first, .first, (.firstFactor 13), (.secondFactor 7), .first, .second, (.firstFactor 7), .first, (.secondFactor 7), .first, .first, .first, .first, (.secondFactor 17), .first, (.secondFactor 13), (.firstFactor 11), .first, (.firstFactor 7), .first, (.firstFactor 29), .first, .first, .first, (.firstFactor 13), .first, (.firstFactor 7), .first, .first, (.firstFactor 11), .first, (.secondFactor 13), (.firstFactor 13), .first, .first, .first, (.firstFactor 17), .first, (.firstFactor 7), .first, .first, .first, (.firstFactor 7), .first, .first, .first, .first, (.secondFactor 7), .first, .first, .first, .first, .first, (.firstFactor 23), .first, (.secondFactor 7), .first, .first, .first, (.firstFactor 17), .first, (.secondFactor 7), (.secondFactor 37), .first, (.firstFactor 11), .first, (.firstFactor 7), .first, (.secondFactor 31), (.firstFactor 17), .first, (.secondFactor 13), .first, .first, (.firstFactor 31), .first, .second, (.firstFactor 7), .first, (.firstFactor 11), .first, .first, (.firstFactor 13), (.firstFactor 7), .first, .first, .first, .first, (.secondFactor 13), .first, (.secondFactor 7), .first, (.firstFactor 7), .first, (.firstFactor 31), .first, .first, .first, .first, .possible, .first, .first, .first, (.firstFactor 7), .first, .first, (.firstFactor 7), .first, .first, .first, (.firstFactor 19), (.secondFactor 29), .first, (.firstFactor 11), .first, .first, (.firstFactor 17), .second, .first, .first, .first, (.secondFactor 23), .first, (.secondFactor 13), .first, .first, (.secondFactor 37), .first, .first]
set_option maxRecDepth 2048 in
theorem row10103_block_007_covers : Budget.RowUpperBound PrimePairVertices.tree_block_007 26034 10103 row10103_block_007 := by
  apply Budget.row_upper_of_plan _ _ _ plan10103_block_007
  · decide
  · rfl

def row10103_block_008 : List Nat := []
def plan10103_block_008 : List Budget.Choice := [.first, (.firstFactor 37), .first, (.secondFactor 7), .first, (.firstFactor 23), (.firstFactor 7), (.secondFactor 23), .first, .first, (.secondFactor 7), .first, (.firstFactor 11), .first, .first, (.firstFactor 7), .first, .first, (.secondFactor 17), .first, (.firstFactor 11), (.firstFactor 7), .first, .second, .first, .first, (.firstFactor 11), .first, .first, (.secondFactor 17), (.firstFactor 13), (.firstFactor 7), (.firstFactor 19), .first, (.secondFactor 7), .first, .first, .first, (.firstFactor 23), .first, .first, (.firstFactor 31), (.secondFactor 17), .first, .first, .first, (.firstFactor 7), .first, (.secondFactor 13), (.secondFactor 7), .first, .first, (.secondFactor 7), .first, (.firstFactor 23), (.secondFactor 31), (.secondFactor 13), .first, .first, (.firstFactor 31), .first, (.firstFactor 11), .first, .first, .first, (.firstFactor 17), .first, (.firstFactor 19), (.secondFactor 7), .first, (.firstFactor 7), (.firstFactor 13), .first, (.secondFactor 7), .first, (.firstFactor 7), .first, .first, .first, .first, (.firstFactor 23), (.secondFactor 7), .first, .first, .second, .second, .first, .first, (.firstFactor 17), .first, (.firstFactor 7), .first, .first, .first, (.firstFactor 19), .first, .first, (.secondFactor 7), .first, .first, (.firstFactor 37), (.firstFactor 7), .first, (.secondFactor 7), .second, .first, .first, (.firstFactor 19), .first, .first, (.firstFactor 7), .first, .first, .first, .first, (.firstFactor 13), .first, (.firstFactor 7), .first, (.secondFactor 7), (.firstFactor 11), .first, .first, .first, .first, .first, (.firstFactor 23), .first, (.firstFactor 11), .first, .first, .first, (.firstFactor 19), .first, (.firstFactor 13), .first, .first, .first, (.secondFactor 7), .first, (.secondFactor 13), (.firstFactor 7), .second, .first, .first, (.secondFactor 23), (.firstFactor 17), (.firstFactor 7), .first, .first, .first, .first, .first, .first, (.firstFactor 7), .first, .first, .first, .first, .first, .first, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside]
set_option maxRecDepth 2048 in
theorem row10103_block_008_covers : Budget.RowUpperBound PrimePairVertices.tree_block_008 26034 10103 row10103_block_008 := by
  apply Budget.row_upper_of_plan _ _ _ plan10103_block_008
  · decide
  · rfl

def row10103_block_009 : List Nat := []
theorem row10103_block_009_covers : Budget.RowUpperBound PrimePairVertices.tree_block_009 26034 10103 row10103_block_009 :=
  Budget.empty_row_after _ _ _ 16514 (fun q member => (Bounds.block_009 q member).1) (by decide)

def row10103_block_010 : List Nat := []
theorem row10103_block_010_covers : Budget.RowUpperBound PrimePairVertices.tree_block_010 26034 10103 row10103_block_010 :=
  Budget.empty_row_after _ _ _ 18562 (fun q member => (Bounds.block_010 q member).1) (by decide)

def row10103_block_011 : List Nat := []
theorem row10103_block_011_covers : Budget.RowUpperBound PrimePairVertices.tree_block_011 26034 10103 row10103_block_011 :=
  Budget.empty_row_after _ _ _ 20610 (fun q member => (Bounds.block_011 q member).1) (by decide)

def row10103_block_012 : List Nat := []
theorem row10103_block_012_covers : Budget.RowUpperBound PrimePairVertices.tree_block_012 26034 10103 row10103_block_012 :=
  Budget.empty_row_after _ _ _ 22658 (fun q member => (Bounds.block_012 q member).1) (by decide)

def row10103_block_013 : List Nat := []
theorem row10103_block_013_covers : Budget.RowUpperBound PrimePairVertices.tree_block_013 26034 10103 row10103_block_013 :=
  Budget.empty_row_after _ _ _ 24706 (fun q member => (Bounds.block_013 q member).1) (by decide)

def row10103_large_left_left : List Nat := row10103_small ++ row10103_block_005
theorem row10103_large_left_left_covers : Budget.RowUpperBound PrimePairVertices.tree_large_left_left 26034 10103 row10103_large_left_left :=
  Budget.RowUpperBound.branch row10103_small_covers row10103_block_005_covers

def row10103_large_left_right_right : List Nat := row10103_block_007 ++ row10103_block_008
theorem row10103_large_left_right_right_covers : Budget.RowUpperBound PrimePairVertices.tree_large_left_right_right 26034 10103 row10103_large_left_right_right :=
  Budget.RowUpperBound.branch row10103_block_007_covers row10103_block_008_covers

def row10103_large_left_right : List Nat := row10103_block_006 ++ row10103_large_left_right_right
theorem row10103_large_left_right_covers : Budget.RowUpperBound PrimePairVertices.tree_large_left_right 26034 10103 row10103_large_left_right :=
  Budget.RowUpperBound.branch row10103_block_006_covers row10103_large_left_right_right_covers

def row10103_large_left : List Nat := row10103_large_left_left ++ row10103_large_left_right
theorem row10103_large_left_covers : Budget.RowUpperBound PrimePairVertices.tree_large_left 26034 10103 row10103_large_left :=
  Budget.RowUpperBound.branch row10103_large_left_left_covers row10103_large_left_right_covers

def row10103_large_right_left : List Nat := row10103_block_009 ++ row10103_block_010
theorem row10103_large_right_left_covers : Budget.RowUpperBound PrimePairVertices.tree_large_right_left 26034 10103 row10103_large_right_left :=
  Budget.RowUpperBound.branch row10103_block_009_covers row10103_block_010_covers

def row10103_large_right_right_right : List Nat := row10103_block_012 ++ row10103_block_013
theorem row10103_large_right_right_right_covers : Budget.RowUpperBound PrimePairVertices.tree_large_right_right_right 26034 10103 row10103_large_right_right_right :=
  Budget.RowUpperBound.branch row10103_block_012_covers row10103_block_013_covers

def row10103_large_right_right : List Nat := row10103_block_011 ++ row10103_large_right_right_right
theorem row10103_large_right_right_covers : Budget.RowUpperBound PrimePairVertices.tree_large_right_right 26034 10103 row10103_large_right_right :=
  Budget.RowUpperBound.branch row10103_block_011_covers row10103_large_right_right_right_covers

def row10103_large_right : List Nat := row10103_large_right_left ++ row10103_large_right_right
theorem row10103_large_right_covers : Budget.RowUpperBound PrimePairVertices.tree_large_right 26034 10103 row10103_large_right :=
  Budget.RowUpperBound.branch row10103_large_right_left_covers row10103_large_right_right_covers

def row10103_large : List Nat := row10103_large_left ++ row10103_large_right
theorem row10103_large_covers : Budget.RowUpperBound PrimePairVertices.tree_large 26034 10103 row10103_large :=
  Budget.RowUpperBound.branch row10103_large_left_covers row10103_large_right_covers

def row10103 : List Nat := row10103_large
theorem row10103_covers : Budget.RowUpperBound PrimePairVertices.largeTree 26034 10103 row10103 := row10103_large_covers

def row10111_small : List Nat := []
theorem row10111_small_covers : Budget.RowUpperBound PrimePairVertices.tree_small 26034 10111 row10111_small :=
  Budget.empty_row_before _ _ _ 8389 (fun q member => (Bounds.small q member).2) (by decide)

def row10111_block_005 : List Nat := [10159]
def plan10111_block_005 : List Budget.Choice := [.outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .first, .first, (.secondFactor 7), .first, .possible, .first, .first, (.firstFactor 7), .first, .first, .first, .first, (.firstFactor 11), .first, .first, .first, (.secondFactor 7), .first, (.secondFactor 23), .first, .first, (.firstFactor 7), .first, (.firstFactor 13), .first, .first, .first, .first, .second, .second]
set_option maxRecDepth 2048 in
theorem row10111_block_005_covers : Budget.RowUpperBound PrimePairVertices.tree_block_005 26034 10111 row10111_block_005 := by
  apply Budget.row_upper_of_plan _ _ _ plan10111_block_005
  · decide
  · rfl

def row10111_block_006 : List Nat := [10957, 11083, 11719]
def plan10111_block_006 : List Budget.Choice := [.first, (.firstFactor 13), .first, (.firstFactor 7), .first, (.secondFactor 31), .first, .second, .first, (.firstFactor 13), .first, .first, (.secondFactor 17), (.firstFactor 7), .first, .first, .first, (.firstFactor 23), .first, (.firstFactor 7), .first, .first, .first, .first, .first, (.firstFactor 7), (.secondFactor 13), .first, (.firstFactor 31), .first, (.secondFactor 7), .first, .first, (.firstFactor 13), (.firstFactor 7), (.firstFactor 37), .first, .first, (.secondFactor 19), (.firstFactor 11), .first, (.firstFactor 13), .first, .first, (.firstFactor 11), .first, .first, .first, .first, (.firstFactor 13), .first, .first, (.firstFactor 7), (.firstFactor 11), (.secondFactor 17), .first, (.secondFactor 7), .first, .possible, .first, .first, .first, .second, .first, .first, .first, .first, (.firstFactor 7), .first, (.firstFactor 29), .possible, .first, .first, (.secondFactor 17), .first, (.firstFactor 23), .first, (.secondFactor 7), .first, .first, .first, (.firstFactor 37), .first, (.secondFactor 13), .first, (.firstFactor 19), .first, .first, (.firstFactor 13), .first, .first, .first, .second, (.firstFactor 11), (.firstFactor 7), (.secondFactor 7), .first, (.firstFactor 17), .first, (.firstFactor 7), .first, (.secondFactor 31), .first, .first, .first, .first, (.firstFactor 7), (.secondFactor 7), .first, (.firstFactor 19), .first, .first, .first, (.firstFactor 13), (.firstFactor 11), .first, .first, (.secondFactor 7), .first, (.secondFactor 19), .first, (.secondFactor 13), (.firstFactor 29), .first, (.firstFactor 37), .first, .first, .first, .first, .first, (.firstFactor 7), .first, .first, .first, .possible, (.firstFactor 7), (.secondFactor 13), .first, (.firstFactor 31), .first, .first, .first, .first, .first, (.secondFactor 7), (.firstFactor 11), .first, .first, (.firstFactor 17), (.secondFactor 7), .first, .second, .first, .first, .first, (.firstFactor 19), .first, .first, .first, (.firstFactor 7), .first, (.firstFactor 11), .first, (.secondFactor 29), .first, .first, (.secondFactor 19), .first, (.firstFactor 13), .first, (.firstFactor 17), .second, .first, (.secondFactor 7), .second, .first, .first, (.firstFactor 7), .first, .first, .first, .first, (.firstFactor 11), .first, .second, .first, .first, (.secondFactor 13), .first, .first, (.secondFactor 7), .first, .first, .first, .first, (.firstFactor 7), .first, (.firstFactor 11), .second, .first, .first, (.secondFactor 23), .first, .first, .first, (.firstFactor 19), .first, .first, (.secondFactor 7), .first]
set_option maxRecDepth 2048 in
theorem row10111_block_006_covers : Budget.RowUpperBound PrimePairVertices.tree_block_006 26034 10111 row10111_block_006 := by
  apply Budget.row_upper_of_plan _ _ _ plan10111_block_006
  · decide
  · rfl

def row10111_block_007 : List Nat := [12967, 13093, 13099, 13219, 13687]
def plan10111_block_007 : List Budget.Choice := [(.firstFactor 11), .first, .first, (.firstFactor 17), .first, .first, .first, (.firstFactor 7), .first, .first, .first, .second, (.secondFactor 37), .first, .first, (.secondFactor 17), .first, (.firstFactor 11), .first, (.secondFactor 7), (.firstFactor 13), .first, (.secondFactor 13), .first, (.firstFactor 7), (.firstFactor 11), (.firstFactor 23), .first, .first, .first, .first, .first, .first, (.firstFactor 7), (.secondFactor 7), .first, (.firstFactor 19), (.firstFactor 7), .first, (.firstFactor 17), .first, (.firstFactor 7), .first, .second, .first, .first, (.firstFactor 7), (.secondFactor 7), (.secondFactor 29), .first, .first, .first, .first, (.firstFactor 7), .first, .first, (.secondFactor 19), .first, .first, .first, .first, .possible, (.firstFactor 13), .first, .first, .first, .first, .first, .first, (.firstFactor 7), .first, .first, .first, (.firstFactor 17), .possible, .possible, .first, .first, .first, .first, (.firstFactor 11), .first, (.firstFactor 7), .first, (.firstFactor 37), (.firstFactor 19), (.secondFactor 37), .first, .first, .possible, .first, .first, (.secondFactor 7), .first, (.firstFactor 17), (.firstFactor 19), .first, (.secondFactor 23), .first, (.firstFactor 7), .first, .first, .first, .first, (.secondFactor 13), .first, .first, (.firstFactor 7), (.secondFactor 7), .first, (.firstFactor 13), .first, .first, .first, .first, (.firstFactor 11), .first, .first, .first, .first, (.firstFactor 7), .first, .first, .first, .first, (.firstFactor 13), .first, .first, (.secondFactor 7), (.firstFactor 19), .first, (.secondFactor 7), .first, (.firstFactor 29), .possible, .first, (.secondFactor 13), .first, .first, (.secondFactor 7), .first, (.secondFactor 23), .first, .first, .first, .first, .first, .first, (.firstFactor 7), .first, (.firstFactor 11), .first, (.firstFactor 7), .first, .first, (.firstFactor 7), .first, (.firstFactor 17), .first, .first, .first, .first, .first, (.secondFactor 7), .first, .first, (.secondFactor 7), .first, .first, (.firstFactor 7), .first, (.firstFactor 31), (.firstFactor 29), .first, .first, .first, (.firstFactor 11), .first, (.firstFactor 7), .first, .first, (.firstFactor 13), .first, .first, .first, (.secondFactor 7), .first, (.firstFactor 31), .first, (.firstFactor 13), .first, .first, (.firstFactor 7), (.firstFactor 37), (.firstFactor 7), .first, .first, .first, .first, (.secondFactor 7), .first, .first, .first, (.firstFactor 17), (.firstFactor 11), (.secondFactor 29), .first, (.firstFactor 7), .first, (.firstFactor 19), .first, .first, .first, (.firstFactor 7)]
set_option maxRecDepth 2048 in
theorem row10111_block_007_covers : Budget.RowUpperBound PrimePairVertices.tree_block_007 26034 10111 row10111_block_007 := by
  apply Budget.row_upper_of_plan _ _ _ plan10111_block_007
  · decide
  · rfl

def row10111_block_008 : List Nat := [14557, 14869, 15493, 15583]
def plan10111_block_008 : List Budget.Choice := [(.secondFactor 17), .first, (.firstFactor 7), .first, (.firstFactor 11), .first, .first, .first, (.firstFactor 29), .possible, .first, .second, .first, (.firstFactor 17), .first, .first, (.firstFactor 7), .first, .first, .first, .first, .first, (.secondFactor 17), .first, (.firstFactor 7), .first, .first, (.firstFactor 11), (.secondFactor 37), .first, .first, .first, .first, (.firstFactor 13), .first, .first, .first, (.firstFactor 7), .first, .first, (.secondFactor 23), .first, .first, .first, .first, .possible, .first, (.firstFactor 19), .first, .first, (.firstFactor 7), (.firstFactor 11), .first, (.firstFactor 37), .first, .first, .first, (.firstFactor 23), (.secondFactor 7), .first, .first, .first, (.firstFactor 11), (.firstFactor 29), .first, .first, (.firstFactor 7), .first, .first, (.firstFactor 23), .first, .first, (.secondFactor 7), .first, .first, .first, .first, (.firstFactor 11), (.secondFactor 19), (.firstFactor 7), .first, .first, (.secondFactor 23), (.firstFactor 7), .first, .first, .second, (.secondFactor 29), .first, (.secondFactor 31), .first, (.firstFactor 17), (.firstFactor 13), .first, .first, (.secondFactor 13), (.secondFactor 7), .first, .first, .first, .first, .first, (.firstFactor 11), .first, .first, (.firstFactor 7), .first, .first, (.secondFactor 29), .first, .first, .first, .possible, .first, (.firstFactor 7), .first, (.secondFactor 19), .first, (.secondFactor 7), .first, .first, .possible, (.secondFactor 7), .first, .first, .first, .first, (.secondFactor 7), .first, .first, (.secondFactor 31), .first, .first, (.firstFactor 7), .first, (.secondFactor 7), .first, .second, .first, .first, .first, .first, .first, .first, (.firstFactor 11), .first, .first, .first, .first, (.firstFactor 17), .second, (.firstFactor 13), (.secondFactor 13), .first, .first, (.firstFactor 7), .second, (.secondFactor 17), (.firstFactor 19), (.firstFactor 11), .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside]
set_option maxRecDepth 2048 in
theorem row10111_block_008_covers : Budget.RowUpperBound PrimePairVertices.tree_block_008 26034 10111 row10111_block_008 := by
  apply Budget.row_upper_of_plan _ _ _ plan10111_block_008
  · decide
  · rfl

def row10111_block_009 : List Nat := []
theorem row10111_block_009_covers : Budget.RowUpperBound PrimePairVertices.tree_block_009 26034 10111 row10111_block_009 :=
  Budget.empty_row_after _ _ _ 16514 (fun q member => (Bounds.block_009 q member).1) (by decide)

def row10111_block_010 : List Nat := []
theorem row10111_block_010_covers : Budget.RowUpperBound PrimePairVertices.tree_block_010 26034 10111 row10111_block_010 :=
  Budget.empty_row_after _ _ _ 18562 (fun q member => (Bounds.block_010 q member).1) (by decide)

def row10111_block_011 : List Nat := []
theorem row10111_block_011_covers : Budget.RowUpperBound PrimePairVertices.tree_block_011 26034 10111 row10111_block_011 :=
  Budget.empty_row_after _ _ _ 20610 (fun q member => (Bounds.block_011 q member).1) (by decide)

def row10111_block_012 : List Nat := []
theorem row10111_block_012_covers : Budget.RowUpperBound PrimePairVertices.tree_block_012 26034 10111 row10111_block_012 :=
  Budget.empty_row_after _ _ _ 22658 (fun q member => (Bounds.block_012 q member).1) (by decide)

def row10111_block_013 : List Nat := []
theorem row10111_block_013_covers : Budget.RowUpperBound PrimePairVertices.tree_block_013 26034 10111 row10111_block_013 :=
  Budget.empty_row_after _ _ _ 24706 (fun q member => (Bounds.block_013 q member).1) (by decide)

def row10111_large_left_left : List Nat := row10111_small ++ row10111_block_005
theorem row10111_large_left_left_covers : Budget.RowUpperBound PrimePairVertices.tree_large_left_left 26034 10111 row10111_large_left_left :=
  Budget.RowUpperBound.branch row10111_small_covers row10111_block_005_covers

def row10111_large_left_right_right : List Nat := row10111_block_007 ++ row10111_block_008
theorem row10111_large_left_right_right_covers : Budget.RowUpperBound PrimePairVertices.tree_large_left_right_right 26034 10111 row10111_large_left_right_right :=
  Budget.RowUpperBound.branch row10111_block_007_covers row10111_block_008_covers

def row10111_large_left_right : List Nat := row10111_block_006 ++ row10111_large_left_right_right
theorem row10111_large_left_right_covers : Budget.RowUpperBound PrimePairVertices.tree_large_left_right 26034 10111 row10111_large_left_right :=
  Budget.RowUpperBound.branch row10111_block_006_covers row10111_large_left_right_right_covers

def row10111_large_left : List Nat := row10111_large_left_left ++ row10111_large_left_right
theorem row10111_large_left_covers : Budget.RowUpperBound PrimePairVertices.tree_large_left 26034 10111 row10111_large_left :=
  Budget.RowUpperBound.branch row10111_large_left_left_covers row10111_large_left_right_covers

def row10111_large_right_left : List Nat := row10111_block_009 ++ row10111_block_010
theorem row10111_large_right_left_covers : Budget.RowUpperBound PrimePairVertices.tree_large_right_left 26034 10111 row10111_large_right_left :=
  Budget.RowUpperBound.branch row10111_block_009_covers row10111_block_010_covers

def row10111_large_right_right_right : List Nat := row10111_block_012 ++ row10111_block_013
theorem row10111_large_right_right_right_covers : Budget.RowUpperBound PrimePairVertices.tree_large_right_right_right 26034 10111 row10111_large_right_right_right :=
  Budget.RowUpperBound.branch row10111_block_012_covers row10111_block_013_covers

def row10111_large_right_right : List Nat := row10111_block_011 ++ row10111_large_right_right_right
theorem row10111_large_right_right_covers : Budget.RowUpperBound PrimePairVertices.tree_large_right_right 26034 10111 row10111_large_right_right :=
  Budget.RowUpperBound.branch row10111_block_011_covers row10111_large_right_right_right_covers

def row10111_large_right : List Nat := row10111_large_right_left ++ row10111_large_right_right
theorem row10111_large_right_covers : Budget.RowUpperBound PrimePairVertices.tree_large_right 26034 10111 row10111_large_right :=
  Budget.RowUpperBound.branch row10111_large_right_left_covers row10111_large_right_right_covers

def row10111_large : List Nat := row10111_large_left ++ row10111_large_right
theorem row10111_large_covers : Budget.RowUpperBound PrimePairVertices.tree_large 26034 10111 row10111_large :=
  Budget.RowUpperBound.branch row10111_large_left_covers row10111_large_right_covers

def row10111 : List Nat := row10111_large
theorem row10111_covers : Budget.RowUpperBound PrimePairVertices.largeTree 26034 10111 row10111 := row10111_large_covers

end ProjectEulerSolutions.PrimePairLarge
