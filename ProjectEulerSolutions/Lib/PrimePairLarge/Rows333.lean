import ProjectEulerSolutions.Lib.PrimePairLarge.Bounds

namespace ProjectEulerSolutions.PrimePairLarge
open PrimePairCertificate

def row10993_small : List Nat := []
theorem row10993_small_covers : Budget.RowUpperBound PrimePairVertices.tree_small 26034 10993 row10993_small :=
  Budget.empty_row_before _ _ _ 8389 (fun q member => (Bounds.small q member).2) (by decide)

def row10993_block_005 : List Nat := []
theorem row10993_block_005_covers : Budget.RowUpperBound PrimePairVertices.tree_block_005 26034 10993 row10993_block_005 :=
  Budget.empty_row_before _ _ _ 10370 (fun q member => (Bounds.block_005 q member).2) (by decide)

def row10993_block_006 : List Nat := [11173, 11593]
def plan10993_block_006 : List Budget.Choice := [.outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .first, .first, .second, .first, (.firstFactor 7), .first, .second, (.firstFactor 13), .first, .first, (.secondFactor 13), .first, (.firstFactor 31), .first, (.firstFactor 29), .first, (.firstFactor 13), .first, .possible, .first, (.firstFactor 23), .first, (.firstFactor 13), .first, (.secondFactor 31), (.firstFactor 11), .first, .first, .first, (.secondFactor 23), .second, (.firstFactor 7), (.firstFactor 13), .first, (.firstFactor 19), .first, (.firstFactor 7), .first, .first, .first, .first, .first, .first, (.firstFactor 7), (.firstFactor 17), .first, (.secondFactor 37), .first, .first, .first, (.firstFactor 31), (.firstFactor 29), (.secondFactor 13), .first, (.secondFactor 7), .first, (.firstFactor 13), .first, (.firstFactor 11), .possible, .first, .first, .first, .first, .first, (.firstFactor 31), .first, (.firstFactor 7), .first, (.secondFactor 23), .first, (.firstFactor 11), (.firstFactor 7), .first, .first, (.secondFactor 7), .first, .first, .first, .first, .first, (.secondFactor 7), (.secondFactor 19), .first, (.secondFactor 17), (.secondFactor 23), (.firstFactor 13), .first, (.firstFactor 23), .first, .first, .first, .first, .first, .first, .first, (.firstFactor 7), (.firstFactor 17), .first, .first, (.secondFactor 13), .first, .first, .first, .first, (.secondFactor 17), .first, .first, (.firstFactor 11), .first, (.secondFactor 7), (.firstFactor 13), .first, .first, (.firstFactor 7), .first, .first, .first, .first, (.firstFactor 17), .first, (.firstFactor 23), .first, .first, .first, .first, .first, (.firstFactor 19), .first, (.firstFactor 13), .first, .first, (.firstFactor 7), .first, .second, (.firstFactor 23), .first, .first, (.secondFactor 17), .first, .second, .first, (.firstFactor 11), (.secondFactor 23), .first, (.firstFactor 13), .first]
set_option maxRecDepth 2048 in
theorem row10993_block_006_covers : Budget.RowUpperBound PrimePairVertices.tree_block_006 26034 10993 row10993_block_006 := by
  apply Budget.row_upper_of_plan _ _ _ plan10993_block_006
  · decide
  · rfl

def row10993_block_007 : List Nat := [12589, 13009, 13183, 13807, 14347]
def plan10993_block_007 : List Budget.Choice := [(.firstFactor 31), .first, .first, (.secondFactor 7), .second, .first, .first, (.firstFactor 7), .first, .first, .first, (.firstFactor 11), (.secondFactor 13), .first, .first, (.firstFactor 29), (.secondFactor 17), (.secondFactor 31), .first, (.firstFactor 11), (.firstFactor 19), .possible, .second, .first, (.firstFactor 7), (.secondFactor 7), .second, .first, .first, .first, .first, .first, .first, (.firstFactor 7), (.secondFactor 7), .first, (.firstFactor 13), (.firstFactor 7), .first, .first, .first, (.firstFactor 7), .first, (.firstFactor 13), .first, .first, (.firstFactor 7), (.secondFactor 7), (.firstFactor 11), (.firstFactor 23), (.firstFactor 29), .first, .first, (.firstFactor 7), .first, .first, .second, .first, .first, .first, .first, (.secondFactor 19), (.firstFactor 11), (.firstFactor 31), .first, .first, .first, .first, .possible, (.firstFactor 7), .first, .first, .first, (.firstFactor 29), .first, .first, .first, .first, .first, .first, .first, .first, (.firstFactor 7), .first, (.firstFactor 11), (.firstFactor 17), .possible, .first, .first, (.secondFactor 13), .first, .first, (.secondFactor 7), .first, (.firstFactor 13), (.secondFactor 7), (.secondFactor 13), (.secondFactor 19), .first, (.firstFactor 7), .first, .first, .first, .first, (.firstFactor 17), .first, .first, (.firstFactor 7), (.secondFactor 7), .first, .second, .first, .first, .first, .first, .first, .first, .first, .first, .first, (.firstFactor 7), .first, (.firstFactor 11), .first, .second, .first, .first, .first, (.secondFactor 7), (.firstFactor 11), .first, (.secondFactor 7), .first, (.firstFactor 23), (.firstFactor 17), .first, .first, .first, .first, (.secondFactor 7), .first, (.firstFactor 19), .second, .first, .first, (.firstFactor 29), .first, .first, (.firstFactor 7), .first, .possible, .first, (.firstFactor 7), .first, .first, (.firstFactor 7), .first, (.secondFactor 7), .first, .first, .second, .first, .first, (.secondFactor 7), .first, (.firstFactor 29), (.firstFactor 11), .first, .first, (.firstFactor 7), .first, .first, (.firstFactor 11), .first, .first, .first, .second, .first, (.firstFactor 7), .first, (.firstFactor 29), .second, .first, .first, .first, (.secondFactor 7), .first, (.firstFactor 17), .first, (.secondFactor 19), .first, .first, (.firstFactor 7), (.firstFactor 13), (.firstFactor 7), .first, .first, (.secondFactor 23), .first, (.secondFactor 7), .possible, .first, .first, (.secondFactor 13), (.firstFactor 17), (.firstFactor 19), .first, (.firstFactor 7), .first, .first, (.firstFactor 13), .first, (.secondFactor 19), (.firstFactor 7)]
set_option maxRecDepth 2048 in
theorem row10993_block_007_covers : Budget.RowUpperBound PrimePairVertices.tree_block_007 26034 10993 row10993_block_007 := by
  apply Budget.row_upper_of_plan _ _ _ plan10993_block_007
  · decide
  · rfl

def row10993_block_008 : List Nat := [14851, 14947]
def plan10993_block_008 : List Budget.Choice := [.first, .first, (.firstFactor 7), .first, .first, .first, .first, .first, (.secondFactor 7), (.firstFactor 11), .first, (.secondFactor 19), .first, (.firstFactor 13), .first, .first, (.firstFactor 7), .first, .first, (.firstFactor 31), .first, .first, .first, .first, (.firstFactor 7), .first, .first, .first, (.secondFactor 23), .first, .first, .first, .first, .second, .first, (.secondFactor 13), .first, (.firstFactor 7), .first, (.firstFactor 11), (.firstFactor 13), .first, .first, .possible, .first, .first, .first, (.firstFactor 11), .first, .first, (.firstFactor 7), (.secondFactor 7), .first, .possible, .first, .first, .first, (.firstFactor 13), (.firstFactor 17), .first, .first, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside]
set_option maxRecDepth 2048 in
theorem row10993_block_008_covers : Budget.RowUpperBound PrimePairVertices.tree_block_008 26034 10993 row10993_block_008 := by
  apply Budget.row_upper_of_plan _ _ _ plan10993_block_008
  · decide
  · rfl

def row10993_block_009 : List Nat := []
theorem row10993_block_009_covers : Budget.RowUpperBound PrimePairVertices.tree_block_009 26034 10993 row10993_block_009 :=
  Budget.empty_row_after _ _ _ 16514 (fun q member => (Bounds.block_009 q member).1) (by decide)

def row10993_block_010 : List Nat := []
theorem row10993_block_010_covers : Budget.RowUpperBound PrimePairVertices.tree_block_010 26034 10993 row10993_block_010 :=
  Budget.empty_row_after _ _ _ 18562 (fun q member => (Bounds.block_010 q member).1) (by decide)

def row10993_block_011 : List Nat := []
theorem row10993_block_011_covers : Budget.RowUpperBound PrimePairVertices.tree_block_011 26034 10993 row10993_block_011 :=
  Budget.empty_row_after _ _ _ 20610 (fun q member => (Bounds.block_011 q member).1) (by decide)

def row10993_block_012 : List Nat := []
theorem row10993_block_012_covers : Budget.RowUpperBound PrimePairVertices.tree_block_012 26034 10993 row10993_block_012 :=
  Budget.empty_row_after _ _ _ 22658 (fun q member => (Bounds.block_012 q member).1) (by decide)

def row10993_block_013 : List Nat := []
theorem row10993_block_013_covers : Budget.RowUpperBound PrimePairVertices.tree_block_013 26034 10993 row10993_block_013 :=
  Budget.empty_row_after _ _ _ 24706 (fun q member => (Bounds.block_013 q member).1) (by decide)

def row10993_large_left_left : List Nat := row10993_small ++ row10993_block_005
theorem row10993_large_left_left_covers : Budget.RowUpperBound PrimePairVertices.tree_large_left_left 26034 10993 row10993_large_left_left :=
  Budget.RowUpperBound.branch row10993_small_covers row10993_block_005_covers

def row10993_large_left_right_right : List Nat := row10993_block_007 ++ row10993_block_008
theorem row10993_large_left_right_right_covers : Budget.RowUpperBound PrimePairVertices.tree_large_left_right_right 26034 10993 row10993_large_left_right_right :=
  Budget.RowUpperBound.branch row10993_block_007_covers row10993_block_008_covers

def row10993_large_left_right : List Nat := row10993_block_006 ++ row10993_large_left_right_right
theorem row10993_large_left_right_covers : Budget.RowUpperBound PrimePairVertices.tree_large_left_right 26034 10993 row10993_large_left_right :=
  Budget.RowUpperBound.branch row10993_block_006_covers row10993_large_left_right_right_covers

def row10993_large_left : List Nat := row10993_large_left_left ++ row10993_large_left_right
theorem row10993_large_left_covers : Budget.RowUpperBound PrimePairVertices.tree_large_left 26034 10993 row10993_large_left :=
  Budget.RowUpperBound.branch row10993_large_left_left_covers row10993_large_left_right_covers

def row10993_large_right_left : List Nat := row10993_block_009 ++ row10993_block_010
theorem row10993_large_right_left_covers : Budget.RowUpperBound PrimePairVertices.tree_large_right_left 26034 10993 row10993_large_right_left :=
  Budget.RowUpperBound.branch row10993_block_009_covers row10993_block_010_covers

def row10993_large_right_right_right : List Nat := row10993_block_012 ++ row10993_block_013
theorem row10993_large_right_right_right_covers : Budget.RowUpperBound PrimePairVertices.tree_large_right_right_right 26034 10993 row10993_large_right_right_right :=
  Budget.RowUpperBound.branch row10993_block_012_covers row10993_block_013_covers

def row10993_large_right_right : List Nat := row10993_block_011 ++ row10993_large_right_right_right
theorem row10993_large_right_right_covers : Budget.RowUpperBound PrimePairVertices.tree_large_right_right 26034 10993 row10993_large_right_right :=
  Budget.RowUpperBound.branch row10993_block_011_covers row10993_large_right_right_right_covers

def row10993_large_right : List Nat := row10993_large_right_left ++ row10993_large_right_right
theorem row10993_large_right_covers : Budget.RowUpperBound PrimePairVertices.tree_large_right 26034 10993 row10993_large_right :=
  Budget.RowUpperBound.branch row10993_large_right_left_covers row10993_large_right_right_covers

def row10993_large : List Nat := row10993_large_left ++ row10993_large_right
theorem row10993_large_covers : Budget.RowUpperBound PrimePairVertices.tree_large 26034 10993 row10993_large :=
  Budget.RowUpperBound.branch row10993_large_left_covers row10993_large_right_covers

def row10993 : List Nat := row10993_large
theorem row10993_covers : Budget.RowUpperBound PrimePairVertices.largeTree 26034 10993 row10993 := row10993_large_covers

def row11003_small : List Nat := []
theorem row11003_small_covers : Budget.RowUpperBound PrimePairVertices.tree_small 26034 11003 row11003_small :=
  Budget.empty_row_before _ _ _ 8389 (fun q member => (Bounds.small q member).2) (by decide)

def row11003_block_005 : List Nat := []
theorem row11003_block_005_covers : Budget.RowUpperBound PrimePairVertices.tree_block_005 26034 11003 row11003_block_005 :=
  Budget.empty_row_before _ _ _ 10370 (fun q member => (Bounds.block_005 q member).2) (by decide)

def row11003_block_006 : List Nat := [11321, 11579, 11657, 12119]
def plan11003_block_006 : List Budget.Choice := [.outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .first, .first, .first, .first, (.firstFactor 11), .first, .first, .first, (.firstFactor 7), .first, (.firstFactor 31), .first, .first, .first, .second, .first, .first, .first, (.firstFactor 7), .first, (.firstFactor 17), .first, (.firstFactor 29), .first, .first, (.firstFactor 7), (.secondFactor 7), (.secondFactor 19), .first, .first, .first, .first, .possible, .first, (.secondFactor 13), .first, .first, .first, (.secondFactor 19), (.firstFactor 11), .second, .first, .first, .first, .first, .first, (.firstFactor 7), (.secondFactor 7), (.firstFactor 19), .first, .first, .first, (.firstFactor 17), .first, .first, .first, .possible, .first, .first, (.firstFactor 7), .first, (.firstFactor 17), (.secondFactor 17), .possible, .first, (.firstFactor 7), .first, .first, .first, (.firstFactor 19), .first, .first, .first, (.secondFactor 7), .first, .first, (.firstFactor 23), .first, (.firstFactor 7), .first, .first, .first, (.firstFactor 19), .first, .first, .first, .second, .first, (.secondFactor 13), (.secondFactor 7), .first, .first, (.firstFactor 11), (.firstFactor 7), (.firstFactor 29), .first, .first, .first, .first, .first, .first, (.secondFactor 7), .first, .second, .first, (.secondFactor 17), .first, .first, (.secondFactor 7), .first, .first, (.firstFactor 7), (.secondFactor 37), .first, (.firstFactor 29), .possible, (.firstFactor 7), .first, .first, (.firstFactor 13), .first, (.secondFactor 7), (.firstFactor 23), .first, (.firstFactor 7), (.firstFactor 13), .first, (.secondFactor 29), .first, .first, (.firstFactor 7), .first, (.secondFactor 7), .first, .first, (.firstFactor 11), (.secondFactor 37), .first, (.secondFactor 17), .first, .first, .first, .first, (.firstFactor 19), .first, .first]
set_option maxRecDepth 2048 in
theorem row11003_block_006_covers : Budget.RowUpperBound PrimePairVertices.tree_block_006 26034 11003 row11003_block_006 := by
  apply Budget.row_upper_of_plan _ _ _ plan11003_block_006
  · decide
  · rfl

def row11003_block_007 : List Nat := [13127, 13523, 14249]
def plan11003_block_007 : List Budget.Choice := [.first, .first, (.firstFactor 7), .first, .first, (.firstFactor 13), (.firstFactor 7), .first, (.secondFactor 7), (.secondFactor 23), (.secondFactor 31), .first, .first, .first, (.firstFactor 17), .first, .first, .first, .first, .first, .first, .first, .first, .second, .first, .first, .first, (.firstFactor 17), (.firstFactor 7), (.firstFactor 11), (.secondFactor 7), .first, (.firstFactor 7), .first, .first, .first, .first, .first, (.firstFactor 17), .first, .first, .first, (.firstFactor 31), .first, (.firstFactor 29), .first, .first, .first, .first, .first, .first, (.firstFactor 23), (.firstFactor 7), .first, (.secondFactor 7), (.firstFactor 11), .first, .first, (.firstFactor 7), (.secondFactor 7), (.secondFactor 17), .first, .first, .first, (.firstFactor 7), .first, .first, .first, .first, .first, (.secondFactor 7), .first, (.firstFactor 11), .first, .first, .first, (.secondFactor 19), (.firstFactor 7), (.secondFactor 7), .possible, .first, (.firstFactor 7), .first, (.firstFactor 31), .first, .first, .first, (.secondFactor 23), (.secondFactor 19), .first, .second, .first, .first, .first, .first, .first, .first, .first, (.firstFactor 11), .first, (.firstFactor 13), .first, .first, (.secondFactor 17), .first, .second, .first, .first, .first, .first, .first, .first, (.firstFactor 17), (.secondFactor 23), (.secondFactor 17), .first, (.firstFactor 7), (.secondFactor 7), .first, .possible, .first, .first, .first, (.firstFactor 11), .first, .first, (.firstFactor 7), (.secondFactor 31), .first, .first, .first, .first, (.firstFactor 29), .first, .first, (.secondFactor 13), .first, (.firstFactor 7), (.firstFactor 11), .first, (.firstFactor 13), .first, .first, (.secondFactor 7), .first, .first, (.firstFactor 17), (.firstFactor 7), .first, (.firstFactor 13), .first, .second, .first, (.firstFactor 11), (.firstFactor 23), .first, (.firstFactor 13), .first, (.firstFactor 19), (.secondFactor 19), .first, (.firstFactor 7), .first, .first, .second, .first, .first, (.firstFactor 17), (.firstFactor 19), .first, .first, .first, .first, (.firstFactor 7), .first, .first, .first, (.secondFactor 13), .first, (.secondFactor 7), .first, .first, .first, (.secondFactor 23), (.firstFactor 7), .first, (.firstFactor 37), .first, .first, .first, (.firstFactor 7), .possible, .first, .first, .first, (.firstFactor 11), .first, .first, (.firstFactor 7), .first, .first, (.firstFactor 7), (.secondFactor 17), .first, .first, .first, (.firstFactor 7), .first, (.firstFactor 13), .first, .first, .second, .first, .first]
set_option maxRecDepth 2048 in
theorem row11003_block_007_covers : Budget.RowUpperBound PrimePairVertices.tree_block_007 26034 11003 row11003_block_007 := by
  apply Budget.row_upper_of_plan _ _ _ plan11003_block_007
  · decide
  · rfl

def row11003_block_008 : List Nat := []
def plan11003_block_008 : List Budget.Choice := [.first, (.secondFactor 17), .first, .first, .first, (.firstFactor 7), .first, (.firstFactor 23), .first, .first, .first, .first, (.secondFactor 7), .first, (.firstFactor 7), (.secondFactor 13), .first, (.firstFactor 11), .first, .first, (.firstFactor 13), .first, .first, (.firstFactor 11), .first, (.secondFactor 7), (.firstFactor 29), .first, .first, .second, (.firstFactor 7), .first, (.secondFactor 7), .first, (.secondFactor 37), .first, (.firstFactor 17), .first, (.firstFactor 13), .first, .first, (.firstFactor 7), (.firstFactor 37), .first, .first, .first, .first, .first, (.firstFactor 13), (.firstFactor 11), .first, .first, (.secondFactor 13), .first, .second, (.firstFactor 7), (.firstFactor 13), .first, .first, (.secondFactor 13), .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside]
set_option maxRecDepth 2048 in
theorem row11003_block_008_covers : Budget.RowUpperBound PrimePairVertices.tree_block_008 26034 11003 row11003_block_008 := by
  apply Budget.row_upper_of_plan _ _ _ plan11003_block_008
  · decide
  · rfl

def row11003_block_009 : List Nat := []
theorem row11003_block_009_covers : Budget.RowUpperBound PrimePairVertices.tree_block_009 26034 11003 row11003_block_009 :=
  Budget.empty_row_after _ _ _ 16514 (fun q member => (Bounds.block_009 q member).1) (by decide)

def row11003_block_010 : List Nat := []
theorem row11003_block_010_covers : Budget.RowUpperBound PrimePairVertices.tree_block_010 26034 11003 row11003_block_010 :=
  Budget.empty_row_after _ _ _ 18562 (fun q member => (Bounds.block_010 q member).1) (by decide)

def row11003_block_011 : List Nat := []
theorem row11003_block_011_covers : Budget.RowUpperBound PrimePairVertices.tree_block_011 26034 11003 row11003_block_011 :=
  Budget.empty_row_after _ _ _ 20610 (fun q member => (Bounds.block_011 q member).1) (by decide)

def row11003_block_012 : List Nat := []
theorem row11003_block_012_covers : Budget.RowUpperBound PrimePairVertices.tree_block_012 26034 11003 row11003_block_012 :=
  Budget.empty_row_after _ _ _ 22658 (fun q member => (Bounds.block_012 q member).1) (by decide)

def row11003_block_013 : List Nat := []
theorem row11003_block_013_covers : Budget.RowUpperBound PrimePairVertices.tree_block_013 26034 11003 row11003_block_013 :=
  Budget.empty_row_after _ _ _ 24706 (fun q member => (Bounds.block_013 q member).1) (by decide)

def row11003_large_left_left : List Nat := row11003_small ++ row11003_block_005
theorem row11003_large_left_left_covers : Budget.RowUpperBound PrimePairVertices.tree_large_left_left 26034 11003 row11003_large_left_left :=
  Budget.RowUpperBound.branch row11003_small_covers row11003_block_005_covers

def row11003_large_left_right_right : List Nat := row11003_block_007 ++ row11003_block_008
theorem row11003_large_left_right_right_covers : Budget.RowUpperBound PrimePairVertices.tree_large_left_right_right 26034 11003 row11003_large_left_right_right :=
  Budget.RowUpperBound.branch row11003_block_007_covers row11003_block_008_covers

def row11003_large_left_right : List Nat := row11003_block_006 ++ row11003_large_left_right_right
theorem row11003_large_left_right_covers : Budget.RowUpperBound PrimePairVertices.tree_large_left_right 26034 11003 row11003_large_left_right :=
  Budget.RowUpperBound.branch row11003_block_006_covers row11003_large_left_right_right_covers

def row11003_large_left : List Nat := row11003_large_left_left ++ row11003_large_left_right
theorem row11003_large_left_covers : Budget.RowUpperBound PrimePairVertices.tree_large_left 26034 11003 row11003_large_left :=
  Budget.RowUpperBound.branch row11003_large_left_left_covers row11003_large_left_right_covers

def row11003_large_right_left : List Nat := row11003_block_009 ++ row11003_block_010
theorem row11003_large_right_left_covers : Budget.RowUpperBound PrimePairVertices.tree_large_right_left 26034 11003 row11003_large_right_left :=
  Budget.RowUpperBound.branch row11003_block_009_covers row11003_block_010_covers

def row11003_large_right_right_right : List Nat := row11003_block_012 ++ row11003_block_013
theorem row11003_large_right_right_right_covers : Budget.RowUpperBound PrimePairVertices.tree_large_right_right_right 26034 11003 row11003_large_right_right_right :=
  Budget.RowUpperBound.branch row11003_block_012_covers row11003_block_013_covers

def row11003_large_right_right : List Nat := row11003_block_011 ++ row11003_large_right_right_right
theorem row11003_large_right_right_covers : Budget.RowUpperBound PrimePairVertices.tree_large_right_right 26034 11003 row11003_large_right_right :=
  Budget.RowUpperBound.branch row11003_block_011_covers row11003_large_right_right_right_covers

def row11003_large_right : List Nat := row11003_large_right_left ++ row11003_large_right_right
theorem row11003_large_right_covers : Budget.RowUpperBound PrimePairVertices.tree_large_right 26034 11003 row11003_large_right :=
  Budget.RowUpperBound.branch row11003_large_right_left_covers row11003_large_right_right_covers

def row11003_large : List Nat := row11003_large_left ++ row11003_large_right
theorem row11003_large_covers : Budget.RowUpperBound PrimePairVertices.tree_large 26034 11003 row11003_large :=
  Budget.RowUpperBound.branch row11003_large_left_covers row11003_large_right_covers

def row11003 : List Nat := row11003_large
theorem row11003_covers : Budget.RowUpperBound PrimePairVertices.largeTree 26034 11003 row11003 := row11003_large_covers

def row11027_small : List Nat := []
theorem row11027_small_covers : Budget.RowUpperBound PrimePairVertices.tree_small 26034 11027 row11027_small :=
  Budget.empty_row_before _ _ _ 8389 (fun q member => (Bounds.small q member).2) (by decide)

def row11027_block_005 : List Nat := []
theorem row11027_block_005_covers : Budget.RowUpperBound PrimePairVertices.tree_block_005 26034 11027 row11027_block_005 :=
  Budget.empty_row_before _ _ _ 10370 (fun q member => (Bounds.block_005 q member).2) (by decide)

def row11027_block_006 : List Nat := [11087, 11903]
def plan11027_block_006 : List Budget.Choice := [.outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .first, (.firstFactor 7), .first, (.firstFactor 17), .first, .first, .possible, (.firstFactor 11), .first, (.secondFactor 7), .first, .first, .first, (.firstFactor 11), .first, (.firstFactor 17), .first, .second, .first, .first, .first, (.secondFactor 7), .first, .first, (.firstFactor 31), (.firstFactor 17), .first, .first, .first, .first, .first, .first, .first, (.firstFactor 7), .first, (.secondFactor 7), .first, (.firstFactor 7), (.firstFactor 29), (.secondFactor 7), (.firstFactor 11), .first, .first, (.firstFactor 31), .first, (.firstFactor 23), .first, (.firstFactor 11), .first, .first, .first, (.firstFactor 7), .first, .first, .first, (.firstFactor 17), .first, .first, (.firstFactor 13), .first, (.firstFactor 11), (.firstFactor 31), (.secondFactor 13), .first, (.firstFactor 17), .first, .first, .first, .first, .first, .first, .first, (.firstFactor 37), .first, (.firstFactor 17), (.secondFactor 7), .first, .first, (.firstFactor 7), .first, .first, (.firstFactor 13), .first, .first, .first, (.secondFactor 17), .first, (.firstFactor 7), .possible, (.firstFactor 13), .first, .second, .first, (.firstFactor 7), .first, .first, .first, (.secondFactor 13), .first, (.firstFactor 7), (.firstFactor 13), .first, (.secondFactor 31), .first, (.secondFactor 7), .first, .first, (.secondFactor 17), .first, .first, (.firstFactor 19), (.firstFactor 7), .first, .first, .first, (.firstFactor 13), (.firstFactor 7), .first, (.firstFactor 23), .first, (.secondFactor 31), (.secondFactor 13), .first, (.secondFactor 23), .first, .first, (.secondFactor 7), .first, .first, (.firstFactor 29), .first, (.firstFactor 11), .first, .first, .second, (.firstFactor 19), .first, (.firstFactor 11), .first, (.firstFactor 13), .first, .first, (.firstFactor 7), .first, (.firstFactor 11)]
set_option maxRecDepth 2048 in
theorem row11027_block_006_covers : Budget.RowUpperBound PrimePairVertices.tree_block_006 26034 11027 row11027_block_006 := by
  apply Budget.row_upper_of_plan _ _ _ plan11027_block_006
  · decide
  · rfl

def row11027_block_007 : List Nat := [12491, 12911, 13037, 13151, 13229, 14321]
def plan11027_block_007 : List Budget.Choice := [.first, .first, (.firstFactor 23), .first, .first, .second, (.firstFactor 11), .first, .possible, (.firstFactor 17), (.secondFactor 7), .first, .first, (.firstFactor 7), .second, .first, .first, .first, (.firstFactor 7), .first, .first, .first, .first, (.firstFactor 7), .first, .first, .first, (.secondFactor 23), .first, (.firstFactor 7), .first, (.firstFactor 19), (.firstFactor 13), .first, .first, (.firstFactor 23), .first, .first, (.firstFactor 11), .first, .first, .first, (.firstFactor 29), .first, (.firstFactor 11), (.firstFactor 7), .first, .first, .first, .first, .first, .first, (.firstFactor 19), .first, .possible, (.secondFactor 23), .first, (.firstFactor 13), (.firstFactor 11), .first, .first, .first, .first, .first, (.secondFactor 13), (.firstFactor 13), .first, (.firstFactor 11), .first, .first, .possible, .first, (.secondFactor 7), .first, .first, .first, .first, (.firstFactor 17), (.firstFactor 31), (.firstFactor 19), .first, .possible, .first, .first, .first, .first, .first, .first, (.secondFactor 7), .first, .possible, (.firstFactor 7), .first, (.secondFactor 7), .first, .first, .first, .first, (.firstFactor 13), .first, (.firstFactor 37), (.firstFactor 11), .first, (.firstFactor 7), .first, (.secondFactor 17), .first, .first, .first, (.secondFactor 37), .first, (.firstFactor 7), .first, .second, (.firstFactor 11), .first, (.firstFactor 29), (.secondFactor 17), .first, .first, .first, (.firstFactor 37), .first, (.firstFactor 7), .first, .first, .first, (.firstFactor 7), .first, .first, .first, .first, (.firstFactor 23), .first, .first, .first, .first, (.firstFactor 19), .first, .first, (.firstFactor 17), .first, .first, .first, .first, .first, (.secondFactor 7), (.firstFactor 13), .first, (.firstFactor 11), .first, (.firstFactor 7), .first, (.secondFactor 13), (.firstFactor 13), .first, .first, .first, (.secondFactor 23), (.secondFactor 29), .first, (.secondFactor 17), (.firstFactor 7), .first, (.firstFactor 11), .first, .first, .first, (.firstFactor 7), .first, (.firstFactor 29), .first, .first, .first, (.firstFactor 31), (.secondFactor 7), .first, (.firstFactor 7), .first, (.secondFactor 37), .first, .first, .first, (.firstFactor 19), (.secondFactor 19), .first, .second, .first, (.firstFactor 7), .first, (.secondFactor 31), (.firstFactor 7), .first, .first, .first, .first, .possible, .first, (.firstFactor 11), .first, .first, (.firstFactor 23), (.secondFactor 13), .first, .first, .first, .second, .first, (.firstFactor 31), .first, .first, .first, .first, .first]
set_option maxRecDepth 2048 in
theorem row11027_block_007_covers : Budget.RowUpperBound PrimePairVertices.tree_block_007 26034 11027 row11027_block_007 := by
  apply Budget.row_upper_of_plan _ _ _ plan11027_block_007
  · decide
  · rfl

def row11027_block_008 : List Nat := [14867]
def plan11027_block_008 : List Budget.Choice := [.first, .first, .first, (.secondFactor 7), .first, (.firstFactor 17), (.firstFactor 7), .first, .first, .first, (.firstFactor 13), .first, (.firstFactor 11), .first, (.secondFactor 13), (.firstFactor 7), .first, .second, (.firstFactor 13), .first, (.firstFactor 11), (.firstFactor 7), .first, (.secondFactor 13), .first, (.firstFactor 13), (.firstFactor 11), .first, .first, (.firstFactor 17), .first, (.firstFactor 7), .first, .first, (.secondFactor 7), .first, (.firstFactor 23), .first, (.secondFactor 7), .first, .first, .first, (.firstFactor 17), .first, .possible, .first, (.firstFactor 7), .first, .first, (.secondFactor 7), .first, .first, (.secondFactor 7), .first, (.firstFactor 13), (.secondFactor 19), .second, .first, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside]
set_option maxRecDepth 2048 in
theorem row11027_block_008_covers : Budget.RowUpperBound PrimePairVertices.tree_block_008 26034 11027 row11027_block_008 := by
  apply Budget.row_upper_of_plan _ _ _ plan11027_block_008
  · decide
  · rfl

def row11027_block_009 : List Nat := []
theorem row11027_block_009_covers : Budget.RowUpperBound PrimePairVertices.tree_block_009 26034 11027 row11027_block_009 :=
  Budget.empty_row_after _ _ _ 16514 (fun q member => (Bounds.block_009 q member).1) (by decide)

def row11027_block_010 : List Nat := []
theorem row11027_block_010_covers : Budget.RowUpperBound PrimePairVertices.tree_block_010 26034 11027 row11027_block_010 :=
  Budget.empty_row_after _ _ _ 18562 (fun q member => (Bounds.block_010 q member).1) (by decide)

def row11027_block_011 : List Nat := []
theorem row11027_block_011_covers : Budget.RowUpperBound PrimePairVertices.tree_block_011 26034 11027 row11027_block_011 :=
  Budget.empty_row_after _ _ _ 20610 (fun q member => (Bounds.block_011 q member).1) (by decide)

def row11027_block_012 : List Nat := []
theorem row11027_block_012_covers : Budget.RowUpperBound PrimePairVertices.tree_block_012 26034 11027 row11027_block_012 :=
  Budget.empty_row_after _ _ _ 22658 (fun q member => (Bounds.block_012 q member).1) (by decide)

def row11027_block_013 : List Nat := []
theorem row11027_block_013_covers : Budget.RowUpperBound PrimePairVertices.tree_block_013 26034 11027 row11027_block_013 :=
  Budget.empty_row_after _ _ _ 24706 (fun q member => (Bounds.block_013 q member).1) (by decide)

def row11027_large_left_left : List Nat := row11027_small ++ row11027_block_005
theorem row11027_large_left_left_covers : Budget.RowUpperBound PrimePairVertices.tree_large_left_left 26034 11027 row11027_large_left_left :=
  Budget.RowUpperBound.branch row11027_small_covers row11027_block_005_covers

def row11027_large_left_right_right : List Nat := row11027_block_007 ++ row11027_block_008
theorem row11027_large_left_right_right_covers : Budget.RowUpperBound PrimePairVertices.tree_large_left_right_right 26034 11027 row11027_large_left_right_right :=
  Budget.RowUpperBound.branch row11027_block_007_covers row11027_block_008_covers

def row11027_large_left_right : List Nat := row11027_block_006 ++ row11027_large_left_right_right
theorem row11027_large_left_right_covers : Budget.RowUpperBound PrimePairVertices.tree_large_left_right 26034 11027 row11027_large_left_right :=
  Budget.RowUpperBound.branch row11027_block_006_covers row11027_large_left_right_right_covers

def row11027_large_left : List Nat := row11027_large_left_left ++ row11027_large_left_right
theorem row11027_large_left_covers : Budget.RowUpperBound PrimePairVertices.tree_large_left 26034 11027 row11027_large_left :=
  Budget.RowUpperBound.branch row11027_large_left_left_covers row11027_large_left_right_covers

def row11027_large_right_left : List Nat := row11027_block_009 ++ row11027_block_010
theorem row11027_large_right_left_covers : Budget.RowUpperBound PrimePairVertices.tree_large_right_left 26034 11027 row11027_large_right_left :=
  Budget.RowUpperBound.branch row11027_block_009_covers row11027_block_010_covers

def row11027_large_right_right_right : List Nat := row11027_block_012 ++ row11027_block_013
theorem row11027_large_right_right_right_covers : Budget.RowUpperBound PrimePairVertices.tree_large_right_right_right 26034 11027 row11027_large_right_right_right :=
  Budget.RowUpperBound.branch row11027_block_012_covers row11027_block_013_covers

def row11027_large_right_right : List Nat := row11027_block_011 ++ row11027_large_right_right_right
theorem row11027_large_right_right_covers : Budget.RowUpperBound PrimePairVertices.tree_large_right_right 26034 11027 row11027_large_right_right :=
  Budget.RowUpperBound.branch row11027_block_011_covers row11027_large_right_right_right_covers

def row11027_large_right : List Nat := row11027_large_right_left ++ row11027_large_right_right
theorem row11027_large_right_covers : Budget.RowUpperBound PrimePairVertices.tree_large_right 26034 11027 row11027_large_right :=
  Budget.RowUpperBound.branch row11027_large_right_left_covers row11027_large_right_right_covers

def row11027_large : List Nat := row11027_large_left ++ row11027_large_right
theorem row11027_large_covers : Budget.RowUpperBound PrimePairVertices.tree_large 26034 11027 row11027_large :=
  Budget.RowUpperBound.branch row11027_large_left_covers row11027_large_right_covers

def row11027 : List Nat := row11027_large
theorem row11027_covers : Budget.RowUpperBound PrimePairVertices.largeTree 26034 11027 row11027 := row11027_large_covers

def row11047_small : List Nat := []
theorem row11047_small_covers : Budget.RowUpperBound PrimePairVertices.tree_small 26034 11047 row11047_small :=
  Budget.empty_row_before _ _ _ 8389 (fun q member => (Bounds.small q member).2) (by decide)

def row11047_block_005 : List Nat := []
theorem row11047_block_005_covers : Budget.RowUpperBound PrimePairVertices.tree_block_005 26034 11047 row11047_block_005 :=
  Budget.empty_row_before _ _ _ 10370 (fun q member => (Bounds.block_005 q member).2) (by decide)

def row11047_block_006 : List Nat := [11677, 12277]
def plan11047_block_006 : List Budget.Choice := [.outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .first, (.secondFactor 17), .first, (.firstFactor 31), (.firstFactor 7), .first, .first, (.firstFactor 11), .first, (.firstFactor 17), .first, (.secondFactor 37), .first, (.secondFactor 17), .first, .first, .first, (.secondFactor 7), .first, (.firstFactor 29), .first, (.firstFactor 7), (.firstFactor 13), .first, .first, .first, .first, .second, (.firstFactor 11), .first, .first, .first, .first, (.secondFactor 13), .first, .first, .first, .first, .first, .first, .first, (.firstFactor 11), .first, (.secondFactor 17), .first, .first, .first, (.firstFactor 13), .first, (.firstFactor 7), .first, (.firstFactor 17), .first, (.firstFactor 23), .first, (.firstFactor 7), (.secondFactor 29), .first, (.secondFactor 7), .first, .first, .first, .possible, .first, (.firstFactor 23), .first, (.firstFactor 37), .first, .first, (.firstFactor 17), (.secondFactor 7), .first, .second, .first, .first, .first, .first, .first, (.secondFactor 13), (.firstFactor 23), .first, (.firstFactor 17), (.firstFactor 7), .first, .first, (.secondFactor 23), .first, .first, .first, (.firstFactor 7), .first, .first, .first, (.secondFactor 29), (.secondFactor 7), (.firstFactor 13), .first, (.firstFactor 11), .first, .first, (.firstFactor 7), .first, (.firstFactor 11), .first, .first, (.firstFactor 7), .first, .first, (.secondFactor 31), .first, .first, (.firstFactor 29), .first, .first, .first, .first, .second, .first, (.secondFactor 7), .first, .first, (.secondFactor 13), .first, .first, (.firstFactor 17), .first, .first, .first, .first, .possible, .first, (.secondFactor 7), (.firstFactor 7), .first, .first, (.firstFactor 7), .first, (.firstFactor 31), .first, (.firstFactor 23), .first, .first, .second, .first]
set_option maxRecDepth 2048 in
theorem row11047_block_006_covers : Budget.RowUpperBound PrimePairVertices.tree_block_006 26034 11047 row11047_block_006 := by
  apply Budget.row_upper_of_plan _ _ _ plan11047_block_006
  · decide
  · rfl

def row11047_block_007 : List Nat := [13933, 14293]
def plan11047_block_007 : List Budget.Choice := [(.firstFactor 19), (.firstFactor 11), .first, .first, (.firstFactor 29), .first, .first, (.secondFactor 17), .first, .first, .first, (.firstFactor 7), (.firstFactor 23), .first, .first, (.secondFactor 7), (.firstFactor 17), (.firstFactor 7), .first, (.secondFactor 23), (.firstFactor 13), (.firstFactor 37), (.secondFactor 13), .first, .first, .first, (.firstFactor 7), .first, .first, .first, .first, .first, .first, (.firstFactor 11), (.secondFactor 37), .first, (.firstFactor 7), (.firstFactor 13), .first, (.secondFactor 13), (.firstFactor 7), .second, .first, .first, .first, .first, .second, (.firstFactor 11), (.secondFactor 31), (.firstFactor 17), (.firstFactor 7), .first, .first, .first, .first, .first, (.secondFactor 7), .first, .first, .first, .first, .first, (.firstFactor 7), (.firstFactor 29), .first, .first, (.secondFactor 7), .first, .second, (.firstFactor 37), .first, .first, .first, (.secondFactor 19), (.firstFactor 11), (.firstFactor 7), .first, .first, .first, .first, (.secondFactor 13), .first, (.firstFactor 11), .first, (.secondFactor 7), (.secondFactor 19), (.firstFactor 7), .first, .first, (.firstFactor 19), .first, .first, .first, .first, (.firstFactor 7), (.firstFactor 11), (.secondFactor 7), (.firstFactor 7), .first, (.firstFactor 29), .first, .first, (.secondFactor 7), .first, (.secondFactor 7), .first, (.secondFactor 31), .first, .first, .first, (.firstFactor 13), .first, .first, .first, .first, (.firstFactor 7), .first, .first, .first, .first, (.secondFactor 13), .first, (.firstFactor 17), .first, (.secondFactor 7), (.firstFactor 13), .first, .first, .second, (.secondFactor 7), .first, (.firstFactor 17), .first, (.secondFactor 23), (.firstFactor 7), .first, (.secondFactor 13), .first, .first, (.secondFactor 17), .first, .first, (.firstFactor 7), .first, .first, (.firstFactor 23), .first, .first, (.firstFactor 19), .first, .first, .first, (.firstFactor 13), .first, .first, (.firstFactor 17), .first, .first, .first, .first, (.firstFactor 19), .first, .first, (.firstFactor 37), .first, .possible, .first, .first, .first, .second, .first, (.secondFactor 7), (.secondFactor 29), .first, .first, .first, .second, .first, (.firstFactor 11), .first, (.firstFactor 7), (.firstFactor 13), (.firstFactor 7), .first, .first, (.firstFactor 23), .first, (.firstFactor 29), .first, (.firstFactor 13), .first, .first, .first, (.firstFactor 11), .possible, .first, .first, (.secondFactor 17), .first, .first, (.firstFactor 11), .first, .first, (.secondFactor 7), (.firstFactor 7), .first, .first, (.firstFactor 31), .first, (.secondFactor 7), .first, .first, (.firstFactor 23), .first]
set_option maxRecDepth 2048 in
theorem row11047_block_007_covers : Budget.RowUpperBound PrimePairVertices.tree_block_007 26034 11047 row11047_block_007 := by
  apply Budget.row_upper_of_plan _ _ _ plan11047_block_007
  · decide
  · rfl

def row11047_block_008 : List Nat := []
def plan11047_block_008 : List Budget.Choice := [(.firstFactor 11), .first, .first, .first, (.firstFactor 13), .first, .first, .first, (.secondFactor 13), (.secondFactor 7), .first, .first, .first, .first, .first, .first, (.secondFactor 13), .first, .first, (.firstFactor 7), .first, .first, (.secondFactor 7), .first, .second, .first, .first, (.secondFactor 17), (.firstFactor 7), .first, .first, .first, .first, (.firstFactor 13), .first, (.firstFactor 7), .first, .second, .first, (.firstFactor 7), .first, .first, .first, (.secondFactor 7), .first, .first, .first, (.secondFactor 19), .first, .first, (.firstFactor 13), (.firstFactor 19), .first, (.firstFactor 7), .first, .first, .first, .second, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside]
set_option maxRecDepth 2048 in
theorem row11047_block_008_covers : Budget.RowUpperBound PrimePairVertices.tree_block_008 26034 11047 row11047_block_008 := by
  apply Budget.row_upper_of_plan _ _ _ plan11047_block_008
  · decide
  · rfl

def row11047_block_009 : List Nat := []
theorem row11047_block_009_covers : Budget.RowUpperBound PrimePairVertices.tree_block_009 26034 11047 row11047_block_009 :=
  Budget.empty_row_after _ _ _ 16514 (fun q member => (Bounds.block_009 q member).1) (by decide)

def row11047_block_010 : List Nat := []
theorem row11047_block_010_covers : Budget.RowUpperBound PrimePairVertices.tree_block_010 26034 11047 row11047_block_010 :=
  Budget.empty_row_after _ _ _ 18562 (fun q member => (Bounds.block_010 q member).1) (by decide)

def row11047_block_011 : List Nat := []
theorem row11047_block_011_covers : Budget.RowUpperBound PrimePairVertices.tree_block_011 26034 11047 row11047_block_011 :=
  Budget.empty_row_after _ _ _ 20610 (fun q member => (Bounds.block_011 q member).1) (by decide)

def row11047_block_012 : List Nat := []
theorem row11047_block_012_covers : Budget.RowUpperBound PrimePairVertices.tree_block_012 26034 11047 row11047_block_012 :=
  Budget.empty_row_after _ _ _ 22658 (fun q member => (Bounds.block_012 q member).1) (by decide)

def row11047_block_013 : List Nat := []
theorem row11047_block_013_covers : Budget.RowUpperBound PrimePairVertices.tree_block_013 26034 11047 row11047_block_013 :=
  Budget.empty_row_after _ _ _ 24706 (fun q member => (Bounds.block_013 q member).1) (by decide)

def row11047_large_left_left : List Nat := row11047_small ++ row11047_block_005
theorem row11047_large_left_left_covers : Budget.RowUpperBound PrimePairVertices.tree_large_left_left 26034 11047 row11047_large_left_left :=
  Budget.RowUpperBound.branch row11047_small_covers row11047_block_005_covers

def row11047_large_left_right_right : List Nat := row11047_block_007 ++ row11047_block_008
theorem row11047_large_left_right_right_covers : Budget.RowUpperBound PrimePairVertices.tree_large_left_right_right 26034 11047 row11047_large_left_right_right :=
  Budget.RowUpperBound.branch row11047_block_007_covers row11047_block_008_covers

def row11047_large_left_right : List Nat := row11047_block_006 ++ row11047_large_left_right_right
theorem row11047_large_left_right_covers : Budget.RowUpperBound PrimePairVertices.tree_large_left_right 26034 11047 row11047_large_left_right :=
  Budget.RowUpperBound.branch row11047_block_006_covers row11047_large_left_right_right_covers

def row11047_large_left : List Nat := row11047_large_left_left ++ row11047_large_left_right
theorem row11047_large_left_covers : Budget.RowUpperBound PrimePairVertices.tree_large_left 26034 11047 row11047_large_left :=
  Budget.RowUpperBound.branch row11047_large_left_left_covers row11047_large_left_right_covers

def row11047_large_right_left : List Nat := row11047_block_009 ++ row11047_block_010
theorem row11047_large_right_left_covers : Budget.RowUpperBound PrimePairVertices.tree_large_right_left 26034 11047 row11047_large_right_left :=
  Budget.RowUpperBound.branch row11047_block_009_covers row11047_block_010_covers

def row11047_large_right_right_right : List Nat := row11047_block_012 ++ row11047_block_013
theorem row11047_large_right_right_right_covers : Budget.RowUpperBound PrimePairVertices.tree_large_right_right_right 26034 11047 row11047_large_right_right_right :=
  Budget.RowUpperBound.branch row11047_block_012_covers row11047_block_013_covers

def row11047_large_right_right : List Nat := row11047_block_011 ++ row11047_large_right_right_right
theorem row11047_large_right_right_covers : Budget.RowUpperBound PrimePairVertices.tree_large_right_right 26034 11047 row11047_large_right_right :=
  Budget.RowUpperBound.branch row11047_block_011_covers row11047_large_right_right_right_covers

def row11047_large_right : List Nat := row11047_large_right_left ++ row11047_large_right_right
theorem row11047_large_right_covers : Budget.RowUpperBound PrimePairVertices.tree_large_right 26034 11047 row11047_large_right :=
  Budget.RowUpperBound.branch row11047_large_right_left_covers row11047_large_right_right_covers

def row11047_large : List Nat := row11047_large_left ++ row11047_large_right
theorem row11047_large_covers : Budget.RowUpperBound PrimePairVertices.tree_large 26034 11047 row11047_large :=
  Budget.RowUpperBound.branch row11047_large_left_covers row11047_large_right_covers

def row11047 : List Nat := row11047_large
theorem row11047_covers : Budget.RowUpperBound PrimePairVertices.largeTree 26034 11047 row11047 := row11047_large_covers

end ProjectEulerSolutions.PrimePairLarge
