class A:
    def __getitem__(self,key):
        return getattr(self,key)
    def __setitem__(self,key,value):
        setattr(self,key,value)
    def __delitem__(self,key):
        delattr(self,key)
dic = A()
dic["name"] = "华山大弟子"
