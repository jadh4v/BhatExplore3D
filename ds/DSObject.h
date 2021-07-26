#pragma once 

namespace sjDS{

class DSObject
{
public:
    DSObject() { }
    //virtual ~DSObject() {}
    bool ValidConstruction() const { return m_ValidConstruction; }

protected:
    virtual void SetValidConstruction()   final   { m_ValidConstruction = true;  }
    virtual void SetInvalidConstruction() final   { m_ValidConstruction = false; }

private:
    bool m_ValidConstruction = true;

};

}

